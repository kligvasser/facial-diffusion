import argparse
import logging
import math
import os
import omegaconf
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import VQModel, DDIMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from PIL import Image

import models
import data

from models import ConditionalDDPMPipeline

logger = get_logger(__name__, log_level='INFO')


def get_arguments():
    example_text = '''
    example:
        accelerate launch --config_file configs/accelerate.yaml train_conditioned.py --config configs/ffhq-vqvae-clip-landmark-fp16.yaml
    '''
    parser = argparse.ArgumentParser(description='Facial conditional training', epilog=example_text)
    parser.add_argument(
        '--config', type=str, default='configs/ffhq-vqvae-clip-landmark-arcface-fp16.yaml'
    )
    args = parser.parse_args()

    return args


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def config_to_dict(config):
    config_dict = dict(config)

    def to_dict_saveable(value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, omegaconf.listconfig.ListConfig):
            value = list(value)
        return value

    config_dict = {k: to_dict_saveable(v) for k, v in config_dict.items()}

    return config_dict


def evaluate(config, epoch, pipeline, generator, encoder_hidden_states, inputs=None):
    images = pipeline(
        encoder_hidden_states=encoder_hidden_states,
        generator=generator,
        num_inference_steps=config.training.num_inference_steps,
    )[0]

    # Make a grid out of the images
    image_grid = make_grid(images, rows=config.validation_data.batch_size // 4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, 'samples')
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save('{}/{:04d}-sample.png'.format(test_dir, epoch + 1))

    if not inputs is None:
        inputs = (inputs / 2 + 0.5).clamp(0, 1)
        inputs = inputs.cpu().permute(0, 2, 3, 1).float().numpy()
        inputs = pipeline.numpy_to_pil(inputs)

        image_grid = make_grid(inputs, rows=config.validation_data.batch_size // 4, cols=4)
        image_grid.save('{}/{:04d}-input.png'.format(test_dir, epoch + 1))


def main(config):
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs('{}/samples'.format(config.output_dir), exist_ok=True)
        omegaconf.OmegaConf.save(config, os.path.join(config.output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**config_to_dict(config.scheduler.configuration))
    unet = UNet2DConditionModel(**config_to_dict(config.unet.configuration))

    if config.training.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError('xformers is not available. Make sure it is installed correctly')

    learning_rate = config.training.learning_rate
    if config.training.scale_lr:
        learning_rate = (
            config.training.learning_rate
            * config.training.gradient_accumulation_steps
            * config.train_data.batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.adam_weight_decay,
        eps=config.training.adam_epsilon,
    )

    # Get the training dataset
    train_dataset = data.__dict__[config.train_data.model_name]
    train_dataset = train_dataset(**config_to_dict(config.train_data.configuration))

    eval_dataset = data.__dict__[config.validation_data.model_name]
    eval_dataset = eval_dataset(**config_to_dict(config.validation_data.configuration))

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_data.batch_size,
        num_workers=config.train_data.num_workers,
        shuffle=True,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.validation_data.batch_size,
        shuffle=True,
    )

    # Scheduler
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader)
        / config.training.gradient_accumulation_steps
        / accelerator.num_processes
    )
    max_train_steps = num_update_steps_per_epoch * config.training.num_epochs

    lr_scheduler = get_scheduler(
        config.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.lr_warmup_steps
        * config.training.gradient_accumulation_steps,
        num_training_steps=max_train_steps * config.training.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16

    # VQVAE
    if 'vqvae' in config:
        vqvae = VQModel.from_pretrained(config.vqvae.model_name, subfolder='vqvae')
        vqvae.requires_grad_(False)
        vqvae.to(accelerator.device, dtype=weight_dtype)
    else:
        vqvae = None

    # Embbeder
    embedder = models.__dict__[config.embbeder.model_name]
    embedder = embedder(**config_to_dict(config.embbeder.configuration))
    embedder.requires_grad_(False)
    embedder.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers('pixel-training')

    # Train!
    total_batch_size = (
        config.train_data.batch_size
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )

    logger.info('***** Running training *****')
    logger.info(' Num examples = {}'.format(len(train_dataset)))
    logger.info(' Num Epochs = {}'.format(config.training.num_epochs))
    logger.info(' Instantaneous batch size per device = {}'.format(config.train_data.batch_size))
    logger.info(
        ' Total train batch size (w. parallel, distributed & accumulation) = {}'.format(
            total_batch_size
        )
    )
    logger.info(
        ' Gradient Accumulation steps = {}'.format(config.training.gradient_accumulation_steps)
    )
    logger.info(' Total optimization steps = {}'.format(max_train_steps))
    logger.info(
        ' Number of parameters in unet: {}'.format(sum([l.nelement() for l in unet.parameters()]))
    )

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if config.training.resume_from_checkpoint:
        if config.training.resume_from_checkpoint != 'latest':
            path = config.training.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith('checkpoint')]
            dirs = sorted(dirs, key=lambda x: int(x.split('-')[1]))
            path = dirs[-1]
            path = os.path.join(config.output_dir, path)
        accelerator.print('Resuming from checkpoint {}'.format(path))
        accelerator.load_state(path)
        global_step = int(path.split('checkpoint-')[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description('Steps')

    for epoch in range(first_epoch, config.training.num_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                config.training.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % config.training.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                pixel_values = batch['image'].to(weight_dtype)

                conditioned_values = dict()
                conditioned_values['image'] = batch['image'].float()
                for key in config.embbeder.extract_keys:
                    conditioned_values[key] = batch[key].float()

                with torch.no_grad():
                    encoder_hidden_states = embedder(conditioned_values).to(weight_dtype)

                if vqvae:
                    latents = (
                        vqvae.encode(pixel_values, return_dict=False)[0]
                        * vqvae.config.scaling_factor
                    )
                else:
                    latents = pixel_values

                noise = torch.randn_like(latents)
                bs = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (bs,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        'Unknown prediction type {}'.format(noise_scheduler.prediction_type)
                    )

                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states, return_dict=False
                )[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.train_data.batch_size)).mean()
                train_loss += avg_loss.item() / config.training.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({'train_loss': train_loss}, step=global_step)
                train_loss = 0.0

            logs = {'step_loss': loss.detach().item(), 'lr': lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if (
                (epoch + 1) % config.training.save_image_epochs == 0
                or epoch == config.training.num_epochs - 1
            ):
                pipeline = ConditionalDDPMPipeline(
                    unet=accelerator.unwrap_model(unet),
                    scheduler=noise_scheduler,
                    vqvae=vqvae,
                )
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(config.training.seed)

                batch = next(iter(eval_dataloader))
                conditioned_values = dict()
                conditioned_values['image'] = batch['image'].float().to(latents.device)
                for key in config.embbeder.extract_keys:
                    conditioned_values[key] = batch[key].float().to(latents.device)
                with torch.no_grad():
                    encoder_hidden_states = embedder(conditioned_values).to(weight_dtype)
                evaluate(config, epoch, pipeline, generator, encoder_hidden_states, batch['image'])

            if (
                (epoch + 1) % config.training.save_model_epochs == 0
                or epoch == config.training.num_epochs - 1
            ):
                save_path = os.path.join(config.output_dir, 'checkpoint-{:04d}'.format(epoch + 1))
                accelerator.save_state(save_path)
                logger.info('Saved state to {}'.format(save_path))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = ConditionalDDPMPipeline(unet=unet, scheduler=noise_scheduler, vqvae=vqvae)
        pipeline.save_pretrained(os.path.join(config.output_dir, 'last-checkpoint'))

    accelerator.end_training()


if __name__ == '__main__':
    args = get_arguments()
    config = omegaconf.OmegaConf.load(args.config)

    main(config)
