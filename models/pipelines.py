import torch
import inspect

from typing import List, Optional, Tuple, Union
from diffusers.utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class ConditionalDDPMPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler, vqvae):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vqvae=vqvae)

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        # check if the scheduler accepts generator
        accepts_generator = 'generator' in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            self.unet.config.in_channels,
            self.unet.config.sample_size,
            self.unet.config.sample_size,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents):
        latents /= self.vqvae.config.scaling_factor
        images = self.vqvae.decode(latents.type(self.vqvae.dtype), return_dict=False)[0]
        return images

    @torch.no_grad()
    def __call__(
        self,
        encoder_hidden_states,
        num_inference_steps: int = 100,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # Prepare latent variables
        device = self._execution_device
        batch_size = len(encoder_hidden_states)
        latents = self.prepare_latents(
            batch_size,
            self.unet.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype

        # Define call parameters
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # Prepare hidden states
        encoder_hidden_states = encoder_hidden_states.to(device).type(latents_dtype)

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        for t in self.progress_bar(self.scheduler.timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            # predict the noise residual
            noise_prediction = self.unet(latent_model_input, t, encoder_hidden_states).sample.to(
                dtype=latents_dtype
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_prediction, t, latents, **extra_step_kwargs
            ).prev_sample

        if self.vqvae:
            images = self.decode_latents(latents)
        else:
            images = latents

        if output_type == 'torch':
            return images.cpu()

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            images = self.numpy_to_pil(images)

        if not return_dict:
            return (images,)

        return ImagePipelineOutput(images=images)
