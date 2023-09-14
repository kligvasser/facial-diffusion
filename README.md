# Facial Diffussion
Welcome to the Conditional Diffusion Model Training Repository!


Explore the world of diffusion models with conditional input. Several trained models are available in the Diffuser Hub. For detailed guidance, refer to the "Run-a-Model" notebook provided. Additionally, you can easily access and download the Facial 40 Attributes model from this [repository](https://github.com/kligvasser/facial-attributes). Here, "landmark" are 468 XYZ facial points extracted by the Google [MediaPipe](https://github.com/google/mediapipe) model. 


Input condition (clip-landmark-arcface):
![Screenshot](images/input.png)

Sample:
![Screenshot](images/output.png)

Installing:
```
conda env create -f environment.yaml
conda activate diffusers
```

Optional:
```
pip install -U xformers
```

Training:
```
accelerate launch --config_file configs/accelerate.yaml train_conditioned.py --config ./configs/ffhq-vqvae-clip.yaml
