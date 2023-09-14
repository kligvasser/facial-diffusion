# Facial Diffussion

Installing:
```
conda env create -f environment.yaml
conda activate diffusers
```

Optional:
```
pip install -U xformers
```

Running:
```
accelerate launch --config_file configs/accelerate.yaml train_conditioned.py --config ./configs/ffhq-vae-dense.yaml
```