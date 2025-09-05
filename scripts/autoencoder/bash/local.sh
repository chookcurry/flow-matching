python scripts/autoencoder/train.py --multirun hydra/launcher=submitit_local \
model.architecure=cae \
model.latent_n_channels=24 