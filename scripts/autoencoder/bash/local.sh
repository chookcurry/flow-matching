python scripts/autoencoder/train.py --multirun hydra/launcher=submitit_local \
model.architecure=cae \
model.num_channels_latent=24 