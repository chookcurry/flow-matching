python scripts/autoencoder/train.py --multirun hydra/launcher=basic \
wandb.project=debug \
train.num_epochs=1 \
model.architecure=cae \
model.num_channels_latent=24