python scripts/autoencoder/train.py --multirun hydra/launcher=slurm \
model.architecure=cae \
model.size_latent=4,8 \
model.num_channels_latent=24,28,32,36,40,44,48,52,56,60,64 \