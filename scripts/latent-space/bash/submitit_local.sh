python scripts/latent-space/train_cae.py --multirun hydra/launcher=submitit_local launcher=local \
model.latent_n_channels=24,32,48,64