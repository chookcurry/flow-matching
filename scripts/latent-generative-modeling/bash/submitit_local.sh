python scripts/autoencoder/train.py --multirun hydra/launcher=submitit_local \
hydra.launcher.n_jobs=-1 \
model.architecure=ae,cae,aec,caec \
model.latent_n_channels=24,32,48,64 