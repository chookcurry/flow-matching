python scripts/autoencoder/train.py --multirun hydra/launcher=joblib \
hydra.launcher.n_jobs=-1 \
model.architecure=cae \
model.latent_n_channels=24,64 