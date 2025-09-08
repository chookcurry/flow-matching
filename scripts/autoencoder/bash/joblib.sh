python scripts/autoencoder/train.py --multirun hydra/launcher=joblib \
+hydra.launcher.joblib.backend=multiprocessing \
hydra.launcher.n_jobs=-1 \
model.architecure=cae,aec,caec \
model.latent_n_channels=24,32,48,64 