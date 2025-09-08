python scripts/autoencoder/train.py --multirun hydra/launcher=joblib \
+hydra.launcher.joblib.backend=multiprocessing \
hydra.launcher.n_jobs=-1 \
model.architecure=cae,aec,caec \
model.num_channels_latent=24,32,48,64 