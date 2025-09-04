python scripts/latent-space/train.py --multirun hydra/launcher=submitit_local launcher=local \
model.architecure=ae,cae,aec,caec \
model.latent_n_channels=24,32,48,64