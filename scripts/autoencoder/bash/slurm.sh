python scripts/autoencoder/train.py --multirun hydra/launcher=slurm \
hydra.launcher.partition=normal \
hydra.launcher.nodes=1 \
hydra.launcher.tasks_per_node=1 \
hydra.launcher.gpus_per_node=1 \
hydra.launcher.timeout_min=60 \
model.architecure=cae \
model.num_channels_latent=24