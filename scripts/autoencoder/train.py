import json
import logging
import os
import shutil
from math import log2

import hydra
import torch
import wandb
from diffusion.architectures.latent.autoencoder import AE, AEC, CAE, CAEC
from diffusion.whar.models.autoencoder_dynamic import (
    SpectrogramAE,
    SpectrogramAEC,
    SpectrogramCAE,
    SpectrogramCAEC,
)
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from whar_datasets.adapters.pytorch import PytorchAdapter
from whar_datasets.support.getter import WHARDatasetID, get_whar_cfg

from diffusion.training.trainer_cae import CAETrainer

# Set up logging
log = logging.getLogger(__name__)

# Load environment variables
load_dotenv(".env")


def make_model_name(cfg: DictConfig) -> str:
    return "_".join(
        [
            cfg.model.architecure,  # e.g., cae
            f"nc{cfg.model.num_channels_spect}",  # number of input channels
            f"lc{cfg.model.num_channels_latent}",  # latent channels
            f"ls{cfg.model.size_latent}",  # latent size
        ]
    )


@hydra.main(version_base=None, config_path="config", config_name="config")
def run_script(cfg: DictConfig) -> None:
    # Login to wandb
    wandb.login(key=os.environ["WANDB_API_KEY"])

    # Set up device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # Set up run
    run_id = make_model_name(cfg)  # HydraConfig.get().job.id
    output_dir = HydraConfig.get().runtime.output_dir
    # cache_dir = f"{output_dir}/cache"
    datasets_dir = os.environ.get("DATASETS_DIR") or cfg.data.datasets_dir
    num_downsamples = int(log2(cfg.model.size_spect / cfg.model.size_latent))
    # cfg.model.size_latent * 2 ** num_downsamples == cfg.model.size_spect
    cfg_dict: dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore

    # Log info
    log.info(f"datasets_dir: {datasets_dir}")
    log.info(f"device: {device}")
    log.info(f"run id: {run_id}")
    log.info(f"output dir: {output_dir}")
    log.info(f"cfg: {OmegaConf.to_yaml(cfg, resolve=True)}")
    log.info(f"num_downsamples: {num_downsamples}")

    # Load and configure dataset
    dataset_cfg = get_whar_cfg(WHARDatasetID(cfg.data.dataset_id), datasets_dir)
    dataset_cfg.seed = cfg.train.seed
    dataset_cfg.in_memory = True
    dataset_cfg.in_parallel = False
    dataset = PytorchAdapter(dataset_cfg)

    # Get dataloaders
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        train_batch_size=cfg.train.batch_size, scv_group_index=cfg.data.scv_group_index
    )
    num_classes = len(dataset.get_class_weights(train_loader))

    # Initialize model
    ae: AE | CAE | AEC | CAEC
    match cfg.model.architecure:
        case "ae":
            ae = SpectrogramAE(
                num_channels_spect=cfg.model.num_channels_spect,
                num_channels_latent=cfg.model.num_channels_latent,
                num_downsamples=num_downsamples,
            )
        case "aec":
            ae = SpectrogramAEC(
                num_channels_spect=cfg.model.num_channels_spect,
                num_channels_latent=cfg.model.num_channels_latent,
                num_classes=num_classes + 1,  # due to null class
                embedding_dim=cfg.model.embedding_dim,
                num_downsamples=num_downsamples,
            )
        case "cae":
            ae = SpectrogramCAE(
                num_channels_spect=cfg.model.num_channels_spect,
                num_channels_latent=cfg.model.num_channels_latent,
                num_classes=num_classes + 1,  # due to null class
                embedding_dim=cfg.model.embedding_dim,
                num_downsamples=num_downsamples,
            )
        case "caec":
            ae = SpectrogramCAEC(
                num_channels_spect=cfg.model.num_channels_spect,
                num_channels_latent=cfg.model.num_channels_latent,
                num_classes=num_classes + 1,  # due to null class
                embedding_dim=cfg.model.embedding_dim,
                num_downsamples=num_downsamples,
            )
        case _:
            raise NotImplementedError

    # Initialize trainer
    trainer = CAETrainer(
        model=ae,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        eta=(1 / num_classes),
        null_class=num_classes,
    )

    # Initialize wandb run
    run = wandb.init(
        project=cfg.wandb.project,
        name=run_id,
        group=cfg.data.dataset_id,
        config=cfg_dict,
        job_type="train",
    )

    # Perform training
    ae_state_dict = trainer.train(
        num_epochs=cfg.train.num_epochs,
        device=device,
        lr=cfg.train.learning_rate,
        patience=cfg.train.patience,
        run=run,
    )

    # Perform eval
    metrics = trainer.eval(device)

    # Save and log autoencoder
    ae_path = f"{output_dir}/ae.pt"
    torch.save(ae_state_dict, ae_path)
    run.log_artifact(ae_path, name=f"model_{run_id}", type="model")

    # save and log metrics
    metrics_path = f"{output_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    run.log_artifact(metrics_path, name=f"metrics_{run_id}", type="metrics")

    # Save and log config
    cfg_path = f"{output_dir}/config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f)
    run.log_artifact(cfg_path, name=f"config_{run_id}", type="config")

    # Clean up
    run.finish()
    # shutil.rmtree(cache_dir)


if __name__ == "__main__":
    run_script()
