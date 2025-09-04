import os
import hydra
import torch
import logging
from omegaconf import DictConfig, OmegaConf
from whar_datasets.adapters.pytorch import PytorchAdapter
from whar_datasets.support.getter import WHARDatasetID, get_whar_cfg
from flow_matching.latent.autoencoder import AE, AEC, CAE, CAEC
from flow_matching.latent.training_cae import CAETrainer
from flow_matching.whar.models.autoencoder import (
    SpectrogramAE,
    SpectrogramCAE,
    SpectrogramAEC,
    SpectrogramCAEC,
)
from hydra.core.hydra_config import HydraConfig
import shutil
import wandb
from dotenv import load_dotenv

# Set up logging
log = logging.getLogger(__name__)

# Load environment variables
load_dotenv(".env")


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
    run_id = HydraConfig.get().job.id
    output_dir = HydraConfig.get().runtime.output_dir
    cache_dir = f"{output_dir}/cache"

    # Log info
    log.info(f"device: {device}")
    log.info(f"run id: {run_id}")
    log.info(f"output dir: {output_dir}")
    log.info(OmegaConf.to_yaml(cfg))

    # Load dataset
    dataset = PytorchAdapter(
        get_whar_cfg(
            WHARDatasetID(cfg.data.dataset_id),
            datasets_dir="datasets",
            cache_dir=cache_dir,
        )
    )

    # Get dataloaders
    train_loader, val_loader, _ = dataset.get_dataloaders(
        train_batch_size=cfg.train.batch_size, scv_group_index=cfg.data.scv_group_index
    )

    # Get number of classes
    num_classes = len(dataset.get_class_weights(train_loader))

    # Initialize model
    ae: AE | CAE | AEC | CAEC
    match cfg.model.architecure:
        case "ae":
            ae = SpectrogramAE(
                spect_c=cfg.model.spect_n_channels,
                latent_c=cfg.model.latent_n_channels,
            )
        case "aec":
            ae = SpectrogramAEC(
                spect_c=cfg.model.spect_n_channels,
                latent_c=cfg.model.latent_n_channels,
            )
        case "cae":
            ae = SpectrogramCAE(latent_c=cfg.model.latent_n_channels)
        case "caec":
            ae = SpectrogramCAEC(latent_c=cfg.model.latent_n_channels)
        case _:
            raise NotImplementedError

    # Initialize trainer
    trainer = CAETrainer(
        model=ae,
        train_loader=train_loader,
        val_loader=val_loader,
        eta=(1 / num_classes),
        null_class=num_classes,
    )

    # Initialize wandb run
    run = wandb.init(
        project=cfg.wandb.project,
        job_type="train",
        name=run_id,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
    )

    # Perform training
    trainer.train(
        num_epochs=cfg.train.num_epochs,
        device=device,
        lr=cfg.train.learning_rate,
        run=run,
    )

    # Save autoencoder
    ae_path = f"{output_dir}/ae.pt"
    torch.save(ae.state_dict(), ae_path)

    # Log autoencoder
    run.log_artifact(ae_path, name="ae", type="model")
    run.finish()

    # Clean up
    shutil.rmtree(cache_dir)


if __name__ == "__main__":
    run_script()
