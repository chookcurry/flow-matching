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

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg: DictConfig) -> None:
    run_id = HydraConfig.get().job.name
    output_dir = HydraConfig.get().runtime.output_dir

    log.info(f"run id: {run_id}")
    log.info(f"output dir: {output_dir}")

    log.info(OmegaConf.to_yaml(cfg))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    log.info(f"device: {device}")

    dataset_cfg = get_whar_cfg(
        WHARDatasetID(cfg.data.dataset_id),
        datasets_dir="datasets",
        cache_dir=f"{output_dir}/cache",
    )

    dataset = PytorchAdapter(dataset_cfg)

    train_loader, val_loader, _ = dataset.get_dataloaders(
        train_batch_size=cfg.train.batch_size, scv_group_index=cfg.data.scv_group_index
    )

    num_classes = len(dataset.get_class_weights(train_loader))

    cae: AE | CAE | AEC | CAEC

    match cfg.model.architecure:
        case "ae":
            cae = SpectrogramAE(
                spect_c=cfg.model.spect_n_channels,
                latent_c=cfg.model.latent_n_channels,
            )
        case "aec":
            cae = SpectrogramAEC(
                spect_c=cfg.model.spect_n_channels,
                latent_c=cfg.model.latent_n_channels,
            )
        case "cae":
            cae = SpectrogramCAE(latent_c=cfg.model.latent_n_channels)
        case "caec":
            cae = SpectrogramCAEC(latent_c=cfg.model.latent_n_channels)
        case _:
            raise NotImplementedError

    trainer = CAETrainer(
        model=cae,
        train_loader=train_loader,
        val_loader=val_loader,
        eta=(1 / num_classes),
        null_class=num_classes,
        track=False,
    )

    trainer.train(
        num_epochs=cfg.train.num_epochs, device=device, lr=cfg.train.learning_rate
    )

    torch.save(cae.state_dict(), f"{output_dir}/ae.pt")


if __name__ == "__main__":
    run()
