import os
import json
import logging
import torch
from math import log2
from matplotlib import pyplot as plt
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig

from whar_datasets.adapters.pytorch import PytorchAdapter
from whar_datasets.support.getter import WHARDatasetID, get_whar_cfg
from flow_matching.latent.autoencoder import AE, AEC, CAE, CAEC
from flow_matching.latent.training_cae import collate_fn, detransform
from flow_matching.whar.models.autoencoder_dynamic import (
    SpectrogramAE,
    SpectrogramCAE,
    SpectrogramAEC,
    SpectrogramCAEC,
)
from flow_matching.whar.stft import decompress_stft, plot_spectrogram_grid

# -------------------
# Setup
# -------------------
project = "scripts/autoencoder/artifacts:v0"
model = "cae_nc18_lc20_ls4"
config_path = f"{project}/config_{model}/config.json"
model_path = f"{project}/model_{model}/ae.pt"

# Logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv(".env")


def make_model_name(cfg: DictConfig) -> str:
    return "_".join(
        [
            cfg.model.architecure,  # e.g., cae
            f"nc{cfg.model.num_channels_spect}",
            f"lc{cfg.model.num_channels_latent}",
            f"ls{cfg.model.size_latent}",
        ]
    )


def run_script(cfg: DictConfig) -> None:
    # Set up device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    datasets_dir = os.environ.get("DATASETS_DIR") or cfg.data.datasets_dir
    num_downsamples = int(log2(cfg.model.size_spect / cfg.model.size_latent))

    # Log info
    log.info(f"datasets_dir: {datasets_dir}")
    log.info(f"device: {device}")
    log.info(f"cfg: {OmegaConf.to_yaml(cfg, resolve=True)}")
    log.info(f"num_downsamples: {num_downsamples}")

    # Load and configure dataset
    dataset_cfg = get_whar_cfg(WHARDatasetID(cfg.data.dataset_id), datasets_dir)
    dataset_cfg.seed = cfg.train.seed
    dataset_cfg.in_memory = True
    dataset_cfg.in_parallel = True
    dataset = PytorchAdapter(dataset_cfg)

    # Get dataloaders
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        train_batch_size=cfg.train.batch_size,
        scv_group_index=cfg.data.scv_group_index,
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
                num_classes=num_classes + 1,
                embedding_dim=cfg.model.embedding_dim,
                num_downsamples=num_downsamples,
            )
        case "cae":
            ae = SpectrogramCAE(
                num_channels_spect=cfg.model.num_channels_spect,
                num_channels_latent=cfg.model.num_channels_latent,
                num_classes=num_classes + 1,
                embedding_dim=cfg.model.embedding_dim,
                num_downsamples=num_downsamples,
            )
        case "caec":
            ae = SpectrogramCAEC(
                num_channels_spect=cfg.model.num_channels_spect,
                num_channels_latent=cfg.model.num_channels_latent,
                num_classes=num_classes + 1,
                embedding_dim=cfg.model.embedding_dim,
                num_downsamples=num_downsamples,
            )
        case _:
            raise NotImplementedError

    # Load model weights
    ae.load_state_dict(torch.load(model_path, map_location=device))
    ae.to(device)
    ae.eval()

    # Run one batch through
    test_loader.collate_fn = collate_fn
    x, y = next(iter(test_loader))
    print(x.shape, y.shape)
    recon, z = ae(x.to(device), y.to(device))

    print(recon.min(), recon.max())

    x = x[0].detach().cpu()
    recon = recon[0].detach().cpu()

    C, H, W = recon.shape

    x_reshaped = x.reshape(C // 2, 2, H, W)
    recon_reshaped = recon.reshape(C // 2, 2, H, W)

    plot_spectrogram_grid(x_reshaped)
    plot_spectrogram_grid(recon_reshaped)

    plot_spectrogram_grid(decompress_stft(x_reshaped))
    plot_spectrogram_grid(decompress_stft(recon_reshaped))

    plt.plot(detransform(x))
    plt.show()

    plt.plot(detransform(recon))
    plt.show()


if __name__ == "__main__":
    # Load config.json directly
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    # Convert to OmegaConf DictConfig for compatibility
    cfg = OmegaConf.create(cfg_dict)
    assert isinstance(cfg, DictConfig)

    run_script(cfg)
