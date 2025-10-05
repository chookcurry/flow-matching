import os
from typing import Tuple
import hydra
import torch
import logging
from omegaconf import DictConfig, OmegaConf
from flow_matching.architectures.autoencoder import AE, AEC, CAE, CAEC
from flow_matching.supervised.alphas_betas import LinearAlpha, LinearBeta
from flow_matching.supervised.odes_sdes import Backbone
from flow_matching.supervised.prob_paths import GaussianCondProbPath
from flow_matching.whar.autoencoder.autoencoder import (
    SpectrogramAE,
    SpectrogramCAE,
    SpectrogramAEC,
    SpectrogramCAEC,
)
from hydra.core.hydra_config import HydraConfig
import shutil
import wandb
from dotenv import load_dotenv
from flow_matching.architectures.latent_cnn import FiLMNetMultiBlock
from flow_matching.architectures.latent_transformer import FlowTransformerBackbone
from flow_matching.whar.whar_sampler import WHARSampler
from flow_matching.training.training_latent_flow import LatentFlowTrainer

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
    sampler = WHARSampler(subject_id=cfg.data.subject_id)
    num_classes = sampler.get_num_classes(sampler.train_indices)
    latent_shape: Tuple[int, ...] = (
        cfg.autoencoder.latent_n_channels,
        cfg.autoencoder.latent_size,
        cfg.autoencoder.latent_size,
    )

    # Initialize autoencoder
    ae: AE | CAE | AEC | CAEC
    match cfg.model.architecure:
        case "ae":
            ae = SpectrogramAE(
                num_channels_spect=cfg.model.spect_n_channels,
                num_channels_latent=cfg.model.latent_n_channels,
            )
        case "aec":
            ae = SpectrogramAEC(
                num_channels_spect=cfg.model.spect_n_channels,
                num_channels_latent=cfg.model.latent_n_channels,
                num_classes=num_classes,
                embedding_dim=cfg.model.embedding_dim,
            )
        case "cae":
            ae = SpectrogramCAE(
                num_channels_spect=cfg.model.spect_n_channels,
                num_channels_latent=cfg.model.latent_n_channels,
                num_classes=num_classes,
                embedding_dim=cfg.model.embedding_dim,
            )
        case "caec":
            ae = SpectrogramCAEC(
                num_channels_spect=cfg.model.spect_n_channels,
                num_channels_latent=cfg.model.latent_n_channels,
                num_classes=num_classes,
                embedding_dim=cfg.model.embedding_dim,
            )
        case _:
            raise NotImplementedError

    # Initialize vector field
    vf: Backbone
    match cfg.model.architecure:
        case "transformer":
            vf = FlowTransformerBackbone(
                latent_channels=cfg.autoencoder.latent_n_channels,
                num_classes=num_classes,
            ).to(device)
        case "cnn":
            vf = FiLMNetMultiBlock(
                in_channels=latent_shape[0], y_classes=num_classes
            ).to(device)
        case _:
            raise NotImplementedError

    # Initialize probability path
    path = GaussianCondProbPath(
        p_data=sampler,
        p_simple_shape=latent_shape,
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    ).to(device)

    # Initialize trainer
    trainer = LatentFlowTrainer(
        path=path,
        model=vf,
        ae=ae,
        eta=(1 / num_classes),
        null_class=num_classes,
        num_classes=num_classes,
        guidance_scale=cfg.val.guidance_scale,
        num_timesteps=cfg.val.num_timesteps,
        num_samples=cfg.val.num_samples,
    )

    # Initialize wandb run
    run = wandb.init(
        project=cfg.wandb.project,
        job_type="train",
        name=run_id,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
    )

    # Load autoencoder
    ae_path = run.use_model(f"{cfg.wandb.project}/{run_id}/ae")
    ae.load_state_dict(torch.load(ae_path, map_location=device))

    # Perform training
    trainer.train(
        num_epochs=cfg.train.num_epochs,
        device=device,
        batch_size=cfg.train.batch_size,
        lr=cfg.train.learning_rate,
        steps_per_epoch=cfg.val.val_every_n_epochs,
        run=run,
    )

    # Save ae and vf
    ae_path = f"{output_dir}/ae.pt"
    vf_path = f"{output_dir}/vf.pt"
    torch.save(ae.state_dict(), ae_path)
    torch.save(vf.state_dict(), vf_path)

    # Log vf
    run.log_artifact(vf_path, name="vf", type="model")
    run.finish()

    # Clean up
    shutil.rmtree(cache_dir)


if __name__ == "__main__":
    run_script()
