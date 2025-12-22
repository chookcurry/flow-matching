import torch

from diffusion.backbones.res_unet import ResUnet
from diffusion.classifiers.encoder import compute_features
from diffusion.classifiers.encoder_mnist import MNISTClassifier
from diffusion.evaluation.evaluate import (
    compare_metrics_per_class,
    evaluate_features,
    plot_metrics_per_class,
)
from diffusion.evaluation.visualize import (
    visualize_samples_per_class,
    visualize_tsne_per_class,
)
from diffusion.flows.prob_paths import GaussianCondProbPath
from diffusion.generation.generator import generate_samples
from diffusion.generation.generator_flow import FlowGenerator
from diffusion.generation.generator_score import ScoreGenerator
from diffusion.sampleables.sampleable_mnist import MNISTSampleable

root_dir = "./scripts/mnist"
backbone_path = f"{root_dir}/models/backbone_flow.pt"
encoder_path = f"{root_dir}/models/classifier.pt"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)


sampeable = MNISTSampleable(train=False)
p_data_test = MNISTSampleable(train=False)

path = GaussianCondProbPath(p_data=sampeable, p_simple_shape=sampeable.shape).to(device)

backbone = ResUnet(
    in_channels=1,
    channels=[16, 32, 64],
    num_classes=sampeable.num_classes,
    t_dim=64,
    y_dim=32,
    cond_dim=64,
).to(device)

encoder = MNISTClassifier(num_classes=sampeable.num_classes).to(device)


backbone.load_state_dict(torch.load(backbone_path))
encoder.load_state_dict(torch.load(encoder_path))

num_timesteps = 10
samples_per_class = 200
guidance_scale = 2.0
seed = 42

flow_generator = FlowGenerator(
    path=path,
    backbone=backbone,
    num_timesteps=num_timesteps,
    null_class=sampeable.num_classes,
    device=device,
)

score_generator = ScoreGenerator(
    path=path,
    backbone=backbone,
    num_timesteps=num_timesteps,
    null_class=sampeable.num_classes,
    device=device,
)

if __name__ == "__main__":
    synth_samples_flow, real_samples_flow = generate_samples(
        generator=flow_generator,
        p_data=p_data_test,
        samples_per_class=samples_per_class,
        num_classes=sampeable.num_classes,
        device=device,
        guidance_scale=guidance_scale,
    )

    synth_samples_score, real_samples_score = generate_samples(
        generator=score_generator,
        p_data=p_data_test,
        samples_per_class=samples_per_class,
        num_classes=sampeable.num_classes,
        device=device,
        guidance_scale=guidance_scale,
    )

    visualize_samples_per_class(
        synth_samples_flow,
        real_samples_flow,
        save_path=f"{root_dir}/plots/flow_samples_scale_{guidance_scale}_steps_{num_timesteps}.png",
    )
    visualize_samples_per_class(
        synth_samples_score,
        real_samples_score,
        save_path=f"{root_dir}/plots/score_samples_scale_{guidance_scale}_steps_{num_timesteps}.png",
    )

    synth_features_flow, real_features_flow = compute_features(
        synth_samples_flow, real_samples_flow, encoder
    )

    synth_features_score, real_features_score = compute_features(
        synth_samples_score, real_samples_score, encoder
    )

    visualize_tsne_per_class(
        synth_features_flow,
        real_features_flow,
        save_path=f"{root_dir}/plots/tsne_flow_scale_{guidance_scale}_steps_{num_timesteps}.png",
    )
    visualize_tsne_per_class(
        synth_features_score,
        real_features_score,
        save_path=f"{root_dir}/plots/tsne_score_scale_{guidance_scale}_steps_{num_timesteps}.png",
    )

    metrics_per_class_flow = evaluate_features(synth_features_flow, real_features_flow)
    metrics_per_class_score = evaluate_features(
        synth_features_score, real_features_score
    )

    plot_metrics_per_class(
        metrics_per_class_flow,
        save_path=f"{root_dir}/plots/metrics_flow_scale_{guidance_scale}_steps_{num_timesteps}.png",
    )
    plot_metrics_per_class(
        metrics_per_class_score,
        save_path=f"{root_dir}/plots/metrics_score_scale_{guidance_scale}_steps_{num_timesteps}.png",
    )

    compare_metrics_per_class(
        metrics_per_class_flow,
        metrics_per_class_score,
        ("Flow", "Score"),
        save_path=f"{root_dir}/plots/metrics_comparison_scale_{guidance_scale}_steps_{num_timesteps}.png",
    )
