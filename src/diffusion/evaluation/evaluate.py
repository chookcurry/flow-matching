from typing import Dict, Tuple
from torch import Tensor, device
import torch
from diffusion.architectures.classifiers.mnist_classifier import Encoder
from diffusion.evaluation.f1 import f1_score, precision_recall_knn
from diffusion.evaluation.generator import Generator
from diffusion.evaluation.kid import (
    kernel_inception_distance_poly,
    kernel_inception_distance_rbf,
)
import matplotlib.pyplot as plt
from diffusion.sampleables.sampleable import Sampleable


def compute_samples(
    generator: Generator,
    p_data: Sampleable,
    samples_per_class: int,
    num_classes: int,
    device: device,
    guidance_scale: float = 3.0,
) -> Tuple[list[Tensor], list[Tensor]]:
    x1_list = []
    real_list = []

    for class_label in range(num_classes):
        # Generate label tensor for this class
        y = torch.full(
            (samples_per_class,), class_label, device=device, dtype=torch.long
        )

        # Generate synthetic samples for this class
        x1 = generator.generate(y, guidance_scale=guidance_scale)
        x1 = x1.to(device)
        x1_list.append(x1)

        # Sample real data for this class
        real_samples, _ = p_data.sample(samples_per_class, class_label)
        real_samples = real_samples.to(device)
        real_list.append(real_samples)

    return x1_list, real_list


# def visualize_samples_per_class(
#     x1_list: list[torch.Tensor], real_list: list[torch.Tensor], n: int = 5
# ) -> None:
#     num_classes = len(x1_list)
#     n = min(n, x1_list[0].shape[0])  # cap n by available samples

#     _, axes = plt.subplots(num_classes, n * 2, figsize=(n * 2.5, num_classes * 2.5))

#     if num_classes == 1:
#         axes = axes[None, :]  # handle 1-class edge case

#     for class_idx in range(num_classes):
#         gen_samples = x1_list[class_idx][:n]
#         real_samples = real_list[class_idx][:n]

#         for i in range(n):
#             # Plot real samples
#             ax_real = axes[class_idx, i]
#             img_real = real_samples[i].detach().cpu()
#             if img_real.ndim == 3 and img_real.shape[0] in [1, 3]:
#                 img_real = img_real.permute(1, 2, 0)
#             ax_real.imshow(img_real.squeeze(), cmap="gray")
#             ax_real.axis("off")
#             if i == 0:
#                 ax_real.set_ylabel(f"Class {class_idx}", fontsize=12)

#             # Plot generated samples
#             ax_gen = axes[class_idx, i + n]
#             img_gen = gen_samples[i].detach().cpu()
#             if img_gen.ndim == 3 and img_gen.shape[0] in [1, 3]:
#                 img_gen = img_gen.permute(1, 2, 0)
#             ax_gen.imshow(img_gen.squeeze(), cmap="gray")
#             ax_gen.axis("off")

#     # Column labels
#     for i in range(n):
#         axes[0, i].set_title(f"Real {i + 1}", fontsize=10)
#     for i in range(n, n * 2):
#         axes[0, i].set_title(f"Generated {i - n + 1}", fontsize=10)

#     plt.suptitle("Real vs Generated Samples per Class", fontsize=16, weight="bold")
#     plt.tight_layout()
#     plt.show()


def visualize_samples_per_class(
    x1_list: list[torch.Tensor], real_list: list[torch.Tensor], n: int = 5
) -> None:
    num_classes = len(x1_list)
    n = min(n, x1_list[0].shape[0])

    fig, axes = plt.subplots(
        num_classes,
        n * 2,
        figsize=(n * 1.8, num_classes * 1.8),
        gridspec_kw={"hspace": 0.3, "wspace": 0.1},
    )

    if num_classes == 1:
        axes = axes[None, :]

    for class_idx in range(num_classes):
        gen_samples = x1_list[class_idx][:n]
        real_samples = real_list[class_idx][:n]

        for i in range(n):
            # Real
            ax_real = axes[class_idx, i]
            img_real = real_samples[i].detach().cpu()
            if img_real.ndim == 3 and img_real.shape[0] in [1, 3]:
                img_real = img_real.permute(1, 2, 0)
            ax_real.imshow(img_real.squeeze(), cmap="gray")
            ax_real.axis("off")
            if i == 0:
                ax_real.set_ylabel(f"Class {class_idx}", fontsize=10)

            # Generated
            ax_gen = axes[class_idx, i + n]
            img_gen = gen_samples[i].detach().cpu()
            if img_gen.ndim == 3 and img_gen.shape[0] in [1, 3]:
                img_gen = img_gen.permute(1, 2, 0)
            ax_gen.imshow(img_gen.squeeze(), cmap="gray")
            ax_gen.axis("off")

    # Column titles
    for i in range(n):
        axes[0, i].set_title(f"Real {i + 1}", fontsize=9)
    for i in range(n, n * 2):
        axes[0, i].set_title(f"Generated {i - n + 1}", fontsize=9)

    plt.suptitle("Real vs Generated Samples per Class", fontsize=14, weight="bold")
    plt.show()


def evaluate_approach(
    x1_list: list[Tensor],
    real_list: list[Tensor],
    encoder: Encoder,
) -> Dict[int, Dict[str, float]]:
    assert len(x1_list) == len(real_list)

    metrics_per_class: Dict[int, Dict[str, float]] = {}

    for class_idx, x1, real in zip(range(len(x1_list)), x1_list, real_list):
        assert x1.shape == real.shape
        # (samples_per_class, ?)

        x1_features = encoder(x1)
        real_features = encoder(real)
        # (samples_per_class, latent_channels)

        kid_poly = kernel_inception_distance_poly(real_features, x1_features)
        kid_rbf = kernel_inception_distance_rbf(real_features, x1_features)
        precision, recall = precision_recall_knn(real_features, x1_features)
        f1 = f1_score(precision, recall)

        metrics_per_class[class_idx] = {
            "kid_poly": kid_poly.item(),
            "kid_rbf": kid_rbf.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
        }

    return metrics_per_class


def plot_metrics_per_class(metrics_per_class: Dict[int, Dict[str, float]]):
    # Extract metric names (assume all classes have the same metrics)
    metric_names = list(next(iter(metrics_per_class.values())).keys())
    num_metrics = len(metric_names)

    # Prepare class indices and metric values
    class_indices = list(metrics_per_class.keys())
    values_per_metric = {
        metric: [metrics_per_class[c][metric] for c in class_indices]
        for metric in metric_names
    }

    # Create figure and subplots
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5), sharex=False)
    if num_metrics == 1:
        axes = [axes]  # ensure iterable if only one metric

    # Plot each metric in its own subplot
    for ax, metric in zip(axes, metric_names):
        ax.bar(
            class_indices, values_per_metric[metric], color="skyblue", edgecolor="black"
        )
        ax.set_title(f"{metric.upper()}", fontsize=14)
        ax.set_xlabel("Class")
        ax.set_ylabel(metric)
        ax.set_xticks(class_indices)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.suptitle("Per-Class Evaluation Metrics", fontsize=16, weight="bold")
    fig.tight_layout()
    plt.show()
