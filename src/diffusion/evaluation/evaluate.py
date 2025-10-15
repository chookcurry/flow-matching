from typing import Dict, Tuple
from torch import Tensor, device
import torch
from diffusion.architectures.classifiers.mnist_classifier import Encoder
from diffusion.evaluation.density_coverage import density_coverage_knn
from diffusion.evaluation.precision_recall import f1_score, precision_recall_knn
from diffusion.evaluation.generator import Generator
from diffusion.evaluation.kid import (
    kernel_inception_distance_poly,
    kernel_inception_distance_rbf,
)
import matplotlib.pyplot as plt
from diffusion.sampleables.sampleable import Sampleable
from sklearn.manifold import TSNE  # type: ignore
import numpy as np
import umap  # type: ignore


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


def compute_features(
    x1_list: list[Tensor], real_list: list[Tensor], encoder: Encoder
) -> Tuple[list[Tensor], list[Tensor]]:
    assert len(x1_list) == len(real_list)

    x1_features = [encoder(x1) for x1 in x1_list]
    real_features = [encoder(real) for real in real_list]

    return x1_features, real_features


def visualize_tsne_per_class(
    x1_features: list[torch.Tensor],
    real_features: list[torch.Tensor],
    perplexity: float = 30.0,
    n_iter: int = 1000,
    title: str = "t-SNE Visualization of Real vs Generated Features per Class",
):
    """
    Visualize feature embeddings using t-SNE, coloring each class differently and
    distinguishing real vs generated samples.

    Args:
        x1_features: List of generated feature tensors, one per class
        real_features: List of real feature tensors, one per class
        perplexity: t-SNE perplexity parameter (default: 30.0)
        n_iter: Number of t-SNE optimization iterations (default: 1000)
        title: Title of the plot
    """
    assert len(x1_features) == len(real_features), "Feature lists must have same length"

    num_classes = len(x1_features)

    # Combine all features into one array
    x1_all = torch.cat(x1_features, dim=0).detach().cpu().numpy()
    real_all = torch.cat(real_features, dim=0).detach().cpu().numpy()

    # Create labels for t-SNE visualization
    gen_labels = np.concatenate(
        [np.full(len(x1_features[i]), i) for i in range(num_classes)]
    )
    real_labels = np.concatenate(
        [np.full(len(real_features[i]), i) for i in range(num_classes)]
    )

    # Combine real and generated for joint visualization
    all_features = np.concatenate([real_all, x1_all], axis=0)
    all_labels = np.concatenate([real_labels, gen_labels], axis=0)
    domain_labels = np.concatenate(
        [np.zeros(len(real_all)), np.ones(len(x1_all))]
    )  # 0 = real, 1 = generated

    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=42,
        verbose=0,
    )
    tsne_results = tsne.fit_transform(all_features)

    # Plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap("tab10", num_classes)

    for class_idx in range(num_classes):
        # Real
        mask_real = (all_labels == class_idx) & (domain_labels == 0)
        plt.scatter(
            tsne_results[mask_real, 0],
            tsne_results[mask_real, 1],
            color=colors(class_idx),
            label=f"Class {class_idx} (Real)",
            alpha=0.6,
            marker="o",
            edgecolor="black",
            linewidths=0.3,
        )

        # Generated
        mask_gen = (all_labels == class_idx) & (domain_labels == 1)
        plt.scatter(
            tsne_results[mask_gen, 0],
            tsne_results[mask_gen, 1],
            color=colors(class_idx),
            label=f"Class {class_idx} (Gen)",
            alpha=0.6,
            marker="^",
            edgecolor="black",
            linewidths=0.3,
        )

    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()


# def visualize_umap_per_class(
#     x1_features: list[torch.Tensor],
#     real_features: list[torch.Tensor],
#     n_neighbors: int = 15,
#     min_dist: float = 0.1,
#     metric: str = "euclidean",
#     title: str = "UMAP Visualization of Real vs Generated Features per Class",
# ):
#     """
#     Visualize feature embeddings using UMAP, coloring each class differently and
#     distinguishing real vs generated samples.

#     Args:
#         x1_features: List of generated feature tensors, one per class
#         real_features: List of real feature tensors, one per class
#         n_neighbors: Number of neighbors to consider for UMAP (default: 15)
#         min_dist: Minimum distance between points in low-dimensional space (default: 0.1)
#         metric: Distance metric for UMAP (default: "euclidean")
#         title: Title for the plot
#     """
#     assert len(x1_features) == len(real_features), "Feature lists must have same length"

#     num_classes = len(x1_features)

#     # Combine all features into one array
#     x1_all = torch.cat(x1_features, dim=0).detach().cpu().numpy()
#     real_all = torch.cat(real_features, dim=0).detach().cpu().numpy()

#     # Create labels
#     gen_labels = np.concatenate(
#         [np.full(len(x1_features[i]), i) for i in range(num_classes)]
#     )
#     real_labels = np.concatenate(
#         [np.full(len(real_features[i]), i) for i in range(num_classes)]
#     )

#     # Combine real and generated
#     all_features = np.concatenate([real_all, x1_all], axis=0)
#     all_labels = np.concatenate([real_labels, gen_labels], axis=0)
#     domain_labels = np.concatenate(
#         [np.zeros(len(real_all)), np.ones(len(x1_all))]
#     )  # 0 = real, 1 = generated

#     # Apply UMAP
#     reducer = umap.UMAP(
#         n_neighbors=n_neighbors,
#         min_dist=min_dist,
#         metric=metric,
#         random_state=42,
#     )
#     result = reducer.fit_transform(all_features)

#     # Handle both return types (tuple or single ndarray)
#     if isinstance(result, tuple):
#         umap_results = result[0]
#     else:
#         umap_results = result

#     # Plot
#     plt.figure(figsize=(10, 8))
#     colors = plt.cm.get_cmap("tab10", num_classes)

#     for class_idx in range(num_classes):
#         # Real samples
#         mask_real = (all_labels == class_idx) & (domain_labels == 0)
#         plt.scatter(
#             umap_results[mask_real, 0],
#             umap_results[mask_real, 1],
#             color=colors(class_idx),
#             label=f"Class {class_idx} (Real)",
#             alpha=0.6,
#             marker="o",
#             edgecolor="black",
#             linewidths=0.3,
#         )

#         # Generated samples
#         mask_gen = (all_labels == class_idx) & (domain_labels == 1)
#         plt.scatter(
#             umap_results[mask_gen, 0],
#             umap_results[mask_gen, 1],
#             color=colors(class_idx),
#             label=f"Class {class_idx} (Gen)",
#             alpha=0.6,
#             marker="^",
#             edgecolor="black",
#             linewidths=0.3,
#         )

#     plt.title(title, fontsize=14, weight="bold")
#     plt.xlabel("UMAP Dim 1")
#     plt.ylabel("UMAP Dim 2")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
#     plt.tight_layout()
#     plt.show()


def evaluate_features(
    x1_features: list[Tensor], real_features: list[Tensor]
) -> Dict[int, Dict[str, float]]:
    assert len(x1_features) == len(real_features)

    metrics_per_class: Dict[int, Dict[str, float]] = {}

    for class_idx, x1, real in zip(range(len(x1_features)), x1_features, real_features):
        kid_poly = kernel_inception_distance_poly(real, x1)
        kid_rbf = kernel_inception_distance_rbf(real, x1)
        precision, recall = precision_recall_knn(real, x1)
        f1 = f1_score(precision, recall)
        density, coverage = density_coverage_knn(real, x1)

        metrics_per_class[class_idx] = {
            "kid_poly": kid_poly.item(),
            "kid_rbf": kid_rbf.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
            "density": density.item(),
            "coverage": coverage.item(),
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


def compare_metrics_per_class(
    metrics_a: Dict[int, Dict[str, float]],
    metrics_b: Dict[int, Dict[str, float]],
    labels=("Model A", "Model B"),
):
    """
    Compare per-class evaluation metrics between two models.

    Args:
        metrics_a: Dict[class_index, Dict[metric_name, value]] for model A
        metrics_b: Dict[class_index, Dict[metric_name, value]] for model B
        labels: Tuple of names for the two models (default: ("Model A", "Model B"))
    """

    # Extract metric names (assume all classes have the same metrics)
    metric_names = list(next(iter(metrics_a.values())).keys())
    num_metrics = len(metric_names)

    # Prepare class indices and values for both models
    class_indices = list(metrics_a.keys())
    values_a = {m: [metrics_a[c][m] for c in class_indices] for m in metric_names}
    values_b = {m: [metrics_b[c][m] for c in class_indices] for m in metric_names}

    # Create figure and subplots
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5), sharex=False)
    if num_metrics == 1:
        axes = [axes]  # ensure iterable

    # Bar width and x offsets for side-by-side bars
    x = range(len(class_indices))
    width = 0.35

    # Plot comparison for each metric
    for ax, metric in zip(axes, metric_names):
        ax.bar(
            [i - width / 2 for i in x],
            values_a[metric],
            width=width,
            label=labels[0],
            color="skyblue",
            edgecolor="black",
        )
        ax.bar(
            [i + width / 2 for i in x],
            values_b[metric],
            width=width,
            label=labels[1],
            color="salmon",
            edgecolor="black",
        )

        ax.set_title(metric.upper(), fontsize=14, weight="bold")
        ax.set_xlabel("Class")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(class_indices)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend()

    fig.suptitle("Per-Class Metric Comparison", fontsize=16, weight="bold")
    fig.tight_layout()
    plt.show()
