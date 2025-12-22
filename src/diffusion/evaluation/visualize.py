from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap  # type: ignore
from sklearn.discriminant_analysis import StandardScaler  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from torch import Tensor


def visualize_samples_per_class(
    samples_synth: Dict[int, Tensor],
    samples_real: Dict[int, Tensor],
    n: int = 5,
    save_path: str | None = None,
) -> None:
    num_classes = len(samples_synth)
    n = min(n, samples_synth[0].shape[0])

    fig, axes = plt.subplots(
        num_classes,
        n * 2,
        figsize=(n * 1.8, num_classes * 1.8),
        gridspec_kw={"hspace": 0.3, "wspace": 0.1},
    )

    if num_classes == 1:
        axes = axes[None, :]

    for class_idx in range(num_classes):
        gen_samples = samples_synth[class_idx][:n]
        real_samples = samples_real[class_idx][:n]

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

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def visualize_tsne_per_class(
    features_synth: Dict[int, Tensor],
    features_real: Dict[int, Tensor],
    perplexity: float = 30.0,
    n_iter: int = 1000,
    save_path: str | None = None,
) -> None:
    assert len(features_synth) == len(features_real)

    num_classes = len(features_synth)

    # Combine all features into one array
    synth_all = torch.cat(list(features_synth.values()), dim=0).detach().cpu().numpy()
    real_all = torch.cat(list(features_real.values()), dim=0).detach().cpu().numpy()

    # Create labels for t-SNE visualization
    synth_labels = np.concatenate(
        [np.full(len(features_synth[i]), i) for i in features_synth.keys()]
    )
    real_labels = np.concatenate(
        [np.full(len(features_real[i]), i) for i in features_real.keys()]
    )

    # Combine real and generated for joint visualization
    all_features = np.concatenate([real_all, synth_all], axis=0)
    all_labels = np.concatenate([real_labels, synth_labels], axis=0)
    domain_labels = np.concatenate([np.zeros(len(real_all)), np.ones(len(synth_all))])
    # 0 = real, 1 = generated

    # 3. Standardize the data (Added Step)
    # This ensures zero mean and unit variance across the combined dataset
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)

    # Apply t-SNE
    results = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=42,
        verbose=0,
        init="pca",
    ).fit_transform(all_features)

    # Plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap("tab10", num_classes)

    for class_idx in range(num_classes):
        class_color = colors(class_idx / num_classes if num_classes > 10 else class_idx)

        # Real
        mask_real = (all_labels == class_idx) & (domain_labels == 0)
        plt.scatter(
            results[mask_real, 0],
            results[mask_real, 1],
            color=class_color,
            label=f"Class {class_idx} (Real)",
            alpha=0.5,
            marker="o",
            edgecolor="black",
            linewidths=0.3,
        )

        # Generated
        mask_gen = (all_labels == class_idx) & (domain_labels == 1)
        plt.scatter(
            results[mask_gen, 0],
            results[mask_gen, 1],
            color=class_color,
            label=f"Class {class_idx} (Gen)",
            alpha=0.5,
            marker="^",
            edgecolor="black",
            linewidths=0.3,
        )

    plt.title(
        "t-SNE Visualization of Real vs Synthetic Features per Class",
        fontsize=14,
        weight="bold",
    )
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def visualize_umap_per_class(
    features_synth: Dict[int, Tensor],
    features_real: Dict[int, Tensor],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    save_path: str | None = None,
) -> None:
    assert len(features_synth) == len(features_real)

    num_classes = len(features_synth)

    # Combine all features into one array
    synth_all = torch.cat(list(features_synth.values()), dim=0).detach().cpu().numpy()
    real_all = torch.cat(list(features_real.values()), dim=0).detach().cpu().numpy()

    # Create labels for t-SNE visualization
    synth_labels = np.concatenate(
        [np.full(len(features_synth[i]), i) for i in features_synth.keys()]
    )
    real_labels = np.concatenate(
        [np.full(len(features_real[i]), i) for i in features_real.keys()]
    )

    # Combine real and generated for joint visualization
    all_features = np.concatenate([real_all, synth_all], axis=0)
    all_labels = np.concatenate([real_labels, synth_labels], axis=0)
    domain_labels = np.concatenate([np.zeros(len(real_all)), np.ones(len(synth_all))])
    # 0 = real, 1 = generated

    # Apply UMAP
    results: Any = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    ).fit_transform(all_features)
    results = results if not isinstance(results, tuple) else results[0]

    # Plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap("tab10", num_classes)

    for class_idx in range(num_classes):
        # Real
        mask_real = (all_labels == class_idx) & (domain_labels == 0)
        plt.scatter(
            results[mask_real, 0],
            results[mask_real, 1],
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
            results[mask_gen, 0],
            results[mask_gen, 1],
            color=colors(class_idx),
            label=f"Class {class_idx} (Gen)",
            alpha=0.6,
            marker="^",
            edgecolor="black",
            linewidths=0.3,
        )

    plt.title(
        "UMAP Visualization of Real vs Synthetic Features per Class",
        fontsize=14,
        weight="bold",
    )
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
