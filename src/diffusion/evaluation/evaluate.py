from typing import Dict, Tuple

import matplotlib.pyplot as plt
from torch import Tensor

from diffusion.evaluation.density_coverage import density_coverage_knn
from diffusion.evaluation.kid import kid_poly, kid_rbf
from diffusion.evaluation.precision_recall import f1_score, precision_recall_knn


def evaluate_features(
    features_synth: Dict[int, Tensor],
    features_real: Dict[int, Tensor],
) -> Dict[int, Dict[str, float]]:
    assert features_synth.keys() == features_real.keys()

    metrics_per_class: Dict[int, Dict[str, float]] = {}

    combined: Dict[int, Tuple[Tensor, Tensor]] = {
        class_label: (features_synth[class_label], features_real[class_label])
        for class_label in features_synth.keys()
    }

    for class_label, (synth, real) in combined.items():
        kid_poly_result = kid_poly(real, synth)
        kid_rbf_result = kid_rbf(real, synth)
        precision, recall = precision_recall_knn(real, synth)
        f1 = f1_score(precision, recall)
        density, coverage = density_coverage_knn(real, synth)

        metrics_per_class[class_label] = {
            "kid_poly": kid_poly_result.item(),
            "kid_rbf": kid_rbf_result.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
            "density": density.item(),
            "coverage": coverage.item(),
        }

    return metrics_per_class


def plot_metrics_per_class(
    metrics_per_class: Dict[int, Dict[str, float]], save_path: str | None = None
) -> None:
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

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close(fig)


def compare_metrics_per_class(
    metrics_a: Dict[int, Dict[str, float]],
    metrics_b: Dict[int, Dict[str, float]],
    labels: Tuple[str, str],
    save_path: str | None = None,
) -> None:
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

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close(fig)
