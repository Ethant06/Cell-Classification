"""Generate the retained intra-class PCA density figure.

The current two datasets contain four classes in total, producing four panels.
PCA operates on grayscale 128×128 tensors scaled to [0, 1] without the
normalization used during CNN training.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from torchvision import transforms
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "cells": ROOT / "data2",
    "pneumonia": ROOT / "data4",
}
OUTPUT = ROOT / "visualizations" / "pca" / "figureB_density_scatter.png"

CLASS_COLORS = {
    "aneuploid": "#4C9ED9",
    "euploid": "#FF9F55",
    "normal": "#69C36D",
    "pneumonia": "#E84A5F",
}

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)


def canonical_class_name(folder_name: str) -> str:
    """Map dataset-specific folder names to canonical display/color labels."""
    name = folder_name.lower()
    for class_name in CLASS_COLORS:
        if class_name in name:
            return class_name
    return folder_name


def load_class_images(data_dir: Path, class_name: str) -> np.ndarray:
    """Load an exact ImageFolder class as an ``(n_samples, n_pixels)`` matrix."""
    dataset = ImageFolder(root=data_dir, transform=IMAGE_TRANSFORM)
    target = dataset.class_to_idx[class_name]
    images = [
        image.numpy().ravel()
        for image, label in dataset
        if label == target
    ]
    return np.asarray(images, dtype=np.float32)


def components_for_variance(cumulative_variance: np.ndarray, threshold: float) -> int:
    """Return the smallest component count reaching a variance threshold."""
    return int(np.searchsorted(cumulative_variance, threshold, side="left") + 1)


def analyze_class(images: np.ndarray) -> dict[str, Any]:
    """Fit intra-class PCA and calculate the statistics displayed in one panel.

    PCA is capped at 500 components to bound memory and runtime. The returned
    coordinates contain only PC1 and PC2, while ``k90`` is calculated from all
    fitted components.

    Args:
        images: Flattened image matrix with one image per row.

    Returns:
        Plot coordinates, centroid, variance percentages, ``k90``, mean radial
        distance in PC1–PC2 space, and the standard deviation of those radial
        distances as ``mean_spread``.

    Raises:
        ValueError: If fewer than three images are supplied.
    """
    if len(images) < 3:
        raise ValueError("At least three images are required for PCA")

    component_count = min(len(images) - 1, images.shape[1], 500)
    pca = PCA(n_components=component_count)
    projected = pca.fit_transform(images)
    explained = pca.explained_variance_ratio_
    coordinates = projected[:, :2]
    center = coordinates.mean(axis=0)
    distances = np.linalg.norm(coordinates - center, axis=1)

    return {
        "coordinates": coordinates,
        "center": center,
        "explained": explained[:2] * 100,
        "k90": components_for_variance(np.cumsum(explained), 0.90),
        "mean_distance": float(distances.mean()),
        "mean_spread": float(distances.std()),
    }


def add_density_contours(
    ax: Axes, coordinates: np.ndarray, color: str
) -> None:
    """Overlay quantile contours from a two-dimensional Gaussian KDE.

    Singular point clouds cannot support KDE; those panels retain their scatter
    points and silently omit contours.
    """
    if len(coordinates) < 4:
        return

    x = coordinates[:, 0]
    y = coordinates[:, 1]
    x_pad = max(float(np.ptp(x)) * 0.08, 1e-6)
    y_pad = max(float(np.ptp(y)) * 0.08, 1e-6)
    grid_x, grid_y = np.mgrid[
        x.min() - x_pad : x.max() + x_pad : 90j,
        y.min() - y_pad : y.max() + y_pad : 90j,
    ]

    try:
        density = gaussian_kde(np.vstack([x, y]))(
            np.vstack([grid_x.ravel(), grid_y.ravel()])
        ).reshape(grid_x.shape)
    except np.linalg.LinAlgError:
        return

    positive_density = density[density > 0]
    if positive_density.size:
        levels = np.quantile(positive_density, [0.70, 0.82, 0.90, 0.96])
        ax.contour(
            grid_x,
            grid_y,
            density,
            levels=np.unique(levels),
            colors=color,
            linestyles="--",
            linewidths=0.8,
        )


def plot_panel(ax: Axes, result: dict[str, Any]) -> None:
    """Render one class's scatter, density contours, centroid, and statistics.

    ``result`` must contain the analysis keys from ``analyze_class`` plus
    ``class_name``, ``sample_count``, and ``color``.
    """
    coordinates = result["coordinates"]
    color = result["color"]
    class_name = result["class_name"]
    explained = result["explained"]

    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=10,
        alpha=0.45,
        color=color,
        edgecolors="none",
    )
    add_density_contours(ax, coordinates, color)
    ax.scatter(
        *result["center"],
        s=55,
        color=color,
        edgecolor="white",
        linewidth=1.2,
        zorder=4,
    )

    ax.set_title(
        f"{class_name}\n"
        f"k90 = {result['k90']}  ·  mean dist = {result['mean_distance']:.1f}",
        fontsize=10,
        weight="bold",
    )
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.grid(alpha=0.25)

    legend_handle = Line2D(
        [], [], linestyle="--", color=color, label=(
            f"n = {result['sample_count']}\n"
            f"mean spread = {result['mean_spread']:.2f}"
        )
    )
    ax.legend(handles=[legend_handle], loc="upper right", fontsize=7)


def main() -> None:
    """Load both datasets and write the retained four-panel PCA figure."""
    missing = [path for path in DATASETS.values() if not path.is_dir()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise SystemExit(f"Missing required dataset directories: {missing_list}")

    results = []
    for dataset_name, data_dir in DATASETS.items():
        dataset = ImageFolder(root=data_dir)
        for folder_name in dataset.classes:
            images = load_class_images(data_dir, folder_name)
            analysis = analyze_class(images)
            canonical_name = canonical_class_name(folder_name)
            analysis.update(
                {
                    "class_name": folder_name,
                    "sample_count": len(images),
                    "color": CLASS_COLORS.get(canonical_name, "#777777"),
                }
            )
            results.append(analysis)

    fig, axes = plt.subplots(1, len(results), figsize=(20.48, 5.27))
    axes = np.atleast_1d(axes)
    for ax, result in zip(axes, results):
        plot_panel(ax, result)

    fig.suptitle(
        "Intra-class PCA scatter — each dot is one image\n"
        "Tight clusters with concentric density rings indicate low image diversity",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
