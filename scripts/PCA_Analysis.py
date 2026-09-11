"""
Raw-pixel intra-class PCA: one PNG per class (PC1 vs PC2), axis labels only.

Prints k90/k95/k99 metrics to the console for each class.

Outputs:
    visualizations/pca/pca_scatter_<dataset>_<class>.png — one panel per class (tight per-class zoom)

    visualizations/pca/intraclass_pca_scatter_2x2.svg — four-panel grid (vector, transparent)
    visualizations/pca/intraclass_pca_scatter_2x2.png — same layout raster preview

The 2×2 uses **shared** square PC limits (all classes pooled) so panels align; legend at bottom.

Run from project root:
    python scripts/PCA_Analysis.py

Edit DATASETS at the top if your folders differ (e.g. cells under data2/).
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Typography (matplotlib uses points)
FONT_FAMILY = "Arial"
AXIS_TICK_FONT_PT = 15.0   # single-panel PNGs: tick numerals
AXIS_LABEL_FONT_PT = 18.0  # single-panel PNGs: “PC1 (..% variance)”

# 2×2 poster (matplotlib fontsize is in pt; px @ 96dpi → pt via px * 72/96)
POSTER_30PX_PT = 30.0 * 72.0 / 96.0
POSTER_36PX_PT = 36.0 * 72.0 / 96.0  # PC1/PC2 lines include variance %
AXIS_LABEL_FONT_2X2_PT = POSTER_36PX_PT
AXIS_TICK_FONT_2X2_PT = POSTER_30PX_PT
LEGEND_FONT_2X2_PT = POSTER_30PX_PT

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "Liberation Sans",
    "DejaVu Sans",
    "sans-serif",
]

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASETS = {
    "cells": "data2",  # subfolders: e.g. Euploid2/ Aneuploid2/
    "pneumonia": "data4",  # subfolders: normal/ pneumonia/
}

OUT_DIR = os.path.join("visualizations", "pca")
os.makedirs(OUT_DIR, exist_ok=True)

# Scatter color per canonical class (matches legend_label(); folders may be Euploid2/, normal/, etc.).
CLASS_COLORS = {
    "normal": "#F4B942",
    "pneumonia": "#B7510A",
    "aneuploid": "#1A5276",
    "euploid": "#5DADE2",
}

# Fixed output geometry: inch × dpi = pixel size (same for every PNG).
FIG_INCHES = 7.0
FIG_DPI = 200

# Combined 2×2 figure (SVG + PNG): grid size; tight pooled zoom for poster.
FIG_2X2_INCHES = (15.5, 12.0)
GRID_SHARED_PAD_FRAC = 0.01
# Trim pooled PC coordinates before shared square box (stronger zoom).
GRID_POOL_PERCENTILES: tuple[float, float] | None = (2.5, 97.5)
GRID_POOL_PERCENTILE_MIN_N = 100

# Tighter zoom: trim outliers from axis range + small padding (less empty margin).
AXIS_PAD_FRAC = 0.015
# Use (low, high) percentiles for limits; set to None to use min/max only.
AXIS_PERCENTILES: tuple[float, float] | None = (1.0, 99.0)
MIN_SAMPLES_FOR_PERCENTILES = 20

LEGEND_ORDER = ("normal", "pneumonia", "aneuploid", "euploid")


def legend_label(class_name: str) -> str:
    s = class_name.lower()
    if "aneuploid" in s:
        return "aneuploid"
    if "euploid" in s:
        return "euploid"
    if "pneumonia" in s:
        return "pneumonia"
    if "normal" in s:
        return "normal"
    return class_name


def color_for_class_folder(class_name: str) -> str:
    lab = legend_label(class_name)
    c = CLASS_COLORS.get(lab)
    if c is None:
        print(f"    WARNING: no CLASS_COLORS entry for label {lab!r} ({class_name}); using gray")
        return "#888888"
    return c


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "class"


def style_pc_axes(
    ax,
    ve: np.ndarray,
    *,
    label_pt: float | None = None,
    tick_pt: float | None = None,
) -> None:
    """Arial; label_pt / tick_pt default to single-panel constants."""
    lp = AXIS_LABEL_FONT_PT if label_pt is None else label_pt
    tp = AXIS_TICK_FONT_PT if tick_pt is None else tick_pt
    ax.set_xlabel(
        f"PC1 ({ve[0]:.1f}% variance)",
        fontsize=lp,
        fontfamily=FONT_FAMILY,
    )
    ax.set_ylabel(
        f"PC2 ({ve[1]:.1f}% variance)",
        fontsize=lp,
        fontfamily=FONT_FAMILY,
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tp,
    )
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        t.set_fontfamily(FONT_FAMILY)


def load_class(data_dir: str, class_name: str) -> np.ndarray:
    """All images in one class folder → (n, 16384) float32."""
    dataset = ImageFolder(root=data_dir, transform=transform)
    target = dataset.class_to_idx[class_name]
    images = [
        img.numpy().flatten()
        for img, label in dataset
        if label == target
    ]
    X = np.array(images, dtype=np.float32)
    print(f"    {class_name:20s}  n={X.shape[0]}")
    return X


def _k_at_cumulative_threshold(cumvar: np.ndarray, th: float) -> int:
    """Smallest k with cumulative variance ≥ th; if never, len(cumvar)+1."""
    if cumvar.size == 0:
        return 0
    i = int(np.searchsorted(cumvar, th, side="left"))
    return i + 1


def intraclass_pca_full(X: np.ndarray) -> dict | None:
    """
    One PCA fit per class for k90 metrics and first two PCs for scatter.
    Returns None if <3 samples or <2 PCs.
    """
    if X.shape[0] < 3:
        return None
    n_full = min(X.shape[0] - 1, X.shape[1], 500)
    if n_full < 2:
        return None
    pca = PCA(n_components=n_full)
    Z = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    k90 = _k_at_cumulative_threshold(cumvar, 0.90)
    k95 = _k_at_cumulative_threshold(cumvar, 0.95)
    k99 = _k_at_cumulative_threshold(cumvar, 0.99)
    pc1_pct = float(evr[0]) * 100.0
    coords = Z[:, :2]
    ve2 = evr[:2] * 100
    return {
        "coords": coords,
        "ve": ve2,
        "k90": k90,
        "k95": k95,
        "k99": k99,
        "pc1_pct": pc1_pct,
        "n_comp_fit": n_full,
        "cumvar_last_pct": float(cumvar[-1]) * 100.0,
    }


def _axis_limits_1d(arr: np.ndarray) -> tuple[float, float]:
    """PC coordinate range: percentile trim (optional) + thin padding → tighter zoom."""
    n = arr.shape[0]
    if (
        AXIS_PERCENTILES is not None
        and n >= MIN_SAMPLES_FOR_PERCENTILES
        and AXIS_PERCENTILES[1] > AXIS_PERCENTILES[0]
    ):
        lo = float(np.percentile(arr, AXIS_PERCENTILES[0]))
        hi = float(np.percentile(arr, AXIS_PERCENTILES[1]))
    else:
        lo, hi = float(arr.min()), float(arr.max())
    span = max(hi - lo, 1e-9)
    pad = span * AXIS_PAD_FRAC
    return lo - pad, hi + pad


def _shared_xy_limits_pool(
    all_results: list[dict],
    pad_frac: float,
    percentiles: tuple[float, float] | None,
    min_n_percentile: int,
) -> tuple[float, float, float, float]:
    xs = np.concatenate([r["coords"][:, 0] for r in all_results])
    ys = np.concatenate([r["coords"][:, 1] for r in all_results])
    n = xs.shape[0]
    if (
        percentiles is not None
        and n >= min_n_percentile
        and percentiles[1] > percentiles[0]
    ):
        x_min = float(np.percentile(xs, percentiles[0]))
        x_max = float(np.percentile(xs, percentiles[1]))
        y_min = float(np.percentile(ys, percentiles[0]))
        y_max = float(np.percentile(ys, percentiles[1]))
    else:
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
    rx = max(x_max - x_min, 1e-9)
    ry = max(y_max - y_min, 1e-9)
    px = rx * pad_frac
    py = ry * pad_frac
    return x_min - px, x_max + px, y_min - py, y_max + py


def _square_axis_limits_pool(all_results: list[dict], pad_frac: float) -> tuple[float, float, float, float]:
    """Square window in PC space covering all classes (for comparable 2×2 panels)."""
    x0, x1, y0, y1 = _shared_xy_limits_pool(
        all_results,
        pad_frac,
        GRID_POOL_PERCENTILES,
        GRID_POOL_PERCENTILE_MIN_N,
    )
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2
    return cx - half, cx + half, cy - half, cy + half


def print_metrics_console(all_results: list[dict]) -> None:
    # ASCII only: Windows consoles often use cp1252 and cannot print box-drawing chars.
    print("\n" + "=" * 92)
    print("  RAW-PIXEL INTRA-CLASS PCA METRICS  (16,384-D flattened; sklearn PCA centers columns)")
    print("=" * 92)
    print(
        f"  {'Class':<18} {'Dataset':<12} {'n':>7}  {'k90':>6} {'k95':>6} {'k99':>6}  "
        f"{'PC1%':>8}  {'n_PC_fit':>9}  {'cumvar@fit':>11}"
    )
    print("  " + "-" * 88)
    for r in all_results:
        print(
            f"  {r['class_name']:<18} {r['dataset']:<12} {r['n_samples']:>7}  "
            f"{r['k90']:>6} {r['k95']:>6} {r['k99']:>6}  "
            f"{r['pc1_pct']:>7.2f}%  {r['n_comp_fit']:>9}  {r['cumvar_last_pct']:>10.2f}%"
        )
    print("=" * 92)
    print(
        "  k90/k95/k99: smallest # of PCs with cumulative variance >= 90/95/99% (within class).\n"
        "  If not reached within n_PC_fit components, value is n_PC_fit+1.\n"
        "  cumvar@fit: cumulative variance from all fitted PCs."
    )


def save_individual_scatters(all_results: list[dict]) -> None:
    """One transparent PNG per class: only PC1/PC2 axis labels (variance %)."""
    print("\n-- Saving one scatter plot per class --")

    px = int(round(FIG_INCHES * FIG_DPI))
    zoom_note = (
        f"zoom: pad={AXIS_PAD_FRAC:g}"
        + (
            f", percentiles={AXIS_PERCENTILES}"
            if AXIS_PERCENTILES
            else ", limits=min/max"
        )
        + f" (if n>={MIN_SAMPLES_FOR_PERCENTILES})"
    )
    print(
        f"  Same canvas every file: {FIG_INCHES:g}x{FIG_INCHES:g} in @ {FIG_DPI} dpi -> {px}x{px} px. "
        f"{zoom_note}"
    )

    for r in all_results:
        coords = r["coords"]
        ve = r["ve"]
        color = r["color"]

        x_lo, x_hi = _axis_limits_1d(coords[:, 0])
        y_lo, y_hi = _axis_limits_1d(coords[:, 1])

        fig, ax = plt.subplots(figsize=(FIG_INCHES, FIG_INCHES))
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color,
            alpha=0.4,
            s=14,
            edgecolors="none",
        )
        style_pc_axes(ax, ve)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

        ax.set_facecolor("none")
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0)

        slug = safe_filename(r["class_name"])
        fname = f"pca_scatter_{r['dataset']}_{slug}.png"
        path = os.path.join(OUT_DIR, fname)

        plt.subplots_adjust(left=0.11, right=0.98, top=0.98, bottom=0.11)
        # Do not use bbox_inches="tight" — it crops differently per figure → uneven PNG sizes.
        plt.savefig(
            path,
            dpi=FIG_DPI,
            facecolor="none",
            transparent=True,
        )
        plt.close(fig)
        print(f"  Saved -> {path}")


def save_scatter_2x2_grid(all_results: list[dict]) -> None:
    """
    One 2×2 figure: shared square limits, PC axis labels, legend, no title.
    Writes SVG (vector) and PNG for quick viewing.
    """
    if len(all_results) > 4:
        print(
            f"\nWARNING: {len(all_results)} classes; 2x2 uses the first 4 in DATASETS / class order only."
        )
    rows = all_results[:4]
    if not rows:
        return

    print("\n-- Saving combined 2x2 (SVG + PNG) --")
    ax_lo, ax_hi, ay_lo, ay_hi = _square_axis_limits_pool(rows, GRID_SHARED_PAD_FRAC)
    print(
        f"  Shared square limits: PC1 in [{ax_lo:.2f}, {ax_hi:.2f}], "
        f"PC2 in [{ay_lo:.2f}, {ay_hi:.2f}]"
    )
    if GRID_POOL_PERCENTILES:
        print(
            f"  Pooled zoom: percentiles={GRID_POOL_PERCENTILES} (if n>={GRID_POOL_PERCENTILE_MIN_N}), "
            f"pad_frac={GRID_SHARED_PAD_FRAC}"
        )

    fig, axes = plt.subplots(2, 2, figsize=FIG_2X2_INCHES)
    axes_flat = axes.flatten()

    for ax, r in zip(axes_flat, rows):
        coords = r["coords"]
        ve = r["ve"]
        color = r["color"]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color,
            alpha=0.4,
            s=18,
            edgecolors="none",
        )
        style_pc_axes(
            ax,
            ve,
            label_pt=AXIS_LABEL_FONT_2X2_PT,
            tick_pt=AXIS_TICK_FONT_2X2_PT,
        )
        ax.set_xlim(ax_lo, ax_hi)
        ax.set_ylim(ay_lo, ay_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_facecolor("none")

    for j in range(len(rows), 4):
        axes_flat[j].set_visible(False)

    label_to_color: dict[str, str] = {}
    for r in rows:
        label_to_color[legend_label(r["class_name"])] = r["color"]
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=label_to_color[lb],
            markeredgecolor="none",
            markersize=11,
            linestyle="none",
            label=lb,
        )
        for lb in LEGEND_ORDER
        if lb in label_to_color
    ]
    leg = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(4, len(legend_handles)),
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        fontsize=LEGEND_FONT_2X2_PT,
        prop={"family": FONT_FAMILY},
    )
    for text in leg.get_texts():
        text.set_fontfamily(FONT_FAMILY)

    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)

    plt.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.97,
        bottom=0.12,
        hspace=0.12,
        wspace=0.26,
    )

    base = os.path.join(OUT_DIR, "intraclass_pca_scatter_2x2")
    svg_path = base + ".svg"
    png_path = base + ".png"

    plt.savefig(
        svg_path,
        format="svg",
        facecolor="none",
        transparent=True,
    )
    plt.savefig(
        png_path,
        dpi=FIG_DPI,
        facecolor="none",
        transparent=True,
    )
    plt.close(fig)
    print(f"  Saved -> {svg_path}")
    print(f"  Saved -> {png_path}")


if __name__ == "__main__":
    print("Loading images (flattened raw pixels, intra-class PCA)...\n")

    all_results = []

    for ds_name, data_dir in DATASETS.items():
        if not os.path.exists(data_dir):
            print(f"WARNING: '{data_dir}' not found - skipping {ds_name}")
            continue

        dummy = ImageFolder(root=data_dir)
        class_names = dummy.classes

        print(f"{ds_name.upper()} - {data_dir}")
        print(f"  classes: {class_names}")

        for cls_name in class_names:
            X = load_class(data_dir, cls_name)
            stats = intraclass_pca_full(X)

            if stats is None:
                print(f"    [skip] {cls_name}: need >=3 samples for intra-class PCA")
                continue

            all_results.append(
                {
                    "class_name": cls_name,
                    "dataset": ds_name,
                    "color": color_for_class_folder(cls_name),
                    "n_samples": X.shape[0],
                    "coords": stats["coords"],
                    "ve": stats["ve"],
                    "k90": stats["k90"],
                    "k95": stats["k95"],
                    "k99": stats["k99"],
                    "pc1_pct": stats["pc1_pct"],
                    "n_comp_fit": stats["n_comp_fit"],
                    "cumvar_last_pct": stats["cumvar_last_pct"],
                }
            )

    if len(all_results) == 0:
        print("\nNo data loaded. Update DATASETS paths at top of script.")
    else:
        print_metrics_console(all_results)
        save_scatter_2x2_grid(all_results)
        save_individual_scatters(all_results)
        print(f"\nOutput: {OUT_DIR}/")
