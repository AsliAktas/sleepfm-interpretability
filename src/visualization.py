"""
Visualization utilities for SleepFM Interpretability Project.

Static PNG outputs via Seaborn + Matplotlib.
(Plotly interactive plots are reserved for later days.)
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_umap_scatter(
    umap_coords: np.ndarray,
    labels: np.ndarray,
    disease_names: Dict[int, str],
    title: str = "UMAP Projection",
    highlight_idx: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    """Seaborn scatter plot — 12 disease groups in distinct colors.

    highlight_idx, verilen noktayı büyük siyah yıldız (★) ile işaretler.
    Pipeline demo'da 'bu hasta UMAP uzayında nerede?' sorusuna görsel cevap.

    Args:
        umap_coords:   (n_samples, 2) — UMAP projected coordinates.
        labels:        (n_samples,)   — integer disease labels.
        disease_names: Dict mapping label int → disease name string.
        title:         Plot title.
        highlight_idx: Sample index to mark with a star marker.
        save_path:     If given, saves PNG to this path (dpi=150).
    """
    n_classes = len(disease_names)
    palette = sns.color_palette("tab20", n_classes)

    df = pd.DataFrame(
        {
            "x": umap_coords[:, 0],
            "y": umap_coords[:, 1],
            "label": labels,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 9))

    for i, (label_idx, disease_name) in enumerate(sorted(disease_names.items())):
        mask = df["label"] == label_idx
        ax.scatter(
            df.loc[mask, "x"],
            df.loc[mask, "y"],
            color=palette[i],
            label=disease_name,
            s=12,
            alpha=0.7,
            edgecolors="none",
        )

    if highlight_idx is not None:
        ax.scatter(
            umap_coords[highlight_idx, 0],
            umap_coords[highlight_idx, 1],
            marker="*",
            s=350,
            color="black",
            zorder=5,
            label="Query Patient",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        markerscale=1.5,
        title="Disease Group",
        title_fontsize=9,
    )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_modality_importance_heatmap(
    weight_matrix: np.ndarray,
    disease_names: List[str],
    modality_names: List[str],
    title: str = "Modality Importance Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Seaborn heatmap — rows=disease, columns=modality.

    annot=True: numbers visible in cells.
    cmap='YlOrRd': zero=light yellow, high=dark red.

    Args:
        weight_matrix:  (n_diseases, n_modalities) — importance values.
        disease_names:  Row labels (length must equal weight_matrix.shape[0]).
        modality_names: Column labels, e.g. ["EEG", "ECG", "Resp", "EMG"].
        title:          Plot title.
        save_path:      If given, saves PNG to this path (dpi=150).
    """
    df = pd.DataFrame(
        weight_matrix,
        index=disease_names,
        columns=modality_names,
    )

    fig, ax = plt.subplots(figsize=(7, 10))

    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={"label": "Mean |Weight|", "shrink": 0.8},
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Modality", fontsize=11, labelpad=8)
    ax.set_ylabel("Disease", fontsize=11, labelpad=8)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)
