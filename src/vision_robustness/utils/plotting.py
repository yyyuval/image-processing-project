"""Plotting helpers for README / report figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vision_robustness.utils.io import ensure_dir


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def show_image_grid(
    images: Iterable[np.ndarray],
    titles: Iterable[str] | None = None,
    ncols: int = 4,
    save_path: str | Path | None = None,
    figsize_scale: float = 3.0,
) -> Path | None:
    images = list(images)
    titles = list(titles) if titles is not None else [""] * len(images)
    n = len(images)
    if n == 0:
        return None
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize_scale * ncols, figsize_scale * nrows)
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    if save_path is None:
        plt.show()
        return None
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_metric_vs_intensity(
    df: pd.DataFrame,
    metric: str,
    group_col: str = "distortion",
    x_col: str = "intensity",
    save_path: str | Path | None = None,
    title: str | None = None,
) -> Path | None:
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x=x_col, y=metric, hue=group_col, marker="o", ax=ax)
    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} vs {x_col}")
    fig.tight_layout()
    if save_path is None:
        plt.show()
        return None
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_bar_comparison(
    df: pd.DataFrame,
    metric: str,
    category: str = "stage",
    hue: str | None = "task",
    save_path: str | Path | None = None,
    title: str | None = None,
) -> Path | None:
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x=category, y=metric, hue=hue, ax=ax)
    ax.set_title(title or f"{metric} by {category}")
    fig.tight_layout()
    if save_path is None:
        plt.show()
        return None
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
