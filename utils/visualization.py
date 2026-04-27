"""Visualisation utilities for the VAE / DRAW comparison project."""

import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_grid(
    images: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
) -> np.ndarray:
    """Convert a batch of tensors (B, C, H, W) to a single numpy HWC grid."""
    B, C, H, W = images.shape
    ncol = math.ceil(B / nrow)
    grid_h = H * ncol + padding * (ncol + 1)
    grid_w = W * nrow + padding * (nrow + 1)
    grid = np.ones((grid_h, grid_w, max(C, 1)), dtype=np.float32)

    imgs = images.detach().cpu().float().numpy()
    for idx, img in enumerate(imgs):
        row = idx // nrow
        col = idx % nrow
        y = padding + row * (H + padding)
        x = padding + col * (W + padding)
        if C == 1:
            grid[y : y + H, x : x + W, :] = np.transpose(img, (1, 2, 0))
        else:
            grid[y : y + H, x : x + W, :] = np.transpose(img, (1, 2, 0))
    return grid.squeeze(-1) if C == 1 else grid


def plot_reconstructions(
    originals: torch.Tensor,
    vae_recons: torch.Tensor,
    draw_recons: torch.Tensor,
    n: int = 8,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot original images alongside VAE and DRAW reconstructions.

    Parameters
    ----------
    originals:
        Ground-truth images ``(B, C, H, W)``.
    vae_recons:
        VAE reconstructions, same shape.
    draw_recons:
        DRAW VAE reconstructions, same shape.
    n:
        Number of images to display (up to min(B, n)).
    save_path:
        If given, saves the figure to this path.
    """
    n = min(n, originals.size(0))

    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 4.5))
    titles = ["Original", "VAE", "DRAW VAE"]
    rows = [originals[:n], vae_recons[:n], draw_recons[:n]]

    for row_idx, (ax_row, title, imgs) in enumerate(zip(axes, titles, rows)):
        ax_row[0].set_ylabel(title, fontsize=10)
        for col_idx, ax in enumerate(ax_row):
            img = imgs[col_idx].detach().cpu().squeeze()
            cmap = "gray" if img.ndim == 2 else None
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_loss_curves(
    vae_train: Sequence[float],
    vae_val: Sequence[float],
    draw_train: Sequence[float],
    draw_val: Sequence[float],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot training and validation loss curves for both models.

    Parameters
    ----------
    vae_train, vae_val:
        Per-epoch losses for the standard VAE.
    draw_train, draw_val:
        Per-epoch losses for the DRAW VAE.
    save_path:
        If given, saves the figure to this path.
    """
    epochs = range(1, len(vae_train) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, vae_train, label="Train", color="steelblue")
    ax1.plot(epochs, vae_val, label="Val", color="steelblue", linestyle="--")
    ax1.set_title("VAE Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ELBO loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, draw_train, label="Train", color="coral")
    ax2.plot(epochs, draw_val, label="Val", color="coral", linestyle="--")
    ax2.set_title("DRAW VAE Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Training curves: VAE vs DRAW VAE", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_latent_interpolation(
    model,
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    steps: int = 10,
    device: Optional[torch.device] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Interpolate between two latent vectors and display the results.

    Works with the standard VAE whose ``decode`` method accepts a single z.

    Parameters
    ----------
    model:
        A trained ``VAE`` instance.
    z_start, z_end:
        Latent vectors of shape ``(latent_dim,)``.
    steps:
        Number of interpolation steps (includes endpoints).
    device:
        Device to run decoding on.
    save_path:
        Optional path to save the figure.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    alphas = torch.linspace(0, 1, steps, device=device)
    zs = torch.stack(
        [z_start.to(device) * (1 - a) + z_end.to(device) * a for a in alphas]
    )  # (steps, latent_dim)

    with torch.no_grad():
        imgs = model.decode(zs)  # (steps, C, H, W)

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.5, 2.5))
    for ax, img in zip(axes, imgs):
        ax.imshow(img.cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle("Latent space interpolation (VAE)", fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_samples(
    vae_samples: torch.Tensor,
    draw_samples: torch.Tensor,
    n: int = 8,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot random samples from both models side by side.

    Parameters
    ----------
    vae_samples, draw_samples:
        Tensors of shape ``(B, C, H, W)``.
    n:
        Number of samples per model to display.
    save_path:
        Optional path to save the figure.
    """
    n = min(n, vae_samples.size(0), draw_samples.size(0))

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))
    titles = ["VAE samples", "DRAW VAE samples"]
    for row_idx, (ax_row, title, imgs) in enumerate(
        zip(axes, titles, [vae_samples[:n], draw_samples[:n]])
    ):
        ax_row[0].set_ylabel(title, fontsize=9)
        for col_idx, ax in enumerate(ax_row):
            img = imgs[col_idx].detach().cpu().squeeze()
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

    fig.suptitle("Generated samples", fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
