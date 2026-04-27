"""Evaluate and compare a trained VAE and DRAW VAE on licence plate reconstruction.

This script loads pre-trained checkpoints produced by ``train.py``, runs both
models on the test split of the synthetic dataset, and computes:

* Mean Squared Error (MSE)
* Structural Similarity Index (SSIM, via scikit-image)
* Average ELBO loss

It also generates comparison plots and saves them to ``results/``.

Usage
-----
python evaluate.py \\
    --vae-ckpt   checkpoints/vae_best.pt \\
    --draw-ckpt  checkpoints/draw_best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

import config
from data.dataset import SyntheticLicensePlateDataset
from models.draw_vae import DRAWVAE
from models.vae import VAE
from utils.visualization import plot_reconstructions, plot_samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mse(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    return F.mse_loss(x_hat, x).item()


def _mean_ssim(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    x_np = x.detach().cpu().numpy()       # (B, C, H, W)
    x_hat_np = x_hat.detach().cpu().numpy()
    scores = []
    for orig, recon in zip(x_np, x_hat_np):
        # skimage expects HW for grayscale or HWC for colour
        orig_img = orig.squeeze()
        recon_img = recon.squeeze()
        score = ssim(
            orig_img,
            recon_img,
            data_range=1.0,
        )
        scores.append(score)
    return float(np.mean(scores))


@torch.no_grad()
def _evaluate_vae(model: VAE, loader: DataLoader, device: torch.device):
    model.eval()
    mse_list, ssim_list, elbo_list = [], [], []
    all_orig, all_recon = [], []

    for x in loader:
        x = x.to(device)
        x_hat, mu, logvar = model(x)
        loss, _, _ = VAE.loss(x, x_hat, mu, logvar)
        mse_list.append(_mse(x, x_hat))
        ssim_list.append(_mean_ssim(x, x_hat))
        elbo_list.append(loss.item())
        if len(all_orig) < 64:
            all_orig.append(x.cpu())
            all_recon.append(x_hat.cpu())

    return (
        np.mean(mse_list), np.mean(ssim_list), np.mean(elbo_list),
        torch.cat(all_orig)[:64], torch.cat(all_recon)[:64],
    )


@torch.no_grad()
def _evaluate_draw(model: DRAWVAE, loader: DataLoader, device: torch.device):
    model.eval()
    mse_list, ssim_list, elbo_list = [], [], []
    all_orig, all_recon = [], []

    for x in loader:
        x = x.to(device)
        x_hat, mus, logvars = model(x)
        loss, _, _ = DRAWVAE.loss(x, x_hat, mus, logvars)
        mse_list.append(_mse(x, x_hat))
        ssim_list.append(_mean_ssim(x, x_hat))
        elbo_list.append(loss.item())
        if len(all_orig) < 64:
            all_orig.append(x.cpu())
            all_recon.append(x_hat.cpu())

    return (
        np.mean(mse_list), np.mean(ssim_list), np.mean(elbo_list),
        torch.cat(all_orig)[:64], torch.cat(all_recon)[:64],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # ---- Test dataset ----
    test_ds = SyntheticLicensePlateDataset(
        num_samples=config.NUM_TEST,
        width=config.IMG_WIDTH,
        height=config.IMG_HEIGHT,
        seed=config.SEED + 99,  # different seed from training
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load and evaluate VAE ----
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae_args = argparse.Namespace(**vae_ckpt["args"])
    vae = VAE(
        img_channels=config.IMG_CHANNELS,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        hidden_dims=config.VAE_HIDDEN_DIMS,
        latent_dim=getattr(vae_args, "latent_dim", config.VAE_LATENT_DIM),
    ).to(device)
    vae.load_state_dict(vae_ckpt["model_state"])

    vae_mse, vae_ssim, vae_elbo, vae_orig, vae_recon = _evaluate_vae(vae, test_loader, device)

    # ---- Load and evaluate DRAW ----
    draw_ckpt = torch.load(args.draw_ckpt, map_location=device)
    draw_args = argparse.Namespace(**draw_ckpt["args"])
    draw = DRAWVAE(
        img_channels=config.IMG_CHANNELS,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        T=getattr(draw_args, "T", config.DRAW_T),
        z_dim=getattr(draw_args, "z_dim", config.DRAW_Z_DIM),
        h_dim=getattr(draw_args, "h_dim", config.DRAW_H_DIM),
        N=getattr(draw_args, "N", config.DRAW_ATTN_N),
    ).to(device)
    draw.load_state_dict(draw_ckpt["model_state"])

    draw_mse, draw_ssim, draw_elbo, _, draw_recon = _evaluate_draw(draw, test_loader, device)

    # ---- Print results ----
    print("\n" + "=" * 60)
    print(f"{'Metric':<20} {'VAE':>15} {'DRAW VAE':>15}")
    print("-" * 60)
    print(f"{'MSE':<20} {vae_mse:>15.6f} {draw_mse:>15.6f}")
    print(f"{'SSIM':<20} {vae_ssim:>15.4f} {draw_ssim:>15.4f}")
    print(f"{'ELBO loss':<20} {vae_elbo:>15.4f} {draw_elbo:>15.4f}")
    print("=" * 60)

    # ---- Save results to CSV ----
    csv_path = results_dir / "comparison.csv"
    with open(csv_path, "w") as f:
        f.write("metric,vae,draw_vae\n")
        f.write(f"mse,{vae_mse:.6f},{draw_mse:.6f}\n")
        f.write(f"ssim,{vae_ssim:.4f},{draw_ssim:.4f}\n")
        f.write(f"elbo,{vae_elbo:.4f},{draw_elbo:.4f}\n")
    print(f"\nComparison results saved to: {csv_path}")

    # ---- Reconstruction plot ----
    recon_plot = results_dir / "reconstructions.png"
    plot_reconstructions(
        vae_orig, vae_recon, draw_recon, n=8, save_path=recon_plot
    )
    print(f"Reconstruction plot saved to: {recon_plot}")

    # ---- Sample plot ----
    with torch.no_grad():
        vae_samples = vae.sample(16, device)
        draw_samples = draw.sample(16, device)
    sample_plot = results_dir / "samples.png"
    plot_samples(vae_samples, draw_samples, n=8, save_path=sample_plot)
    print(f"Sample plot saved to:         {sample_plot}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate VAE vs DRAW VAE on licence plate reconstruction"
    )
    p.add_argument(
        "--vae-ckpt",
        default=str(Path(config.CHECKPOINT_DIR) / "vae_best.pt"),
        help="Path to the VAE checkpoint",
    )
    p.add_argument(
        "--draw-ckpt",
        default=str(Path(config.CHECKPOINT_DIR) / "draw_best.pt"),
        help="Path to the DRAW VAE checkpoint",
    )
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
