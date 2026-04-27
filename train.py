"""Train either the VAE or DRAW VAE on the synthetic licence plate dataset.

Usage examples
--------------
# Train the standard VAE for 50 epochs:
python train.py --model vae --epochs 50 --batch-size 32 --latent-dim 128

# Train DRAW VAE:
python train.py --model draw --epochs 50 --batch-size 32 --T 10 --z-dim 32

Both scripts save a checkpoint to ``checkpoints/<model>_best.pt`` and write
per-epoch losses to ``results/<model>_losses.csv``.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import config
from data.dataset import SyntheticLicensePlateDataset
from models.draw_vae import DRAWVAE
from models.vae import VAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_dataloaders(args: argparse.Namespace):
    dataset = SyntheticLicensePlateDataset(
        num_samples=config.NUM_TRAIN + config.NUM_VAL,
        width=config.IMG_WIDTH,
        height=config.IMG_HEIGHT,
        seed=config.SEED,
    )
    n_val = config.NUM_VAL
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config.SEED),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


def _build_vae(args: argparse.Namespace) -> VAE:
    return VAE(
        img_channels=config.IMG_CHANNELS,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        hidden_dims=config.VAE_HIDDEN_DIMS,
        latent_dim=args.latent_dim,
    )


def _build_draw(args: argparse.Namespace) -> DRAWVAE:
    return DRAWVAE(
        img_channels=config.IMG_CHANNELS,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        T=args.T,
        z_dim=args.z_dim,
        h_dim=args.h_dim,
        N=args.N,
    )


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _train_epoch_vae(model, loader, optimizer, device, beta):
    model.train()
    total, recon_sum, kl_sum = 0.0, 0.0, 0.0
    for x in loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(x)
        loss, recon, kl = VAE.loss(x, x_hat, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()
        total += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
    n = len(loader)
    return total / n, recon_sum / n, kl_sum / n


@torch.no_grad()
def _val_epoch_vae(model, loader, device, beta):
    model.eval()
    total, recon_sum, kl_sum = 0.0, 0.0, 0.0
    for x in loader:
        x = x.to(device)
        x_hat, mu, logvar = model(x)
        loss, recon, kl = VAE.loss(x, x_hat, mu, logvar, beta=beta)
        total += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
    n = len(loader)
    return total / n, recon_sum / n, kl_sum / n


def _train_epoch_draw(model, loader, optimizer, device, beta):
    model.train()
    total, recon_sum, kl_sum = 0.0, 0.0, 0.0
    for x in loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_hat, mus, logvars = model(x)
        loss, recon, kl = DRAWVAE.loss(x, x_hat, mus, logvars, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
    n = len(loader)
    return total / n, recon_sum / n, kl_sum / n


@torch.no_grad()
def _val_epoch_draw(model, loader, device, beta):
    model.eval()
    total, recon_sum, kl_sum = 0.0, 0.0, 0.0
    for x in loader:
        x = x.to(device)
        x_hat, mus, logvars = model(x)
        loss, recon, kl = DRAWVAE.loss(x, x_hat, mus, logvars, beta=beta)
        total += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
    n = len(loader)
    return total / n, recon_sum / n, kl_sum / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    torch.manual_seed(config.SEED)
    device = torch.device(args.device)

    train_loader, val_loader = _build_dataloaders(args)

    # Build model
    if args.model == "vae":
        model = _build_vae(args).to(device)
        train_epoch = _train_epoch_vae
        val_epoch = _val_epoch_vae
    else:
        model = _build_draw(args).to(device)
        train_epoch = _train_epoch_draw
        val_epoch = _val_epoch_draw

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model.upper()}  |  Parameters: {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )

    ckpt_dir = Path(config.CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"{args.model}_losses.csv"
    best_val = float("inf")
    best_ckpt = ckpt_dir / f"{args.model}_best.pt"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_total", "train_recon", "train_kl",
                         "val_total", "val_recon", "val_kl"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_total, tr_recon, tr_kl = train_epoch(
                model, train_loader, optimizer, device, args.beta
            )
            vl_total, vl_recon, vl_kl = val_epoch(
                model, val_loader, device, args.beta
            )
            elapsed = time.time() - t0

            writer.writerow([epoch, tr_total, tr_recon, tr_kl,
                             vl_total, vl_recon, vl_kl])
            f.flush()

            print(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"Train: {tr_total:.4f} (recon={tr_recon:.4f}, kl={tr_kl:.4f})  "
                f"Val: {vl_total:.4f} (recon={vl_recon:.4f}, kl={vl_kl:.4f})  "
                f"[{elapsed:.1f}s]"
            )

            scheduler.step(vl_total)

            if vl_total < best_val:
                best_val = vl_total
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "args": vars(args)}, best_ckpt)

    print(f"\nBest validation loss: {best_val:.4f}")
    print(f"Checkpoint saved to: {best_ckpt}")
    print(f"Loss CSV saved to:   {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VAE or DRAW VAE on licence plates")
    p.add_argument("--model", choices=["vae", "draw"], default="vae",
                   help="Which model to train (default: vae)")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--beta", type=float, default=1.0,
                   help="KL weight in ELBO (beta-VAE). Default: 1.0")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # VAE-specific
    p.add_argument("--latent-dim", type=int, default=config.VAE_LATENT_DIM)

    # DRAW-specific
    p.add_argument("--T", type=int, default=config.DRAW_T,
                   help="Number of DRAW timesteps")
    p.add_argument("--z-dim", type=int, default=config.DRAW_Z_DIM)
    p.add_argument("--h-dim", type=int, default=config.DRAW_H_DIM)
    p.add_argument("--N", type=int, default=config.DRAW_ATTN_N,
                   help="Attention grid size")

    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
