# license_plates_vae

A deep learning project comparing two generative models — a standard **Variational Autoencoder (VAE)** and a **DRAW VAE** (Deep Recurrent Attentive Writer) — for licence plate image reconstruction.

---

## Overview

| Model | Architecture | Latent space | Attention |
|-------|-------------|--------------|-----------|
| **VAE** | Conv encoder → FC heads → ConvTranspose decoder | Single Gaussian | ✗ |
| **DRAW VAE** | LSTM encoder/decoder with sequential read/write | Sum of T Gaussians | ✓ (Gaussian filterbanks) |

Both models are trained on a synthetic greyscale licence plate dataset (format `AB-1234`) that is generated on the fly — no external data download required.

---

## Project structure

```
license_plates_vae/
├── config.py            # Shared hyper-parameters
├── train.py             # Training script (VAE or DRAW VAE)
├── evaluate.py          # Comparison metrics & plots
├── requirements.txt
├── data/
│   └── dataset.py       # Synthetic dataset + real-image loader
├── models/
│   ├── vae.py           # Standard convolutional VAE
│   └── draw_vae.py      # DRAW VAE with attention
├── utils/
│   └── visualization.py # Plotting utilities
└── tests/               # pytest unit tests
    ├── test_dataset.py
    ├── test_vae.py
    └── test_draw_vae.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

Train the **standard VAE** for 50 epochs:

```bash
python train.py --model vae --epochs 50 --batch-size 32 --latent-dim 128
```

Train the **DRAW VAE** (10 sequential attention steps):

```bash
python train.py --model draw --epochs 50 --batch-size 32 --T 10 --z-dim 32
```

Checkpoints are saved to `checkpoints/<model>_best.pt` and per-epoch losses
are written to `results/<model>_losses.csv`.

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `vae` | `vae` or `draw` |
| `--epochs` | `50` | Training epochs |
| `--batch-size` | `32` | Mini-batch size |
| `--lr` | `1e-3` | Learning rate |
| `--beta` | `1.0` | β-VAE KL weight |
| `--latent-dim` | `128` | VAE latent dim |
| `--T` | `10` | DRAW timesteps |
| `--z-dim` | `32` | DRAW per-step latent dim |
| `--h-dim` | `256` | DRAW LSTM hidden size |
| `--N` | `12` | DRAW attention grid size |
| `--device` | auto | `cpu` or `cuda` |

---

## Evaluation

After training both models, run:

```bash
python evaluate.py \
    --vae-ckpt  checkpoints/vae_best.pt \
    --draw-ckpt checkpoints/draw_best.pt
```

This prints a comparison table of **MSE**, **SSIM**, and **ELBO loss**, saves
`results/comparison.csv`, `results/reconstructions.png`, and
`results/samples.png`.

---

## Running the tests

```bash
python -m pytest tests/ -v
```

---

## Models

### VAE

The encoder applies three strided Conv2d blocks (channels: 32 → 64 → 128) to
produce μ and log σ² vectors.  The decoder mirrors this with ConvTranspose2d
blocks.  The ELBO objective is:

```
L = E[BCE(x̂, x)] − β · KL(q(z|x) ‖ p(z))
```

### DRAW VAE

DRAW (Gregor et al., 2015) builds a reconstruction iteratively over T steps.
At each step the model:
1. **Reads** a glimpse from the input and error image using N×N Gaussian filterbanks.
2. **Encodes** the glimpse with an LSTM to produce a per-step latent zₜ.
3. **Decodes** zₜ with a second LSTM.
4. **Writes** a patch back to a shared canvas via transposed attention.

The loss sums KL divergences across all T steps:

```
L = E[BCE(x̂_T, x)] + Σ_t KL(q_t(z) ‖ p(z))
```

---

## References

* Kingma & Welling (2013). *Auto-Encoding Variational Bayes.* [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
* Gregor et al. (2015). *DRAW: A Recurrent Neural Network For Image Generation.* [arXiv:1502.04623](https://arxiv.org/abs/1502.04623)
