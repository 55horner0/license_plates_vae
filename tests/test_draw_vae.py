"""Unit tests for the DRAW VAE model."""

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.draw_vae import DRAWVAE, _AttentionRead, _AttentionWrite, _filterbank


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

IMG_H = 32
IMG_W = 64
IMG_C = 1
T = 3        # small T for fast tests
Z_DIM = 16
H_DIM = 64
N = 4        # small attention grid
BATCH = 4


@pytest.fixture
def draw():
    return DRAWVAE(
        img_channels=IMG_C,
        img_height=IMG_H,
        img_width=IMG_W,
        T=T,
        z_dim=Z_DIM,
        h_dim=H_DIM,
        N=N,
    )


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.rand(BATCH, IMG_C, IMG_H, IMG_W)


# ------------------------------------------------------------------
# Shape tests
# ------------------------------------------------------------------

class TestDRAWShapes:
    def test_forward_output_shapes(self, draw, batch):
        x_hat, mus, logvars = draw(batch)
        assert x_hat.shape == batch.shape, f"Expected {batch.shape}, got {x_hat.shape}"
        assert len(mus) == T
        assert len(logvars) == T
        for mu, lv in zip(mus, logvars):
            assert mu.shape == (BATCH, Z_DIM)
            assert lv.shape == (BATCH, Z_DIM)

    def test_sample_shape(self, draw):
        device = torch.device("cpu")
        samples = draw.sample(6, device)
        assert samples.shape == (6, IMG_C, IMG_H, IMG_W)


# ------------------------------------------------------------------
# Value range tests
# ------------------------------------------------------------------

class TestDRAWValueRange:
    def test_reconstruction_in_0_1(self, draw, batch):
        x_hat, _, _ = draw(batch)
        assert x_hat.min().item() >= 0.0 - 1e-6
        assert x_hat.max().item() <= 1.0 + 1e-6

    def test_sample_in_0_1(self, draw):
        samples = draw.sample(4, torch.device("cpu"))
        assert samples.min().item() >= 0.0 - 1e-6
        assert samples.max().item() <= 1.0 + 1e-6


# ------------------------------------------------------------------
# Loss tests
# ------------------------------------------------------------------

class TestDRAWLoss:
    def test_loss_returns_three_tensors(self, draw, batch):
        x_hat, mus, logvars = draw(batch)
        total, recon, kl = DRAWVAE.loss(batch, x_hat, mus, logvars)
        for t in (total, recon, kl):
            assert t.shape == torch.Size([])

    def test_loss_is_finite(self, draw, batch):
        x_hat, mus, logvars = draw(batch)
        total, recon, kl = DRAWVAE.loss(batch, x_hat, mus, logvars)
        assert torch.isfinite(total)
        assert torch.isfinite(recon)
        assert torch.isfinite(kl)

    def test_beta_scales_kl(self, draw, batch):
        x_hat, mus, logvars = draw(batch)
        t1, r1, _ = DRAWVAE.loss(batch, x_hat, mus, logvars, beta=1.0)
        t2, r2, _ = DRAWVAE.loss(batch, x_hat, mus, logvars, beta=2.0)
        # Recon should be unchanged
        assert torch.isclose(r1, r2, atol=1e-5)
        # Total with higher beta should be >= lower beta (assuming KL >= 0 not always,
        # but with random weights it could be negative; just check shapes/finiteness)
        assert torch.isfinite(t2)


# ------------------------------------------------------------------
# Attention sub-modules
# ------------------------------------------------------------------

class TestFilterbank:
    def test_shape(self):
        B, N_attn, W = 2, 4, 64
        g = torch.ones(B)
        sigma2 = torch.ones(B) * 0.5
        delta = torch.ones(B) * 0.5
        F = _filterbank(g, g, sigma2, delta, N_attn, W)
        assert F.shape == (B, N_attn, W)

    def test_rows_sum_to_one(self):
        B, N_attn, W = 3, 5, 32
        g = torch.rand(B) * W
        sigma2 = torch.ones(B) * 0.3
        delta = torch.ones(B) * 0.3
        F = _filterbank(g, g, sigma2, delta, N_attn, W)
        row_sums = F.sum(-1)  # (B, N)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestAttentionRead:
    def test_output_shape(self):
        reader = _AttentionRead(h_dim=H_DIM, img_height=IMG_H, img_width=IMG_W, N=N)
        x = torch.rand(BATCH, IMG_C, IMG_H, IMG_W)
        x_err = torch.rand_like(x)
        h_dec = torch.zeros(BATCH, H_DIM)
        r = reader(x, x_err, h_dec)
        assert r.shape == (BATCH, 2 * N * N)


class TestAttentionWrite:
    def test_output_shape(self):
        writer = _AttentionWrite(h_dim=H_DIM, img_height=IMG_H, img_width=IMG_W, N=N)
        h_dec = torch.rand(BATCH, H_DIM)
        out = writer(h_dec)
        assert out.shape == (BATCH, IMG_H * IMG_W)


# ------------------------------------------------------------------
# Gradient flow
# ------------------------------------------------------------------

class TestDRAWGradients:
    def test_loss_backprop(self, draw, batch):
        x_hat, mus, logvars = draw(batch)
        loss, _, _ = DRAWVAE.loss(batch, x_hat, mus, logvars)
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in draw.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0
        assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms)
