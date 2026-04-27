"""Unit tests for the standard VAE model."""

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae import VAE, VAEEncoder, VAEDecoder


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

IMG_H = 32
IMG_W = 64
IMG_C = 1
HIDDEN_DIMS = [32, 64, 128]
LATENT_DIM = 64
BATCH = 4


@pytest.fixture
def vae():
    return VAE(
        img_channels=IMG_C,
        img_height=IMG_H,
        img_width=IMG_W,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
    )


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.rand(BATCH, IMG_C, IMG_H, IMG_W)


# ------------------------------------------------------------------
# Shape tests
# ------------------------------------------------------------------

class TestVAEShapes:
    def test_encoder_output_shapes(self, vae, batch):
        mu, logvar = vae.encode(batch)
        assert mu.shape == (BATCH, LATENT_DIM), f"Expected ({BATCH}, {LATENT_DIM}), got {mu.shape}"
        assert logvar.shape == (BATCH, LATENT_DIM)

    def test_decoder_output_shape(self, vae):
        z = torch.randn(BATCH, LATENT_DIM)
        x_hat = vae.decode(z)
        assert x_hat.shape == (BATCH, IMG_C, IMG_H, IMG_W)

    def test_forward_shapes(self, vae, batch):
        x_hat, mu, logvar = vae(batch)
        assert x_hat.shape == batch.shape
        assert mu.shape == (BATCH, LATENT_DIM)
        assert logvar.shape == (BATCH, LATENT_DIM)

    def test_sample_shape(self, vae):
        device = torch.device("cpu")
        samples = vae.sample(8, device)
        assert samples.shape == (8, IMG_C, IMG_H, IMG_W)


# ------------------------------------------------------------------
# Value range tests
# ------------------------------------------------------------------

class TestVAEValueRange:
    def test_reconstruction_in_0_1(self, vae, batch):
        x_hat, _, _ = vae(batch)
        assert x_hat.min().item() >= 0.0 - 1e-6
        assert x_hat.max().item() <= 1.0 + 1e-6

    def test_sample_in_0_1(self, vae):
        samples = vae.sample(4, torch.device("cpu"))
        assert samples.min().item() >= 0.0 - 1e-6
        assert samples.max().item() <= 1.0 + 1e-6


# ------------------------------------------------------------------
# Loss tests
# ------------------------------------------------------------------

class TestVAELoss:
    def test_loss_returns_three_tensors(self, vae, batch):
        x_hat, mu, logvar = vae(batch)
        total, recon, kl = VAE.loss(batch, x_hat, mu, logvar)
        assert total.shape == torch.Size([])
        assert recon.shape == torch.Size([])
        assert kl.shape == torch.Size([])

    def test_loss_is_finite(self, vae, batch):
        x_hat, mu, logvar = vae(batch)
        total, recon, kl = VAE.loss(batch, x_hat, mu, logvar)
        assert torch.isfinite(total)
        assert torch.isfinite(recon)
        assert torch.isfinite(kl)

    def test_total_equals_recon_plus_kl(self, vae, batch):
        x_hat, mu, logvar = vae(batch)
        total, recon, kl = VAE.loss(batch, x_hat, mu, logvar, beta=1.0)
        assert torch.isclose(total, recon + kl, atol=1e-4)

    def test_beta_scales_kl(self, vae, batch):
        x_hat, mu, logvar = vae(batch)
        total_b1, recon_b1, kl_b1 = VAE.loss(batch, x_hat, mu, logvar, beta=1.0)
        total_b2, recon_b2, kl_b2 = VAE.loss(batch, x_hat, mu, logvar, beta=2.0)
        assert torch.isclose(recon_b1, recon_b2, atol=1e-5)
        # raw KL should be identical regardless of beta
        assert torch.isclose(kl_b1, kl_b2, atol=1e-5)
        # difference in totals should equal one additional kl term
        assert torch.isclose(total_b2 - total_b1, kl_b1, atol=1e-4)


# ------------------------------------------------------------------
# Reparameterisation
# ------------------------------------------------------------------

class TestReparameterise:
    def test_shape(self):
        mu = torch.zeros(4, 32)
        logvar = torch.zeros(4, 32)
        z = VAE.reparameterise(mu, logvar)
        assert z.shape == (4, 32)

    def test_deterministic_at_zero_variance(self):
        mu = torch.ones(4, 16) * 3.0
        logvar = torch.full((4, 16), -20.0)  # very small variance
        z = VAE.reparameterise(mu, logvar)
        assert torch.allclose(z, mu, atol=1e-2)


# ------------------------------------------------------------------
# Gradient flow
# ------------------------------------------------------------------

class TestVAEGradients:
    def test_loss_backprop(self, vae, batch):
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        x_hat, mu, logvar = vae(batch)
        loss, _, _ = VAE.loss(batch, x_hat, mu, logvar)
        loss.backward()
        # At least some parameters should have gradients
        grad_norms = [
            p.grad.norm().item()
            for p in vae.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0
        assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms)
