"""DRAW: A Recurrent Neural Network For Image Generation (Gregor et al., 2015).

This module implements a DRAW-style VAE with selective attention for
licence-plate image reconstruction.

Architecture overview
---------------------
For each timestep t = 1 … T:

  1. **Read**:   r_t = read(x, x̂_{t-1}, h_dec_{t-1})
                 Extracts a glimpse from the input and the current error image
                 using a bank of N×N Gaussian filters.

  2. **Encode**: h_enc_t, c_enc_t = LSTM_enc([r_t, h_dec_{t-1}], (h_enc_{t-1}, c_enc_{t-1}))

  3. **Sample**: z_t ~ q_t(z) = N(μ_t, σ_t)
                 μ_t, log σ²_t = Linear(h_enc_t)

  4. **Decode**: h_dec_t, c_dec_t = LSTM_dec(z_t, (h_dec_{t-1}, c_dec_{t-1}))

  5. **Write**:  canvas_t = canvas_{t-1} + write(h_dec_t)
                 Writes a patch back to the canvas using transposed attention.

Final reconstruction: x̂ = σ(canvas_T)

Loss
----
  L = E[BCE(x̂, x)] + Σ_t KL(q_t(z) ‖ p(z))
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------

def _attention_params(
    h: torch.Tensor,
    img_height: int,
    img_width: int,
    N: int,
) -> Tuple[torch.Tensor, ...]:
    """Compute Gaussian filter bank parameters from a hidden vector.

    Returns (gx, gy, sigma², delta, gamma) each of shape ``(B,)``.

    The parameters follow the notation in Section 3.2 of the DRAW paper.
    """
    # Five scalar outputs per sample
    params = nn.functional.linear(
        h,
        h.new_zeros(5, h.size(-1)),  # placeholder; weights live in the module
    )
    # NOTE: The actual linear is defined in DRAWAttention; this function is
    #       called with already-projected params (see _read / _write below).
    g_x_hat, g_y_hat, log_sigma2, log_delta, log_gamma = params.unbind(-1)

    g_x = ((img_width + 1) / 2) * (g_x_hat + 1)
    g_y = ((img_height + 1) / 2) * (g_y_hat + 1)
    sigma2 = log_sigma2.exp()
    delta = ((max(img_height, img_width) - 1) / (N - 1)) * log_delta.exp()
    gamma = log_gamma.exp()
    return g_x, g_y, sigma2, delta, gamma


def _filterbank(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    sigma2: torch.Tensor,
    delta: torch.Tensor,
    N: int,
    img_size: int,
) -> torch.Tensor:
    """Return a (B, N, img_size) Gaussian filterbank matrix F_x or F_y."""
    i = torch.arange(1, N + 1, dtype=g_x.dtype, device=g_x.device)  # (N,)
    # mean of i-th filter: μ_i = g + (i - N/2 - 0.5) · δ
    mu = g_x.unsqueeze(1) + (i - N / 2.0 - 0.5) * delta.unsqueeze(1)  # (B, N)
    j = torch.arange(1, img_size + 1, dtype=g_x.dtype, device=g_x.device)  # (W,)
    # (B, N, W)
    f = torch.exp(-0.5 * (j.view(1, 1, -1) - mu.unsqueeze(-1)) ** 2 / sigma2.view(-1, 1, 1))
    # Normalise each filter to sum to 1
    f = f / (f.sum(-1, keepdim=True) + 1e-8)
    return f


class _AttentionRead(nn.Module):
    """Selective-attention read operation."""

    def __init__(self, h_dim: int, img_height: int, img_width: int, N: int) -> None:
        super().__init__()
        self.N = N
        self.H = img_height
        self.W = img_width
        self.fc_params = nn.Linear(h_dim, 5)

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        h_dec: torch.Tensor,
    ) -> torch.Tensor:
        """Return read vector of shape ``(B, 2*N*N)``."""
        B = x.size(0)
        p = self.fc_params(h_dec)  # (B, 5)
        g_x_hat, g_y_hat, log_sigma2, log_delta, log_gamma = p.unbind(-1)

        g_x = ((self.W + 1) / 2.0) * (torch.tanh(g_x_hat) + 1)
        g_y = ((self.H + 1) / 2.0) * (torch.tanh(g_y_hat) + 1)
        sigma2 = log_sigma2.exp()
        delta = ((max(self.H, self.W) - 1) / (self.N - 1)) * log_delta.exp()
        gamma = log_gamma.exp()

        Fx = _filterbank(g_x, g_y, sigma2, delta, self.N, self.W)  # (B, N, W)
        Fy = _filterbank(g_y, g_x, sigma2, delta, self.N, self.H)  # (B, N, H)

        # Apply filters: glimpse = Fy · img · Fx^T
        def _apply(img: torch.Tensor) -> torch.Tensor:
            # img: (B, H, W)
            tmp = torch.bmm(Fy, img)          # (B, N, W)
            tmp = torch.bmm(tmp, Fx.permute(0, 2, 1))  # (B, N, N)
            return gamma.view(B, 1, 1) * tmp

        x_2d = x.view(B, self.H, self.W)
        x_hat_2d = x_hat.view(B, self.H, self.W)
        r_x = _apply(x_2d).view(B, -1)
        r_x_hat = _apply(x_hat_2d).view(B, -1)
        return torch.cat([r_x, r_x_hat], dim=1)  # (B, 2*N*N)


class _AttentionWrite(nn.Module):
    """Selective-attention write operation."""

    def __init__(self, h_dim: int, img_height: int, img_width: int, N: int) -> None:
        super().__init__()
        self.N = N
        self.H = img_height
        self.W = img_width
        self.fc_patch = nn.Linear(h_dim, N * N)
        self.fc_params = nn.Linear(h_dim, 5)

    def forward(self, h_dec: torch.Tensor) -> torch.Tensor:
        """Return a write patch of shape ``(B, H*W)``."""
        B = h_dec.size(0)
        w = self.fc_patch(h_dec).view(B, self.N, self.N)  # (B, N, N)

        p = self.fc_params(h_dec)  # (B, 5)
        g_x_hat, g_y_hat, log_sigma2, log_delta, log_gamma = p.unbind(-1)

        g_x = ((self.W + 1) / 2.0) * (torch.tanh(g_x_hat) + 1)
        g_y = ((self.H + 1) / 2.0) * (torch.tanh(g_y_hat) + 1)
        sigma2 = log_sigma2.exp()
        delta = ((max(self.H, self.W) - 1) / (self.N - 1)) * log_delta.exp()
        gamma = log_gamma.exp()

        Fx = _filterbank(g_x, g_y, sigma2, delta, self.N, self.W)  # (B, N, W)
        Fy = _filterbank(g_y, g_x, sigma2, delta, self.N, self.H)  # (B, N, H)

        # Write: patch_full = Fy^T · w · Fx
        out = torch.bmm(Fy.permute(0, 2, 1), w)   # (B, H, N)
        out = torch.bmm(out, Fx)                    # (B, H, W)
        out = (1.0 / (gamma.view(B, 1, 1) + 1e-8)) * out
        return out.view(B, -1)  # (B, H*W)


# ---------------------------------------------------------------------------
# DRAW VAE
# ---------------------------------------------------------------------------

class DRAWVAE(nn.Module):
    """DRAW Variational Autoencoder for licence-plate image reconstruction.

    Parameters
    ----------
    img_channels:
        Number of image channels (1 for grayscale).
    img_height, img_width:
        Spatial dimensions of the input image.
    T:
        Number of sequential read/write steps.
    z_dim:
        Latent dimensionality per step.
    h_dim:
        LSTM hidden state size for both encoder and decoder.
    N:
        Attention grid size (N×N Gaussian filters).
    """

    def __init__(
        self,
        img_channels: int = 1,
        img_height: int = 32,
        img_width: int = 64,
        T: int = 10,
        z_dim: int = 32,
        h_dim: int = 256,
        N: int = 12,
    ) -> None:
        super().__init__()
        self.T = T
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.N = N
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self._img_flat = img_channels * img_height * img_width

        # Attention modules
        self.read = _AttentionRead(h_dim, img_height, img_width, N)
        self.write = _AttentionWrite(h_dim, img_height, img_width, N)

        # Encoder LSTM: input = [read_out, h_dec_prev]
        read_size = 2 * N * N  # two N×N glimpses concatenated
        self.encoder_lstm = nn.LSTMCell(read_size + h_dim, h_dim)

        # Latent projections
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

        # Decoder LSTM: input = z_t
        self.decoder_lstm = nn.LSTMCell(z_dim, h_dim)

        # Write projection: canvas update is written to the flat image
        self.fc_write = nn.Linear(h_dim, self._img_flat // img_channels)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Run DRAW for T timesteps.

        Parameters
        ----------
        x:
            Input images, shape ``(B, C, H, W)``.

        Returns
        -------
        x_hat:
            Reconstructed images, shape ``(B, C, H, W)``.
        mus:
            List of T mean tensors each of shape ``(B, z_dim)``.
        logvars:
            List of T log-variance tensors, same shape.
        """
        B = x.size(0)
        device = x.device

        # Flatten to (B, H*W) for canvas arithmetic (single channel assumed)
        x_flat = x.view(B, -1)

        # Initial states
        h_enc = x.new_zeros(B, self.h_dim)
        c_enc = x.new_zeros(B, self.h_dim)
        h_dec = x.new_zeros(B, self.h_dim)
        c_dec = x.new_zeros(B, self.h_dim)
        canvas = x.new_zeros(B, self._img_flat // self.img_channels)

        mus: list[torch.Tensor] = []
        logvars: list[torch.Tensor] = []

        for _ in range(self.T):
            # Error image
            x_hat_t = torch.sigmoid(canvas).view(B, self.img_channels, self.img_height, self.img_width)
            x_err = x - x_hat_t  # (B, C, H, W)

            # Read
            r = self.read(x, x_err, h_dec)  # (B, 2*N*N)

            # Encode
            h_enc, c_enc = self.encoder_lstm(torch.cat([r, h_dec], dim=1), (h_enc, c_enc))

            # Latent sample
            mu = self.fc_mu(h_enc)
            logvar = self.fc_logvar(h_enc)
            z = self._reparameterise(mu, logvar)
            mus.append(mu)
            logvars.append(logvar)

            # Decode
            h_dec, c_dec = self.decoder_lstm(z, (h_dec, c_dec))

            # Write
            canvas = canvas + self.write(h_dec)

        x_hat = torch.sigmoid(canvas).view(B, self.img_channels, self.img_height, self.img_width)
        return x_hat, mus, logvars

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Draw images from the prior p(z) = N(0, I)."""
        B = num_samples
        h_dec = torch.zeros(B, self.h_dim, device=device)
        c_dec = torch.zeros(B, self.h_dim, device=device)
        canvas = torch.zeros(B, self._img_flat // self.img_channels, device=device)

        for _ in range(self.T):
            z = torch.randn(B, self.z_dim, device=device)
            h_dec, c_dec = self.decoder_lstm(z, (h_dec, c_dec))
            canvas = canvas + self.write(h_dec)

        return torch.sigmoid(canvas).view(
            B, self.img_channels, self.img_height, self.img_width
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mus: list[torch.Tensor],
        logvars: list[torch.Tensor],
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """DRAW ELBO loss.

        Returns
        -------
        total, recon_loss, kl_loss – all scalar tensors.
        """
        recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
        kl = sum(
            -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            for mu, lv in zip(mus, logvars)
        )
        return recon + beta * kl, recon, kl
