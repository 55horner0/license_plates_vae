"""Standard convolutional Variational Autoencoder (VAE).

Architecture
------------
* **Encoder**: three Conv2d blocks (each Conv → BatchNorm → ReLU) followed by
  two parallel linear heads that produce the mean (μ) and log-variance
  (log σ²) of the approximate posterior q(z|x).
* **Reparameterisation**: z = μ + ε · σ, ε ~ N(0, I).
* **Decoder**: linear projection → three ConvTranspose2d blocks → sigmoid
  output in [0, 1].

Loss
----
  ELBO = E[log p(x|z)] − KL(q(z|x) ‖ p(z))
       = −BCE(x̂, x)   − 0.5 · Σ(1 + log σ² − μ² − σ²)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    """Convolutional encoder that outputs q(z|x) parameters."""

    def __init__(
        self,
        in_channels: int,
        hidden_dims: list[int],
        latent_dim: int,
        img_height: int,
        img_width: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        layers: list[nn.Module] = []
        ch = in_channels
        for h in hidden_dims:
            layers += [
                nn.Conv2d(ch, h, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h),
                nn.ReLU(inplace=True),
            ]
            ch = h
        self.convs = nn.Sequential(*layers)

        # Compute flattened size after convolutions
        scale = 2 ** len(hidden_dims)
        self._flat_h = img_height // scale
        self._flat_w = img_width // scale
        flat_size = hidden_dims[-1] * self._flat_h * self._flat_w

        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.convs(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    """Transposed-convolutional decoder that maps z → x̂."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        out_channels: int,
        img_height: int,
        img_width: int,
    ) -> None:
        super().__init__()

        scale = 2 ** len(hidden_dims)
        self._flat_h = img_height // scale
        self._flat_w = img_width // scale
        flat_size = hidden_dims[-1] * self._flat_h * self._flat_w
        self._first_ch = hidden_dims[-1]

        self.fc = nn.Linear(latent_dim, flat_size)

        layers: list[nn.Module] = []
        rev = list(reversed(hidden_dims))
        for i in range(len(rev) - 1):
            layers += [
                nn.ConvTranspose2d(rev[i], rev[i + 1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(rev[i + 1]),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.ConvTranspose2d(rev[-1], out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        ]
        self.deconvs = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), self._first_ch, self._flat_h, self._flat_w)
        return self.deconvs(h)


class VAE(nn.Module):
    """Variational Autoencoder for licence-plate image reconstruction.

    Parameters
    ----------
    img_channels:
        Number of input/output image channels (1 for grayscale).
    img_height, img_width:
        Spatial dimensions of the input image.
    hidden_dims:
        List of channel widths for each encoder convolutional block.  The
        decoder mirrors these in reverse order.
    latent_dim:
        Dimensionality of the latent space z.
    """

    def __init__(
        self,
        img_channels: int = 1,
        img_height: int = 32,
        img_width: int = 64,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width

        self.encoder = VAEEncoder(img_channels, hidden_dims, latent_dim, img_height, img_width)
        self.decoder = VAEDecoder(latent_dim, hidden_dims, img_channels, img_height, img_width)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ q(z|x) using the reparameterisation trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (μ, log σ²) for the approximate posterior."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct an image from a latent code z."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: returns (x̂, μ, log σ²)."""
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Draw *num_samples* images from the prior p(z) = N(0, I)."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ELBO loss (negated for minimisation).

        Parameters
        ----------
        x:
            Original images, shape ``(B, C, H, W)``.
        x_hat:
            Reconstructed images, same shape.
        mu, logvar:
            Encoder outputs.
        beta:
            Weight on the KL term (β-VAE formulation).

        Returns
        -------
        total, recon_loss, kl_loss – all scalar tensors.
        """
        recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kl, recon, kl
