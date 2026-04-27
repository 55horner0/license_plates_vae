"""Synthetic license plate dataset.

Generates grayscale license plate images on the fly using PIL so the project
works without downloading an external dataset.  Real images stored on disk can
also be loaded via ``LicensePlateImageFolder``.

Plate format used for synthesis: ``AB-1234`` (2 letters, hyphen, 4 digits).
"""

import random
import string
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision import transforms


def _random_plate_text() -> str:
    letters = "".join(random.choices(string.ascii_uppercase, k=2))
    digits = "".join(random.choices(string.digits, k=4))
    return f"{letters}-{digits}"


def _render_plate(
    text: str,
    width: int = 64,
    height: int = 32,
    noise_std: float = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """Render a licence plate as a grayscale PIL Image."""
    img = Image.new("L", (width, height), color=220)
    draw = ImageDraw.Draw(img)

    # Border
    draw.rectangle([1, 1, width - 2, height - 2], outline=0, width=1)

    # Text – use the built-in bitmap font so no font file is needed
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) // 2
    y = (height - text_h) // 2
    draw.text((x, y), text, fill=0, font=font)

    # Mild Gaussian noise for realism
    _rng = rng if rng is not None else np.random.default_rng()
    arr = np.array(img, dtype=np.float32)
    arr += _rng.normal(0, noise_std, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class SyntheticLicensePlateDataset(Dataset):
    """Generates synthetic grayscale licence plate images on the fly.

    Parameters
    ----------
    num_samples:
        Number of samples in the dataset.
    width, height:
        Output image dimensions (pixels).
    transform:
        Optional torchvision transform applied after converting to tensor.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        width: int = 64,
        height: int = 32,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.num_samples = num_samples
        self.width = width
        self.height = height
        self.transform = transform
        self._seed = seed if seed is not None else 0

        rng = random.Random(seed)
        self._texts: list[str] = [
            "".join(rng.choices(string.ascii_uppercase, k=2))
            + "-"
            + "".join(rng.choices(string.digits, k=4))
            for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # Per-index seeded RNG for fully reproducible samples
        item_rng = np.random.default_rng(self._seed + idx)
        noise_std = float(item_rng.uniform(5, 15))
        img = _render_plate(
            self._texts[idx],
            width=self.width,
            height=self.height,
            noise_std=noise_std,
            rng=item_rng,
        )
        to_tensor = transforms.ToTensor()
        tensor = to_tensor(img)  # shape (1, H, W), values in [0, 1]
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor


class LicensePlateImageFolder(Dataset):
    """Load licence plate images from a directory of PNG/JPEG files.

    All images are resized to ``(height, width)`` and converted to grayscale.

    Parameters
    ----------
    root:
        Path to the directory containing image files.
    width, height:
        Target image dimensions.
    transform:
        Optional torchvision transform applied after the default pipeline.
    extensions:
        Allowed file extensions (case-insensitive).
    """

    _EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(
        self,
        root: str | Path,
        width: int = 64,
        height: int = 32,
        transform: Optional[Callable] = None,
        extensions: Optional[set[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.width = width
        self.height = height
        self.transform = transform
        exts = extensions if extensions is not None else self._EXTENSIONS
        self._paths = sorted(
            p for p in self.root.iterdir() if p.suffix.lower() in exts
        )
        if not self._paths:
            raise FileNotFoundError(
                f"No image files found in '{self.root}'. "
                "Supported extensions: " + ", ".join(sorted(exts))
            )
        self._pipeline = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((height, width)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int):
        img = Image.open(self._paths[idx]).convert("RGB")
        tensor = self._pipeline(img)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor
