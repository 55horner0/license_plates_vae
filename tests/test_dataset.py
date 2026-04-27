"""Unit tests for the synthetic licence plate dataset."""

import pytest
import torch
from torch.utils.data import DataLoader

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import SyntheticLicensePlateDataset


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

IMG_H = 32
IMG_W = 64
NUM_SAMPLES = 20


@pytest.fixture
def dataset():
    return SyntheticLicensePlateDataset(
        num_samples=NUM_SAMPLES,
        width=IMG_W,
        height=IMG_H,
        seed=42,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestSyntheticDataset:
    def test_len(self, dataset):
        assert len(dataset) == NUM_SAMPLES

    def test_item_shape(self, dataset):
        sample = dataset[0]
        assert sample.shape == (1, IMG_H, IMG_W), f"Expected (1, {IMG_H}, {IMG_W}), got {sample.shape}"

    def test_item_value_range(self, dataset):
        sample = dataset[0]
        assert sample.min().item() >= 0.0 - 1e-6
        assert sample.max().item() <= 1.0 + 1e-6

    def test_deterministic_with_seed(self):
        ds1 = SyntheticLicensePlateDataset(num_samples=5, seed=7)
        ds2 = SyntheticLicensePlateDataset(num_samples=5, seed=7)
        for i in range(5):
            assert torch.allclose(ds1[i], ds2[i])

    def test_different_seeds_differ(self):
        ds1 = SyntheticLicensePlateDataset(num_samples=5, seed=1)
        ds2 = SyntheticLicensePlateDataset(num_samples=5, seed=2)
        # At least one sample should differ
        diffs = [not torch.allclose(ds1[i], ds2[i]) for i in range(5)]
        assert any(diffs)

    def test_dataloader_batch_shape(self, dataset):
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch.shape == (4, 1, IMG_H, IMG_W)

    def test_all_items_accessible(self, dataset):
        for i in range(len(dataset)):
            item = dataset[i]
            assert item.shape == (1, IMG_H, IMG_W)
