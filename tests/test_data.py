"""
Unit tests for data loading and preprocessing.

Tests data loaders, augmentation pipelines, and dataset classes.
"""

from typing import DefaultDict
import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
from pathlib import Path

from channelsformer.config import get_config


class TestImageNetDataset:
    """Test cases for ImageNet dataset."""

    @pytest.fixture
    def temp_imagenet_data(self):
        """Create temporary ImageNet dataset for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small parquet file with fake image paths
            data = {
                "path": [f"{tmpdir}/img_{i}.jpg" for i in range(10)],
                "label": [i % 10 for i in range(10)],
            }
            df = pd.DataFrame(data)

            # Create dummy images
            for path in data["path"]:
                img = Image.fromarray(
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                )
                img.save(path)

            # Save parquet file
            parquet_path = Path(tmpdir) / "train.parquet"
            df.to_parquet(parquet_path)

            yield str(parquet_path), tmpdir

    def test_imagenet_dataset_loading(self, temp_imagenet_data):
        """Test ImageNet dataset can load data."""
        from channelsformer.data.imagenet import ImageNet

        parquet_path, _ = temp_imagenet_data
        dataset = ImageNet(data_path=parquet_path)

        assert len(dataset) == 10

    def test_imagenet_dataset_getitem(self, temp_imagenet_data):
        """Test ImageNet dataset __getitem__ returns correct format."""
        from channelsformer.data.imagenet import ImageNet

        parquet_path, _ = temp_imagenet_data
        dataset = ImageNet(data_path=parquet_path)

        img, label = dataset[0]

        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)  # C, H, W format after transform
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long


class TestDataAugmentation:
    """Test cases for data augmentation."""

    def test_build_transform_train(self):
        """Test training data augmentation pipeline."""
        from channelsformer.data.build import build_transform
        # create a namespace-like object with attributes
        args = type('Args', (), {'cfg': '', 'opts': False})()
        config = get_config(args)
        transform = build_transform(is_train=True, config=config)

        # Create a dummy image
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

        transformed = transform(img)

        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape[0] == 3  # 3 channels

    def test_build_transform_eval(self):
        """Test evaluation data augmentation pipeline."""
        from channelsformer.data.build import build_transform
        args = type('Args', (), {'cfg': '', 'opts': False})()
        config = get_config(args)
        transform = build_transform(is_train=False, config=config)

        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

        transformed = transform(img)

        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape[0] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
