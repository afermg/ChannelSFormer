"""
Unit tests for ChannelSFormer model.

Tests the core functionality of the ChannelSFormer architecture including:
- Model instantiation with various configurations
- Forward pass with different input shapes
- Attention mechanisms (divided, space-only, joint)
- Channel embedding behavior
- Classification token handling
"""

import pytest
import torch
import torch.nn as nn
from channelsformer.models.channelsformer import (
    ChannelSFormer,
    ChannelBlock,
    ChannelPatchEmbed,
    Attention,
)


class TestAttention:
    """Test cases for the Attention module."""

    def test_attention_init(self):
        """Test Attention module initialization."""
        attn = Attention(dim=192, num_heads=3)
        assert attn.num_heads == 3
        assert attn.head_dim == 64
        assert attn.scale == 64**-0.5

    def test_attention_forward(self):
        """Test Attention forward pass."""
        attn = Attention(dim=192, num_heads=3)
        x = torch.randn(2, 197, 192)  # (batch, seq_len, dim)
        output = attn(x)
        assert output.shape == x.shape

    def test_attention_with_dropout(self):
        """Test Attention with dropout enabled."""
        attn = Attention(dim=192, num_heads=3, attn_drop=0.1, proj_drop=0.1)
        attn.train()
        x = torch.randn(2, 197, 192)
        output = attn(x)
        assert output.shape == x.shape


class TestChannelPatchEmbed:
    """Test cases for the ChannelPatchEmbed module."""

    def test_channel_patch_embed_forward(self):
        """Test ChannelPatchEmbed processes channels independently."""
        patch_embed = ChannelPatchEmbed(
            img_size=224, patch_size=16, in_chans=1, embed_dim=192
        )
        x = torch.randn(2, 5, 224, 224)  # (batch, channels, height, width)
        output, num_channels = patch_embed(x)

        # Output should have shape (B*C, num_patches, embed_dim)
        expected_num_patches = (224 // 16) ** 2
        assert output.shape == (2 * 5, expected_num_patches, 192)
        assert num_channels == 5

    def test_channel_patch_embed_different_sizes(self):
        """Test ChannelPatchEmbed with different input sizes."""
        patch_embed = ChannelPatchEmbed(
            img_size=112, patch_size=8, in_chans=1, embed_dim=128, strict_img_size=False
        )
        x = torch.randn(1, 3, 112, 112)
        output, num_channels = patch_embed(x)

        expected_num_patches = (112 // 8) ** 2
        assert output.shape == (1 * 3, expected_num_patches, 128)
        assert num_channels == 3


class TestChannelBlock:
    """Test cases for the ChannelBlock module."""

    def test_channel_block_space_only(self):
        """Test ChannelBlock with space-only attention."""
        block = ChannelBlock(
            dim=192,
            num_heads=3,
            attention_type="space_only",
        )
        x = torch.randn(2, 197, 192)  # (batch, seq_len, dim)
        output = block(x, T=5)
        assert output.shape == x.shape

    def test_channel_block_divided_attention(self):
        """Test ChannelBlock with divided space-channel attention."""
        block = ChannelBlock(
            dim=192,
            num_heads=3,
            attention_type="divided_space_channel",
            separate_cls_for_channel=False,
        )
        # Shape: (batch, 1 + num_patches*channels, dim)
        x = torch.randn(2, 1 + 196 * 5, 192)
        output = block(x, T=5)
        assert output.shape == x.shape

    def test_channel_block_separate_cls(self):
        """Test ChannelBlock with separate cls tokens per channel."""
        block = ChannelBlock(
            dim=192,
            num_heads=3,
            attention_type="divided_space_channel",
            separate_cls_for_channel=True,
            SPACE_CHANNEL_ORDER="channel_first",
        )
        # Shape: (batch, (num_patches+1)*channels, dim)
        x = torch.randn(2, (196 + 1) * 5, 192)
        output = block(x, T=5)
        assert output.shape == x.shape

    def test_channel_block_parallel_attention(self):
        """Test ChannelBlock with parallel space-channel attention."""
        block = ChannelBlock(
            dim=192,
            num_heads=3,
            attention_type="divided_space_channel",
            separate_cls_for_channel=True,
            SPACE_CHANNEL_ORDER="parallel",
            no_additional_mapping=True,
        )
        x = torch.randn(2, (196 + 1) * 5, 192)
        output = block(x, T=5)
        assert output.shape == x.shape


class TestChannelSFormer:
    """Test cases for the ChannelSFormer model."""

    def test_channelsformer_tiny_init(self):
        """Test ChannelSFormer tiny model initialization."""
        model = ChannelSFormer(
            img_size=224,
            patch_size=16,
            in_chans=5,
            num_classes=1000,
            embed_dim=192,
            depth=12,
            num_heads=3,
        )
        assert model.embed_dim == 192
        assert model.in_chans == 5
        assert len(model.blocks) == 12

    def test_channelsformer_forward_pass(self):
        """Test ChannelSFormer forward pass with standard settings."""
        model = ChannelSFormer(
            img_size=224,
            patch_size=16,
            in_chans=5,
            num_classes=100,
            embed_dim=192,
            depth=2,  # Use shallow model for faster testing
            num_heads=3,
            attention_type="divided_space_channel",
        )
        x = torch.randn(2, 5, 224, 224)
        output = model(x)
        assert output.shape == (2, 100)

    def test_channelsformer_with_channel_embedding(self):
        """Test ChannelSFormer with channel embeddings enabled."""
        model = ChannelSFormer(
            img_size=224,
            patch_size=16,
            in_chans=5,
            num_classes=100,
            embed_dim=192,
            depth=2,
            num_heads=3,
            attention_type="divided_space_channel",
            use_channel_embedding=True,
        )
        x = torch.randn(2, 5, 224, 224)
        output = model(x)
        assert output.shape == (2, 100)
        assert hasattr(model, "channel_embed")

    def test_channelsformer_separate_cls_tokens(self):
        """Test ChannelSFormer with separate cls tokens per channel."""
        model = ChannelSFormer(
            img_size=224,
            patch_size=16,
            in_chans=5,
            num_classes=100,
            embed_dim=192,
            depth=2,
            num_heads=3,
            attention_type="divided_space_channel",
            separate_cls_for_channel=True,
            separate_cls_aggregation="mean",
        )
        x = torch.randn(2, 5, 224, 224)
        output = model(x)
        assert output.shape == (2, 100)

    def test_channelsformer_attention_pool(self):
        """Test ChannelSFormer with attention pooling for cls aggregation."""
        model = ChannelSFormer(
            img_size=224,
            patch_size=16,
            in_chans=5,
            num_classes=100,
            embed_dim=192,
            depth=2,
            num_heads=3,
            attention_type="divided_space_channel",
            separate_cls_for_channel=True,
            separate_cls_aggregation="att",
        )
        x = torch.randn(2, 5, 224, 224)
        output = model(x)
        assert output.shape == (2, 100)
        assert model.attn_pool is not None

    def test_channelsformer_space_first_order(self):
        """Test ChannelSFormer with space-first attention order."""
        model = ChannelSFormer(
            img_size=224,
            patch_size=16,
            in_chans=5,
            num_classes=100,
            embed_dim=192,
            depth=2,
            num_heads=3,
            attention_type="divided_space_channel",
            SPACE_CHANNEL_ORDER="space_first",
            separate_cls_for_channel=True,
        )
        x = torch.randn(2, 5, 224, 224)
        output = model(x)
        assert output.shape == (2, 100)

    def test_channelsformer_different_image_sizes(self):
        """Test ChannelSFormer with various input image sizes."""
        model = ChannelSFormer(
            img_size=112,
            patch_size=8,
            in_chans=3,
            num_classes=10,
            embed_dim=128,
            depth=2,
            num_heads=4,
        )
        x = torch.randn(1, 3, 112, 112)
        output = model(x)
        assert output.shape == (1, 10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_channelsformer_cuda(self):
        """Test ChannelSFormer on CUDA device."""
        model = ChannelSFormer(
            img_size=224,
            patch_size=16,
            in_chans=5,
            num_classes=100,
            embed_dim=192,
            depth=2,
            num_heads=3,
        ).cuda()
        x = torch.randn(2, 5, 224, 224).cuda()
        output = model(x)
        assert output.shape == (2, 100)
        assert output.is_cuda

    def test_channelsformer_no_class_token(self):
        """Test ChannelSFormer without class token (using global pooling)."""
        model = ChannelSFormer(
            img_size=224,
            patch_size=16,
            in_chans=5,
            num_classes=100,
            embed_dim=192,
            depth=2,
            num_heads=3,
            class_token=True,
            global_pool="avg",
        )
        x = torch.randn(2, 5, 224, 224)
        output = model(x)
        assert output.shape == (2, 100)

    def test_channelsformer_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = ChannelSFormer(
            img_size=224,
            patch_size=16,
            in_chans=5,
            num_classes=100,
            embed_dim=192,
            depth=2,
            num_heads=3,
        )
        x = torch.randn(2, 5, 224, 224, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert model.head.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
