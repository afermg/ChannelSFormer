import math
import warnings
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

# from timm.models.layers.mlp import Mlp
from timm.layers.mlp import Mlp
from timm.models.layers import lecun_normal_, trunc_normal_


class AttentionPoolLatentCus(nn.Module):
    """Attention pooling w/ latent query
    Sigmoid Loss for Language Image Pre-Training

    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        latent_len: int = 1,
        latent_dim: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
        act_layer: Optional[Type[nn.Module]] = nn.GELU,
        map_q: bool = True,
        use_proj: bool = True,
        use_mlp: bool = True,
        post_norm: bool = True,
        drop: float = 0.0,
        fused_attn: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        assert latent_len == 1
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim))

        if map_q:
            self.q: nn.Module = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        else:
            self.q = nn.Identity()

        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        if qk_norm:
            qk_norm_layer = norm_layer or nn.LayerNorm
            self.q_norm = qk_norm_layer(self.head_dim)
            self.k_norm = qk_norm_layer(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        if use_proj:
            self.proj: nn.Module = nn.Linear(embed_dim, embed_dim)
            self.proj_drop: nn.Module = nn.Dropout(drop)
        else:
            self.proj = nn.Identity()
            self.proj_drop = nn.Identity()

        if use_mlp:
            self.norm: Optional[nn.Module] = (
                norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
            )
            self.mlp: Optional[nn.Module] = Mlp(
                embed_dim, int(embed_dim * mlp_ratio), act_layer=act_layer
            )
        else:
            self.norm = None
            self.mlp = None

        if post_norm:
            post_norm_layer = norm_layer or nn.LayerNorm
            self.post_norm = post_norm_layer(embed_dim)
        else:
            self.post_norm = nn.Identity()

        self.init_weights()

    def init_weights(self):
        trunc_normal_tf_(self.latent, std=self.latent_dim**-0.5)

        self.q.apply(segm_init_weights)
        self.kv.apply(segm_init_weights)

        self.q_norm.apply(segm_init_weights)
        self.k_norm.apply(segm_init_weights)

        self.proj.apply(segm_init_weights)
        self.post_norm.apply(segm_init_weights)

    def forward(self, x):
        B, N, C = x.shape

        q_latent = self.latent.expand(B, -1, -1)
        q = (
            self.q(q_latent)
            .reshape(B, self.latent_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        # as self.latent_len == 1
        x = x.reshape(B, self.latent_len, C)
        # x = x.transpose(1, 2).reshape(B, self.latent_len, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.mlp is not None:
            x = x + self.mlp(self.norm(x))

        x = self.post_norm(x)

        return x.squeeze(1)


def trunc_normal_tf_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    r"""
    from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/weight_init.py#L70
    """
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)
    return tensor


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l_ = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l_ - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
