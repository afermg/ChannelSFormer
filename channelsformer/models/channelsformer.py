"""
ChannelSFormer: A Channel Agnostic Vision Transformer for Multi-Channel Images

This module implements the ChannelSFormer architecture, which extends the Vision Transformer
to handle multi-channel images (e.g., microscopy data with 5+ channels) in a channel-agnostic way.

Key innovations:
- Divided space-channel attention: Separates spatial and channel-wise attention
- Channel embeddings: Optional learnable embeddings for each input channel
- Flexible cls token handling: Supports separate cls tokens per channel or shared
- Multiple attention orders: channel-first, space-first, or parallel processing

Architecture overview:
1. ChannelPatchEmbed: Processes each channel independently through patch embedding
2. ChannelBlock: Transformer blocks with divided space-channel attention
3. ChannelSFormer: Full model combining embedding, blocks, and classification head
"""

import math
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Set, Tuple, Type, Union

import torch
from einops import rearrange
from timm.layers import (
    AttentionPoolLatent,
    DropPath,
    LayerType,
    Mlp,
    PatchDropout,
    get_act_layer,
    get_norm_layer,
    resample_abs_pos_embed,
    trunc_normal_,
)

from timm.layers.patch_embed import F, Format, PatchEmbed, _assert, nchw_to
from timm.models.vision_transformer import (
    _load_weights,
    get_init_weights_vit,
    global_pool_nlc,
    init_weights_vit_timm,
    named_apply,
)
from torch import nn

from channelsformer.models.models_utils import (
    AttentionPoolLatentCus,
)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        fused_attn: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ChannelPatchEmbed(PatchEmbed):
    """
    Channel-aware patch embedding layer.

    Extends PatchEmbed to process multi-channel images by treating each channel
    independently. Input shape (B, C, H, W) is reshaped to (B*C, 1, H, W) so each
    channel goes through patch embedding separately.

    Returns
    -------
    tuple
        (embedded_patches, num_channels) where embedded_patches has shape
        (B*C, num_patches, embed_dim) and num_channels is C
    """
    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(
                    H == self.img_size[0],
                    f"Input height ({H}) doesn't match model ({self.img_size[0]}).",
                )
                _assert(
                    W == self.img_size[1],
                    f"Input width ({W}) doesn't match model ({self.img_size[1]}).",
                )
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size \
                        ({self.patch_size[0]}).",
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size \
                        ({self.patch_size[1]}).",
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Reshape to process each channel independently: (B, C, H, W) -> (B*C, 1, H, W)
        x = x.reshape(B * C, 1, H, W)
        # Project patches: (B*C, 1, H, W) -> (B*C, embed_dim, H_patches, W_patches)
        x = self.proj(x)
        if self.flatten:
            # Flatten spatial dimensions: (B*C, D, H_p, W_p) -> (B*C, num_patches, D)
            x = x.flatten(2).transpose(1, 2)
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        # Return embeddings and number of channels for downstream processing
        return x, C


class ChannelBlock(nn.Module):
    """
    Transformer block with divided space-channel attention.

    This block extends the standard ViT block to handle multi-channel images by
    applying separate attention mechanisms for spatial and channel dimensions.

    Parameters
    ----------
    dim : int
        Feature dimension
    num_heads : int
        Number of attention heads
    mlp_ratio : float, default=4.0
        MLP hidden dimension ratio
    qkv_bias : bool, default=False
        Use bias in QKV projection
    qk_norm : bool, default=False
        Apply layer norm to Q and K
    proj_bias : bool, default=True
        Use bias in output projection
    proj_drop : float, default=0.0
        Projection dropout rate
    attn_drop : float, default=0.0
        Attention dropout rate
    init_values : Optional[float], default=None
        Initial values for layer scale (not implemented)
    drop_path : float, default=0.0
        Stochastic depth rate
    act_layer : Type[nn.Module], default=nn.GELU
        Activation layer
    norm_layer : Type[nn.Module], default=nn.LayerNorm
        Normalization layer
    mlp_layer : Type[nn.Module], default=Mlp
        MLP layer
    attention_type : {'divided_space_channel', 'space_only', 'joint_space_channel'}, default='divided_space_channel'
        Type of attention mechanism to use
    separate_cls_for_channel : bool, default=False
        Whether to use separate cls tokens per channel
    SPACE_CHANNEL_ORDER : {'channel_first', 'space_first', 'parallel'}, default='channel_first'
        Order of spatial and channel attention application
    no_additional_mapping : bool, default=False
        Skip additional linear mapping after channel attention
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
        # added for processing channel dimension
        attention_type: Literal[
            "divided_space_channel", "space_only", "joint_space_channel"
        ] = "divided_space_channel",
        separate_cls_for_channel=False,
        SPACE_CHANNEL_ORDER: Literal["channel_first", "space_first", "parallel"] = "channel_first",
        no_additional_mapping: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            # fused_attn=False,
            # packed_attn=True,
        )
        if init_values is not None:
            raise NotImplementedError
        # self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # channel attention
        self.attention_type = attention_type
        assert attention_type in [
            "divided_space_channel",
            "joint_space_channel",
            "space_only",
        ]  # double check "space_only"
        if self.attention_type == "divided_space_channel":
            self.channel_norm = norm_layer(dim)
            self.channel_attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                fused_attn=False,
                # because of the short length of channel dimension,
                # we do not use flash attention
            )
            self.channel_drop = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            if no_additional_mapping:
                self.channel_fc: nn.Module = nn.Identity()
            else:
                # this is the mapping from channel attention to spatial
                self.channel_fc = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        # self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.separate_cls_for_channel = separate_cls_for_channel
        self.SPACE_CHANNEL_ORDER = SPACE_CHANNEL_ORDER
        # if SPACE_CHANNEL_ORDER != "channel_first" and not separate_cls_for_channel:
        #     raise NotImplementedError(\n        #
        # \"Spatial first attention is not implemented
        #  for non-separate cls token\"\n        #     )
        if SPACE_CHANNEL_ORDER == "parallel":
            assert no_additional_mapping, (
                "Parallel attention is only supported with no additional mapping"
            )

    def forward_channelsformer_separate_cls(
        self, x: torch.Tensor, B: int, L: int, T: int
    ) -> torch.Tensor:
        # x has shape (B, (L+1) * T, D) -> (B, L+1, T, D)
        x = rearrange(x, "b (l t) d -> b l t d", b=B, t=T)  # l = L + 1
        x.shape[1]
        x.shape[2]
        if self.SPACE_CHANNEL_ORDER == "space_first":
            ## Spatial first
            xs = x
            xs = rearrange(xs, "b l t d -> (b t) l d", b=B, t=T)  # l = L + 1
            xs = self.drop_path1(self.attn(self.norm1(xs)))
            xs = rearrange(xs, "(b t) l d -> b l t d", b=B, t=T)
            xs = self.channel_fc(xs)  # here the channel_fc goes to spatial
            x = x + xs

            ## Channel
            xt = x
            xt = rearrange(xt, "b l t d -> (b l) t d", b=B, t=T)  # l = L + 1
            xt = self.channel_drop(self.channel_attn(self.channel_norm(xt)))

            xt = rearrange(xt, "(b l) t d -> b l t d", b=B, t=T)  # l = L + 1
            x = x + xt
            x = rearrange(x, "b l t d -> b (l t) d", b=B, t=T)
        elif self.SPACE_CHANNEL_ORDER == "channel_first":
            ## Channel first
            xt = x
            xt = rearrange(xt, "b l t d -> (b l) t d", b=B, t=T)  # l = L + 1
            xt = self.channel_drop(self.channel_attn(self.channel_norm(xt)))
            xt = rearrange(xt, "(b l) t d -> b l t d", b=B, t=T)  # l = L + 1
            xt = self.channel_fc(xt)
            x = x + xt

            ## Spatial
            xs = x
            xs = rearrange(xs, "b l t d -> (b t) l d", b=B, t=T)  # l = L + 1
            xs = self.drop_path1(self.attn(self.norm1(xs)))
            xs = rearrange(xs, "(b t) l d -> b (l t) d", b=B, t=T)

            # res
            x = rearrange(x, "b l t d -> b (l t) d", b=B, t=T)
            x = x + xs
        elif self.SPACE_CHANNEL_ORDER == "parallel":
            ## Spatial
            xs = rearrange(x, "b l t d -> (b t) l d", b=B, t=T)  # l = L + 1
            xs = self.drop_path1(self.attn(self.norm1(xs)))
            xs = rearrange(xs, "(b t) l d -> b l t d", b=B, t=T)

            ## Channel
            xt = rearrange(x, "b l t d -> (b l) t d", b=B, t=T)  # l = L + 1
            xt = self.channel_drop(self.channel_attn(self.channel_norm(xt)))
            xt = rearrange(xt, "(b l) t d -> b l t d", b=B, t=T)  # l = L + 1

            # average the two attentions and residual connection
            x = x + (xs + xt) / 2
            x = rearrange(x, "b l t d -> b (l t) d", b=B, t=T)
        else:
            raise NotImplementedError
        return x

    def forward_channelsformer(self, x: torch.Tensor, B: int, L: int, T: int) -> torch.Tensor:
        if self.SPACE_CHANNEL_ORDER == "channel_first":
            ## Channel
            xt = x[:, 1:, :]
            xt = rearrange(xt, "b (l t) m -> (b l) t m", b=B, l=L, t=T)
            res_channel = self.channel_drop(self.channel_attn(self.channel_norm(xt)))
            res_channel = rearrange(res_channel, "(b l) t m -> b (l t) m", b=B, l=L, t=T)
            res_channel = self.channel_fc(res_channel)
            xt = x[:, 1:, :] + res_channel

            ## Spatial
            init_cls_token = x[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, "b t m -> (b t) m", b=B, t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, "b (l t) m -> (b t) l m", b=B, l=L, t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path1(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:, 0, :]
            cls_token = rearrange(cls_token, "(b t) m -> b t m", b=B, t=T)
            cls_token = torch.mean(cls_token, 1, True)  ## averaging for every frame
            res_spatial = res_spatial[:, 1:, :]
            res_spatial = rearrange(res_spatial, "(b t) l m -> b (l t) m", b=B, l=L, t=T)
            res = res_spatial
            x = xt

            # res
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
        elif self.SPACE_CHANNEL_ORDER == "space_first":
            ## Spatial
            init_cls_token = x[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, "b t m -> (b t) m", b=B, t=T).unsqueeze(1)
            xs = x[:, 1:, :]
            xs = rearrange(xs, "b (l t) m -> (b t) l m", b=B, l=L, t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path1(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:, 0, :]
            cls_token = rearrange(cls_token, "(b t) m -> b t m", b=B, t=T)
            cls_token = torch.mean(cls_token, 1, True)  ## averaging for every frame
            res_spatial = res_spatial[:, 1:, :]
            res_spatial = rearrange(res_spatial, "(b t) l m -> b (l t) m", b=B, l=L, t=T)
            res = res_spatial
            x = x[:, 1:, :]

            # res
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)

            ## Channel
            xt = x[:, 1:, :]
            xt = rearrange(xt, "b (l t) m -> (b l) t m", b=B, l=L, t=T)
            res_channel = self.channel_drop(self.channel_attn(self.channel_norm(xt)))
            res_channel = rearrange(res_channel, "(b l) t m -> b (l t) m", b=B, l=L, t=T)
            res_channel = self.channel_fc(res_channel)
            xt = x[:, 1:, :] + res_channel

            x = torch.cat((x[:, [0], :], xt), 1)
        else:
            raise NotImplementedError
        return x

    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        # original code:
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        # return x
        if T <= 0:
            raise ValueError(f"Number of channels T must be positive, got {T}")

        B = x.shape[0]

        # Only validate for divided_space_channel attention that uses T parameter
        if self.attention_type == "divided_space_channel":
            if self.separate_cls_for_channel:
                # x has shape (B, (L+1) * T, D)
                if x.size(1) % T != 0:
                    raise ValueError(
                        f"Input sequence length {x.size(1)} incompatible with T={T} "
                        f"for separate cls tokens. Expected (L+1)*T, but got "
                        f"remainder {x.size(1) % T}"
                    )
                L = x.size(1) // T - 1
            else:
                # x has shape (B, 1 + L * T, D)
                if (x.size(1) - 1) % T != 0:
                    raise ValueError(
                        f"Input sequence length {x.size(1)} incompatible with T={T}. "
                        f"Expected 1 + L*T, but got remainder {(x.size(1) - 1) % T}"
                    )
                L = (x.size(1) - 1) // T
        else:
            # For space_only and joint_space_channel, we still compute L but don't enforce strict validation
            if self.separate_cls_for_channel:
                L = x.size(1) // T - 1 if x.size(1) % T == 0 else (x.size(1) - 1) // T
            else:
                L = (x.size(1) - 1) // T

        if self.attention_type in ["space_only", "joint_space_channel"]:
            x = x + self.drop_path1(self.attn(self.norm1(x)))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        elif self.attention_type == "divided_space_channel":
            if self.separate_cls_for_channel:
                x = self.forward_channelsformer_separate_cls(x, B, L, T)
            else:
                x = self.forward_channelsformer(x, B, L, T)

            ## Mlp
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
        return x


class ChannelSFormer(nn.Module):
    """Vision Transformer (cloned from timm.models.vision_transformer.VisionTransformer)

    A PyTorch impl of : `An Image is Worth 16x16 Words:
        Transformers for Image\n    Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "avgmax", "max", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: Callable = ChannelPatchEmbed,
        embed_norm_layer: Optional[LayerType] = None,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = ChannelBlock,
        mlp_layer: Type[nn.Module] = Mlp,
        # added for processing channel dimension
        attention_type: Literal[
            "divided_space_channel", "space_only", "joint_space_channel"
        ] = "divided_space_channel",
        zero_init_channel: Optional[bool] = False,
        use_channel_embedding: Optional[bool] = True,
        separate_cls_for_channel: Optional[bool] = False,
        separate_cls_aggregation: str = "mean",  # 'mean', 'att'
        SPACE_CHANNEL_ORDER: Literal["channel_first", "space_first", "parallel"] = "channel_first",
        no_additional_mapping: bool = False,  # for divided space channel attention
        separate_cls_init: bool = False,
        use_channel_embedding_for_cls: Optional[
            bool
        ] = False,  # if True, use channel embedding for cls token
    ) -> None:
        super().__init__()
        assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool in ("avg", "avgmax", "max") if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        embed_norm_layer = get_norm_layer(embed_norm_layer)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.in_chans = in_chans
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = (
            embed_dim  # for consistency with other models
        )
        self.num_prefix_tokens = 1 if class_token else 0
        self.has_class_token = class_token
        self.dynamic_img_size = dynamic_img_size

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        if embed_norm_layer is not None:
            embed_args["norm_layer"] = embed_norm_layer
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        reduction = (
            self.patch_embed.feat_ratio()
            if hasattr(self.patch_embed, "feat_ratio")
            else patch_size
        )

        num_cls = in_chans if separate_cls_init else 1
        self.cls_token = nn.Parameter(torch.zeros(1, num_cls, embed_dim)) if class_token else None

        # Note that for every channel, the initial cls token is the same
        self.separate_cls_for_channel = separate_cls_for_channel
        self.separate_cls_aggregation = separate_cls_aggregation
        self.separate_cls_init = separate_cls_init

        embed_len = num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        # added for processing channel dimension
        self.attention_type = attention_type
        self.use_channel_embedding = use_channel_embedding
        self.use_channel_embedding_for_cls = use_channel_embedding_for_cls
        if (
            self.attention_type != "space_only" and use_channel_embedding
        ) or self.use_channel_embedding_for_cls:
            self.channel_embed = nn.Parameter(
                torch.randn(1, in_chans, embed_dim) * 0.02
            )  # channel embedding
            self.channel_drop = nn.Dropout(p=drop_rate)

        if patch_drop_rate > 0:
            self.patch_drop: nn.Module = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
            raise NotImplementedError("PatchDropout not implemented for ChannelPatchEmbed")
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [  # change to ModuleList as we need to pass T to blocks
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_bias=proj_bias,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    # added for processing channel dimension
                    attention_type=attention_type,
                    separate_cls_for_channel=separate_cls_for_channel,
                    SPACE_CHANNEL_ORDER=SPACE_CHANNEL_ORDER,
                    no_additional_mapping=no_additional_mapping,
                )
                for i in range(depth)
            ]
        )
        self.feature_info = [
            dict(module=f"blocks.{i}", num_chs=embed_dim, reduction=reduction)
            for i in range(depth)
        ]
        self.norm = norm_layer(embed_dim) if final_norm and not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == "map":
            self.attn_pool: Optional[nn.Module] = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,  # type: ignore
                act_layer=act_layer,
            )
        elif separate_cls_aggregation == "att":
            self.attn_pool = AttentionPoolLatentCus(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,  # type: ignore
                act_layer=act_layer,
                map_q=False,
                use_proj=False,
                use_mlp=False,
                post_norm=True,
                fused_attn=False,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights("")

        ## initialization of channel attention weights
        if zero_init_channel:
            if self.attention_type == "divided_space_channel":
                i = 0
                for m in self.blocks.modules():
                    m_str = str(m)
                    if "Block" in m_str:
                        if i > 0:
                            assert isinstance(m.channel_fc, nn.Linear)
                            nn.init.constant_(m.channel_fc.weight, 0)
                            nn.init.constant_(m.channel_fc.bias, 0)
                        i += 1

    def init_weights(self, mode: str = "") -> None:
        assert mode in ("jax", "jax_nlhb", "moco", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if hasattr(self, "channel_embed") and self.channel_embed is not None:
            nn.init.normal_(self.channel_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m: nn.Module) -> None:
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = "") -> None:
        _load_weights(self, checkpoint_path, prefix)  # type: ignore

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token", "dist_token", "channel_embed"}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r"^cls_token|pos_embed|channel_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    def _pos_embed(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        # x has shape B * T, L, D
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        # pos embeddings
        if self.dynamic_img_size:
            # no need to change anything as T is in Batch size
            BT, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=[H, W],
                old_size=prev_grid_size,
                num_prefix_tokens=self.num_prefix_tokens,
            )
            x = x.view(BT, -1, C)
        else:
            pos_embed = self.pos_embed

        if self.separate_cls_for_channel:
            if self.attention_type != "space_only":
                x = rearrange(x, "(b t) l d -> b t l d", b=B, t=T)

                if self.use_channel_embedding:
                    channel_embed = self.channel_embed.unsqueeze(-2)
                    x = x + channel_embed  # (B, T, L+1, D) + (1, T, 1, D)
                    x = self.channel_drop(x)

                # cls token has shape (1, T, D) -> (B, T, 1, D) or (1, 1, D) -> (B, 1, 1, D)
                cls_token = self.cls_token.unsqueeze(-2)  # type: ignore
                # expand cls_token to match batch size and channel dimension
                cls_token = cls_token.expand(x.shape[0], T, 1, self.cls_token.shape[-1])  # type: ignore
                x = torch.cat([cls_token, x], dim=2)  # (B, T, L+1, D)
                x = x + pos_embed.unsqueeze(1)  # (B, T, L+1, D) + (1, 1, L+1, D)

                x = rearrange(x, "b t l d -> b (l t) d")  # (B, (L+1) * T, D)
            else:
                # x has shape (B * T, L, D)
                # cls token has shape (1, D) -> (B, 1, D)
                cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # type: ignore
                x = torch.cat([cls_token, x], dim=1)
                x = x + pos_embed
        else:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # type: ignore
            x = torch.cat([cls_token, x], dim=1)
            x = x + pos_embed

            ## Channel Embeddings
            if self.attention_type != "space_only":
                if self.use_channel_embedding:
                    cls_tokens = x[:B, 0, :].unsqueeze(1)
                    x = x[:, 1:]
                    x = rearrange(x, "(b t) l d -> (b l) t d", b=B, t=T)
                    x = x + self.channel_embed
                    x = self.channel_drop(x)
                    x = rearrange(x, "(b l) t d -> b (l t) d", b=B, t=T)
                    x = torch.cat((cls_tokens, x), dim=1)
                else:
                    cls_tokens = x[:B, 0, :].unsqueeze(1)
                    x = x[:, 1:]
                    x = rearrange(x, "(b t) l d -> b (l t) d", b=B, t=T)
                    x = torch.cat((cls_tokens, x), dim=1)

        return x

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        B = x.shape[0]
        x, T = self.patch_embed(x)
        x = self._pos_embed(x, B, T)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for block in self.blocks:
            # pass T to each block
            x = block(x, T)
        x = self.norm(x)
        return x, T

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None, T=None) -> torch.Tensor:
        if self.separate_cls_for_channel:
            # first reshape x back to (B, L+1, T, D), then get the cls token for each channel
            if self.attention_type != "space_only":
                x = rearrange(x, "b (l t) d -> b l t d", t=T)
                x_cls = x[:, 0, :, :]  # (B, T, D)
            else:
                x = rearrange(x, "(b t) l d -> b t l d", t=T)
                x_cls = x[:, :, 0, :]  # (B, T, D)
            if self.use_channel_embedding_for_cls:
                channel_embed = self.channel_embed
                x_cls = x_cls + channel_embed  # (B, T, D) + (1, T, D)
                x_cls = self.channel_drop(x_cls)
            if self.separate_cls_aggregation == "mean":
                x = torch.mean(x_cls, dim=1)
            elif self.separate_cls_aggregation == "att" and self.attn_pool is not None:
                x = self.attn_pool(x_cls)
            else:
                raise NotImplementedError
            return x

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False, T=None) -> torch.Tensor:
        x = self.pool(x, T=T)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, T = self.forward_features(x)
        x = self.forward_head(x, T=T)
        return x


if __name__ == "__main__":
    # set cuda device to be 3 using pytorch
    torch.cuda.set_device(3)  # Set CUDA device to be 3

    model = ChannelSFormer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=192,
        depth=12,
        num_heads=3,
        attention_type="divided_space_channel",
        drop_rate=0.0,
        drop_path_rate=0.1,
        use_channel_embedding=False,
        separate_cls_for_channel=True,
        separate_cls_aggregation="mean",
        SPACE_CHANNEL_ORDER="space_first",  #'parallel', #"channel_first",
        no_additional_mapping=False,
    ).cuda()

    # test the model
    x = torch.randn(2, 3, 224, 224).cuda()  # batch size 2, 3 channels, 224x224 image
    output = model(x)

    print(output.shape)
