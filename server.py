"""Nahual server for ChannelSFormer.

Loads a `ChannelSFormer` (channel-agnostic ViT for multi-channel cell painting
images) and runs it on incoming `NCZYX` tensors. The Z dimension is dropped
before forward.

Run with:
    nix run . -- ipc:///tmp/channelsformer.ipc
or:
    python server.py ipc:///tmp/channelsformer.ipc
"""

import os
import sys
from functools import partial
from typing import Callable

import numpy
import pynng
import torch
import trio
from nahual.preprocess import pad_channel_dim, validate_input_shape
from nahual.server import responder

# Make the local channelsformer package importable.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from channelsformer.models.channelsformer import ChannelSFormer  # noqa: E402

address = sys.argv[1]


def setup(
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 5,
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 6,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    attention_type: str = "divided_space_channel",
    use_channel_embedding: bool = True,
    separate_cls_for_channel: bool = False,
    separate_cls_aggregation: str = "mean",
    SPACE_CHANNEL_ORDER: str = "channel_first",
    no_additional_mapping: bool = False,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    weights: str | None = None,
    checkpoint_key: str | None = None,
    device: int | None = None,
) -> tuple[Callable, dict]:
    """Build a ChannelSFormer and (optionally) load a checkpoint.

    Parameters
    ----------
    img_size : int
        Square input image size (must be divisible by ``patch_size``).
    patch_size : int
        ViT patch size.
    in_chans : int
        Number of input channels expected by the model. Inputs with fewer
        channels are zero-padded; inputs with more are an error.
    embed_dim, depth, num_heads, mlp_ratio :
        Standard ViT hyperparameters.
    attention_type : str
        ``divided_space_channel`` (default), ``space_only``, or
        ``joint_space_channel``.
    use_channel_embedding, separate_cls_for_channel, separate_cls_aggregation,
    SPACE_CHANNEL_ORDER, no_additional_mapping :
        ChannelSFormer-specific knobs (see ``ChannelSFormer.__init__``).
    weights : str | None
        Path to a ``.pth`` checkpoint. If None the model uses random init —
        useful for smoke tests.
    checkpoint_key : str | None
        Optional key inside the checkpoint dict to load (e.g. ``model``,
        ``teacher``, ``state_dict``).
    device : int | None
        CUDA device index. None → cuda:0 if available, else cpu.
    """
    if device is None:
        device = 0
    if torch.cuda.is_available():
        torch_device = torch.device(int(device))
    else:
        torch_device = torch.device("cpu")

    # num_classes=0 → head becomes nn.Identity so forward() returns features.
    model = ChannelSFormer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=0,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        attention_type=attention_type,
        use_channel_embedding=use_channel_embedding,
        separate_cls_for_channel=separate_cls_for_channel,
        separate_cls_aggregation=separate_cls_aggregation,
        SPACE_CHANNEL_ORDER=SPACE_CHANNEL_ORDER,
        no_additional_mapping=no_additional_mapping,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    if weights is not None and os.path.exists(weights):
        state_dict = torch.load(weights, map_location="cpu")
        if isinstance(state_dict, dict) and checkpoint_key and checkpoint_key in state_dict:
            state_dict = state_dict[checkpoint_key]
        elif isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Strip common prefixes.
        state_dict = {
            k.replace("module.", "").replace("backbone.", ""): v
            for k, v in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        load_info = {"missing": len(missing), "unexpected": len(unexpected)}
    else:
        load_info = {"missing": 0, "unexpected": 0, "weights": "random"}

    model.to(torch_device).eval()

    info = {
        "device": str(torch_device),
        "img_size": img_size,
        "patch_size": patch_size,
        "in_chans": in_chans,
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "attention_type": attention_type,
        "load": load_info,
    }
    processor = partial(
        process,
        model=model,
        device=torch_device,
        expected_tile_size=patch_size,
        expected_channels=in_chans,
    )
    return processor, info


def process(
    pixels: numpy.ndarray,
    model,
    device: torch.device,
    expected_tile_size: int,
    expected_channels: int,
) -> torch.Tensor:
    """Forward an NCZYX numpy array through ChannelSFormer, returning pooled features.

    With ``num_classes=0`` the model's head is an Identity, so ``forward`` returns
    the pooled (cls-token-style) feature tensor of shape (N, embed_dim).
    """
    if pixels.ndim != 5:
        raise ValueError(
            f"Expected NCZYX (5D) array, got shape {pixels.shape}"
        )
    _, _, _, *input_yx = pixels.shape
    validate_input_shape(input_yx, expected_tile_size)

    # pad_channel_dim drops the Z axis (returns NCYX) and pads channels up to
    # ``expected_channels`` if needed.
    pixels = pad_channel_dim(pixels, expected_channels)
    torch_tensor = torch.from_numpy(pixels.copy()).float().to(device)

    with torch.no_grad():
        feats = model(torch_tensor)  # (N, embed_dim)
    return feats


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"ChannelSFormer server listening on {address}", flush=True)
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
