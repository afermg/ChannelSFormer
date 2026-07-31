#!/usr/bin/env python3
"""End-to-end inference against the ChannelSFormer OCI container."""

import json
import os

os.environ.setdefault("NAHUAL_IPC_TIMEOUT_MS", "900000")

import numpy as np
from nahual.process import dispatch_setup_process


def main() -> None:
    address = os.environ.get("NAHUAL_ADDRESS", "tcp://127.0.0.1:5555")
    device = os.environ.get("NAHUAL_DEVICE", "cpu")
    setup, process = dispatch_setup_process("channelsformer")
    info = setup(
        {
            "img_size": 224,
            "patch_size": 16,
            "in_chans": 5,
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "device": device,
        },
        address=address,
    )
    pixels = np.random.default_rng(42).random((1, 5, 1, 224, 224), dtype=np.float32)
    result = process(pixels, address=address)
    assert info["device"] == device, info
    assert info["load"]["weights"] == "random", info
    assert result.shape == (1, 384), result.shape
    assert np.isfinite(result).all()
    print(json.dumps({"setup": info, "shape": list(result.shape)}))


if __name__ == "__main__":
    main()
