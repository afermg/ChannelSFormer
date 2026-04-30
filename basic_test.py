"""Standalone smoke test for ChannelSFormer.

Instantiates a ChannelSFormer with random weights and runs a single forward
pass through ``forward_features`` to confirm the model loads and the imports
in this environment work end-to-end (no Nahual server involved).

Run with:
    nix develop --impure --command python basic_test.py
"""

import numpy
import torch

from channelsformer.models.channelsformer import ChannelSFormer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChannelSFormer(
    img_size=224,
    patch_size=16,
    in_chans=5,
    embed_dim=384,
    depth=12,
    num_heads=6,
).to(device).eval()

numpy.random.seed(seed=42)
data = numpy.random.random_sample((1, 5, 224, 224)).astype("float32")
x = torch.from_numpy(data).to(device)

with torch.no_grad():
    feats, T = model.forward_features(x)

print("forward_features shape:", tuple(feats.shape), "T:", T)
