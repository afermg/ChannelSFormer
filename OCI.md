# ChannelSFormer Nahual OCI image

Build the reproducible archive and load it into Podman or Docker:

```console
nix build .#oci-image
podman load < result                         # or: docker load < result
```

The image is tagged `nahual/channelsformer:local` and listens on TCP port 5555.

```console
podman run --rm --device nvidia.com/gpu=all -p 5555:5555 \
  nahual/channelsformer:local
```

For Docker, replace the CDI device option with `--gpus all`. CPU operation is
supported. With Nahual and NumPy installed on the host, run:

```console
NAHUAL_DEVICE=cpu python oci/smoke_test.py
```

The server uses a randomly initialized network unless the setup request's
`weights` parameter names a checkpoint mounted in the container. Mount trusted
checkpoints read-only (for example, `-v "$PWD/checkpoints:/models:ro"`) because
PyTorch full checkpoints may contain pickle-based metadata.
