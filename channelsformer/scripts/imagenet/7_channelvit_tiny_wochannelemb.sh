torchrun --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  ./channelsformer/main.py \
  --cfg ./channelsformer/configs/imagenet/channelvit_tiny_wochannelemb.yaml \
  --dataset imagenet \
  --data-path path/to/imagenet_df \
  --batch-size 128 \
  --output /tmp/channelsformer_imagenet/channelvit_tiny_wochannelemb \
  --project channelsformer_imagenet \
  --tag channelvit_tiny_wochannelemb \
  --wandb true