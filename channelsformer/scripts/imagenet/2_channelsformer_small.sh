torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  ./channelsformer/main.py \
  --cfg ./channelsformer/configs/imagenet/channelsformer_small.yaml \
  --dataset imagenet \
  --data-path path/to/imagenet_df \
  --batch-size 128 \
  --output /tmp/channelsformer_imagenet/channelsformer_small \
  --project channelsformer_imagenet \
  --tag channelsformer_small \
  --wandb true