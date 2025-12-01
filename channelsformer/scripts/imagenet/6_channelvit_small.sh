torchrun --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  ./channelsformer/main.py \
  --cfg ./channelsformer/configs/imagenet/channelvit_small.yaml \
  --dataset imagenet \
  --data-path path/to/imagenet_df \
  --batch-size 128 \
  --output /tmp/channelsformer_imagenet/channelvit_small \
  --project channelsformer_imagenet \
  --tag channelvit_small \
  --wandb true