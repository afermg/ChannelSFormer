torchrun --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  ./channelsformer/main.py \
  --dataset jumpcp \
  --data-path s3://insitro-research-2023-context-vit/jumpcp/BR00116991.pq \
  --cfg ./channelsformer/configs/jumpcp/channelsformer_small.yaml \
  --batch-size 32 \
  --model_ema_decay 0.9990 \
  --output /tmp/channelsformer/channelsformer_small \
  --project channelsformer_jumpcp \
  --tag channelsformer_small \
  --wandb true