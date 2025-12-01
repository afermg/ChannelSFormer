# --------------------------------------------------------
# Speed test script for model throughput evaluation
# --------------------------------------------------------
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from config import get_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser("Model throughput testing script", add_help=True)
    parser.add_argument("--cfg", type=str, default="", help="path to config file", required=True)
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for testing")
    parser.add_argument("--image-size", type=int, default=224, help="input image size")
    parser.add_argument("--num-channels", type=int, default=3, help="number of input channels")
    parser.add_argument("--patch-size", type=int, default=16, help="patch size for CMViT")
    parser.add_argument(
        "--repeat", type=int, default=50, help="number of repeat runs for accurate timing"
    )
    parser.add_argument("--warmup", type=int, default=5, help="number of warmup iterations")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # assign cuda device
    parser.add_argument("--gpu", type=str, default="0", help="gpu ids")

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs",
        default=None,
        nargs="+",
    )
    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True


def measure_throughput(model, dummy_input, args):
    """Measure model throughput in images/second"""
    print(f"Running throughput test with batch size: {args.batch_size}")
    batch_size = args.batch_size

    # Warmup runs
    print(f"Warming up for {args.warmup} iterations...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(dummy_input)

    # Synchronize before timing
    torch.cuda.synchronize()

    # Actual timing runs
    print(f"Measuring performance over {args.repeat} iterations...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(args.repeat):
            _ = model(dummy_input)

    # Synchronize after timing
    torch.cuda.synchronize()

    end_time = time.time()

    elapsed_time = end_time - start_time
    images_per_second = args.repeat * batch_size / elapsed_time
    ms_per_batch = 1000 * elapsed_time / args.repeat

    print(f"Throughput: {images_per_second:.2f} images/second")
    print(f"Latency: {ms_per_batch:.2f} ms/batch")

    return images_per_second, ms_per_batch


def main():
    args = parse_args()
    set_seed(args.seed)

    config = get_config(args)
    # Update config with command line options
    config.defrost()
    config.DATA.IMG_SIZE = args.image_size
    config.MODEL.CMVIT.PATCH_SIZE = args.patch_size
    config.MODEL.CMVIT.IN_CHANS = args.num_channels
    config.freeze()

    # set cuda device using torch
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Using GPU: {args.gpu}")
    else:
        raise Exception("No GPU found")

    print(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    model = model.cuda()
    model.eval()

    # Print model parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_parameters:,}")

    # Create dummy input data
    dummy_input = torch.rand(args.batch_size, args.num_channels, args.image_size, args.image_size)
    dummy_input = dummy_input.cuda()

    images_per_second, ms_per_batch = measure_throughput(model, dummy_input, args)

    # Print results summary
    print("\nTest Results Summary:")
    print(f"Model: {config.MODEL.NAME}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Input channels: {args.num_channels}")
    print(f"Throughput: {images_per_second:.2f} images/second")
    print(f"Latency: {ms_per_batch:.2f} ms/batch")


if __name__ == "__main__":
    main()
