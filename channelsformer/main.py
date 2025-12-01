# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import argparse
import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from config import get_config
from data.build import build_loader
from fvcore.nn import FlopCountAnalysis, flop_count_str
from models import build_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter, accuracy
from timm.utils import ModelEma as ModelEma
from utils.logger import create_logger
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.utils import (
    NativeScalerWithGradNormCount,
    auto_resume_helper,
    load_checkpoint_ema,
    load_pretrained_ema,
    reduce_tensor,
    save_checkpoint_ema,
)

try:
    import wandb  # noqa: F401
    wandb_available = True
    from utils.utils import WandbLogger
except ImportError:
    wandb_available = False

if torch.multiprocessing.get_start_method() != "spawn":
    print(f"||{torch.multiprocessing.get_start_method()}||", end="")
    torch.multiprocessing.set_start_method("spawn", force=True)


def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_option():
    parser = argparse.ArgumentParser(
        "Swin Transformer training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        metavar="FILE",
        default="",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--dataset", type=str, default="imagenet", help="name of dataset")
    parser.add_argument(
        "--data-path",
        type=str,
        default="path_to_imagenet",
        help="path to dataset",
    )
    parser.add_argument(
        "--zip", action="store_true", help="use zipped dataset instead of folder dataset"
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument(
        "--pretrained",
        help="pretrained weight from checkpoint, could be imagenet22k pretrained weight",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument("--disable_amp", action="store_true", help="Disable pytorch amp")
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> ",
    )
    parser.add_argument(
        "--tag", default=time.strftime("%Y%m%d%H%M%S", time.localtime()), help="tag of experiment"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--throughput", action="store_true", help="Test throughput only")

    parser.add_argument("--fused_layernorm", action="store_true", help="Use fused layernorm.")
    parser.add_argument(
        "--optim", type=str, help="overwrite optimizer if provided, can be adamw/sgd."
    )

    # EMA related parameters
    parser.add_argument("--model_ema", type=str2bool, default=True)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999, help="")
    parser.add_argument("--model_ema_force_cpu", type=str2bool, default=False, help="")

    parser.add_argument(
        "--memory_limit_rate", type=float, default=-1, help="limitation of gpu memory use"
    )

    # added
    parser.add_argument("--wandb", type=str2bool, default=False, help="use wandb for logging")
    parser.add_argument(
        "--project", type=str, default="channel_mha_vit", help="wandb project name"
    )
    parser.add_argument("--num_worker", type=int, default=16, help="number of workers")
    parser.add_argument(
        "--re_root", type=str, default=None, help="redirect the path of images for preloading"
    )

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, args):
    datasets, data_loaders, mixup_fn = build_loader(config)
    _, dataset_val = datasets[0], datasets[1]
    data_loader_train, data_loader_val = data_loaders[0], data_loaders[1]
    if len(datasets) > 2:
        dataset_test = datasets[2]
        data_loader_test = data_loaders[2]
    else:
        dataset_test = None
        data_loader_test = None

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    if dist.get_rank() == 0:
        if hasattr(model, "flops"):
            logger.info(str(model))
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"number of params: {n_parameters}")
            flops = model.flops()
            logger.info(f"number of GFLOPs: {flops / 1e9}")
        else:
            logger.info(flop_count_str(FlopCountAnalysis(model, (dataset_val[0][0][None],))))
    torch.cuda.empty_cache()
    dist.barrier()
    model.cuda()
    model_without_ddp = model

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper,
        # and AMP\n        # but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    optimizer = build_optimizer(config, model, logger)
    model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(
            config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS
        )
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    # only used when dataset_test is not None
    # and does not work with resume currently
    max_accuracy_test = 0.0
    max_accuracy_ema_test = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    if config.MODEL.RESUME:
        max_accuracy, max_accuracy_ema = load_checkpoint_ema(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger, model_ema
        )
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(
                f"Accuracy of the network ema on the {len(dataset_val)} test "
                f"images: {acc1_ema:.1f}%"
            )

        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained_ema(config, model_without_ddp, logger, model_ema)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(
                f"Accuracy of the network ema on the {len(dataset_val)} test "
                f"images: {acc1_ema:.1f}%"
            )

        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE and (dist.get_rank() == 0):
        logger.info("throughput mode ==============================")
        throughput(data_loader_val, model, logger)
        if model_ema is not None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            throughput(data_loader_val, model_ema.ema, logger)
        return

    if wandb_logger is not None:
        wandb_logger.define_metric("Train/epoch")
        for metric in ["acc1", "acc5", "loss", "max_accuracy"]:
            wandb_logger.define_metric(name=f"Valid/{metric}", step_metric="Train/epoch")
            wandb_logger.define_metric(name=f"Valid_ema/{metric}", step_metric="Train/epoch")

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        last_log_step = train_one_epoch(
            config,
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            mixup_fn,
            lr_scheduler,
            loss_scaler,
            model_ema,
            wandb_logger=wandb_logger,
        )
        if not config.WANDB:
            # use normal save checkpoint function if wandb is not enabled
            if dist.get_rank() == 0 and (
                epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)
            ):
                save_checkpoint_ema(
                    config,
                    epoch,
                    model_without_ddp,
                    max_accuracy,
                    optimizer,
                    lr_scheduler,
                    loss_scaler,
                    logger,
                    model_ema,
                    max_accuracy_ema,
                )

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        # max_accuracy = max(max_accuracy, acc1)
        if acc1 > max_accuracy:
            max_accuracy = acc1
            update_test = True
        else:
            update_test = False
        logger.info(f"Max accuracy: {max_accuracy:.2f}%")
        if wandb_logger:
            wandb_logger.log(
                {
                    "Valid/acc1": acc1,
                    "Valid/acc5": acc5,
                    "Valid/loss": loss,
                    "Valid/max_accuracy": max_accuracy,
                },
                step=last_log_step,
            )

        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(
                f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_ema:.1f}%"
            )
            # max_accuracy_ema = max(max_accuracy_ema, acc1_ema)
            if acc1_ema > max_accuracy_ema:
                max_accuracy_ema = acc1_ema
                update_test_ema = True
            else:
                update_test_ema = False
            logger.info(f"Max accuracy ema: {max_accuracy_ema:.2f}%")

            if wandb_logger:
                wandb_logger.log(
                    {
                        "Valid_ema/acc1": acc1_ema,
                        "Valid_ema/acc5": acc5_ema,
                        "Valid_ema/loss": loss_ema,
                        "Valid_ema/max_accuracy": max_accuracy_ema,
                    },
                    step=last_log_step,
                )
        else:
            acc1_ema = None

        if dataset_test is not None:
            acc1, acc5, loss = validate(config, data_loader_test, model)
            logger.info(
                f"Accuracy of the network on the {len(dataset_test)} test test images: {acc1:.1f}%"
            )
            max_accuracy_test = acc1 if update_test else max_accuracy_test
            logger.info(f"Accuracy of the best val model: {max_accuracy_test:.2f}%")
            if wandb_logger:
                wandb_logger.log(
                    {
                        "Test/acc1": acc1,
                        "Test/acc5": acc5,
                        "Test/loss": loss,
                        "Test/max_val_accuracy": max_accuracy_test,
                    },
                    step=last_log_step,
                )

            if model_ema is not None:
                acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_test, model_ema.ema)
                logger.info(
                    f"Accuracy of the network ema on the {len(dataset_test)} test test "
                    f"images: {acc1_ema:.1f}%"
                )
                max_accuracy_ema_test = acc1_ema if update_test_ema else max_accuracy_ema_test
                logger.info(f"Accuracy of the best val model ema: {max_accuracy_ema_test:.2f}%")
                if wandb_logger:
                    wandb_logger.log(
                        {
                            "Test_ema/acc1": acc1_ema,
                            "Test_ema/acc5": acc5_ema,
                            "Test_ema/loss": loss_ema,
                            "Test_ema/max_val_accuracy": max_accuracy_ema_test,
                        },
                        step=last_log_step,
                    )

        if wandb_logger:  # save the model to wandb
            if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
                # save the model to wandb
                # epoch, model, accuracy, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                # logger, model_ema = None, accuracy_ema = None, max_accuracy_ema = None):
                wandb_logger.log_model(
                    epoch,
                    model_without_ddp,
                    acc1,
                    max_accuracy,
                    optimizer,
                    lr_scheduler,
                    loss_scaler,
                    logger,
                    model_ema,
                    acc1_ema,
                    max_accuracy_ema,
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
    loss_scaler,
    model_ema=None,
    model_time_warmup=50,
    wandb_logger=None,
):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    last_log_step = 0
    for idx, (samples, targets) in enumerate(data_loader):
        torch.cuda.reset_peak_memory_stats()
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=model.parameters(),
            create_graph=is_second_order,
            update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0,
        )
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx > model_time_warmup:
            model_time.update(batch_time.val - data_time.val)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            wd = optimizer.param_groups[0]["weight_decay"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"data time {data_time.val:.4f} ({data_time.avg:.4f})\t"
                f"model time {model_time.val:.4f} ({model_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
            if wandb_logger is not None:
                wandb_logger.log(
                    {
                        "Train/lr": lr,
                        "Train/wd": wd,
                        "Train/loss": loss_meter.val,
                        "Train/loss_avg": loss_meter.avg,
                        "Train/grad_norm": norm_meter.val,
                        "Train/grad_norm_avg": norm_meter.avg,
                        "Train/loss_scale": scaler_meter.val,
                        "Train/loss_scale_avg": scaler_meter.avg,
                        "Train/epoch": epoch,
                    },
                    step=epoch * num_steps + idx,
                )  # wandb step must be monotonically increasing every time called log
            last_log_step = epoch * num_steps + idx
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return last_log_step


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
                f"Mem {memory_used:.0f}MB"
            )
    logger.info(f" * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}")
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info("throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == "__main__":
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        # rank = -1
        # world_size = -1
        print("Will run the code on one GPU.")
        rank, world_size = 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    dist.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if True:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = (
        config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    linear_scaled_warmup_lr = (
        config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    linear_scaled_min_lr = (
        config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # to make sure all the config.OUTPUT are the same
    config.defrost()
    if dist.get_rank() == 0:
        obj = [config.OUTPUT]
        # obj = [str(random.randint(0, 100))] # for test
    else:
        obj = [None]
    dist.broadcast_object_list(obj)
    dist.barrier()
    config.OUTPUT = obj[0]
    print(config.OUTPUT, flush=True)
    config.freeze()
    os.makedirs(config.OUTPUT, exist_ok=True)

    if config.WANDB and not wandb_available:
        raise ImportError("wandb is not installed, please install it to use wandb logging.")

    if wandb_available:
        # initialize wandb logger when the config.WANDB == True
        wandb_logger = WandbLogger(config)
    else:
        wandb_logger = None

    logger = create_logger(
        output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}"
    )

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    if args.memory_limit_rate > 0 and args.memory_limit_rate < 1:
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_rate)
        usable_memory = (
            torch.cuda.get_device_properties(0).total_memory * args.memory_limit_rate / 1e6
        )
        print(f"===========> GPU memory is limited to {usable_memory}MB", flush=True)

    main(config, args)

    if wandb_logger:
        # close wandb logger
        wandb_logger.finish(clean_up_model=True)
