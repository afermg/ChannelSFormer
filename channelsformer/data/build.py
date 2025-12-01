# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import os

import torch
import torch.distributed as dist
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from channelsformer.data.aug_cell import transforms_eval, transforms_train
from channelsformer.data.imagenet import ImageNet
from channelsformer.data.jumpcp import (
    DATASET_MEAN,
    DATASET_STD,
    Jumpcp,
)

try:
    from torchvision.transforms import InterpolationMode
    def _pil_interp(method):
        if method == "bicubic":
            return InterpolationMode.BICUBIC
        elif method == "lanczos":
            return InterpolationMode.LANCZOS
        elif method == "hamming":
            return InterpolationMode.HAMMING
        else:
            # default bilinear
            return InterpolationMode.BILINEAR
    import timm.data.transforms as timm_transforms
    timm_transforms.str_to_pil_interp = _pil_interp
except:  # noqa: E722
    pass


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    if isinstance(dataset_val, list):
        dataset_val, dataset_test = dataset_val
        has_test = True
    else:
        has_test = False
    print(f"rank {dist.get_rank()} successfully build val dataset")
    if has_test:
        print(f"rank {dist.get_rank()} successfully build test dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )
    if has_test:
        if config.TEST.SEQUENTIAL:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        else:
            sampler_test = torch.utils.data.distributed.DistributedSampler(
                dataset_test, shuffle=config.TEST.SHUFFLE
            )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        persistent_workers=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True,
    )
    if has_test:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            persistent_workers=True,
        )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = (
        config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0.0 or config.AUG.CUTMIX_MINMAX is not None
    )
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES,
        )

    datasets = [dataset_train, dataset_val]
    data_loaders = [data_loader_train, data_loader_val]
    if has_test:
        datasets.append(dataset_test)
        data_loaders.append(data_loader_test)
    return datasets, data_loaders, mixup_fn


def build_dataset(is_train, config):
    if config.DATA.DATASET not in ["jumpcp"]:
        transform = build_transform(is_train, config)
    else:
        transform = None
    if config.DATA.DATASET == "imagenet":
        prefix = "train" if is_train else "validation"
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = ImageNet(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == "jumpcp":
        prefix = "train" if is_train else "valid"
        if config.AUG.AUTO_AUGMENT != "none":
            transform = None
            transform_init_fn = build_cell_transform
            transform_params = {
                "is_train": is_train,
                "config": config,
                "mean": DATASET_MEAN,
                "std": DATASET_STD,
            }
        else:
            transform = build_cell_transform(is_train, config, mean=DATASET_MEAN, std=DATASET_STD)
            transform_init_fn = None
            transform_params = None
        if is_train:
            prefix = "train"
            dataset = Jumpcp(
                config.DATA.DATA_PATH,
                transform=transform,
                transform_init_fn=transform_init_fn,
                transform_params=transform_params,
                split=prefix,
            )
        else:
            dataset = [
                Jumpcp(
                    config.DATA.DATA_PATH,
                    transform=transform,
                    transform_init_fn=transform_init_fn,
                    transform_params=transform_params,
                    split=prefix,
                )
                for prefix in ["valid", "test"]
            ]

        nb_classes = 161
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != "none" else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=_pil_interp(config.DATA.INTERPOLATION),
                )
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_cell_transform(
    is_train=False, config=None, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):

    print("Building cell transform for is_train:", is_train)
    if is_train:
        return transforms_train(
            img_size=config.DATA.IMG_SIZE,
            hflip=0.5,
            vflip=0.5,
            rotate90=0.75,
            auto_augment=config.AUG.AUTO_AUGMENT != "none",
            mean=mean,
            std=std,
        )
    else:
        return transforms_eval(
            img_size=config.DATA.IMG_SIZE,
            crop=True,
            mean=mean,
            std=std,
        )
