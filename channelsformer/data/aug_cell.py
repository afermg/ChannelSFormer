#!/usr/bin/env python3
from typing import List, Tuple, Union

import kornia.augmentation as K
import torch
from kornia.augmentation.auto import RandAugment
from kornia.augmentation.auto.base import SUBPOLICY_CONFIG
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn

cell_policy: List[SUBPOLICY_CONFIG] = [
    [("auto_contrast", 0, 1)],
    [("equalize", 0, 1)],
    [("rotate", -45.0, 45.0)],
    [("posterize", 0.0, 4)],
    [("solarize", 0.0, 1.0)],
    [("solarize_add", 0.0, 0.43)],
    [("contrast", 0.1, 1.9)],
    [("brightness", 0.1, 1.9)],
    [("sharpness", 0.1, 1.9)],
    [("shear_x", -0.3, 0.3)],
    [("shear_y", -0.3, 0.3)],
    [("translate_x", -0.1, 0.1)],
    [("translate_y", -0.1, 0.1)],
]


def transforms_train(
    img_size: Union[int, Tuple[int, int]] = 224,
    scale: Tuple[float, float] = (0.4, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    hflip: float = 0.5,
    vflip: float = 0.5,
    rotate90: float = 0.75,
    auto_augment: bool = False,
    mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
):
    if isinstance(img_size, int):
        img_size = (int(img_size), int(img_size))

    primary_tfl: List[K.GeometricAugmentationBase2D] = [
        K.RandomResizedCrop(
            size=img_size,
            scale=scale,
            ratio=ratio,
        )
    ]
    if hflip > 0.0:
        primary_tfl += [K.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [K.RandomVerticalFlip(p=vflip)]
    if rotate90 > 0.0:
        primary_tfl += [K.RandomRotation90(times=(1, 3), p=rotate90)]

    secondary_tfl = []
    if auto_augment:
        secondary_tfl += [RandAugment(n=2, m=9, policy=cell_policy)]

    if mean is not None and std is not None:
        norm = [K.Normalize(torch.tensor(mean), torch.tensor(std))]
    else:
        norm = []

    train_augs = nn.Sequential(
        K.Normalize(0.0, 255.0),
        *primary_tfl,
        *secondary_tfl,
        *norm,
    )

    return train_augs


def transforms_eval(
    img_size: Union[int, Tuple[int, int]] = 224,
    crop: bool = True,
    resize: bool = False,
    mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
):
    assert not (resize and crop), "Only one of 'resize' or 'crop' can be True at a time."
    if isinstance(img_size, int):
        img_size = (int(img_size), int(img_size))
    primary_tfl: List[K.GeometricAugmentationBase2D]
    if resize:
        primary_tfl = [
            K.Resize(
                size=img_size,
            )
        ]
    elif crop:
        primary_tfl = [
            K.CenterCrop(
                size=img_size,
            )
        ]
    else:
        primary_tfl = []
    if mean is not None and std is not None:
        norm = [K.Normalize(torch.tensor(mean), torch.tensor(std))]
    else:
        norm = []
    eval_augs = nn.Sequential(
        K.Normalize(0.0, 255.0),
        *primary_tfl,
        *norm,
    )

    return eval_augs
