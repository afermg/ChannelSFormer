# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
from math import inf
from typing import Optional

import torch
import torch.distributed as dist
from timm.utils import ModelEma as ModelEma
from yacs.config import _VALID_TYPES, CfgNode, _assert_with_logging, _valid_type


def load_checkpoint_ema(
    config,
    model,
    optimizer,
    lr_scheduler,
    loss_scaler,
    logger,
    model_ema: Optional[ModelEma] = None,
):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        logger.info(f"resuming model: {msg}")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.RESUME}! ")

    if model_ema is not None:
        if "model_ema" in checkpoint:
            msg = model_ema.ema.load_state_dict(checkpoint["model_ema"], strict=False)
            logger.info(f"resuming model_ema: {msg}")
        else:
            logger.warning(f"No 'model_ema' found in {config.MODEL.RESUME}! ")

    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    if (
        not config.EVAL_MODE
        and "optimizer" in checkpoint
        and "lr_scheduler" in checkpoint
        and "epoch" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        logger.info(
            f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})"
        )
        if "max_accuracy" in checkpoint:
            max_accuracy = checkpoint["max_accuracy"]
        if "max_accuracy_ema" in checkpoint:
            max_accuracy_ema = checkpoint["max_accuracy_ema"]

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, max_accuracy_ema


def load_pretrained_ema(config, model, logger, model_ema: Optional[ModelEma] = None):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded 'model' successfully from '{config.MODEL.PRETRAINED}'")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.PRETRAINED}! ")

    if model_ema is not None:
        if "model_ema" in checkpoint:
            logger.info("=> loading 'model_ema' separately...")
        key = "model_ema" if ("model_ema" in checkpoint) else "model"
        if key in checkpoint:
            msg = model_ema.ema.load_state_dict(checkpoint[key], strict=False)
            logger.warning(msg)
            logger.info(
                f"=> loaded '{key}' successfully from '{config.MODEL.PRETRAINED}' for model_ema"
            )
        else:
            logger.warning(f"No '{key}' found in {config.MODEL.PRETRAINED}! ")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint_ema(
    config,
    epoch,
    model,
    max_accuracy,
    optimizer,
    lr_scheduler,
    loss_scaler,
    logger,
    model_ema: Optional[ModelEma] = None,
    max_accuracy_ema=None,
):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "max_accuracy": max_accuracy,
        "scaler": loss_scaler.state_dict(),
        "epoch": epoch,
        "config": config,
    }

    if model_ema is not None:
        save_state.update(
            {"model_ema": model_ema.ema.state_dict(), "max_accuray_ema": max_accuracy_ema}
        )

    save_path = os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith("pth")]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime
        )
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def yacs_to_dict(cfg_node, key_list):
    if not isinstance(cfg_node, CfgNode):
        _assert_with_logging(
            _valid_type(cfg_node),
            "Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES
            ),
        )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = yacs_to_dict(v, key_list + [k])
        return cfg_dict


try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


class WandbLogger(object):
    def __init__(self, config):
        # initialize wandb logger
        if config.WANDB:
            dist_mode = dist.is_available() and dist.is_initialized()
            if not wandb_available:
                raise ImportError("wandb is not installed. Please install it to use wandb logger.")
            if dist_mode:
                # initialize wandb logger
                if dist.get_rank() == 0:
                    print(
                        f"wandb init with project {config.PROJECT}, tag "
                        f"{config.TAG}, output {config.OUTPUT}"
                    )
                    # print("Config dump:", config.dump())
                    config_dict = yacs_to_dict(config, [])
                    self.run = wandb.init(
                        project=config.PROJECT,
                        name=config.TAG,
                        dir=config.OUTPUT,
                        config=config_dict,
                    )  # config.dump())
                    self.config = config
                    self.model_name = f"model-{self.run.id}"
                    self.model_type = "model"
                    print(f"wandb run initialized with id {self.run.id}")
            else:
                # none distributed training for debugging
                print(
                    f"wandb init with project {config.PROJECT}, tag {config.TAG}, "
                    f"output {config.OUTPUT}"
                )
                # print("Config dump:", config.dump())
                config_dict = yacs_to_dict(config, [])
                self.run = wandb.init(
                    project=config.PROJECT, name=config.TAG, dir=config.OUTPUT, config=config_dict
                )  # config.dump())
                self.config = config
                self.model_name = f"model-{self.run.id}"
                self.model_type = "model"

        if not hasattr(self, "run"):
            self.run = None
            self.config = None
            self.model_name = None
            self.model_type = None

    def define_metric(self, name, **kwargs):
        if self.run is not None:
            self.run.define_metric(name, **kwargs)

    def log(self, data, step=None):
        """
        Log data to wandb.
        Args:
            data: dictionary of data to log
            step: step number to log the data at
        """
        if self.run is not None:
            wandb.log(data, step=step)

    def finish(self, clean_up_model=False):
        """
        Finish the wandb run.
        This should be called at the end of the training script to ensure all data is logged.
        """
        if self.run is not None:
            wandb.finish()
            if clean_up_model:
                self.clean_up_model()
            self.run = None

    # config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler,
    # logger, model_ema: ModelEma=None, max_accuracy_ema=None)
    def log_model(
        self,
        epoch,
        model,
        accuracy,
        max_accuracy,
        optimizer,
        lr_scheduler,
        loss_scaler,
        logger,
        model_ema=None,
        accuracy_ema=None,
        max_accuracy_ema=None,
    ):
        """
        Log model checkpoints to wandb.
        This saves the model state and optimizer state to wandb artifacts.
        """
        if self.run is not None:
            aliases = ["latest"]

            if accuracy == max_accuracy:
                aliases.append("best")

            if accuracy_ema and accuracy_ema == max_accuracy_ema:
                aliases.append("best_ema")

            # Create a model state dictionary with all necessary components
            model_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "max_accuracy": max_accuracy,
                "scaler": loss_scaler.state_dict(),
                "epoch": epoch,
                "config": self.config,
            }

            if model_ema is not None:
                model_state.update(
                    {"model_ema": model_ema.ema.state_dict(), "max_accuracy_ema": max_accuracy_ema}
                )

            # directly save the model_states wandb artifact without additionally saving to disk
            model_artifact = wandb.Artifact(
                name=self.model_name, type=self.model_type, incremental=False
            )

            # Create a temporary buffer to store model state
            with model_artifact.new_file("model.pth", mode="wb") as f:
                torch.save(model_state, f)

            if logger:
                logger.info("Directly saving model to wandb artifact...")

            # Log the artifact with appropriate aliases
            self.run.log_artifact(model_artifact, aliases=aliases, tags=["epoch_{}".format(epoch)])

            # # Always save and log the latest model
            # model_path = os.path.join(self.config.OUTPUT, 'model.pth')
            # torch.save(model_state, model_path)
            #
            # if logger:
            #     logger.info(f"save model {model_path} to wandb...")
            # # model_name = f"model-{self.run.id}"
            # model_artifact = wandb.Artifact(
            #     name=self.model_name, type=self.model_type,
            #     incremental=False
            # )
            # model_artifact.add_file(model_path, name="model.pth")
            # self.run.log_artifact(model_artifact, aliases=aliases)

    def clean_up_model(self):
        """
        Clean up wandb run if it exists.
        """
        if self.run is not None:
            print(f"clean up wandb run {self.run.id}")
            api = wandb.Api()
            artifacts = api.artifacts(
                name=f"{self.run.entity}/{self.run.project}/{self.model_name}",
                type_name=self.model_type,
            )
            for v in artifacts:
                print(f"wandb artifact {v.name} with aliases {v.aliases}")
                if len(v.aliases) == 0:
                    print(f"\tDeleting wandb artifact {v.name} with no aliases")
                    v.delete()
