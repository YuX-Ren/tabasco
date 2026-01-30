import torch
import os

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from tabasco.models.lightning_tabasco import LightningTabasco


from tabasco.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

torch.set_float32_matmul_precision("high")
log = RankedLogger(__name__, rank_zero_only=True)
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def apply_ema_weights_to_model(model: torch.nn.Module, ckpt_path: str, device: str = "cpu"):
    """
    Manually load EMA weights from the checkpoint's optimizer_states and apply them to the model.
    """
    print(f"Loading checkpoint from {ckpt_path} to extract EMA weights...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "optimizer_states" not in checkpoint or not checkpoint["optimizer_states"]:
        print("WARNING: No optimizer_states found in checkpoint! Using original weights.")
        return

    # Assuming single optimizer
    opt_state = checkpoint["optimizer_states"][0]

    if "ema" not in opt_state:
        print("WARNING: No 'ema' key found in optimizer state! Using original weights.")
        return

    ema_params_list = opt_state["ema"]
    model_params = list(model.parameters())

    if len(model_params) != len(ema_params_list):
        print(f"ERROR: Model has {len(model_params)} params, but EMA has {len(ema_params_list)} params.")
        return

    print("Overwriting model weights with EMA weights...")
    with torch.no_grad():
        for param, ema_param in zip(model_params, ema_params_list):
            # Ensure dtype and device match
            param.data.copy_(ema_param.to(device=param.device, dtype=param.dtype))
            
    print("Successfully applied EMA weights to the model!")


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.lightning_module._target_}>")
    lightning_module: LightningModule = hydra.utils.instantiate(cfg.lightning_module)
    checkpoint_path = 'protein_kl_32.ckpt'
    # lightning_module = lightning_module.load_from_checkpoint('outputs/2025-11-13/03-20-07/checkpoints/checkpoint_epoch=1399.ckpt')
    checkpoint = torch.load(checkpoint_path, weights_only=False)  # 加载 checkpoint
    lightning_module.load_state_dict(checkpoint["state_dict"],strict=False)  # 加载 state_dict（模型权重）
    apply_ema_weights_to_model(lightning_module.model, 'protein_kl_32.ckpt')
    print('loaded pretrained model')
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": lightning_module.model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=lightning_module,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
