import torch
import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# 保留你原本的 utils 引用
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
def apply_ema_weights_to_model(model: LightningModule, ckpt_path: str):
    """
    手动从 Checkpoint 的 optimizer_states 中加载 EMA 权重并应用到模型上。
    适用于 trainer.validate 模式。
    """
    log.info(f"Attempting to load EMA weights from {ckpt_path} ...")
    
    # 1. 加载 Checkpoint (CPU)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # 2. 检查是否存在 optimizer_states
    if "optimizer_states" not in checkpoint or not checkpoint["optimizer_states"]:
        log.warning("No optimizer_states found in checkpoint! Using original weights.")
        return

    # 假设只有一个 Optimizer (绝大多数情况)
    opt_state = checkpoint["optimizer_states"][0]
    
    # 3. 检查 EMA 数据是否存在
    # 你的 EMAOptimizer 在 state_dict 中保存了 'ema' 字段
    if "ema" not in opt_state:
        log.warning("No 'ema' key found in optimizer state! Using original weights.")
        return
    
    ema_params_list = opt_state["ema"]
    
    # 4. 获取模型当前参数列表
    # 注意：顺序必须与 Optimizer 初始化时的参数顺序一致
    # 通常 Lightning 默认也是 model.parameters() 这个顺序
    model_params = list(model.parameters())
    
    if len(model_params) != len(ema_params_list):
        log.error(f"Mismatch! Model has {len(model_params)} params, but EMA has {len(ema_params_list)} params.")
        return

    # 5. 覆盖权重
    log.info("Overwriting model weights with EMA weights...")
    for param, ema_param in zip(model_params, ema_params_list):
        # 确保数据类型和设备一致
        param.data.copy_(ema_param.data.to(device=param.device, dtype=param.dtype))
        
    log.info("Successfully applied EMA weights to the model!")

@task_wrapper
def validate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluates the model using a given checkpoint.
    """
    # 1. 设置随机种子 (保持复现性)
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # 2. 实例化 DataModule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # 3. 实例化 Model
    # 注意：这里只是根据 config 实例化模型结构，权重是随机的。
    # 真正的权重会在 trainer.validate 时通过 ckpt_path 加载覆盖。
    log.info(f"Instantiating model <{cfg.lightning_module._target_}>")
    lightning_module: LightningModule = hydra.utils.instantiate(cfg.lightning_module)

    # 4. 实例化 Callbacks (虽然验证不需要 ModelCheckpoint，但可能需要其他的如 RichProgressBar)
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # 5. 实例化 Loggers
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # 6. 实例化 Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": lightning_module, # 注意这里修正了原代码可能的引用问题，直接用 instance
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # --- 核心修改部分 ---
    
    # 获取 Checkpoint 路径
    # ckpt_path = cfg.get("ckpt_path")
    
    # 验证必须要有 ckpt_path
    # if not ckpt_path:
    #     raise ValueError("Check point path must be specified for validation! (cfg.ckpt_path)")

    # 检查文件是否存在
    # if not os.path.exists(ckpt_path):
    #     raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt_path = 'epoch=757-step=285766.ckpt'
    log.info(f"Starting validation using checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    lightning_module.load_state_dict(checkpoint["state_dict"]) 
    
    # 2. 【核心步骤】手动覆盖 EMA 权重
    apply_ema_weights_to_model(lightning_module, ckpt_path)

    # 7. 运行 Validation
    # 这会自动加载 ckpt_path 中的权重到 lightning_module，并运行 validation_step
    trainer.validate(
        model=lightning_module, 
        datamodule=datamodule, 
        ckpt_path=None
    )
    
    # 获取验证指标
    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


# 这里建议单独建立一个 eval.yaml 或者复用 train.yaml 但通过命令行覆盖参数
@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml") 
def main(cfg: DictConfig) -> Optional[float]:
    
    extras(cfg)

    # 调用 validate 而不是 train
    metric_dict, _ = validate(cfg)

    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    print(metric_value)
    return metric_value


if __name__ == "__main__":
    main()