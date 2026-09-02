import os
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


from posedreamer.generation.alignment.dataset import DPOEasyControlDataset, collate_dpo_batch
from posedreamer.generation.alignment.trainer import DPOEasyControlTrainer

logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig):
    """Setup logging configuration."""
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(cfg.logging.log_dir) / "train.log")
        ]
    )


def set_tensor_core_optimization():
    """Optimize tensor cores for H100 GPU performance."""
    torch.set_float32_matmul_precision('medium')
    logger.info("Set tensor core optimization to 'medium' for H100 performance")


def set_seed(seed: int):
    """Set seeds for reproducibility."""
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloader(cfg: DictConfig):
    """Create train and validation datasets."""
    logger.info(f"Creating dataset from {cfg.dataset.metadata_path}")
    dataset = DPOEasyControlDataset(
        metadata_path=cfg.dataset.metadata_path,
        metric_name=cfg.dataset.metric_name,
        metric_mode=cfg.dataset.metric_mode,
        image_size=cfg.dataset.image_size,
        cond_size=cfg.dataset.cond_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_dpo_batch,
    )
    
    return dataloader


def create_model(cfg: DictConfig):
    """Create the DPO trainer model."""
    logger.info("Creating DPO EasyControl trainer model")
    
    model = DPOEasyControlTrainer(
        pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
        control_lora_path=cfg.model.control_lora_path,
        learning_rate=cfg.model.learning_rate,
        beta=cfg.model.beta,
        cond_size=cfg.dataset.cond_size,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
    )
    return model

def create_callbacks(cfg: DictConfig, output_dir: str):
    """Create PyTorch Lightning callbacks."""
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        every_n_train_steps=cfg.training.checkpoint_every_n_steps,
        filename="{epoch}-step={step}-{train_loss:.4f}",
        auto_insert_metric_name=False,
        save_top_k=-1,
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "_"
    callbacks.append(checkpoint_callback)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


def create_trainer(cfg: DictConfig, callbacks, loggers, output_dir: str):
    """Create PyTorch Lightning trainer."""
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        strategy=cfg.trainer.strategy,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        callbacks=callbacks,
        logger=loggers,
        deterministic=True,
        log_every_n_steps=1,
    )
    
    return trainer

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    logger.info("Training configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    setup_logging(cfg)
    set_tensor_core_optimization()
    set_seed(cfg.seed)
    
    logger.info("Starting DPO training for EasyControl")
    dataloader = create_dataloader(cfg)

    current_time = datetime.now()
    date_str = current_time.strftime("%Y_%d_%m-%H_%M_%S")
    output_dir = os.path.join(cfg.logging.log_dir, cfg.logging.experiment_name, date_str)
    os.makedirs(output_dir, exist_ok=True)
    tensorboard_dir = os.path.join(output_dir, "tensorboard_logs")

    tb_logger = TensorBoardLogger(
        save_dir=tensorboard_dir,
        name="reward_lora"
    )
    
    model = create_model(cfg)
    callbacks = create_callbacks(cfg, output_dir)
    trainer = create_trainer(cfg, callbacks, tb_logger, output_dir)
    
    logger.info("Starting training...")
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main() 
