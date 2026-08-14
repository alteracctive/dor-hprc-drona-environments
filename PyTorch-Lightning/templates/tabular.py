#!/usr/bin/env python3
"""
Generated PyTorch Lightning training script — Tabular & Structured Data.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# __LIGHTNING_IMPORT_BLOCK__

# ──────────────────────────────────────────────────────────────────────────────
# 1. Hyperparameters & Configurations
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = "__DATA_DIR__"
LOG_DIR = "__LOG_DIR__"
EXPERIMENT_NAME = "__EXPERIMENT_NAME__"
LR = __LR__
BATCH_SIZE = __BATCH_SIZE__
NUM_WORKERS = __NUM_WORKERS__
MAX_EPOCHS = __MAX_EPOCHS__
ACCELERATOR = "__ACCELERATOR__"
DEVICES = __DEVICES__
PRECISION = __PRECISION__
LOG_EVERY_N_STEPS = __LOG_EVERY_N_STEPS__
SEED = __SEED__

# Tabular Specific Settings
MODEL_TYPE = "__MODEL_TYPE__"  # "mlp", "resnet", "tabnet"
TASK_TYPE = "__TASK_TYPE__"    # "classification", "regression"

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── DataModule ──
class LitDataModule(L.LightningDataModule):
    def setup(self, stage=None):
        # __DATASET_SETUP__
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


# ── Model Architectures ──
# __MODEL_BLOCK__


# ── Logger Setup ─────────────────────────────────────────────────────────────
def build_loggers():
    loggers = [
        # __LOGGER_LINES__
    ]
    return [lg for lg in loggers if lg is not False]


# ── Main Entrypoint & Training Loop ────────────────────────────────────────────
def main():
    L.seed_everything(SEED, workers=True)
    datamodule = LitDataModule()
    datamodule.setup()
    
    in_features = datamodule.in_features
    num_classes = getattr(datamodule, "num_classes", 1)
    
    model = LitModel(in_features=in_features, out_features=num_classes)
    loggers = build_loggers()
    
    # __CALLBACK_BLOCK__
    
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        logger=loggers if loggers else None,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
