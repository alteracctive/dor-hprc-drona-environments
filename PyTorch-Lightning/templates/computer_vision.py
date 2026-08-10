#!/usr/bin/env python3
"""
Generated PyTorch Lightning training script — Computer Vision.

This script demonstrates a complete deep learning workflow:
  1. Data preparation and loading using PyTorch Lightning DataModules.
  2. Model definition (MLP or CNN) subclassing LightningModule.
  3. Configurable training runs with logging (TensorBoard, etc.) and checkpoints.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# __LIGHTNING_IMPORT_BLOCK__
# __ADDITIONAL_IMPORTS__

# ──────────────────────────────────────────────────────────────────────────────
# 1. Hyperparameters & Configurations
#    Modify these variables to adjust model and training settings.
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = "__DATA_DIR__"              # Directory where datasets are downloaded or stored
LOG_DIR = "__LOG_DIR__"                # Parent directory for logger output files
EXPERIMENT_NAME = "__EXPERIMENT_NAME__" # Subdirectory/name for this training run
LR = __LR__                            # Learning rate for the optimizer
BATCH_SIZE = __BATCH_SIZE__            # Input batch size for training and validation
NUM_WORKERS = __NUM_WORKERS__          # Number of CPU subprocesses for data loading
MAX_EPOCHS = __MAX_EPOCHS__            # Maximum number of training epochs
ACCELERATOR = "__ACCELERATOR__"        # Accelerator hardware ("cpu", "gpu", "xpu", etc.)
DEVICES = __DEVICES__                  # Number of device instances to use (e.g. 1, "auto")
PRECISION = __PRECISION__              # Training precision (e.g. 32, "16-mixed", "bf16-mixed")
LOG_EVERY_N_STEPS = __LOG_EVERY_N_STEPS__ # How often to log training metrics
SEED = __SEED__                        # Random seed for reproducibility

# ── Data Helpers & Custom Datasets ────────────────────────────────────────────
# Injected dynamically based on your dataset choice.
# __DATA_HELPERS_BLOCK__

# ── DataModule ────────────────────────────────────────────────────────────────
# Controls dataset setup, downloads, splits, and DataLoader instantiation.
# __DATAMODULE_BLOCK__

# ── Model & Training/Validation Steps ──────────────────────────────────────────
# Custom model architecture (subclass of L.LightningModule).
# Defines forward pass, optimizer, and training/validation step loops.
# __MODEL_BLOCK__

# ── Main Entrypoint & Training Orchestration ─────────────────────────────────
# Sets up seeds, datamodule, model, callbacks, logger, and runs trainer.fit().
# __MAIN_BLOCK__
