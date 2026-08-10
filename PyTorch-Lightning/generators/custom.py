import os
import re
from .helpers import (
    _py_str,
    _get_env_dir,
)

_DEFAULT_CUSTOM_STUB = '''#!/usr/bin/env python3
"""Generated PyTorch Lightning training script — Custom (Bare-bones)."""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
try:
    import pytorch_lightning as L
except ImportError:
    import lightning as L

# Optional: Import standard callbacks for monitoring and early stopping
# try:
#     from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, DeviceStatsMonitor
# except ImportError:
#     from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, DeviceStatsMonitor

DATA_PATH = ""
LOG_DIR = "./lightning_logs"
EXPERIMENT_NAME = "lightning_run"
LR = 1e-3
BATCH_SIZE = 32
NUM_WORKERS = 0
MAX_EPOCHS = 10
ACCELERATOR = "cpu"
DEVICES = 1
PRECISION = 32
LOG_EVERY_N_STEPS = 50
SEED = 42

class MyDataset(Dataset):
    def __init__(self, data_path=DATA_PATH, train=True):
        self.length = 100
    def __len__(self): return self.length
    def __getitem__(self, idx):
        return torch.zeros(10), torch.tensor(0)

class LitDataModule(L.LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
    def setup(self, stage=None):
        self.train_ds = MyDataset()
        self.val_ds = MyDataset()
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

class LitModel(L.LightningModule):
    def __init__(self, lr=LR):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Linear(10, 2)
    def forward(self, x): return self.net(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def build_loggers():
    return []

def main():
    L.seed_everything(SEED, workers=True)
    datamodule = LitDataModule()
    model = LitModel(lr=LR)

    # Configure callbacks if needed for prototyping (e.g. EarlyStopping, LearningRateMonitor)
    # callbacks = [
    #     EarlyStopping(monitor="val_loss", patience=3, mode="min"),
    #     LearningRateMonitor(logging_interval="step"),
    # ]

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS, 
        accelerator=ACCELERATOR, 
        devices=DEVICES,
        # callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
'''

def _gen_custom_script(
    exp_name, ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
    logger_lines, callback_block,
    custom_dataset_path="",
):
    """Returns (train_script, None) — loads the workspace train.py and updates hyperparameters."""
    env_dir = _get_env_dir()
    train_py_path = env_dir / "train.py"
    if train_py_path.is_file():
        with open(train_py_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = _DEFAULT_CUSTOM_STUB

    prec_val = str(prec) if str(prec).isdigit() else f'"{_py_str(prec)}"'

    content = re.sub(r'^(DATA_PATH\s*=\s*).*$', lambda m: m.group(1) + f'"{_py_str(custom_dataset_path)}"', content, flags=re.MULTILINE)
    content = re.sub(r'^(LOG_DIR\s*=\s*).*$', lambda m: m.group(1) + f'"{_py_str(log_dir)}"', content, flags=re.MULTILINE)
    content = re.sub(r'^(EXPERIMENT_NAME\s*=\s*).*$', lambda m: m.group(1) + f'"{_py_str(exp_name)}"', content, flags=re.MULTILINE)
    content = re.sub(r'^(LR\s*=\s*).*$', lambda m: m.group(1) + str(lr), content, flags=re.MULTILINE)
    content = re.sub(r'^(BATCH_SIZE\s*=\s*).*$', lambda m: m.group(1) + str(bs), content, flags=re.MULTILINE)
    content = re.sub(r'^(NUM_WORKERS\s*=\s*).*$', lambda m: m.group(1) + str(nw), content, flags=re.MULTILINE)
    content = re.sub(r'^(MAX_EPOCHS\s*=\s*).*$', lambda m: m.group(1) + str(ep), content, flags=re.MULTILINE)
    content = re.sub(r'^(ACCELERATOR\s*=\s*).*$', lambda m: m.group(1) + f'"{_py_str(acc)}"', content, flags=re.MULTILINE)
    content = re.sub(r'^(DEVICES\s*=\s*).*$', lambda m: m.group(1) + str(dev), content, flags=re.MULTILINE)
    content = re.sub(r'^(PRECISION\s*=\s*).*$', lambda m: m.group(1) + prec_val, content, flags=re.MULTILINE)
    content = re.sub(r'^(LOG_EVERY_N_STEPS\s*=\s*).*$', lambda m: m.group(1) + str(log_n), content, flags=re.MULTILINE)
    content = re.sub(r'^(SEED\s*=\s*).*$', lambda m: m.group(1) + str(seed_val), content, flags=re.MULTILINE)

    # Update build_loggers() list dynamically
    loggers_content = "\n".join(logger_lines)
    content = re.sub(
        r'(def build_loggers\(\):\s*\n\s*loggers\s*=\s*\[)(.*?)(\]\s*\n\s*return)',
        rf'\g<1>\n{loggers_content}\n\g<3>',
        content,
        flags=re.DOTALL
    )

    # Update callbacks assignment in main()
    content = re.sub(
        r'(\n\s*loggers\s*=\s*build_loggers\(\)\s*\n\s*)(callbacks\s*=\s*[^\n]+)(\s*\n\s*trainer\s*=\s*L\.Trainer)',
        rf'\g<1>{callback_block}\g<3>',
        content
    )

    return content, None
