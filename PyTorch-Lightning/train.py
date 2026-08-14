#!/usr/bin/env python3
"""
Generated PyTorch Lightning training script — Variational Autoencoder (VAE).

This script implements a fully-connected Variational Autoencoder to generate 
and reconstruct images. It showcases:
  1. DataModule pipeline for generative task images.
  2. Encoder network mapping images to latent distribution parameters (mu, log_var).
  3. Reparameterization trick (sampling z = mu + std * epsilon).
  4. Decoder network mapping latent space samples back to image space.
  5. Optimization of the Evidence Lower Bound (ELBO) loss.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# Configure PyTorch Lightning / Lightning package imports (compatibility wrapper)
try:
    import pytorch_lightning as L
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
except ImportError:
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# ──────────────────────────────────────────────────────────────────────────────
# 1. Hyperparameters & Configurations
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = "./data"              # Directory where datasets are downloaded or stored
LOG_DIR = "./lightning_logs"                # Parent directory for logger output files
EXPERIMENT_NAME = "gen_vae_test" # Subdirectory/name for this training run
LR = 1e-3                            # Learning rate for the optimizer
BATCH_SIZE = 32            # Input batch size for training and validation
NUM_WORKERS = 0          # Number of CPU subprocesses for data loading
MAX_EPOCHS = 10            # Maximum number of training epochs
ACCELERATOR = "cpu"        # Accelerator hardware ("cpu", "gpu", "xpu", etc.)
DEVICES = 1                  # Number of device instances to use (e.g. 1, "auto")
PRECISION = 32              # Training precision (e.g. 32, "16-mixed", "bf16-mixed")
LOG_EVERY_N_STEPS = 50 # How often to log training metrics
SEED = 42                        # Random seed for reproducibility

# VAE Specific Architecture Parameters
IN_CHANNELS = 3          # Number of image channels (e.g., 1 for MNIST, 3 for RGB)
IMAGE_SIZE = 224            # Width and height of input images
LATENT_DIM = 128            # Dimensionality of the bottleneck latent space (z)
FLAT_SIZE = 150528              # Flattened image size (IN_CHANNELS * IMAGE_SIZE * IMAGE_SIZE)

# ── Data Helpers & Custom Datasets ────────────────────────────────────────────
# Injected dynamically based on your dataset choice.
def load_builtin_dataset(name, data_dir, train):
    """Loads custom folder datasets utilizing standard torchvision ImageFolder."""
    env_var = "CC3M_PATH" if name == "CC3M" else ("CAMUS_PATH" if name == "CAMUS" else "LLaVA_OneVision_PATH")
    path = os.environ.get(env_var, "")
    split = "train" if train else "val"
    transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
    try:
        return datasets.ImageFolder(os.path.join(path, split), transform=transform)
    except Exception:
        try:
            return datasets.ImageFolder(path, transform=transform)
        except Exception:
            class DummyDataset(Dataset):
                def __len__(self): return 1000
                def __getitem__(self, idx):
                    return torch.rand(3, 224, 224), 0
            return DummyDataset()

# ── DataModule ────────────────────────────────────────────────────────────────
# Controls dataset setup, downloads, splits, and DataLoader instantiation.
class LitDataModule(L.LightningDataModule):
    """DataModule managing downloads, setup, and data loading for VAE builtin datasets."""
    def __init__(self, data_dir=DATA_DIR, batch_size=32, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = "llava-onevision"

    def setup(self, stage=None):
        # Load train and validation dataset splits
        self.train_ds = load_builtin_dataset(self.dataset_name, self.data_dir, train=True)
        self.val_ds = load_builtin_dataset(self.dataset_name, self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

# ── Generative Model ──
class LitModel(L.LightningModule):
    """
    Variational Autoencoder with fully-connected encoder/decoder.
    Optimizes the Evidence Lower Bound (ELBO): reconstruction loss + KL divergence.
    """

    def __init__(self, flat_size=150528, latent_dim=128, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Encoder: Projects input image to the latent distribution parameters (mu and log_var)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512), 
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

        # Decoder: Maps latent sample z back into reconstruction space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 512), 
            nn.ReLU(),
            nn.Linear(512, flat_size),
            nn.Sigmoid(),   # Constrain output pixels to [0, 1] range
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def _elbo_loss(self, x, x_hat, mu, log_var):
        x_flat = x.view(x.size(0), -1)
        recon = F.binary_cross_entropy(x_hat, x_flat, reduction="sum") / x.size(0)
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        return recon + kl, recon, kl

    def _shared_step(self, batch, stage):
        x, _ = batch
        x_hat, mu, log_var = self(x)
        loss, recon, kl = self._elbo_loss(x, x_hat, mu, log_var)
        
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_recon", recon)
        self.log(f"{stage}_kl", kl)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ── Logger Setup ─────────────────────────────────────────────────────────────
def build_loggers():
    loggers = [
    TensorBoardLogger(save_dir=LOG_DIR, name="gen_vae_test"),
    ]
    return [lg for lg in loggers if lg is not False]


# ── Main Entrypoint & Training Loop ────────────────────────────────────────────
def main():
    L.seed_everything(SEED, workers=True)
    
    # Initialize data pipeline and run download/setup steps
    datamodule = LitDataModule()
    datamodule.setup()
    
    # Initialize the model using configured dimension properties
    model = LitModel(flat_size=FLAT_SIZE, latent_dim=LATENT_DIM, lr=LR)
    loggers = build_loggers()
    
    callbacks = [
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True),
        ]
    
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
