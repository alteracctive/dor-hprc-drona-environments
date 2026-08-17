#!/usr/bin/env python3
"""
Generated PyTorch Lightning training script — Graph Neural Network (GNN).
Supports GCN, GAT, and GraphSAGE message-passing models.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure PyTorch Lightning / Lightning package imports (compatibility wrapper)
try:
    import pytorch_lightning as L
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
except ImportError:
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint

# Temporary mock of torch.cuda.is_available to allow PyTorch Geometric imports on CPU/login nodes
_orig_cuda_available = torch.cuda.is_available
torch.cuda.is_available = lambda: True
try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
finally:
    torch.cuda.is_available = _orig_cuda_available

# ──────────────────────────────────────────────────────────────────────────────
# 1. Hyperparameters & Configurations
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = "./data"
LOG_DIR = "./lightning_logs"
EXPERIMENT_NAME = "lightning_run"
LR = 2e-3
BATCH_SIZE = 32
NUM_WORKERS = 0
MAX_EPOCHS = 15
ACCELERATOR = "cpu"
DEVICES = 1
PRECISION = 32
LOG_EVERY_N_STEPS = 50
SEED = 100

GNN_HIDDEN_DIM = 64
GNN_NUM_LAYERS = 2

# ── DataModule ──
class LitDataModule(L.LightningDataModule):
    def setup(self, stage=None):
        # Load prepared QM9 dataset from cluster module environment path
        from torch_geometric.datasets import QM9
        path = os.environ.get("QM9_PATH", "./data/QM9")
        dataset = QM9(root=path)
        
        # Split into training and validation sets
        split_idx = int(len(dataset) * 0.8)
        self.train_ds = dataset[:split_idx]
        self.val_ds = dataset[split_idx:]
        
        self.num_features = dataset.num_features
        self.num_classes = 19  # QM9 has 19 regression targets
        self.is_gnn_benchmark = True
        self.is_qm9 = True

    def train_dataloader(self):
        from torch_geometric.loader import DataLoader as PyGDataLoader
        if getattr(self, "is_gnn_benchmark", False):
            return PyGDataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        else:
            return PyGDataLoader([self.data], batch_size=1)

    def val_dataloader(self):
        from torch_geometric.loader import DataLoader as PyGDataLoader
        if getattr(self, "is_gnn_benchmark", False):
            return PyGDataLoader(self.val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        else:
            return PyGDataLoader([self.data], batch_size=1)

# ── Graph Neural Network Model ──
class LitModel(L.LightningModule):
    """Multi-layer Graph Neural Network using GCNConv layers."""

    def __init__(self, in_channels, hidden_channels=64,
                 out_channels=1, num_layers=2, lr=2e-3, is_qm9=False):
        super().__init__()
        self.save_hyperparameters()
        
        conv_cls = {
            "gcn": GCNConv,
            "gat": lambda in_c, out_c: GATConv(in_c, out_c, heads=1),
            "sage": SAGEConv
        }["gcn"]
        
        self.convs = nn.ModuleList()
        self.convs.append(conv_cls(in_channels, hidden_channels))
        for _ in range(max(0, num_layers - 2)):
            self.convs.append(conv_cls(hidden_channels, hidden_channels))
        self.convs.append(conv_cls(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            out = conv(x, edge_index)
            if isinstance(out, tuple): out = out[0]
            x = F.relu(out)
            x = F.dropout(x, p=0.5, training=self.training)
            
        out = self.convs[-1](x, edge_index)
        if isinstance(out, tuple): out = out[0]
        return out

    def _shared_step(self, batch, stage):
        out = self(batch.x, batch.edge_index)
        if self.hparams.is_qm9:
            from torch_geometric.nn import global_mean_pool
            out = global_mean_pool(out, batch.batch)
            loss = F.l1_loss(out, batch.y)
            self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
            self.log(f"{stage}_mae", loss, prog_bar=True, batch_size=batch.num_graphs)
        else:
            mask = getattr(batch, f"{stage}_mask", None)
            if mask is not None:
                loss = F.cross_entropy(out[mask], batch.y[mask])
                acc = (out[mask].argmax(dim=-1) == batch.y[mask]).float().mean()
            else:
                loss = F.cross_entropy(out, batch.y)
                acc = (out.argmax(dim=-1) == batch.y).float().mean()
            self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch.num_graphs if hasattr(batch, 'num_graphs') else 1)
            self.log(f"{stage}_acc", acc, prog_bar=True, batch_size=batch.num_graphs if hasattr(batch, 'num_graphs') else 1)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)


def build_loggers():
    """Build list of active loggers configured in the generator UI."""
    loggers = [
    TensorBoardLogger(save_dir=LOG_DIR, name="lightning_run"),
    ]
    return [lg for lg in loggers if lg is not False]

def main():
    """Main training orchestration function."""
    L.seed_everything(SEED, workers=True)
    
    accelerator = ACCELERATOR
    devices = DEVICES
    


    # Initialize data pipeline
    datamodule = LitDataModule()
    datamodule.setup()
    
    # Initialize the model using configuration settings
    model = LitModel(
        in_channels=datamodule.num_features,
        hidden_channels=GNN_HIDDEN_DIM,
        out_channels=datamodule.num_classes,
        num_layers=GNN_NUM_LAYERS,
        lr=LR,
        is_qm9=getattr(datamodule, "is_qm9", False),
    )
    
    # Configure logging and callback checkpoints
    loggers = build_loggers()
    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True),
    ]
    
    # Set up PyTorch Lightning Trainer with all hyperparameters
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator,
        devices=devices,
        precision=PRECISION,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        logger=loggers if loggers else None,
        callbacks=callbacks,
        enable_checkpointing=True,
    )
    
    # Run the model fitting phase
    trainer.fit(model, datamodule=datamodule)

    # Save checkpoint weights side-by-side as a clean PyTorch state-dict (.pt file)
    checkpoint_callback = trainer.checkpoint_callback
    if checkpoint_callback and getattr(checkpoint_callback, "best_model_path", None):
        ckpt_paths = set()
        if getattr(checkpoint_callback, "best_model_path", None):
            ckpt_paths.add(checkpoint_callback.best_model_path)
        if getattr(checkpoint_callback, "last_model_path", None):
            ckpt_paths.add(checkpoint_callback.last_model_path)
            
        for ckpt_path in filter(None, ckpt_paths):
            if os.path.exists(ckpt_path):
                pt_path = os.path.splitext(ckpt_path)[0] + ".pt"
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    # Save just the weights dictionary (state_dict) if present, else save the whole object
                    weights = ckpt.get("state_dict", ckpt)
                    torch.save(weights, pt_path)
                    print(f"Saved PyTorch weights side-by-side at: {pt_path}")
                except Exception as e:
                    print(f"Could not save side-by-side .pt weights from {ckpt_path}: {e}")

if __name__ == "__main__":
    main()

