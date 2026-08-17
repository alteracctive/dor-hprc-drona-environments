import os
from .helpers import (
    _py_str,
    _trainer_main_block,
)

def _gen_gnn_script(
    exp_name, graph_ds_type, graph_blt, graph_path,
    ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
    logger_lines, callback_block,
    gnn_hidden=64,
    gnn_layers=2,
    gnn_layer="gcn",
):
    """Returns (train_script, None) for Graph Neural Network."""
    use_builtin = (graph_ds_type == "builtin")
    data_dir = "./data"

    # Dataset setup code block
    if use_builtin:
        dataset_setup = f'''# Load prepared QM9 dataset from cluster module environment path
        from torch_geometric.datasets import QM9
        path = os.environ.get("QM9_PATH", "{data_dir}/QM9")
        dataset = QM9(root=path)
        
        # Split into training and validation sets
        split_idx = int(len(dataset) * 0.8)
        self.train_ds = dataset[:split_idx]
        self.val_ds = dataset[split_idx:]
        
        self.num_features = dataset.num_features
        self.num_classes = 19  # QM9 has 19 regression targets
        self.is_gnn_benchmark = True
        self.is_qm9 = True'''
    else:
        dataset_setup = f'''# Load custom graph dataset
        path = "{_py_str(graph_path)}"
        if os.path.isfile(path):
            self.data = torch.load(path)
        else:
            self.data = torch.load(os.path.join(path, "data.pt"))
            
        self.num_features = self.data.num_node_features
        if self.data.y.ndim == 1 and self.data.y.numel() > 1:
            self.num_classes = int(self.data.y.max().item()) + 1
        else:
            self.num_classes = self.data.y.shape[-1] if self.data.y.ndim > 1 else 1
            
        self.is_gnn_benchmark = False
        self.is_qm9 = False'''

    # Precision formatting
    prec_val = str(prec) if str(prec).isdigit() else f'"{_py_str(prec)}"'

    # Model initialization argument block for the trainer main function
    model_init = f'''model = LitModel(
        in_channels=datamodule.num_features,
        hidden_channels=GNN_HIDDEN_DIM,
        out_channels=datamodule.num_classes,
        num_layers=GNN_NUM_LAYERS,
        lr=LR,
        is_qm9=getattr(datamodule, "is_qm9", False),
    )'''

    # Build the main trainer orchestration block
    main_block = _trainer_main_block(
        acc=acc, dev=dev, prec=prec, log_n=log_n,
        model_init=model_init, callback_block=callback_block, logger_lines=logger_lines
    )

    script_content = f'''#!/usr/bin/env python3
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

from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# ──────────────────────────────────────────────────────────────────────────────
# 1. Hyperparameters & Configurations
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = "{data_dir}"
LOG_DIR = "{_py_str(log_dir)}"
EXPERIMENT_NAME = "{_py_str(exp_name)}"
LR = {lr}
BATCH_SIZE = {bs}
NUM_WORKERS = {nw}
MAX_EPOCHS = {ep}
ACCELERATOR = "{_py_str(acc)}"
DEVICES = {dev}
PRECISION = {prec_val}
LOG_EVERY_N_STEPS = {log_n}
SEED = {seed_val}

GNN_HIDDEN_DIM = {gnn_hidden}
GNN_NUM_LAYERS = {gnn_layers}

# ── DataModule ──
class LitDataModule(L.LightningDataModule):
    def setup(self, stage=None):
        {dataset_setup}

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
    """Multi-layer Graph Neural Network using {_py_str(gnn_layer.upper())}Conv layers."""

    def __init__(self, in_channels, hidden_channels={gnn_hidden},
                 out_channels=1, num_layers={gnn_layers}, lr={lr}, is_qm9=False):
        super().__init__()
        self.save_hyperparameters()
        
        conv_cls = {{
            "gcn": GCNConv,
            "gat": lambda in_c, out_c: GATConv(in_c, out_c, heads=1),
            "sage": SAGEConv
        }}["{_py_str(gnn_layer)}"]
        
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
            self.log(f"{{stage}}_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
            self.log(f"{{stage}}_mae", loss, prog_bar=True, batch_size=batch.num_graphs)
        else:
            mask = getattr(batch, f"{{stage}}_mask", None)
            if mask is not None:
                loss = F.cross_entropy(out[mask], batch.y[mask])
                acc = (out[mask].argmax(dim=-1) == batch.y[mask]).float().mean()
            else:
                loss = F.cross_entropy(out, batch.y)
                acc = (out.argmax(dim=-1) == batch.y).float().mean()
            self.log(f"{{stage}}_loss", loss, prog_bar=True, batch_size=batch.num_graphs if hasattr(batch, 'num_graphs') else 1)
            self.log(f"{{stage}}_acc", acc, prog_bar=True, batch_size=batch.num_graphs if hasattr(batch, 'num_graphs') else 1)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)

{main_block}
'''
    return script_content, None
