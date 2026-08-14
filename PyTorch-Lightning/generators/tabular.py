import os
import builtins
from .helpers import (
    _py_str,
    _load_template,
    _LIGHTNING_IMPORT_BLOCK,
)

def _drona_msg(msg, level="warning"):
    if hasattr(builtins, "drona_add_message"):
        builtins.drona_add_message(msg, level)
    else:
        print(f"[{level.upper()}] {msg}")

def _gen_tabular_script(
    exp_name, tab_ds_type, tab_blt, tab_path, tab_tgt, tab_task,
    ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
    logger_lines, callback_block,
    tab_model_type="mlp",
):
    """Returns (train_script, None)."""
    use_builtin = (tab_ds_type == "builtin")
    data_dir = "./data"

    if not use_builtin and not tab_path:
        _drona_msg("Custom Tabular dataset path is required.", "error")
        return None, None

    # Dataset Setup
    if use_builtin:
        dataset_setup = f'''# Generate synthetic tabular dataset
        import numpy as np
        np.random.seed(SEED)
        
        # 1000 samples, 15 features
        n_samples = 1000
        n_features = 15
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        
        is_classification = "{tab_task}" == "classification"
        if is_classification:
            # Generate binary classification labels (0 or 1) based on random hyperplanes
            y = (X[:, 0] + X[:, 1] - X[:, 2] > 0.0).astype(np.int64)
            self.num_classes = 2
        else:
            # Continuous regression target
            y = (X[:, 0] * 2.0 + X[:, 1] - 0.5 * X[:, 2] + np.random.randn(n_samples)).astype(np.float32)
            self.num_classes = 1

        self.in_features = n_features
        
        # Save generated dataset to CSV for user inspection
        try:
            import csv
            csv_path = "synthetic_tabular_dataset.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = [f"feature_{{i}}" for i in range(n_features)] + ["target"]
                writer.writerow(header)
                for idx in range(n_samples):
                    writer.writerow(list(X[idx]) + [float(y[idx])])
            print(f"Saved synthetic tabular dataset for inspection to: {{os.path.abspath(csv_path)}}")
        except Exception as e:
            print(f"Failed to save synthetic dataset CSV: {{e}}")
        
        # Split train/val
        split = int(n_samples * 0.8)
        self.train_ds = TabularDataset(X[:split], y[:split])
        self.val_ds = TabularDataset(X[split:], y[split:])'''
    else:
        dataset_setup = f'''# Load custom CSV tabular dataset
        import numpy as np
        import pandas as pd
        
        csv_file = "{_py_str(tab_path)}"
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Custom CSV file not found: {{csv_file}}")
            
        df = pd.read_csv(csv_file)
        
        # Isolate target and features
        target_col = "{_py_str(tab_tgt)}"
        if target_col not in df.columns:
            raise KeyError(f"Target column '{{target_col}}' not found in CSV. Options: {{list(df.columns)}}")
            
        y_raw = df[target_col].values
        X_raw = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values.astype(np.float32)
        
        is_classification = "{tab_task}" == "classification"
        if is_classification:
            unique_labels = sorted(list(set(y_raw)))
            label_map = {{lbl: i for i, lbl in enumerate(unique_labels)}}
            y = np.array([label_map[lbl] for lbl in y_raw], dtype=np.int64)
            self.num_classes = len(unique_labels)
        else:
            y = y_raw.astype(np.float32).reshape(-1, 1)
            self.num_classes = 1

        # Normalize features
        mean = X_raw.mean(axis=0)
        std = X_raw.std(axis=0) + 1e-8
        X = (X_raw - mean) / std

        self.in_features = X.shape[1]
        
        # Split train/val
        split = int(len(X) * 0.8)
        self.train_ds = TabularDataset(X[:split], y[:split])
        self.val_ds = TabularDataset(X[split:], y[split:])'''

    # Model Blocks
    loss_fn = "F.cross_entropy" if tab_task == "classification" else "F.mse_loss"
    y_type = "long()" if tab_task == "classification" else "float()"
    y_target_reshape = ".squeeze(-1)" if tab_task == "classification" else ""
    metric_calculation = (
        '''acc = (logits.argmax(dim=-1) == y.long().squeeze(-1)).float().mean()
        self.log(f"{stage}_acc", acc, prog_bar=True)'''
        if tab_task == "classification" else
        'self.log(f"{stage}_rmse", loss.sqrt(), prog_bar=True)'
    )

    if tab_model_type == "resnet":
        model_block = f'''class TabularResNetBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.norm1(x))
        x = self.linear1(x)
        x = F.relu(self.norm2(x))
        x = self.linear2(x)
        x = self.dropout(x)
        return residual + x

class LitModel(L.LightningModule):
    """ResNet model customized for tabular classification and regression."""
    def __init__(self, in_features, out_features=1, hidden_dim=128, num_blocks=2, lr={lr}):
        super().__init__()
        self.save_hyperparameters()
        self.in_proj = nn.Linear(in_features, hidden_dim)
        self.blocks = nn.ModuleList([TabularResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = {loss_fn}(logits, y.{y_type}{y_target_reshape})
        self.log(f"{{stage}}_loss", loss, prog_bar=True)
        {metric_calculation}
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)'''
    elif tab_model_type == "tabnet":
        model_block = f'''class AttentiveTransformer(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features)
        self.bn = nn.BatchNorm1d(in_features)

    def forward(self, x):
        coef = self.linear(x)
        coef = self.bn(coef)
        mask = F.softmax(coef, dim=-1)
        return mask

class LitModel(L.LightningModule):
    """TabNet primitive with attentive masking for feature selection."""
    def __init__(self, in_features, out_features=1, lr={lr}):
        super().__init__()
        self.save_hyperparameters()
        self.attentive_transformer = AttentiveTransformer(in_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        mask = self.attentive_transformer(x)
        masked_x = x * mask
        return self.mlp(masked_x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = {loss_fn}(logits, y.{y_type}{y_target_reshape})
        self.log(f"{{stage}}_loss", loss, prog_bar=True)
        {metric_calculation}
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)'''
    else: # mlp
        model_block = f'''class LitModel(L.LightningModule):
    """Deep Multi-Layer Perceptron (DeepMLP) for tabular data learning."""
    def __init__(self, in_features, out_features=1, lr={lr}):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_features)
        )

    def forward(self, x):
        return self.net(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = {loss_fn}(logits, y.{y_type}{y_target_reshape})
        self.log(f"{{stage}}_loss", loss, prog_bar=True)
        {metric_calculation}
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)'''

    template = _load_template("tabular")
    train_script = template.replace("# __LIGHTNING_IMPORT_BLOCK__", _LIGHTNING_IMPORT_BLOCK.strip())
    train_script = train_script.replace("__DATA_DIR__", _py_str(data_dir))
    train_script = train_script.replace("__LOG_DIR__", _py_str(log_dir))
    train_script = train_script.replace("__EXPERIMENT_NAME__", _py_str(exp_name))
    train_script = train_script.replace("__LR__", str(lr))
    train_script = train_script.replace("__BATCH_SIZE__", str(bs))
    train_script = train_script.replace("__NUM_WORKERS__", str(nw))
    train_script = train_script.replace("__MAX_EPOCHS__", str(ep))
    train_script = train_script.replace("__ACCELERATOR__", _py_str(acc))
    train_script = train_script.replace("__DEVICES__", str(dev))
    train_script = train_script.replace("__PRECISION__", str(prec) if str(prec).isdigit() else f'"{_py_str(prec)}"')
    train_script = train_script.replace("__LOG_EVERY_N_STEPS__", str(log_n))
    train_script = train_script.replace("__SEED__", str(seed_val))
    train_script = train_script.replace("__MODEL_TYPE__", _py_str(tab_model_type))
    train_script = train_script.replace("__TASK_TYPE__", _py_str(tab_task))
    train_script = train_script.replace("        # __DATASET_SETUP__", dataset_setup)
    train_script = train_script.replace("# __MODEL_BLOCK__", model_block.strip())
    train_script = train_script.replace("        # __LOGGER_LINES__", "\n".join(logger_lines))
    train_script = train_script.replace("    # __CALLBACK_BLOCK__", "    " + callback_block.replace("\n", "\n    "))

    return train_script, None
