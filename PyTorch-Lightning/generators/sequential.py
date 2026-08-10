import numpy as np
from .helpers import (
    _py_str,
    _load_template,
    _LIGHTNING_IMPORT_BLOCK,
    _trainer_main_block,
)

def _gen_sequential_script(
    exp_name, ts_path, tgt_col, task_type,
    s_len, p_len, h_size, n_layers,
    ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
    logger_lines, callback_block,
    seq_dataset_type="builtin", seq_builtin_dataset="mackey_glass",
    seq_model_type="lstm",
):
    """Returns (train_script, None) — sequential uses synthetic generator or CSV."""
    is_clf = (task_type == "classification")
    loss_line = (
        "loss = F.cross_entropy(y_hat, y.long().squeeze(-1))"
        if is_clf else
        "loss = F.mse_loss(y_hat, y)"
    )
    metric_lines = (
        '''acc = (y_hat.argmax(dim=-1) == y.long().squeeze(-1)).float().mean()
        self.log(f"{stage}_acc", acc, prog_bar=True)'''
        if is_clf else
        'self.log(f"{stage}_rmse", loss.sqrt(), prog_bar=True)'
    )
    out_size_init = "2" if is_clf else str(p_len)   # default 2 classes for synthetic classification

    # Configure dataset loading / creation in script
    if seq_dataset_type == "custom":
        dataset_init_block = f'''import pandas as pd
        df = pd.read_csv("{_py_str(ts_path)}")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != "{_py_str(tgt_col)}"]

        features = df[feature_cols].values.astype(np.float32)
        target = df["{_py_str(tgt_col)}"].values.astype(np.float32)
        
        # Normalize features
        self.feat_mean = features.mean(axis=0)
        self.feat_std = features.std(axis=0) + 1e-8
        features = (features - self.feat_mean) / self.feat_std

        n = len(target)
        split = int(n * train_frac)
        self.features = features[:split] if train else features[split:]
        self.target = target[:split] if train else target[split:]
        self.num_features = len(feature_cols)'''
    else:
        # Synthetic generator branch
        if seq_builtin_dataset in ("mackey_glass", "sine"):
            dataset_init_block = f'''# Generate Mackey-Glass chaotic time series
        np.random.seed(SEED)
        n_samples = 5000
        tau = 17
        x = np.zeros(n_samples)
        # Initialize history with random noise around 1.2
        x[:tau] = 1.2 + 0.1 * np.random.randn(tau)
        for i in range(tau, n_samples):
            x[i] = 0.9 * x[i-1] + 0.2 * x[i-tau] / (1.0 + x[i-tau]**10)
        signal = x.astype(np.float32)
        
        features = signal[:-{p_len}].reshape(-1, 1).astype(np.float32)
        if TASK_TYPE == "classification":
            # Classify whether next sequence value increases (1) or decreases (0)
            diff = np.diff(signal)
            labels = (diff > 0).astype(np.int64)
            target = labels[seq_len - 1 : len(features) - {p_len}]
            features = features[:len(target) + seq_len]
        else:
            target = signal[{p_len}:].astype(np.float32)
        
        n = len(target)
        split = int(n * train_frac)
        
        # Normalize features
        self.feat_mean = features.mean(axis=0)
        self.feat_std = features.std(axis=0) + 1e-8
        features = (features - self.feat_mean) / self.feat_std
        
        self.features = features[:split] if train else features[split:]
        self.target = target[:split] if train else target[split:]
        self.num_features = 1'''
        elif seq_builtin_dataset in ("fastText", "AISHELL"):
            dataset_init_block = f'''# Load from cluster environment variables
        import os
        import pandas as pd
        
        env_var = "fastText_PATH" if "{seq_builtin_dataset}" == "fastText" else "AISHELL_PATH"
        path = os.environ.get(env_var, "")
        
        # Look for any CSV file inside the directory
        csv_file = None
        if os.path.isdir(path):
            for r_dir, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith(".csv"):
                        csv_file = os.path.join(r_dir, f)
                        break
                if csv_file:
                    break
                    
        if csv_file:
            try:
                df = pd.read_csv(csv_file)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 0:
                    signal = df[numeric_cols[0]].values.astype(np.float32)
                else:
                    signal = np.sin(np.linspace(0, 100, 5000)).astype(np.float32)
            except Exception:
                signal = np.sin(np.linspace(0, 100, 5000)).astype(np.float32)
        else:
            signal = np.sin(np.linspace(0, 100, 5000)).astype(np.float32)
            
        features = signal[:-{p_len}].reshape(-1, 1).astype(np.float32)
        if TASK_TYPE == "classification":
            diff = np.diff(signal)
            labels = (diff > 0).astype(np.int64)
            target = labels[seq_len - 1 : len(features) - {p_len}]
            features = features[:len(target) + seq_len]
        else:
            target = signal[{p_len}:].astype(np.float32)
            
        n = len(target)
        split = int(n * train_frac)
        
        # Normalize features
        self.feat_mean = features.mean(axis=0)
        self.feat_std = features.std(axis=0) + 1e-8
        features = (features - self.feat_mean) / self.feat_std
        
        self.features = features[:split] if train else features[split:]
        self.target = target[:split] if train else target[split:]
        self.num_features = 1'''
        else: # sunspots
            dataset_init_block = f'''# Load Monthly Mean Total Sunspot Number dataset (1749 to July 2018)
        import os
        import urllib.request
        from pathlib import Path
        import pandas as pd
        import ssl
        
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        dest = data_dir / "sunspots.csv"
        
        if not dest.exists():
            print("Downloading Sunspots dataset...")
            url = "https://raw.githubusercontent.com/dicodingacademy/assets/main/Simulation/machine_learning/sunspots.csv"
            context = ssl._create_unverified_context()
            req = urllib.request.Request(url, headers={{"User-Agent": "Mozilla/5.0"}})
            try:
                with urllib.request.urlopen(req, context=context) as response:
                    with open(dest, "wb") as f:
                        f.write(response.read())
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download Sunspots dataset. "
                    f"Prepared datasets are prefetched on the submit node. Error: {{e}}"
                )
                
        df = pd.read_csv(dest)
        # Columns: Unnamed: 0, Date, Monthly Mean Total Sunspot Number
        # Use third column as target (sunspot counts)
        signal = df.iloc[:, 2].values.astype(np.float32)
        
        features = signal[:-{p_len}].reshape(-1, 1).astype(np.float32)
        if TASK_TYPE == "classification":
            # Classify whether next sequence value increases (1) or decreases (0)
            diff = np.diff(signal)
            labels = (diff > 0).astype(np.int64)
            target = labels[seq_len - 1 : len(features) - {p_len}]
            features = features[:len(target) + seq_len]
        else:
            target = signal[{p_len}:].astype(np.float32)
            
        n = len(target)
        split = int(n * train_frac)
        
        # Normalize features
        self.feat_mean = features.mean(axis=0)
        self.feat_std = features.std(axis=0) + 1e-8
        features = (features - self.feat_mean) / self.feat_std
        
        self.features = features[:split] if train else features[split:]
        self.target = target[:split] if train else target[split:]
        self.num_features = 1'''

    # Configure model block based on seq_model_type
    if seq_model_type == "gru":
        model_block = f'''class LitModel(L.LightningModule):
    """Stacked GRU model for time-series forecasting or classification."""
    def __init__(self, input_size, hidden_size={h_size}, num_layers={n_layers},
                 output_size={out_size_init}, lr={lr}, task_type="{task_type}"):
        super().__init__()
        self.save_hyperparameters()
        
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers,
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

    def _shared_step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        # __LOSS_LINE__
        self.log(f"{{stage}}_loss", loss, prog_bar=True)
        # __METRIC_LINES__
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        return {{
            "optimizer": optimizer,
            "lr_scheduler": {{"scheduler": scheduler, "monitor": "val_loss"}},
        }}
'''
    elif seq_model_type == "cnn1d":
        model_block = f'''class LitModel(L.LightningModule):
    """1D CNN model for time-series forecasting or classification."""
    def __init__(self, input_size, hidden_size={h_size}, num_layers={n_layers},
                 output_size={out_size_init}, lr={lr}, task_type="{task_type}"):
        super().__init__()
        self.save_hyperparameters()
        
        layers = []
        in_ch = input_size
        out_ch = hidden_size
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            in_ch = out_ch
            
        self.convs = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # input x shape: (batch_size, seq_len, input_size) -> transpose to (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        out = self.convs(x)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)

    def _shared_step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        # __LOSS_LINE__
        self.log(f"{{stage}}_loss", loss, prog_bar=True)
        # __METRIC_LINES__
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        return {{
            "optimizer": optimizer,
            "lr_scheduler": {{"scheduler": scheduler, "monitor": "val_loss"}},
        }}
'''
    elif seq_model_type == "transformer":
        model_block = f'''class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LitModel(L.LightningModule):
    """Transformer Encoder model for time-series forecasting or classification."""
    def __init__(self, input_size, hidden_size={h_size}, num_layers={n_layers},
                 output_size={out_size_init}, lr={lr}, task_type="{task_type}", num_heads=4):
        super().__init__()
        self.save_hyperparameters()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size * 4, 
            batch_first=True, 
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.input_projection(x)
        out = self.pos_encoder(out)
        out = self.transformer(out)
        return self.fc(out[:, -1, :])

    def _shared_step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        # __LOSS_LINE__
        self.log(f"{{stage}}_loss", loss, prog_bar=True)
        # __METRIC_LINES__
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        return {{
            "optimizer": optimizer,
            "lr_scheduler": {{"scheduler": scheduler, "monitor": "val_loss"}},
        }}
'''
    else: # lstm
        model_block = f'''class LitModel(L.LightningModule):
    """Stacked LSTM model for time-series forecasting or classification."""
    def __init__(self, input_size, hidden_size={h_size}, num_layers={n_layers},
                 output_size={out_size_init}, lr={lr}, task_type="{task_type}"):
        super().__init__()
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers,
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def _shared_step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        # __LOSS_LINE__
        self.log(f"{{stage}}_loss", loss, prog_bar=True)
        # __METRIC_LINES__
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        return {{
            "optimizer": optimizer,
            "lr_scheduler": {{"scheduler": scheduler, "monitor": "val_loss"}},
        }}
'''

    model_block = model_block.replace("# __LOSS_LINE__", loss_line.strip())
    model_block = model_block.replace("# __METRIC_LINES__", metric_lines)

    template = _load_template("sequential")
    train_script = template.replace("# __LIGHTNING_IMPORT_BLOCK__", _LIGHTNING_IMPORT_BLOCK.strip())
    train_script = train_script.replace("__TS_DATA_PATH__", _py_str(ts_path))
    train_script = train_script.replace("__TARGET_COLUMN__", _py_str(tgt_col))
    train_script = train_script.replace("__SEQ_TASK_TYPE__", task_type)
    train_script = train_script.replace("__SEQ_LEN__", str(s_len))
    train_script = train_script.replace("__PRED_LEN__", str(p_len))
    train_script = train_script.replace("__HIDDEN_SIZE__", str(h_size))
    train_script = train_script.replace("__NUM_LSTM_LAYERS__", str(n_layers))
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
    train_script = train_script.replace("# __DATASET_INIT_BLOCK__", dataset_init_block.strip())
    train_script = train_script.replace("# __MODEL_BLOCK__", model_block.strip())
    train_script = train_script.replace("__OUT_SIZE_INIT__", out_size_init)
    train_script = train_script.replace("        # __LOGGER_LINES__", "\n".join(logger_lines))
    train_script = train_script.replace("    # __CALLBACK_BLOCK__", "    " + callback_block.replace("\n", "\n    "))

    prefetch_script = None
    if seq_dataset_type == "builtin" and seq_builtin_dataset in ("sunspots", "weather"):
        prefetch_script = f'''#!/usr/bin/env python3
"""Pre-download Monthly Sunspots dataset on the submit node."""

import urllib.request
import ssl
from pathlib import Path

URL = "https://raw.githubusercontent.com/dicodingacademy/assets/main/Simulation/machine_learning/sunspots.csv"
DEST = Path("data") / "sunspots.csv"

def main():
    print(f"Downloading Sunspots dataset to {{DEST}}...")
    DEST.parent.mkdir(exist_ok=True)
    context = ssl._create_unverified_context()
    req = urllib.request.Request(URL, headers={{"User-Agent": "Mozilla/5.0"}})
    try:
        with urllib.request.urlopen(req, context=context) as response:
            with open(DEST, "wb") as f:
                f.write(response.read())
        print(f"Prefetch complete. File saved to {{DEST}}")
    except Exception as e:
        print(f"Error downloading Sunspots dataset: {{e}}")
        raise e

if __name__ == "__main__":
    main()
'''

    return train_script, prefetch_script
