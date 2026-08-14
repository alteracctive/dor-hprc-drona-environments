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

def _gen_audio_script(
    exp_name, audio_ds_type, audio_blt, audio_path,
    ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
    logger_lines, callback_block,
    audio_model_type="classification",
    audio_transform_type="mel_spectrogram",
):
    """Returns (train_script, None)."""
    use_builtin = (audio_ds_type == "builtin")
    data_dir = "./data"

    if not use_builtin and not audio_path:
        _drona_msg("Custom Audio dataset path is required.", "error")
        return None, None

    # Dataset Setup
    if use_builtin:
        dataset_setup = f'''# Index audio files from AISHELL dataset
        import os
        
        path = os.environ.get("AISHELL_PATH", DATA_DIR)
        
        audio_files = []
        labels = []
        if os.path.isdir(path):
            for r_dir, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith((".wav", ".mp3", ".flac", ".ogg")):
                        audio_files.append(os.path.join(r_dir, f))
                        labels.append(os.path.basename(r_dir))
                        if len(audio_files) >= 1000:
                            break
                if len(audio_files) >= 1000:
                    break
        
        if not audio_files:
            audio_files = ["/dummy/path/audio1.wav", "/dummy/path/audio2.wav"] * 50
            labels = ["class0", "class1"] * 50

        unique_labels = sorted(list(set(labels)))
        label_map = {{lbl: i for i, lbl in enumerate(unique_labels)}}
        labels = [label_map[lbl] for lbl in labels]
        self.num_classes = len(unique_labels)

        split = int(len(audio_files) * 0.8)
        self.train_ds = AudioDataset(audio_files[:split], labels[:split], transform=self.transform)
        self.val_ds = AudioDataset(audio_files[split:], labels[split:], transform=self.transform)'''
    else:
        dataset_setup = f'''# Index custom user audio files
        import os
        
        path = "{_py_str(audio_path)}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Custom audio path does not exist: {{path}}")
            
        audio_files = []
        labels = []
        if os.path.isdir(path):
            for r_dir, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith((".wav", ".mp3", ".flac", ".ogg")):
                        audio_files.append(os.path.join(r_dir, f))
                        labels.append(os.path.basename(r_dir))
                        if len(audio_files) >= 1000:
                            break
                if len(audio_files) >= 1000:
                    break
        elif os.path.isfile(path):
            audio_files = [path]
            labels = [os.path.basename(os.path.dirname(path))]

        if not audio_files:
            raise RuntimeError(f"No audio files found in custom directory: {{path}}")

        unique_labels = sorted(list(set(labels)))
        label_map = {{lbl: i for i, lbl in enumerate(unique_labels)}}
        labels = [label_map[lbl] for lbl in labels]
        self.num_classes = len(unique_labels)

        split = int(len(audio_files) * 0.8) if len(audio_files) > 1 else 1
        self.train_ds = AudioDataset(audio_files[:split], labels[:split], transform=self.transform)
        self.val_ds = AudioDataset(audio_files[split:], labels[split:], transform=self.transform) if len(audio_files) > 1 else self.train_ds'''

    # Model Blocks
    if audio_model_type == "synthesis":
        model_block = f'''class LitModel(L.LightningModule):
    """Autoencoder model for audio spectrogram and waveform synthesis."""
    def __init__(self, in_channels, feature_shape, num_classes=2, lr={lr}):
        super().__init__()
        self.save_hyperparameters()
        
        self.is_1d = len(feature_shape) == 2
        if self.is_1d:
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            )

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        if out.shape != x.shape:
            out = F.interpolate(out, size=x.shape[2:])
        return out

    def _shared_step(self, batch, stage):
        x, _ = batch
        reconstructed = self(x)
        loss = F.mse_loss(reconstructed, x)
        self.log(f"{{stage}}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)'''
    else: # classification or speech
        model_block = f'''class LitModel(L.LightningModule):
    """Convolutional model for acoustic or speech classification."""
    def __init__(self, in_channels, feature_shape, num_classes=2, lr={lr}):
        super().__init__()
        self.save_hyperparameters()
        
        self.is_1d = len(feature_shape) == 2
        if self.is_1d:
            self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2, 2)
        else:
            self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
        
        with torch.no_grad():
            dummy_x = torch.zeros(1, *feature_shape)
            dummy_out = self.pool(F.relu(self.conv2(self.pool(F.relu(self.conv1(dummy_x))))))
            flat_dim = dummy_out.numel()
            
        self.fc = nn.Linear(flat_dim, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log(f"{{stage}}_loss", loss, prog_bar=True)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log(f"{{stage}}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)'''

    template = _load_template("audio")
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
    train_script = train_script.replace("__MODEL_TYPE__", _py_str(audio_model_type))
    train_script = train_script.replace("__TRANSFORM_TYPE__", _py_str(audio_transform_type))
    train_script = train_script.replace("        # __DATASET_SETUP__", dataset_setup)
    train_script = train_script.replace("# __MODEL_BLOCK__", model_block.strip())
    train_script = train_script.replace("        # __LOGGER_LINES__", "\n".join(logger_lines))
    train_script = train_script.replace("    # __CALLBACK_BLOCK__", "    " + callback_block.replace("\n", "\n    "))

    return train_script, None
