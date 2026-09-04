import os
from .helpers import (
    _py_str,
    _load_template,
    _LIGHTNING_IMPORT_BLOCK,
    _get_additional_imports,
    _get_data_helpers,
    _trainer_main_block,
)

def _gen_computer_vision_script(
    exp_name, ds_type, builtin, custom_path,
    ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
    logger_lines, callback_block,
    cv_model_arch="cnn",
    cv_dataset_format="tensors",
    cv_image_size="0",
    cv_channels="3",
):
    """Returns (train_script, prefetch_script_or_None)."""
    if cv_model_arch not in ("cnn", "mlp", "vit", "unet"):
        cv_model_arch = "cnn"

    try:
        sz = int(cv_image_size) if cv_image_size is not None else 0
        if sz < 0:
            sz = 0
    except (ValueError, TypeError):
        sz = 0

    try:
        ch = int(cv_channels) if cv_channels else 3
        if ch not in (1, 3):
            ch = 3
    except (ValueError, TypeError):
        ch = 3

    cache_dir = "./data"

    if ds_type == "builtin":
        dm = f'''class LitDataModule(L.LightningDataModule):
    """DataModule managing downloads, setup, and data loading for builtin datasets."""
    def __init__(self, data_dir=DATA_DIR, batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = "{_py_str(builtin)}"

    def setup(self, stage=None):
        # Load train and validation dataset splits
        self.train_ds = load_builtin_dataset(self.dataset_name, self.data_dir, train=True)
        self.val_ds = load_builtin_dataset(self.dataset_name, self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
'''
        if builtin == "ImageNet":
            ch, nc, sz, arch = 3, 1000, 224, "cnn"
        else:
            raise ValueError(f"Prepared dataset '{builtin}' is not available via cluster modules. Direct internet downloads are disabled. Please provide a custom dataset path or select ImageNet.")
    else:
        if cv_model_arch == "unet":
            if cv_dataset_format == "images":
                dm = f'''class LitDataModule(L.LightningDataModule):
    """DataModule managing custom semantic segmentation datasets from raw images."""
    def __init__(self, data_root="{_py_str(custom_path)}", batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = ImageSegmentationDataset(os.path.join(self.data_root, "train"), image_size={sz})
        self.val_ds = ImageSegmentationDataset(os.path.join(self.data_root, "val"), image_size={sz})
        test_dir = os.path.join(self.data_root, "test")
        if os.path.isdir(test_dir):
            self.test_ds = ImageSegmentationDataset(test_dir, image_size={sz})
        else:
            self.test_ds = None

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if getattr(self, "test_ds", None) is not None:
            return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
        return None
'''
            else:
                dm = f'''class LitDataModule(L.LightningDataModule):
    """DataModule managing custom semantic segmentation datasets."""
    def __init__(self, data_root="{_py_str(custom_path)}", batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = TensorSegmentationDataset(os.path.join(self.data_root, "train"), image_size={sz})
        self.val_ds = TensorSegmentationDataset(os.path.join(self.data_root, "val"), image_size={sz})
        test_dir = os.path.join(self.data_root, "test")
        if os.path.isdir(test_dir):
            self.test_ds = TensorSegmentationDataset(test_dir, image_size={sz})
        else:
            self.test_ds = None

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if getattr(self, "test_ds", None) is not None:
            return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
        return None
'''
            ch, nc, sz, arch = ch, 1, sz, "unet"
        else:
            if cv_dataset_format == "images":
                transform_ops = []
                if sz > 0:
                    transform_ops.append(f"transforms.Resize(({sz}, {sz}))")
                if ch == 1:
                    transform_ops.append("transforms.Grayscale(num_output_channels=1)")
                transform_ops.append("transforms.ToTensor()")
                
                transforms_str = "\n            ".join(f"{op}," for op in transform_ops)
                transform_code = f"""transforms.Compose([
            {transforms_str}
        ])"""

                dm = f'''class LitDataModule(L.LightningDataModule):
    """DataModule managing custom folder-based raw image datasets."""
    def __init__(self, data_root="{_py_str(custom_path)}", batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = {transform_code}

    def setup(self, stage=None):
        self.train_ds = ImageFolder(os.path.join(self.data_root, "train"), transform=self.transform)
        self.val_ds = ImageFolder(os.path.join(self.data_root, "val"), transform=self.transform)
        self.num_classes = len(self.train_ds.classes)
        if len(self.train_ds) > 0:
            sample_x, _ = self.train_ds[0]
            self.input_size = sample_x.numel()
            self.image_size = sample_x.shape[-1]
        else:
            self.input_size = 784
            self.image_size = 28
        test_dir = os.path.join(self.data_root, "test")
        if os.path.isdir(test_dir):
            self.test_ds = ImageFolder(test_dir, transform=self.transform)
        else:
            self.test_ds = None

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if getattr(self, "test_ds", None) is not None:
            return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
        return None
'''
            else:
                dm = f'''class LitDataModule(L.LightningDataModule):
    """DataModule managing custom folder-based tensor datasets."""
    def __init__(self, data_root="{_py_str(custom_path)}", batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Loads class-sorted subdirectories of .pt tensors for train and validation
        self.train_ds = TensorFolderDataset(os.path.join(self.data_root, "train"), image_size={sz})
        self.val_ds = TensorFolderDataset(os.path.join(self.data_root, "val"), image_size={sz})
        self.num_classes = self.train_ds.num_classes
        if len(self.train_ds) > 0:
            sample_x, _ = self.train_ds[0]
            self.input_size = sample_x.numel()
            self.image_size = sample_x.shape[-1]
        else:
            self.input_size = 784
            self.image_size = 28
        test_dir = os.path.join(self.data_root, "test")
        if os.path.isdir(test_dir):
            self.test_ds = TensorFolderDataset(test_dir, image_size={sz})
        else:
            self.test_ds = None

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if getattr(self, "test_ds", None) is not None:
            return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
        return None
'''
            ch, nc, sz, arch = ch, "num_classes", sz, cv_model_arch

    if arch == "cnn":
        nc_arg = nc if ds_type == "builtin" else "10"
        model_block = f'''class LitModel(L.LightningModule):
    """Convolutional Neural Network model for multi-class image classification."""
    def __init__(self, lr={lr}, num_classes={nc_arg}, in_channels={ch}):
        super().__init__()
        self.save_hyperparameters()
        
        # Simple convolutional neural network layout
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
'''
        model_init = (
            f"num_classes = datamodule.num_classes\n    model = LitModel(lr=LR, num_classes=num_classes, in_channels={ch})"
            if ds_type == "custom"
            else f"model = LitModel(lr=LR, num_classes={nc}, in_channels={ch})"
        )
    elif arch == "mlp":
        flat = ch * sz * sz if sz > 0 else 784
        nc_arg = nc if ds_type == "builtin" else "10"
        model_block = f'''class LitModel(L.LightningModule):
    """Multi-layer Perceptron (MLP) model for multi-class classification."""
    def __init__(self, lr={lr}, num_classes={nc_arg}, input_size={flat}):
        super().__init__()
        self.save_hyperparameters()
        
        # Fully connected layers with ReLU activations
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
'''
        if ds_type == "custom":
            if sz > 0:
                model_init = f"num_classes = datamodule.num_classes\n    model = LitModel(lr=LR, num_classes=num_classes, input_size={flat})"
            else:
                model_init = f"num_classes = datamodule.num_classes\n    input_size = getattr(datamodule, 'input_size', {flat})\n    model = LitModel(lr=LR, num_classes=num_classes, input_size=input_size)"
        else:
            model_init = f"model = LitModel(lr=LR, num_classes={nc}, input_size={flat})"
    elif arch == "vit":
        nc_arg = nc if ds_type == "builtin" else "10"
        if sz > 0:
            patch_sz = 16 if sz >= 64 else (4 if sz >= 16 else 2)
            pos_embed_init = f"nn.Parameter(torch.zeros(1, 1 + ({sz} // {patch_sz})**2, embed_dim))"
            patch_embed_init = f"self.patch_embed = PatchEmbedding(in_channels, {patch_sz}, embed_dim)"
            vit_init_args = f"lr={lr}, num_classes={nc_arg}, in_channels={ch}, patch_size={patch_sz}, embed_dim=128, num_heads=4, num_layers=4"
            model_init = (
                f"num_classes = datamodule.num_classes\n    model = LitModel(lr=LR, num_classes=num_classes, in_channels={ch})"
                if ds_type == "custom"
                else f"model = LitModel(lr=LR, num_classes={nc}, in_channels={ch})"
            )
        else:
            pos_embed_init = "nn.Parameter(torch.zeros(1, 1 + (image_size // patch_size)**2, embed_dim))"
            patch_embed_init = f"""if patch_size > image_size or image_size % patch_size != 0:
            patch_size = 4 if image_size >= 16 else 2
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)"""
            vit_init_args = f"lr={lr}, num_classes={nc_arg}, in_channels={ch}, image_size=224, patch_size=16, embed_dim=128, num_heads=4, num_layers=4"
            model_init = (
                f"num_classes = datamodule.num_classes\n    image_size = getattr(datamodule, 'image_size', 224)\n    model = LitModel(lr=LR, num_classes=num_classes, in_channels={ch}, image_size=image_size)"
                if ds_type == "custom"
                else f"model = LitModel(lr=LR, num_classes={nc}, in_channels={ch})"
            )

        model_block = f'''class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class LitModel(L.LightningModule):
    """Vision Transformer (ViT) model for multi-class image classification."""
    def __init__(self, {vit_init_args}):
        super().__init__()
        self.save_hyperparameters()
        {patch_embed_init}
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = {pos_embed_init}
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, batch_first=True, dropout=0.1),
            num_layers=num_layers
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
'''
    elif arch == "unet":
        model_block = f'''class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class LitModel(L.LightningModule):
    """U-Net architecture for semantic segmentation."""
    def __init__(self, lr={lr}, in_channels={ch}, out_channels={nc}):
        super().__init__()
        self.save_hyperparameters()
        
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        u1 = self.up1(x3)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.conv_up2(u2)
        
        return self.outc(u2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = (logits > 0).float()
        intersection = (preds * y).sum(dim=(2, 3))
        union = (preds + y).clamp(max=1.0).sum(dim=(2, 3))
        iou = ((intersection + 1e-6) / (union + 1e-6)).mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_iou", iou, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = (logits > 0).float()
        intersection = (preds * y).sum(dim=(2, 3))
        union = (preds + y).clamp(max=1.0).sum(dim=(2, 3))
        iou = ((intersection + 1e-6) / (union + 1e-6)).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = (logits > 0).float()
        intersection = (preds * y).sum(dim=(2, 3))
        union = (preds + y).clamp(max=1.0).sum(dim=(2, 3))
        iou = ((intersection + 1e-6) / (union + 1e-6)).mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_iou", iou, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
'''
        model_init = f"model = LitModel(lr=LR, in_channels={ch}, out_channels=1)"

    main_block = _trainer_main_block(acc, dev, prec, log_n, model_init, callback_block, logger_lines)

    template = _load_template("computer_vision")
    train_script = template.replace("# __LIGHTNING_IMPORT_BLOCK__", _LIGHTNING_IMPORT_BLOCK.strip())
    train_script = train_script.replace("# __ADDITIONAL_IMPORTS__", _get_additional_imports(ds_type, builtin, dataset_format=cv_dataset_format).strip())
    train_script = train_script.replace("__DATA_DIR__", _py_str(cache_dir))
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
    
    # Generate dynamic helpers tailored only to the chosen dataset
    data_helpers = _get_data_helpers(ds_type, builtin, cv_model_arch=cv_model_arch, dataset_format=cv_dataset_format)
    train_script = train_script.replace("# __DATA_HELPERS_BLOCK__", data_helpers.strip())
    
    train_script = train_script.replace("# __DATAMODULE_BLOCK__", dm.strip())
    train_script = train_script.replace("# __MODEL_BLOCK__", model_block.strip())
    train_script = train_script.replace("# __MAIN_BLOCK__", main_block.strip())

    return train_script, None

