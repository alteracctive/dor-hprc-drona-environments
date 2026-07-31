#!/usr/bin/env python3
"""Generated PyTorch Lightning training script — Computer Vision."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import pytorch_lightning as L
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
except ImportError:
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint


DATA_DIR = "./data"
LOG_DIR = "./lightning_logs"
EXPERIMENT_NAME = "pt-mnist-prefetch"
LR = 1e-3
BATCH_SIZE = 32
NUM_WORKERS = 0
MAX_EPOCHS = 1
ACCELERATOR = "cpu"
DEVICES = 1
PRECISION = 32
LOG_EVERY_N_STEPS = 10
SEED = 42


import gzip
import os
import struct
import urllib.request
from pathlib import Path


_PROXY_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")


class _NoProxy:
    def __enter__(self):
        self._saved = {k: os.environ.pop(k) for k in _PROXY_KEYS if k in os.environ}
        return self

    def __exit__(self, exc_type, exc, tb):
        os.environ.update(self._saved)


def _retrieve(url, dest):
    import ssl
    context = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=context, timeout=20) as response:
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)


def _download(url, dest, mirrors=()):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    errors = []
    print(f"Download started for {dest.name}...")
    for candidate in (url,) + tuple(mirrors):
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        for use_proxy in (True, False):
            try:
                if use_proxy:
                    _retrieve(candidate, tmp)
                else:
                    with _NoProxy():
                        _retrieve(candidate, tmp)
                tmp.rename(dest)
                print(f"Download finished successfully for {dest.name}.")
                return
            except Exception as err:
                last_err = err
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
        errors.append(f"{candidate}: {last_err}")
    print(f"Download failed for {dest.name}.")
    raise RuntimeError(
        "Failed to download "
        f"{dest.name}. Prepared datasets are prefetched on the submit node before "
        f"the Slurm job starts; compute nodes have no internet access. "
        f"Tried: {'; '.join(errors)}"
    )


def _idx_dataset_spec(name):
    if name == "MNIST":
        return (
            "MNIST",
            "https://storage.googleapis.com/cvdf-datasets/mnist",
            ("https://yann.lecun.com/exdb/mnist",),
        )
    return (
        "FashionMNIST",
        "https://storage.googleapis.com/cvdf-datasets/fashion-mnist",
        (
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com",
            "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion",
        ),
    )


def _ensure_idx_dataset_files(root, name):
    subdir, primary, fallbacks = _idx_dataset_spec(name)
    files = (
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    )
    root = Path(root) / subdir
    for fname in files:
        _download(
            f"{primary}/{fname}",
            root / fname,
            mirrors=tuple(f"{base}/{fname}" for base in fallbacks),
        )


def _ensure_cifar_dataset_files(root, name):
    import tarfile
    root = Path(root)
    if name == "CIFAR10":
        archive = "cifar-10-python.tar.gz"
        folder = "cifar-10-batches-py"
        primary = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/cifar-10-python.tar.gz"
        mirrors = (
            "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        )
    else:
        archive = "cifar-100-python.tar.gz"
        folder = "cifar-100-python"
        primary = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar-100-python.tar.gz"
        mirrors = (
            "https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz",
            "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        )
    dest_archive = root / archive
    dest_folder = root / folder
    if dest_folder.exists():
        print(f"{name} folder already exists under {root}. Skipping download/extraction.")
        return
    print(f"Downloading {name} dataset from {primary}...")
    _download(primary, dest_archive, mirrors=mirrors)
    print(f"Extracting {name} dataset archive {archive}...")
    with tarfile.open(dest_archive, "r:gz") as tar:
        tar.extractall(path=root)
    print(f"Successfully extracted {name} dataset to {dest_folder}.")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        _magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols).copy()


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        _magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).copy()


class ArrayImageDataset(Dataset):
    def __init__(self, images, labels, channels=1):
        self.images = images
        self.labels = labels
        self.channels = channels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.channels == 1:
            x = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        else:
            if img.ndim == 3 and img.shape[-1] == 3:
                x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            else:
                x = torch.from_numpy(img.reshape(3, 32, 32)).float() / 255.0
        return x, int(self.labels[idx])


def _load_idx_dataset(root, name, train):
    _ensure_idx_dataset_files(root, name)
    subdir, _primary, _fallbacks = _idx_dataset_spec(name)
    files = {
        "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
        "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
    }
    split = "train" if train else "test"
    img_name, lbl_name = files[split]
    root = Path(root) / subdir
    images = _read_idx_images(root / img_name)
    labels = _read_idx_labels(root / lbl_name)
    return ArrayImageDataset(images, labels, channels=1)


def _load_cifar_dataset(root, name, train):
    import pickle
    print(f"Initializing {name} dataset loading...")
    _ensure_cifar_dataset_files(root, name)
    root = Path(root)
    if name == "CIFAR10":
        folder = "cifar-10-batches-py"
        files = (
            ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
            if train
            else ["test_batch"]
        )
    else:
        folder = "cifar-100-python"
        files = ["train" if train else "test"]
    
    images_list = []
    labels_list = []
    print(f"Loading {name} dataset files for split: {'train' if train else 'test'}...")
    for fname in files:
        fpath = root / folder / fname
        print(f"Reading data file {fname}...")
        with open(fpath, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            images_list.append(entry["data"])
            if name == "CIFAR10":
                labels_list.extend(entry["labels"])
            else:
                labels_list.extend(entry["fine_labels"])
    
    print(f"Processing and converting dataset tensors...")
    images = np.vstack(images_list).reshape(-1, 3, 32, 32)
    images = images.transpose((0, 2, 3, 1))
    labels = np.array(labels_list, dtype=np.int64)
    print(f"Successfully loaded {name} dataset with {len(images)} samples.")
    return ArrayImageDataset(images, labels, channels=3)


def load_builtin_dataset(name, data_dir, train):
    if name in ("MNIST", "FashionMNIST"):
        return _load_idx_dataset(data_dir, name, train)
    elif name in ("CIFAR10", "CIFAR100"):
        return _load_cifar_dataset(data_dir, name, train)
    elif name == "ImageNet":
        import importlib
        ImageFolder = importlib.import_module("torch" + "vision.datasets").ImageFolder
        T = importlib.import_module("torch" + "vision.transforms")
        path = "/scratch/data/pytorch-computer-vision-datasets/imagenet-raw-dataset"
        split = "train" if train else "val"
        if train:
            transform = T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return ImageFolder(os.path.join(path, split), transform=transform)
    raise ValueError(f"Unsupported prepared dataset: {name}")


class TensorFolderDataset(Dataset):
    """Load per-sample .pt tensors from class-named subfolders."""

    def __init__(self, root, image_size=224):
        self.root = Path(root)
        self.image_size = image_size
        self.samples = []
        self.class_to_idx = {}
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")
        classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not classes:
            raise ValueError(f"No class subfolders found under {self.root}")
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        for class_name in classes:
            class_idx = self.class_to_idx[class_name]
            for path in sorted((self.root / class_name).glob("*.pt")):
                self.samples.append((path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        item = torch.load(path, map_location="cpu")
        if isinstance(item, dict):
            x = item.get("x", item.get("image"))
            label = int(item.get("y", label))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            x, label = item[0], int(item[1])
        else:
            x = item
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = x.unsqueeze(0) if x.dim() == 3 else x
            x = F.interpolate(x.float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            x = x.squeeze(0)
        return x.float(), label

    @property
    def num_classes(self):
        return len(self.class_to_idx)


class LitDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=32, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = "MNIST"

    def setup(self, stage=None):
        self.train_ds = load_builtin_dataset(self.dataset_name, self.data_dir, train=True)
        self.val_ds = load_builtin_dataset(self.dataset_name, self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)


class LitModel(L.LightningModule):
    def __init__(self, lr=1e-3, num_classes=10, input_size=784):
        super().__init__()
        self.save_hyperparameters()
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def build_loggers():
    loggers = [
    TensorBoardLogger(save_dir="./lightning_logs", name="pt-mnist-prefetch"),
    ]
    return [lg for lg in loggers if lg is not False]

def main():
    L.seed_everything(SEED, workers=True)
    accelerator = ACCELERATOR
    devices = DEVICES
    if accelerator == "gpu" and not torch.cuda.is_available():
        slurm_gpus = os.environ.get("SLURM_JOB_GPUS", "unset")
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
        raise RuntimeError(
            "GPU training was requested but CUDA is not available on this node. "
            "Check that the SLURM script includes --partition=gpu and --gres=gpu:... "
            f"(SLURM_JOB_GPUS={slurm_gpus}, CUDA_VISIBLE_DEVICES={cuda_visible})"
        )
    datamodule = LitDataModule()
    datamodule.setup()
    model = LitModel(lr=LR, num_classes=10)
    loggers = build_loggers()
    callbacks = None
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator,
        devices=devices,
        precision=PRECISION,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        logger=loggers if loggers else None,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)

    # Save a .pt file side-by-side with the best checkpoint .ckpt file
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        best_ckpt = trainer.checkpoint_callback.best_model_path
        best_pt = os.path.splitext(best_ckpt)[0] + ".pt"
        try:
            ckpt = torch.load(best_ckpt, map_location="cpu")
            if "state_dict" in ckpt:
                torch.save(ckpt["state_dict"], best_pt)
            else:
                torch.save(ckpt, best_pt)
            print(f"Saved PyTorch weights side-by-side at: {best_pt}")
        except Exception as e:
            print(f"Could not save side-by-side .pt file: {e}")

    # Save a .pt file side-by-side with the last/final checkpoint .ckpt file
    if trainer.checkpoint_callback and hasattr(trainer.checkpoint_callback, 'last_model_path') and trainer.checkpoint_callback.last_model_path:
        last_ckpt = trainer.checkpoint_callback.last_model_path
        if last_ckpt != trainer.checkpoint_callback.best_model_path:
            last_pt = os.path.splitext(last_ckpt)[0] + ".pt"
            try:
                ckpt = torch.load(last_ckpt, map_location="cpu")
                if "state_dict" in ckpt:
                    torch.save(ckpt["state_dict"], last_pt)
                else:
                    torch.save(ckpt, last_pt)
                print(f"Saved PyTorch weights side-by-side at: {last_pt}")
            except Exception as e:
                print(f"Could not save side-by-side .pt file: {e}")

if __name__ == "__main__":
    main()


