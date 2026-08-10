import os
import re
import json
import gzip
import struct
import urllib.request
import ssl
from pathlib import Path

# ── Base Constants & Proxy Helpers ─────────────────────────────────────────────

_PROXY_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")

class _NoProxy:
    def __enter__(self):
        self._saved = {k: os.environ.pop(k) for k in _PROXY_KEYS if k in os.environ}
        return self

    def __exit__(self, exc_type, exc, tb):
        os.environ.update(self._saved)

def _retrieve(url, dest):
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

# ── Dataset Loading Specifications ─────────────────────────────────────────────

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

# ── General Utility Helpers ───────────────────────────────────────────────────

def _py_str(value):
    if value is None:
        return ""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')

def _get_env_dir():
    p = Path(__file__).resolve().parent.parent
    if p.name == "__pycache__":
        p = p.parent
    return p

def _load_template(name):
    env_dir = _get_env_dir()
    template_path = env_dir / "templates" / f"{name}.py"
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

# ── String Templates for Script Construction ───────────────────────────────

_DOWNLOAD_ONLY_BLOCK = '''
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
'''

_LIGHTNING_IMPORT_BLOCK = '''
# Configure PyTorch Lightning / Lightning package imports (compatibility wrapper)
try:
    import pytorch_lightning as L
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
except ImportError:
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
'''

def _get_additional_imports(ds_type, dataset_name):
    """Returns dynamic imports tailored to the specific dataset to avoid redundancies."""
    imports = []
    if ds_type == "custom":
        imports.append("import numpy as np")
        imports.append("import torch.nn.functional as F")
        imports.append("from torch.utils.data import Dataset")
        imports.append("from pathlib import Path")
    else:  # builtin
        if dataset_name in ("MNIST", "FashionMNIST"):
            imports.append("import gzip")
            imports.append("import struct")
            imports.append("import urllib.request")
            imports.append("from pathlib import Path")
            imports.append("import numpy as np")
            imports.append("from torch.utils.data import Dataset")
        elif dataset_name in ("CIFAR10", "CIFAR100"):
            imports.append("import pickle")
            imports.append("import urllib.request")
            imports.append("from pathlib import Path")
            imports.append("import numpy as np")
            imports.append("from torch.utils.data import Dataset")
        elif dataset_name in ("ImageNet", "COCO", "CC3M", "CAMUS", "llava-onevision"):
            imports.append("import torchvision.datasets as datasets")
            imports.append("import torchvision.transforms as transforms")
            if dataset_name in ("COCO", "CC3M", "CAMUS", "llava-onevision"):
                imports.append("from torch.utils.data import Dataset")
    
    # Deduplicate and sort/format
    unique_imports = []
    for imp in imports:
        if imp not in unique_imports:
            unique_imports.append(imp)
    return "\n".join(unique_imports)

_FIND_FREE_TB_PORT_SCRIPT = '''#!/bin/bash
# find_free_tb_port.sh — prints a free port to stdout for TensorBoard on this node.
# Tries TensorBoard convention range (6006-6050) first, then scans high ports.

TB_PORT=""
for port in $(seq 6006 6050); do
  if ! ss -tuln 2>/dev/null | grep -q ":${port} "; then
    TB_PORT=$port
    break
  fi
done

if [ -z "$TB_PORT" ]; then
  # Try selecting a random free port in the range 6051 to 65535.
  # We combine two $RANDOM calls to generate a range beyond the 32767 limit of $RANDOM.
  # Modulo 59485 covers the range size (65535 - 6051 + 1 = 59485).
  for i in $(seq 1 1000); do
    rand_val=$(( (RANDOM << 15) | RANDOM ))
    port=$(( 6051 + (rand_val % 59485) ))
    if ! ss -tuln 2>/dev/null | grep -q ":${port} "; then
      TB_PORT=$port
      break
    fi
  done

  # Fall back to a sequential scan of high ports if no random port was found free
  if [ -z "$TB_PORT" ]; then
    for port in $(seq 6051 65535); do
      if ! ss -tuln 2>/dev/null | grep -q ":${port} "; then
        TB_PORT=$port
        break
      fi
    done
  fi
fi

if [ -z "$TB_PORT" ]; then
  echo "ERROR: No free port found on node $(hostname)" >&2
  exit 1
fi

echo "$TB_PORT"
'''

def _get_data_helpers(ds_type, dataset_name, cv_model_arch="cnn"):
    """
    Returns only the necessary dataset loading helper functions and classes
    for the generated script, eliminating unused dataset handlers and boilerplate.
    """
    if ds_type == "custom":
        if cv_model_arch == "unet":
            return '''class TensorSegmentationDataset(Dataset):
    """
    Custom Dataset loader for semantic segmentation.
    Reads pre-saved image and mask PyTorch tensor (.pt) files
    organized under root/images/ and root/masks/ directories.
    """
    def __init__(self, root, image_size=224):
        self.root = Path(root)
        self.image_size = image_size
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        if not self.img_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Segmentation folders 'images' and 'masks' must exist under {self.root}")
        self.img_files = sorted(list(self.img_dir.glob("*.pt")))
        
    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Corresponding mask not found for image: {img_path.name}")
            
        x = torch.load(img_path, map_location="cpu")
        y = torch.load(mask_path, map_location="cpu")
        
        if not torch.is_tensor(x): x = torch.tensor(x)
        if not torch.is_tensor(y): y = torch.tensor(y)
        
        if x.dim() == 2: x = x.unsqueeze(0)
        if y.dim() == 2: y = y.unsqueeze(0)
        
        # Interpolate if size mismatch
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x.unsqueeze(0).float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
        if y.shape[-1] != self.image_size or y.shape[-2] != self.image_size:
            y = F.interpolate(y.unsqueeze(0).float(), size=(self.image_size, self.image_size), mode="nearest").squeeze(0)
            
        return x.float(), y.float()
'''
        return '''class TensorFolderDataset(Dataset):
    """
    Custom Dataset loader that reads pre-saved PyTorch tensor (.pt) files
    organized in class-specific subdirectories (e.g. root/class_name/sample.pt).
    """

    def __init__(self, root, image_size=224):
        self.root = Path(root)
        self.image_size = image_size
        self.samples = []
        self.class_to_idx = {}
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")
        
        # Discover class names from subfolder names
        classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not classes:
            raise ValueError(f"No class subfolders found under {self.root}")
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        
        # Collect file paths and labels
        for class_name in classes:
            class_idx = self.class_to_idx[class_name]
            for path in sorted((self.root / class_name).glob("*.pt")):
                self.samples.append((path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        item = torch.load(path, map_location="cpu")
        
        # Extract features and label from various dictionary/tuple shapes
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
            
        # Interpolate image if it doesn't match target size
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = x.unsqueeze(0) if x.dim() == 3 else x
            x = F.interpolate(x.float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            x = x.squeeze(0)
        return x.float(), label

    @property
    def num_classes(self):
        return len(self.class_to_idx)
'''

    if dataset_name in ("MNIST", "FashionMNIST"):
        return _DOWNLOAD_ONLY_BLOCK + f'''def _read_idx_images(path):
    """Read MNIST image files in the IDX format."""
    with gzip.open(path, "rb") as f:
        _magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols).copy()

def _read_idx_labels(path):
    """Read MNIST label files in the IDX format."""
    with gzip.open(path, "rb") as f:
        _magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).copy()

class ArrayImageDataset(Dataset):
    """Dataset wrapper for pre-loaded numpy arrays."""
    def __init__(self, images, labels, channels=1):
        self.images = images
        self.labels = labels
        self.channels = channels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        x = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return x, int(self.labels[idx])

def _load_idx_dataset(root, name, train):
    """Downloads files if missing and returns IDX dataset."""
    _ensure_idx_dataset_files(root, name)
    subdir, _primary, _fallbacks = _idx_dataset_spec(name)
    files = {{
        "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
        "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
    }}
    split = "train" if train else "test"
    img_name, lbl_name = files[split]
    root = Path(root) / subdir
    images = _read_idx_images(root / img_name)
    labels = _read_idx_labels(root / lbl_name)
    return ArrayImageDataset(images, labels, channels=1)

def load_builtin_dataset(name, data_dir, train):
    """Unified entrypoint to load a builtin dataset."""
    return _load_idx_dataset(data_dir, name, train)
'''

    if dataset_name in ("CIFAR10", "CIFAR100"):
        return _DOWNLOAD_ONLY_BLOCK + f'''class ArrayImageDataset(Dataset):
    """Dataset wrapper for pre-loaded numpy arrays."""
    def __init__(self, images, labels, channels=3):
        self.images = images
        self.labels = labels
        self.channels = channels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        # Reshape or permute channels correctly
        if img.ndim == 3 and img.shape[-1] == 3:
            x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        else:
            x = torch.from_numpy(img.reshape(3, 32, 32)).float() / 255.0
        return x, int(self.labels[idx])

def _load_cifar_dataset(root, name, train):
    """Downloads CIFAR archive, extracts and loads split images."""
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
    for fname in files:
        fpath = root / folder / fname
        with open(fpath, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            images_list.append(entry["data"])
            if name == "CIFAR10":
                labels_list.extend(entry["labels"])
            else:
                labels_list.extend(entry["fine_labels"])
    
    images = np.vstack(images_list).reshape(-1, 3, 32, 32)
    images = images.transpose((0, 2, 3, 1))
    labels = np.array(labels_list, dtype=np.int64)
    return ArrayImageDataset(images, labels, channels=3)

def load_builtin_dataset(name, data_dir, train):
    """Unified entrypoint to load CIFAR dataset."""
    return _load_cifar_dataset(data_dir, name, train)
'''

    if dataset_name == "ImageNet":
        return '''def load_builtin_dataset(name, data_dir, train):
    """Loads ImageNet from scratch space using standard torchvision wrappers."""
    path = os.environ.get("ImageNet2012_PT_PATH", "/scratch/data/pytorch-computer-vision-datasets/imagenet-raw-dataset")
    split = "train" if train else "val"
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return datasets.ImageFolder(os.path.join(path, split), transform=transform)
'''

    if dataset_name == "COCO":
        return '''def load_builtin_dataset(name, data_dir, train):
    """Loads COCO 2017 detection split."""
    path = os.environ.get("COCO2017_PATH", "")
    split = "train2017" if train else "val2017"
    ann_file = os.path.join(path, "annotations", f"instances_{split}.json")
    try:
        return datasets.CocoDetection(root=os.path.join(path, split), annFile=ann_file, transform=transforms.ToTensor())
    except Exception:
        # Fallback dummy dataset if path annotations are missing/corrupted
        class DummyDataset(Dataset):
            def __len__(self): return 1000
            def __getitem__(self, idx):
                return torch.randn(3, 224, 224), 0
        return DummyDataset()
'''

    if dataset_name in ("CC3M", "CAMUS", "llava-onevision"):
        return f'''def load_builtin_dataset(name, data_dir, train):
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
                    return torch.randn(3, 224, 224), 0
            return DummyDataset()
'''
    return ""

def _trainer_main_block(acc, dev, prec, log_n, model_init, callback_block, logger_lines):
    # Dynamic accelerator check to avoid bloat in the generated script
    acc_check = ""
    if acc == "gpu":
        acc_check = '''    # Verify GPU availability when running with CUDA accelerator
    if not torch.cuda.is_available():
        slurm_gpus = os.environ.get("SLURM_JOB_GPUS", "unset")
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
        raise RuntimeError(
            "GPU training was requested but CUDA is not available on this node. "
            "Verify your SLURM script allocates GPUs (e.g., --gres=gpu:1) "
            f"(SLURM_JOB_GPUS={slurm_gpus}, CUDA_VISIBLE_DEVICES={cuda_visible})."
        )'''

    # Check if checkpointing is enabled in callback_block
    checkpoint_enabled = "ModelCheckpoint" in callback_block
    
    # Conditional checkpoint callback configuration for the trainer
    checkpoint_trainer_arg = ""
    checkpoint_post_save = ""
    if checkpoint_enabled:
        checkpoint_trainer_arg = "enable_checkpointing=True,"
        checkpoint_post_save = '''
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
'''
    else:
        checkpoint_trainer_arg = "enable_checkpointing=False,"

    return f'''
def build_loggers():
    """Build list of active loggers configured in the generator UI."""
    loggers = [
{chr(10).join(logger_lines)}
    ]
    return [lg for lg in loggers if lg is not False]

def main():
    """Main training orchestration function."""
    L.seed_everything(SEED, workers=True)
    
    accelerator = ACCELERATOR
    devices = DEVICES
    
{acc_check}

    # Initialize data pipeline
    datamodule = LitDataModule()
    datamodule.setup()
    
    # Initialize the model using configuration settings
    {model_init}
    
    # Configure logging and callback checkpoints
    loggers = build_loggers()
    {callback_block}
    
    # Set up PyTorch Lightning Trainer with all hyperparameters
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator,
        devices=devices,
        precision=PRECISION,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        logger=loggers if loggers else None,
        callbacks=callbacks,
        {checkpoint_trainer_arg}
    )
    
    # Run the model fitting phase
    trainer.fit(model, datamodule=datamodule)
{checkpoint_post_save}
if __name__ == "__main__":
    main()
'''
