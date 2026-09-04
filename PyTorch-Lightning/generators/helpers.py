import os
import re
import json
from pathlib import Path

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

_DOWNLOAD_ONLY_BLOCK = ""


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

def _get_additional_imports(ds_type, dataset_name, dataset_format="tensors"):
    """Returns dynamic imports tailored to the specific dataset to avoid redundancies."""
    imports = []
    if ds_type == "custom":
        imports.append("import numpy as np")
        imports.append("import torch.nn.functional as F")
        imports.append("from torch.utils.data import Dataset")
        imports.append("from pathlib import Path")
        if dataset_format == "images":
            imports.append("import torchvision.transforms as transforms")
            imports.append("from torchvision.datasets import ImageFolder")
            imports.append("from PIL import Image")
            imports.append("import torchvision.transforms.functional as TF")
    else:  # builtin
        if dataset_name in ("ImageNet", "COCO", "CC3M", "CAMUS", "llava-onevision"):
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

def _get_data_helpers(ds_type, dataset_name, cv_model_arch="cnn", dataset_format="tensors"):
    """
    Returns only the necessary dataset loading helper functions and classes
    for the generated script, eliminating unused dataset handlers and boilerplate.
    """
    if ds_type == "custom":
        if dataset_format == "images":
            if cv_model_arch == "unet":
                return '''class ImageSegmentationDataset(Dataset):
    """
    Custom Dataset loader for semantic segmentation of standard image files.
    Reads image and mask files (.png, .jpg, .jpeg, .bmp)
    organized under root/images/ and root/masks/ directories.
    """
    def __init__(self, root, image_size=224):
        self.root = Path(root)
        self.image_size = image_size
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        if not self.img_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Segmentation folders 'images' and 'masks' must exist under {self.root}")
        
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        self.img_files = sorted(
            [p for p in self.img_dir.iterdir() if p.suffix.lower() in valid_exts]
        )
        
    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = None
        for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
            candidate = self.mask_dir / f"{img_path.stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
                
        if mask_path is None:
            raise FileNotFoundError(f"Corresponding mask not found for image: {img_path.name}")
            
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.image_size > 0:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        x = TF.to_tensor(img)
        y = TF.to_tensor(mask)
        
        return x, y
'''
            else:
                return ""

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
        
        # Interpolate if size mismatch and target size is specified (> 0)
        if self.image_size > 0:
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
            
        # Interpolate image if target size is specified (> 0) and doesn't match
        if self.image_size > 0 and (x.shape[-1] != self.image_size or x.shape[-2] != self.image_size):
            x = x.unsqueeze(0) if x.dim() == 3 else x
            x = F.interpolate(x.float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            x = x.squeeze(0)
        return x.float(), label

    @property
    def num_classes(self):
        return len(self.class_to_idx)
'''

    if dataset_name in ("MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"):
        return f'''def load_builtin_dataset(name, data_dir, train):
    raise RuntimeError(
        f"Direct internet download of '{{name}}' is disabled on cluster compute nodes. "
        "Please provide your dataset path using 'Import' (custom dataset) or select an available cluster module dataset (e.g. ImageNet)."
    )
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
                return torch.rand(3, 224, 224), 0
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
                    return torch.rand(3, 224, 224), 0
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

    # Run the model testing phase
    if getattr(datamodule, "test_ds", None) is not None:
        trainer.test(model, datamodule=datamodule)
{checkpoint_post_save}
if __name__ == "__main__":
    main()
'''
