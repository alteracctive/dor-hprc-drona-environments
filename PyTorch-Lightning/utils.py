import importlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


# ── Utility helpers ────────────────────────────────────────────────────────────

def _normalize_select_value(value):
    """Extract .value from dynamicSelect JSON; pass through plain strings."""
    if value is None:
        return ""
    if isinstance(value, str) and value.startswith("$"):
        return ""
    if isinstance(value, dict):
        return str(value.get("value", "")).strip()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return str(parsed.get("value", value)).strip()
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return str(value).strip()


def _wants_gpu(gpu, slurm_box):
    if not _checkbox_on(slurm_box):
        return False
    gpu = _normalize_select_value(gpu)
    return gpu not in ("", "none")


def _is_drona_var(val):
    if not val:
        return False
    return str(val).strip().startswith("$")


def _resolve_val(val, default):
    if not val or _is_drona_var(val):
        return default
    return str(val).strip()


def _resolve_int(val, default):
    if not val or _is_drona_var(val):
        return default
    try:
        return int(str(val).strip())
    except ValueError:
        return default


def _py_str(value):
    if value is None:
        return ""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def sanitize_job_name(name):
    if not name:
        return name
    return str(name).strip().replace(" ", "_")


def _checkbox_on(value):
    return value == "Yes" or value is True or value == "true"


_LR_FORMAT = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _parse_learning_rate(value):
    if value is None:
        return "1e-3", None
    text = str(value).strip()
    if not text or text.startswith("$"):
        return "1e-3", None
    if not _LR_FORMAT.match(text):
        return None, (
            "Learning rate must be a number in decimal or scientific notation "
            "(e.g. 0.001 or 1e-3)."
        )
    num = float(text)
    if not math.isfinite(num) or num <= 0:
        return None, "Learning rate must be a positive number."
    return text, None


def _parse_seed(value):
    if value is None:
        return 42, None
    text = str(value).strip()
    if not text or text.startswith("$"):
        return 42, None
    try:
        seed = int(text)
    except (TypeError, ValueError):
        return None, "Random seed must be an integer."
    if seed < 0:
        return None, "Random seed must be zero or greater."
    return seed, None


def _get_env_dir():
    p = Path(__file__).resolve().parent
    if p.name == "__pycache__":
        p = p.parent
    return p


# ── Cluster / module setup ─────────────────────────────────────────────────────

def retrieve_cluster_info():
    cluster_module = None
    module_directory = _get_env_dir()
    if module_directory not in sys.path:
        sys.path.insert(0, f"{module_directory}")

    cluster = subprocess.check_output(["/sw/local/bin/clustername"], text=True).strip()
    if importlib.util.find_spec(f"clusters.{cluster}") is not None:
        cluster_module = importlib.import_module(f"clusters.{cluster}")
    else:
        cluster_module = importlib.import_module("clusters.defaultcluster")
    return cluster, cluster_module


DEFAULT_PT_MODULES = (
    "module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch-Lightning/2.2.1-CUDA-12.1.1\n"
    "module load CUDA/12.1.1 2>/dev/null || true"
)

# GNN requires a separate stack: PyG 2.1.0 was built against PyTorch 1.12 + CUDA 11.7
DEFAULT_GNN_MODULES = (
    "module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch-Lightning/1.8.4-CUDA-11.7.0\n"
    "module load PyTorch-Geometric/2.1.0-PyTorch-1.12.0-CUDA-11.7.0"
)


def _get_torchvision_module(base_modules_str):
    if "CUDA-12.6.0" in base_modules_str:
        return "torchvision/0.16.0-CUDA-12.1.1"
    elif "CUDA-12.1.1" in base_modules_str:
        return "torchvision/0.16.0-CUDA-12.1.1"
    elif "CUDA-11.7.0" in base_modules_str:
        return "torchvision/0.13.1-CUDA-11.7.0"
    elif "CUDA-11.3.1" in base_modules_str:
        return "torchvision/0.11.1-CUDA-11.3.1"
    else:
        return "torchvision/0.16.0-CUDA-12.1.1"


CV_DATASET_MODULES = {
    "ImageNet": "Datasets/ImageNet-PyTorch/2012",
    "COCO": "Datasets/COCO/2017",
    "CC3M": "Datasets/CC3M/1.1",
    "CAMUS": "Datasets/CAMUS/2019",
}

SEQ_DATASET_MODULES = {
    "fastText": "Datasets/fastText/2015",
    "AISHELL": "Datasets/AISHELL/2017",
}

GNN_DATASET_MODULES = {
    "JODIE": "Datasets/JODIE/2012",
    "QM9": "Datasets/QM9/2012",
}

GEN_DATASET_MODULES = {
    "llava-onevision": "Datasets/llava-onevision/2024",
}


def setup_pytorch_modules(
    model_category="computer_vision",
    dataset_type="builtin",
    builtin_dataset="ImageNet",
    seq_dataset="mackey_glass",
    graph_dataset="JODIE",
    gen_dataset="llava-onevision",
    gpu="",
):
    """Return module load commands for the given model category and dataset."""
    cluster, cluster_module = retrieve_cluster_info()
    module_use_cmd = ""
    
    ds_t = (dataset_type or "builtin").strip()
    dataset_cmd = ""
    
    if ds_t == "builtin":
        if model_category == "computer_vision":
            if builtin_dataset in CV_DATASET_MODULES:
                dataset_cmd = f"\nmodule load {CV_DATASET_MODULES[builtin_dataset]} 2>/dev/null || true"
        elif model_category == "sequential":
            if seq_dataset in SEQ_DATASET_MODULES:
                dataset_cmd = f"\nmodule load {SEQ_DATASET_MODULES[seq_dataset]} 2>/dev/null || true"
        elif model_category == "gnn":
            if graph_dataset in GNN_DATASET_MODULES:
                dataset_cmd = f"\nmodule load {GNN_DATASET_MODULES[graph_dataset]} 2>/dev/null || true"
        elif model_category == "generative":
            if gen_dataset in GEN_DATASET_MODULES:
                dataset_cmd = f"\nmodule load {GEN_DATASET_MODULES[gen_dataset]} 2>/dev/null || true"

    gpu_val = (gpu or "").strip().lower()
    if cluster == "aces" and gpu_val == "pvc":
        raise ValueError("Intel PVC GPUs are not supported in the PyTorch-Lightning environment.")

    if model_category == "gnn":
        gnn_mods = getattr(cluster_module, "gnn_modules", DEFAULT_GNN_MODULES)
        return (
            "# Load cluster modules for Graph Neural Network training\n"
            f"{module_use_cmd}{gnn_mods}{dataset_cmd}"
        )

    base = getattr(cluster_module, "pytorch_lightning_modules", DEFAULT_PT_MODULES)

    torchvision_cmd = ""
    if model_category == "computer_vision" and ds_t == "builtin" and builtin_dataset == "ImageNet":
        if "CUDA-12.6.0" in base:
            # Overwrite base modules to GCC/12.3.0 OpenMPI/4.1.5 to make torchvision/0.16.0-CUDA-12.1.1 available
            # and avoid the Lmod toolchain swap that deactivates the CUDA-12.6.0 packages.
            base = "module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch-Lightning/2.2.1-CUDA-12.1.1\nmodule load CUDA/12.1.1 2>/dev/null || true"
        tv_mod = _get_torchvision_module(base)
        torchvision_cmd = (
            f"\n# Load torchvision module\n"
            f"module load {tv_mod} 2>/dev/null || true\n"
        )
        
    return (
        "# Load cluster PyTorch Lightning stack and datasets\n"
        f"{module_use_cmd}{base}{torchvision_cmd}{dataset_cmd}"
    )


def setup_pytorch_modules_if_run(
    mode,
    model_category="computer_vision",
    dataset_type="builtin",
    builtin_dataset="ImageNet",
    seq_dataset="mackey_glass",
    graph_dataset="JODIE",
    gen_dataset="llava-onevision",
    gpu="",
):
    if mode == "monitor":
        return "# monitor mode — no training job"
    return setup_pytorch_modules(
        model_category, dataset_type, builtin_dataset, seq_dataset, graph_dataset, gen_dataset, gpu
    )


def setup_python_env(penv, pythonVersionDropdown, createEnvName, currentEnvDropdown, sharedEnvDropdown):
    if penv == "module":
        return "# Load latest Python module\nmodule load GCCcore/13.3.0 Python/3.12.3 "
    elif penv == "private":
        return "# Setup private virtual environment\n" + currentEnvDropdown
    elif penv == "create":
        if createEnvName == "":
            return ""
        return (
            "# Create new virtual env\n"
            + pythonVersionDropdown
            + "\ncreate_venv "
            + createEnvName
            + f"\nsource activate_venv {createEnvName}"
        )
    elif penv == "shared":
        return "# Setup shared virtual environment\n" + sharedEnvDropdown
    return ""


def retrieve_driver_contents(mode):
    base = _get_env_dir()
    if mode == "monitor":
        file_path = base / "drivers" / "driver-monitor.sh"
    else:
        file_path = base / "drivers" / "driver-run.sh"
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# ── Drona internal helpers ─────────────────────────────────────────────────────

def _get_db_retriever_path():
    runtime_dir = ""
    try:
        from views.utils import get_runtime_dir
        runtime_dir = get_runtime_dir()
    except Exception:
        runtime_dir = os.environ.get("DRONA_RUNTIME_DIR", "")
    if not runtime_dir:
        return ""
    return os.path.join(runtime_dir, "db_access", "drona_db_retriever.py")


def _normalize_job_dir(job_dir):
    match = re.search(r">(/[^<]+)<", job_dir or "")
    return (match.group(1) if match else (job_dir or "").strip())


def _normalize_location(location):
    loc = _normalize_job_dir(location)
    if not loc or loc.startswith("$"):
        return ""
    return loc


def _write_staged_file(env_dir, job_location, filename, content, show_in_preview=True, preview_order=None):
    """Write a generated file directly into the job folder."""
    if env_dir:
        try:
            env_dest = Path(env_dir) / filename
            env_dest.parent.mkdir(parents=True, exist_ok=True)
            with open(env_dest, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
        except OSError as exc:
            drona_add_message(
                f"Could not write {filename} to environment directory ({env_dir}): {exc}",
                "warning",
            )
    if show_in_preview:
        if preview_order is not None:
            drona_add_additional_file(filename, filename, preview_order)
        else:
            drona_add_additional_file(filename, filename)
    loc = _normalize_location(job_location)
    if loc:
        dest = Path(loc) / filename
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
        except OSError as exc:
            drona_add_message(
                f"Could not write {filename} to job directory ({loc}): {exc}",
                "warning",
            )


def _lookup_workflow_location(workflow_id):
    if not workflow_id or str(workflow_id).startswith("$"):
        return ""
    db = _get_db_retriever_path()
    if not db or not os.path.isfile(db):
        return ""
    try:
        out = subprocess.check_output(
            [sys.executable, db, "-i", str(workflow_id)],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        record = json.loads(out.strip()) if out.strip() else {}
        return (record.get("location") or "").strip()
    except Exception:
        return ""


def resolve_monitor_dir(mode, pt_workflow, job_dir):
    if mode != "monitor":
        return ""
    path = _normalize_job_dir(job_dir)
    if path:
        return path
    return _lookup_workflow_location(pt_workflow)


def configure_monitor_mode(mode, pt_workflow, job_dir):
    if mode != "monitor":
        return ""
    drona_add_mapping("JOBNAME", "monitor-session")
    drona_add_mapping("TIME", "00:01")
    drona_add_mapping("MEM", "1G")
    drona_add_mapping("TASKS", "1")
    drona_add_mapping("NODES", "1")
    drona_add_mapping("CPUS", "1")
    drona_add_mapping("PARTITION", "")
    drona_add_mapping("EXTRA", "")
    monitor_dir = resolve_monitor_dir(mode, pt_workflow, job_dir)
    if monitor_dir:
        drona_add_mapping("MONITOR_DIR", monitor_dir)
    elif not pt_workflow or str(pt_workflow).startswith("$"):
        drona_add_message(
            "Select a training run — Slurm status and logs will appear on the form and in the Preview Monitor tab.",
            "note",
        )
    else:
        drona_add_message(
            f"Could not resolve workflow directory for {pt_workflow}.",
            "warning",
        )
    return ""


def build_monitor_preview_html(mode, pt_workflow, job_dir):
    if mode != "monitor":
        return ""
    if not pt_workflow or str(pt_workflow).startswith("$"):
        return ""
    location = resolve_monitor_dir(mode, pt_workflow, job_dir)
    if not location:
        return ""
    env_dir = Path(__file__).resolve().parent
    run_env = os.environ.copy()
    run_env["LOCATION"] = location
    sections = []
    script_name = "retrieve_pt_monitor_dashboard.sh"
    script_path = env_dir / script_name
    if script_path.is_file():
        try:
            out = subprocess.check_output(
                ["bash", str(script_path)],
                env=run_env,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            sections.append(out.strip())
        except subprocess.CalledProcessError:
            sections.append(f"<p><em>Could not load {script_name}</em></p>")
    body = "\n<hr/>\n".join(sections) if sections else "<p><em>No monitor data available.</em></p>"
    html = (
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">"
        "<title>PyTorch-Lightning Monitor</title></head><body>\n"
        f"<h3>Training run monitor</h3>\n"
        f"<p>Workflow: <code>{location}</code></p>\n"
        f"{body}\n"
        "</body></html>\n"
    )
    preview_path = env_dir / "monitor_dashboard.html"
    preview_path.write_text(html, encoding="utf-8")
    drona_add_additional_file("monitor_dashboard.html", "Monitor", 0)
    return ""


def retrieve_monitor_action(mode, pt_workflow, jobs, job_dir):
    if mode != "monitor":
        return ""
    db = _get_db_retriever_path()
    cleanup = (
        f"python3 {db} --delete -i $DRONA_WF_ID 2>/dev/null; "
        "echo 'Monitor session record cleaned up.'"
        if db
        else "echo 'Monitor session finished.'"
    )
    staging_cleanup = 'rm -rf "$STAGING_DIR" 2>/dev/null || true'
    monitor_dir = resolve_monitor_dir(mode, pt_workflow, job_dir)
    if not pt_workflow or str(pt_workflow).startswith("$") or not monitor_dir:
        return (
            "STAGING_DIR=\"$(pwd)\"\n"
            "echo 'Monitor mode: select a training run to view the dashboard on the form.'\n"
            f"{cleanup}\n"
            f"{staging_cleanup}"
        )
    safe_dir = monitor_dir.replace('"', '\\"')
    return f"""STAGING_DIR="$(pwd)"
echo "=== PyTorch-Lightning Monitor (no new job submitted) ==="
echo "Viewing: {safe_dir}"
echo "Slurm status and logs are shown on the form dashboard and Preview Monitor tab."
{cleanup}
{staging_cleanup}
exit 0"""


def retrieve_tasks_and_other_resources(mode, nodes, tasks, cpus, mem, gpu, numgpu, walltime, account, extra, slurmBox):
    if mode == "monitor":
        return ""
    gpu = _normalize_select_value(gpu)
    account = _normalize_select_value(account)
    if slurmBox != "Yes":
        tasks = "1"
        nodes = "1"
        cpus = "1"
        mem = ""
        gpu = ""
        walltime = ""
        account = ""
        extra = ""
    numgpunum = 1
    if gpu != "" and gpu != "none":
        numgpunum = 1 if numgpu == "" else int(numgpu)
    tasknum = int(tasks)
    nodenum = 0 if nodes == "" else int(nodes)
    cpunum = 1 if cpus == "" else int(cpus)
    totalmemnum = 0 if mem == "" else int(mem[:-1])
    timestring = "02:00" if walltime == "" else walltime
    cluster, cluster_module = retrieve_cluster_info()
    cluster_module.cluster_slurm_checks(
        nodenum, tasknum, cpunum, totalmemnum, gpu, numgpunum,
        timestring, account, extra, drona_add_mapping, drona_add_message
    )
    return ""


# ── Shared template blocks embedded in generated scripts ──────────────────────

_DOWNLOAD_ONLY_BLOCK = '''
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
'''


_DATA_HELPERS_BLOCK = _DOWNLOAD_ONLY_BLOCK + '''
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
        path = os.environ.get("ImageNet2012_PT_PATH", "/scratch/data/pytorch-computer-vision-datasets/imagenet-raw-dataset")
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
    elif name == "COCO":
        import importlib
        CocoDetection = importlib.import_module("torch" + "vision.datasets").CocoDetection
        T = importlib.import_module("torch" + "vision.transforms")
        path = os.environ.get("COCO2017_PATH", "")
        split = "train2017" if train else "val2017"
        ann_file = os.path.join(path, "annotations", f"instances_{split}.json")
        try:
            return CocoDetection(root=os.path.join(path, split), annFile=ann_file, transform=T.ToTensor())
        except Exception:
            class DummyDataset(Dataset):
                def __len__(self): return 1000
                def __getitem__(self, idx):
                    return torch.randn(3, 224, 224), 0
            return DummyDataset()
    elif name in ("CC3M", "CAMUS", "llava-onevision"):
        import importlib
        ImageFolder = importlib.import_module("torch" + "vision.datasets").ImageFolder
        T = importlib.import_module("torch" + "vision.transforms")
        env_var = "CC3M_PATH" if name == "CC3M" else ("CAMUS_PATH" if name == "CAMUS" else "LLaVA_OneVision_PATH")
        path = os.environ.get(env_var, "")
        split = "train" if train else "val"
        transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
        try:
            return ImageFolder(os.path.join(path, split), transform=transform)
        except Exception:
            try:
                return ImageFolder(path, transform=transform)
            except Exception:
                class DummyDataset(Dataset):
                    def __len__(self): return 1000
                    def __getitem__(self, idx):
                        return torch.randn(3, 224, 224), 0
                return DummyDataset()
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
'''


_LIGHTNING_IMPORT_BLOCK = '''
try:
    import pytorch_lightning as L
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
except ImportError:
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
'''


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


# ── Helper: shared trainer main() block ───────────────────────────────────────

def _trainer_main_block(acc, dev, prec, log_n, model_init, callback_block, logger_lines):
    return f'''
def build_loggers():
    loggers = [
{chr(10).join(logger_lines)}
    ]
    return [lg for lg in loggers if lg is not False]

def main():
    L.seed_everything(SEED, workers=True)
    accelerator = ACCELERATOR
    devices = DEVICES
    if accelerator == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
        except ImportError:
            pass
    if accelerator == "gpu" and not torch.cuda.is_available():
        slurm_gpus = os.environ.get("SLURM_JOB_GPUS", "unset")
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
        raise RuntimeError(
            "GPU training was requested but CUDA is not available on this node. "
            "Check that the SLURM script includes --partition=gpu and --gres=gpu:... "
            f"(SLURM_JOB_GPUS={{slurm_gpus}}, CUDA_VISIBLE_DEVICES={{cuda_visible}})"
        )
    elif accelerator == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        slurm_gpus = os.environ.get("SLURM_JOB_GPUS", "unset")
        raise RuntimeError(
            "XPU training was requested but Intel XPU (GPU) is not available on this node. "
            f"(SLURM_JOB_GPUS={{slurm_gpus}})"
        )
    datamodule = LitDataModule()
    datamodule.setup()
    {model_init}
    loggers = build_loggers()
    {callback_block}
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
            print(f"Saved PyTorch weights side-by-side at: {{best_pt}}")
        except Exception as e:
            print(f"Could not save side-by-side .pt file: {{e}}")

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
                print(f"Saved PyTorch weights side-by-side at: {{last_pt}}")
            except Exception as e:
                print(f"Could not save side-by-side .pt file: {{e}}")

if __name__ == "__main__":
    main()

'''


# ── Category generators ────────────────────────────────────────────────────────

def _gen_computer_vision_script(
    exp_name, ds_type, builtin, custom_path,
    ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
    logger_lines, callback_block,
):
    """Returns (train_script, prefetch_script_or_None)."""
    cache_dir = "./data"

    if ds_type == "builtin":
        dm = f'''class LitDataModule(L.LightningDataModule):
    def __init__(self, data_dir="{_py_str(cache_dir)}", batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = "{_py_str(builtin)}"

    def setup(self, stage=None):
        self.train_ds = load_builtin_dataset(self.dataset_name, self.data_dir, train=True)
        self.val_ds = load_builtin_dataset(self.dataset_name, self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
'''
        if builtin in ("MNIST", "FashionMNIST"):
            ch, nc, sz, arch = 1, 10, 28, "mlp"
        elif builtin in ("CIFAR10", "CIFAR100"):
            nc = 100 if builtin == "CIFAR100" else 10
            ch, nc, sz, arch = 3, nc, 32, "cnn"
        elif builtin == "ImageNet":
            ch, nc, sz, arch = 3, 1000, 224, "cnn"
        else:
            raise ValueError(f"Unsupported prepared dataset: {builtin}")
    else:
        dm = f'''class LitDataModule(L.LightningDataModule):
    def __init__(self, data_root="{_py_str(custom_path)}", batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = TensorFolderDataset(os.path.join(self.data_root, "train"), image_size=224)
        self.val_ds = TensorFolderDataset(os.path.join(self.data_root, "val"), image_size=224)
        self.num_classes = self.train_ds.num_classes

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
'''
        ch, nc, sz, arch = 3, "num_classes", 224, "cnn"

    if arch == "cnn":
        nc_arg = nc if ds_type == "builtin" else "num_classes"
        model_block = f'''class LitModel(L.LightningModule):
    def __init__(self, lr={lr}, num_classes={nc_arg}, in_channels={ch}):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * ({sz} // 4) * ({sz} // 4), 256), nn.ReLU(),
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
'''
        model_init = (
            "num_classes = datamodule.num_classes\n    model = LitModel(lr=LR, num_classes=num_classes)"
            if ds_type == "custom"
            else f"model = LitModel(lr=LR, num_classes={nc})"
        )
    else:
        flat = sz * sz
        model_block = f'''class LitModel(L.LightningModule):
    def __init__(self, lr={lr}, num_classes={nc}, input_size={flat}):
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
'''
        model_init = f"model = LitModel(lr=LR, num_classes={nc})"

    main_block = _trainer_main_block(acc, dev, prec, log_n, model_init, callback_block, logger_lines)

    train_script = f'''#!/usr/bin/env python3
"""Generated PyTorch Lightning training script — Computer Vision."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
{_LIGHTNING_IMPORT_BLOCK}

DATA_DIR = "{_py_str(cache_dir)}"
LOG_DIR = "{_py_str(log_dir)}"
EXPERIMENT_NAME = "{_py_str(exp_name)}"
LR = {lr}
BATCH_SIZE = {bs}
NUM_WORKERS = {nw}
MAX_EPOCHS = {ep}
ACCELERATOR = "{_py_str(acc)}"
DEVICES = {dev}
PRECISION = {prec if str(prec).isdigit() else f'"{_py_str(prec)}"'}
LOG_EVERY_N_STEPS = {log_n}
SEED = {seed_val}

{_DATA_HELPERS_BLOCK}

{dm}

{model_block}
{main_block}
'''

    prefetch_script = None
    if ds_type == "builtin" and builtin in ("MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"):
        prefetch_script = f'''#!/usr/bin/env python3
"""Pre-download prepared image datasets on the submit node (compute nodes have no internet)."""

DATASET = "{_py_str(builtin)}"
DATA_DIR = "{_py_str(cache_dir)}"

{_DOWNLOAD_ONLY_BLOCK}

def main():
    if DATASET in ("MNIST", "FashionMNIST"):
        _ensure_idx_dataset_files(DATA_DIR, DATASET)
    elif DATASET in ("CIFAR10", "CIFAR100"):
        _ensure_cifar_dataset_files(DATA_DIR, DATASET)
    elif DATASET == "ImageNet":
        print("ImageNet dataset is centrally available on HPRC; skipping download.")
    else:
        raise ValueError(f"Unsupported dataset for prefetch: {{DATASET}}")
    print(f"Prefetched {{DATASET}} under {{DATA_DIR}}")

if __name__ == "__main__":
    main()
'''

    return train_script, prefetch_script


def _gen_llm_transformer_script(
    exp_name, hf_model_name, num_labels, max_seq_len,
    text_dataset_type, hf_dataset_name, custom_text_path, text_col, label_col,
    ep, bs, lr, nw, seed_val, acc, dev, log_n, log_dir,
    logger_lines, callback_block,
):
    """Returns (train_script, prefetch_script)."""
    cache_dir = "./hf_cache"

    # Custom CSV/JSON branch
    if text_dataset_type == "custom_text":
        if not custom_text_path:
            drona_add_message("Custom text dataset path is required.", "error")
            return None, None
        dm = f'''class LitDataModule(L.LightningDataModule):
    def __init__(self, batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        import pandas as pd
        path = "{_py_str(custom_text_path)}"
        df = pd.read_csv(path) if path.endswith(".csv") else pd.read_json(path, lines=True)
        texts = df["{_py_str(text_col)}"].tolist()
        labels = df["{_py_str(label_col)}"].tolist()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        class _DS(Dataset):
            def __init__(self, txt, lab, tok, max_len):
                self.txt = txt
                self.lab = lab
                self.tok = tok
                self.max_len = max_len
            def __len__(self):
                return len(self.lab)
            def __getitem__(self, idx):
                t = str(self.txt[idx])
                enc = self.tok(
                    t, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
                )
                item = {{k: v.squeeze(0) for k, v in enc.items()}}
                item["labels"] = torch.tensor(self.lab[idx], dtype=torch.long)
                return item

        n = len(texts)
        split = int(n * 0.8)
        self.train_ds = _DS(texts[:split], labels[:split], tokenizer, MAX_SEQ_LEN)
        self.val_ds = _DS(texts[split:], labels[split:], tokenizer, MAX_SEQ_LEN)

    @property
    def num_classes(self):
        if hasattr(self, "train_ds") and hasattr(self.train_ds, "lab"):
            try:
                return len(set(self.train_ds.lab))
            except Exception:
                pass
        return None

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
'''
        prefetch_script = None  # no prefetch needed for custom CSV
    else:
        dm = f'''class LitDataModule(L.LightningDataModule):
    def __init__(self, batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        from datasets import load_dataset
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        from transformers import DataCollatorWithPadding
        self.collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

        try:
            ds = load_dataset(HF_DATASET_NAME)
        except ValueError as e:
            if "Available configs in the cache" in str(e):
                parts = str(e).split("Available configs in the cache: ")
                if len(parts) > 1:
                    config_str = parts[1].replace("[", "").replace("]", "").replace("'", "").replace('"', "")
                    config_name = config_str.split(",")[0].strip()
                    if config_name:
                        ds = load_dataset(HF_DATASET_NAME, config_name)
                    else:
                        raise e
                else:
                    raise e
            else:
                raise e
        train_split = "train"
        val_split = "test" if "test" in ds else "validation"

        def _tokenize(batch):
            return tokenizer(
                batch["text"], truncation=True, max_length=MAX_SEQ_LEN
            )

        self.train_ds = ds[train_split].map(_tokenize, batched=True)
        self.val_ds = ds[val_split].map(_tokenize, batched=True)

        # Rename "label" -> "labels" if needed (HF models expect "labels")
        if "label" in self.train_ds.column_names and "labels" not in self.train_ds.column_names:
            self.train_ds = self.train_ds.rename_column("label", "labels")
        if "label" in self.val_ds.column_names and "labels" not in self.val_ds.column_names:
            self.val_ds = self.val_ds.rename_column("label", "labels")

        keep = ["input_ids", "attention_mask", "labels"]
        self.train_ds.set_format("torch", columns=[c for c in keep if c in self.train_ds.column_names])
        self.val_ds.set_format("torch", columns=[c for c in keep if c in self.val_ds.column_names])

    @property
    def num_classes(self):
        if hasattr(self, "train_ds"):
            if hasattr(self.train_ds, "features") and "labels" in self.train_ds.features:
                feat = self.train_ds.features["labels"]
                if hasattr(feat, "num_classes"):
                    return feat.num_classes
            try:
                labels = self.train_ds["labels"]
                if hasattr(labels, "unique"):
                    return int(labels.unique().numel())
                else:
                    return len(set(labels))
            except Exception:
                pass
        return None

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
'''
        prefetch_script = f'''#!/usr/bin/env python3
"""
Pre-download HuggingFace model weights, tokenizer, and dataset on the submit node.
Compute nodes have no internet access — all assets must be cached here first.
"""
import os
import urllib.request

os.environ["HF_HOME"] = "{_py_str(cache_dir)}"

def check_connectivity(url="https://huggingface.co", timeout=3):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False

if not check_connectivity():
    print("Hugging Face Hub is not reachable. Enabling offline mode.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    print("Hugging Face Hub is reachable. Online mode enabled.")

MODEL_NAME = "{_py_str(hf_model_name)}"
NUM_LABELS = {num_labels}
HF_DATASET_NAME = "{_py_str(hf_dataset_name)}"

print(f"Caching tokenizer: {{MODEL_NAME}}")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Caching model weights: {{MODEL_NAME}} ({{NUM_LABELS}} labels)")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
print("Model and tokenizer cached successfully.")

if HF_DATASET_NAME:
    print(f"Caching dataset: {{HF_DATASET_NAME}}")
    from datasets import load_dataset
    dataset = load_dataset(HF_DATASET_NAME)
    print(f"Dataset cached: {{dataset}}")

print("Prefetch complete. All assets are ready for offline compute nodes.")
'''


    model_block = f'''class LitTransformer(L.LightningModule):
    """Fine-tune a HuggingFace sequence classification model with PyTorch Lightning."""

    def __init__(self, model_name=MODEL_NAME, num_labels=NUM_LABELS, lr=LR):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def _shared_step(self, batch, stage):
        labels = batch.get("labels", batch.get("label"))
        outputs = self(batch["input_ids"], batch["attention_mask"], labels=labels)
        loss = outputs.loss
        acc = (outputs.logits.argmax(dim=-1) == labels).float().mean()
        self.log(f"{{stage}}_loss", loss, prog_bar=True)
        self.log(f"{{stage}}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        from torch.optim import AdamW
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=MAX_EPOCHS
        )
        return [optimizer], [scheduler]
'''

    model_init = '''num_labels = getattr(datamodule, "num_classes", None) or NUM_LABELS
    model = LitTransformer(model_name=MODEL_NAME, num_labels=num_labels, lr=LR)'''
    main_block = _trainer_main_block(acc, dev, "bf16-mixed", log_n, model_init, callback_block, logger_lines)

    train_script = f'''#!/usr/bin/env python3
"""Generated PyTorch Lightning training script — LLM / Transformer Fine-tuning."""

import os
# Force offline mode on compute nodes (all assets cached by prefetch_data.py)
os.environ.setdefault("HF_HOME", "{_py_str(cache_dir)}")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
{_LIGHTNING_IMPORT_BLOCK}
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "{_py_str(hf_model_name)}"
HF_DATASET_NAME = "{_py_str(hf_dataset_name)}"
NUM_LABELS = {num_labels}
MAX_SEQ_LEN = {max_seq_len}

LOG_DIR = "{_py_str(log_dir)}"
EXPERIMENT_NAME = "{_py_str(exp_name)}"
LR = {lr}
BATCH_SIZE = {bs}
NUM_WORKERS = {nw}
MAX_EPOCHS = {ep}
ACCELERATOR = "{_py_str(acc)}"
DEVICES = {dev}
PRECISION = "bf16-mixed"   # recommended for Transformer fine-tuning
LOG_EVERY_N_STEPS = {log_n}
SEED = {seed_val}


{dm}

{model_block}
{main_block}
'''

    return train_script, prefetch_script


def _gen_sequential_script(
    exp_name, ts_data_path, target_col, seq_task_type,
    seq_len, pred_len, hidden_size, num_lstm_layers,
    ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
    logger_lines, callback_block,
    seq_dataset_type="builtin", seq_builtin_dataset="mackey_glass",
):
    """Returns (train_script, None) — sequential uses synthetic generator or CSV."""
    is_clf = (seq_task_type == "classification")
    loss_line = (
        "loss = F.cross_entropy(y_hat, y.long().squeeze(-1))"
        if is_clf else
        "loss = F.mse_loss(y_hat, y)"
    )
    metric_lines = (
        '''        acc = (y_hat.argmax(dim=-1) == y.long().squeeze(-1)).float().mean()
        self.log(f"{stage}_acc", acc, prog_bar=True)'''
        if is_clf else
        '        self.log(f"{stage}_rmse", loss.sqrt(), prog_bar=True)'
    )
    out_size = f"num_classes" if is_clf else f"{pred_len}"
    out_size_arg = "num_classes=10" if is_clf else f"output_size={pred_len}"
    out_size_init = "2" if is_clf else str(pred_len)   # default 2 classes for synthetic classification

    # Configure dataset loading / creation in script
    if seq_dataset_type == "custom":
        if not ts_data_path:
            drona_add_message("A CSV dataset path is required for custom Sequential / Time-Series training.", "error")
            return None, None
        dataset_init_block = f'''
        import pandas as pd
        df = pd.read_csv("{_py_str(ts_data_path)}")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != "{_py_str(target_col)}"]

        features = df[feature_cols].values.astype(np.float32)
        target = df["{_py_str(target_col)}"].values.astype(np.float32)
        
        # Normalize features
        self.feat_mean = features.mean(axis=0)
        self.feat_std = features.std(axis=0) + 1e-8
        features = (features - self.feat_mean) / self.feat_std

        n = len(target)
        split = int(n * train_frac)
        self.features = features[:split] if train else features[split:]
        self.target = target[:split] if train else target[split:]
        self.num_features = len(feature_cols)
'''
    else:
        # Synthetic generator branch
        if seq_builtin_dataset in ("mackey_glass", "sine"):
            dataset_init_block = f'''
        # Generate Mackey-Glass chaotic time series
        np.random.seed(SEED)
        n_samples = 5000
        tau = 17
        x = np.zeros(n_samples)
        # Initialize history with random noise around 1.2
        x[:tau] = 1.2 + 0.1 * np.random.randn(tau)
        for i in range(tau, n_samples):
            x[i] = 0.9 * x[i-1] + 0.2 * x[i-tau] / (1.0 + x[i-tau]**10)
        signal = x.astype(np.float32)
        
        features = signal[:-{pred_len}].reshape(-1, 1).astype(np.float32)
        if TASK_TYPE == "classification":
            # Classify whether next sequence value increases (1) or decreases (0)
            diff = np.diff(signal)
            labels = (diff > 0).astype(np.int64)
            target = labels[seq_len - 1 : len(features) - {pred_len}]
            features = features[:len(target) + seq_len]
        else:
            target = signal[{pred_len}:].astype(np.float32)
        
        n = len(target)
        split = int(n * train_frac)
        
        # Normalize features
        self.feat_mean = features.mean(axis=0)
        self.feat_std = features.std(axis=0) + 1e-8
        features = (features - self.feat_mean) / self.feat_std
        
        self.features = features[:split] if train else features[split:]
        self.target = target[:split] if train else target[split:]
        self.num_features = 1
'''
        elif seq_builtin_dataset in ("fastText", "AISHELL"):
            dataset_init_block = f'''
        # Load from cluster environment variables
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
            
        features = signal[:-{pred_len}].reshape(-1, 1).astype(np.float32)
        if TASK_TYPE == "classification":
            diff = np.diff(signal)
            labels = (diff > 0).astype(np.int64)
            target = labels[seq_len - 1 : len(features) - {pred_len}]
            features = features[:len(target) + seq_len]
        else:
            target = signal[{pred_len}:].astype(np.float32)
            
        n = len(target)
        split = int(n * train_frac)
        
        # Normalize features
        self.feat_mean = features.mean(axis=0)
        self.feat_std = features.std(axis=0) + 1e-8
        features = (features - self.feat_mean) / self.feat_std
        
        self.features = features[:split] if train else features[split:]
        self.target = target[:split] if train else target[split:]
        self.num_features = 1
'''
        else: # sunspots
            dataset_init_block = f'''
        # Load Monthly Mean Total Sunspot Number dataset (1749 to July 2018)
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
        
        features = signal[:-{pred_len}].reshape(-1, 1).astype(np.float32)
        if TASK_TYPE == "classification":
            # Classify whether next sequence value increases (1) or decreases (0)
            diff = np.diff(signal)
            labels = (diff > 0).astype(np.int64)
            target = labels[seq_len - 1 : len(features) - {pred_len}]
            features = features[:len(target) + seq_len]
        else:
            target = signal[{pred_len}:].astype(np.float32)
            
        n = len(target)
        split = int(n * train_frac)
        
        # Normalize features
        self.feat_mean = features.mean(axis=0)
        self.feat_std = features.std(axis=0) + 1e-8
        features = (features - self.feat_mean) / self.feat_std
        
        self.features = features[:split] if train else features[split:]
        self.target = target[:split] if train else target[split:]
        self.num_features = 1
'''

    train_script = f'''#!/usr/bin/env python3
"""Generated PyTorch Lightning training script — Sequential / Time-Series (LSTM)."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
{_LIGHTNING_IMPORT_BLOCK}

CSV_PATH = "{_py_str(ts_data_path)}"
TARGET_COLUMN = "{_py_str(target_col)}"
TASK_TYPE = "{seq_task_type}"   # "regression" or "classification"
SEQ_LEN = {seq_len}
PRED_LEN = {pred_len}
HIDDEN_SIZE = {hidden_size}
NUM_LSTM_LAYERS = {num_lstm_layers}

LOG_DIR = "{_py_str(log_dir)}"
EXPERIMENT_NAME = "{_py_str(exp_name)}"
LR = {lr}
BATCH_SIZE = {bs}
NUM_WORKERS = {nw}
MAX_EPOCHS = {ep}
ACCELERATOR = "{_py_str(acc)}"
DEVICES = {dev}
PRECISION = {prec if str(prec).isdigit() else f'"{_py_str(prec)}"'}
LOG_EVERY_N_STEPS = {log_n}
SEED = {seed_val}


class TimeSeriesDataset(Dataset):
    """Dataset with sliding window indexing for sequence modeling."""

    def __init__(self, train=True, train_frac=0.8):
        seq_len = SEQ_LEN
        pred_len = PRED_LEN
        {dataset_init_block}
        
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return max(0, len(self.target) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]                  # (seq_len, num_features)
        y = self.target[idx + self.seq_len : idx + self.seq_len + self.pred_len]  # (pred_len,)
        return torch.tensor(x), torch.tensor(y)


class LitDataModule(L.LightningDataModule):
    def __init__(self, batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = TimeSeriesDataset(train=True)
        self.val_ds   = TimeSeriesDataset(train=False)
        self.num_features = self.train_ds.num_features

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)


class LitLSTM(L.LightningModule):
    """Stacked LSTM for time-series regression or classification."""

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LSTM_LAYERS,
                 output_size={pred_len}, lr=LR, task_type=TASK_TYPE):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # use last time-step hidden state

    def _shared_step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        {loss_line}
        self.log(f"{{stage}}_loss", loss, prog_bar=True)
{metric_lines}
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


def build_loggers():
    loggers = [
{chr(10).join(logger_lines)}
    ]
    return [lg for lg in loggers if lg is not False]

def main():
    L.seed_everything(SEED, workers=True)
    datamodule = LitDataModule()
    datamodule.setup()
    model = LitLSTM(
        input_size=datamodule.num_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LSTM_LAYERS,
        output_size={out_size_init},
        lr=LR,
        task_type=TASK_TYPE,
    )
    loggers = build_loggers()
    {callback_block}
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
'''
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


def _gen_gnn_script(
    exp_name, graph_dataset_type, graph_data_path, gnn_hidden_dim, gnn_num_layers,
    ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
    logger_lines, callback_block,
):
    """Returns (train_script, prefetch_script_or_None)."""
    use_builtin = graph_dataset_type in ("JODIE", "QM9")
    pyg_name = graph_dataset_type
    data_dir = "./data"

    if not use_builtin and not graph_data_path:
        drona_add_message("Custom graph dataset path is required.", "error")
        return None, None

    if use_builtin:
        prefetch_script = None
        if pyg_name == "QM9":
            dataset_setup = f'''
        from torch_geometric.datasets import QM9
        import os
        path = os.environ.get("QM9_PATH", "{_py_str(data_dir)}")
        self.train_ds = QM9(root=path)
        self.val_ds = self.train_ds
        self.num_features = self.train_ds.num_features
        self.num_classes = 19
        self.is_gnn_benchmark = True'''
        else:  # JODIE
            dataset_setup = f'''
        from torch_geometric.datasets import JODIE
        import os
        path = os.environ.get("JODIE_PATH", os.environ.get("ImageNet2012_PT_PATH", "{_py_str(data_dir)}"))
        dataset = JODIE(root=path, name="wikipedia")
        self.data = dataset[0]
        self.num_features = dataset.num_features
        self.num_classes = 2
        self.is_gnn_benchmark = False'''
    else:
        dataset_setup = f'''
        data_path = "{_py_str(graph_data_path)}"
        self.data = torch.load(os.path.join(data_path, "data.pt"))
        self.num_features = self.data.num_node_features
        self.num_classes = int(self.data.y.max().item()) + 1
        self.is_gnn_benchmark = False'''
        prefetch_script = None

    train_script = f'''#!/usr/bin/env python3
"""
Generated PyTorch Lightning training script — Graph Neural Network (GCN).

NOTE: This script uses PyTorch-Geometric/2.1.0 which requires the PyTorch 1.12 +
CUDA 11.7 module stack (PyTorch-Lightning/1.8.4). See the generated SBATCH script
for the correct module load commands.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
{_LIGHTNING_IMPORT_BLOCK}
from torch_geometric.nn import GCNConv

DATA_DIR = "{_py_str(data_dir)}"
LOG_DIR = "{_py_str(log_dir)}"
EXPERIMENT_NAME = "{_py_str(exp_name)}"
LR = {lr}
BATCH_SIZE = 1          # GNN batching: one full graph per step
NUM_WORKERS = {nw}
MAX_EPOCHS = {ep}
ACCELERATOR = "{_py_str(acc)}"
DEVICES = {dev}
PRECISION = {prec if str(prec).isdigit() else f'"{_py_str(prec)}"'}
LOG_EVERY_N_STEPS = {log_n}
SEED = {seed_val}
GNN_HIDDEN_DIM = {gnn_hidden_dim}
GNN_NUM_LAYERS = {gnn_num_layers}


class LitDataModule(L.LightningDataModule):
    def setup(self, stage=None):{dataset_setup}

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


class LitGCN(L.LightningModule):
    """Multi-layer GCN for node classification."""

    def __init__(self, in_channels, hidden_channels=GNN_HIDDEN_DIM,
                 out_channels=10, num_layers=GNN_NUM_LAYERS, lr=LR):
        super().__init__()
        self.save_hyperparameters()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(max(0, num_layers - 2)):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, edge_index)

    def _shared_step(self, batch, stage):
        out = self(batch.x, batch.edge_index)
        mask = getattr(batch, f"{{stage}}_mask", None)
        if mask is not None:
            loss = F.cross_entropy(out[mask], batch.y[mask])
            acc = (out[mask].argmax(dim=-1) == batch.y[mask]).float().mean()
        else:
            loss = F.cross_entropy(out, batch.y)
            acc = (out.argmax(dim=-1) == batch.y).float().mean()
        batch_size = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        self.log(f"{{stage}}_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{{stage}}_acc", acc, prog_bar=True, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)


def build_loggers():
    loggers = [
{chr(10).join(logger_lines)}
    ]
    return [lg for lg in loggers if lg is not False]

def main():
    L.seed_everything(SEED, workers=True)
    datamodule = LitDataModule()
    datamodule.setup()
    model = LitGCN(
        in_channels=datamodule.num_features,
        hidden_channels=GNN_HIDDEN_DIM,
        out_channels=datamodule.num_classes,
        num_layers=GNN_NUM_LAYERS,
        lr=LR,
    )
    loggers = build_loggers()
    {callback_block}
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
'''
    return train_script, prefetch_script


def _gen_generative_script(
    exp_name, gen_dataset_type, gen_custom_path, latent_dim,
    ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
    logger_lines, callback_block,
):
    """Returns (train_script, prefetch_script_or_None)."""
    cache_dir = "./data"
    use_builtin = gen_dataset_type != "custom"

    if use_builtin:
        if gen_dataset_type in ("MNIST", "FashionMNIST"):
            in_ch, img_sz = 1, 28
            prefetch_script = f'''#!/usr/bin/env python3
"""Pre-download prepared dataset for VAE training."""

DATASET = "{_py_str(gen_dataset_type)}"
DATA_DIR = "{_py_str(cache_dir)}"

{_DOWNLOAD_ONLY_BLOCK}

def main():
    if DATASET in ("MNIST", "FashionMNIST"):
        _ensure_idx_dataset_files(DATA_DIR, DATASET)
    print(f"Prefetched {{DATASET}} under {{DATA_DIR}}")

if __name__ == "__main__":
    main()
'''
        elif gen_dataset_type == "llava-onevision":
            in_ch, img_sz = 3, 224
            prefetch_script = None
        else:
            raise ValueError(f"Unsupported prepared generative dataset: {gen_dataset_type}")
        dm = f'''class LitDataModule(L.LightningDataModule):
    def __init__(self, data_dir="{_py_str(cache_dir)}", batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = "{_py_str(gen_dataset_type)}"

    def setup(self, stage=None):
        self.train_ds = load_builtin_dataset(self.dataset_name, self.data_dir, train=True)
        self.val_ds = load_builtin_dataset(self.dataset_name, self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
'''
    else:
        if not gen_custom_path:
            drona_add_message("Custom image folder path is required for Generative model.", "error")
            return None, None
        in_ch, img_sz = 3, 64   # sensible default for custom folders
        dm = f'''class LitDataModule(L.LightningDataModule):
    def __init__(self, data_root="{_py_str(gen_custom_path)}", batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = TensorFolderDataset(os.path.join(self.data_root, "train"), image_size={img_sz})
        self.val_ds = TensorFolderDataset(os.path.join(self.data_root, "val"), image_size={img_sz})

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
'''
        prefetch_script = None

    flat_size = in_ch * img_sz * img_sz

    train_script = f'''#!/usr/bin/env python3
"""Generated PyTorch Lightning training script — Variational Autoencoder (VAE)."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
{_LIGHTNING_IMPORT_BLOCK}

DATA_DIR = "{_py_str(cache_dir)}"
LOG_DIR = "{_py_str(log_dir)}"
EXPERIMENT_NAME = "{_py_str(exp_name)}"
LR = {lr}
BATCH_SIZE = {bs}
NUM_WORKERS = {nw}
MAX_EPOCHS = {ep}
ACCELERATOR = "{_py_str(acc)}"
DEVICES = {dev}
PRECISION = {prec if str(prec).isdigit() else f'"{_py_str(prec)}"'}
LOG_EVERY_N_STEPS = {log_n}
SEED = {seed_val}

IN_CHANNELS = {in_ch}
IMAGE_SIZE = {img_sz}
LATENT_DIM = {latent_dim}
FLAT_SIZE = {flat_size}   # IN_CHANNELS * IMAGE_SIZE * IMAGE_SIZE

{_DATA_HELPERS_BLOCK}

{dm}

class LitVAE(L.LightningModule):
    """
    Variational Autoencoder with fully-connected encoder/decoder.
    Optimises the Evidence Lower Bound (ELBO): reconstruction loss + KL divergence.
    """

    def __init__(self, flat_size=FLAT_SIZE, latent_dim=LATENT_DIM, lr=LR):
        super().__init__()
        self.save_hyperparameters()

        # Encoder: image → (mu, log_var)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

        # Decoder: z → reconstructed image
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, flat_size),
            nn.Sigmoid(),   # pixel values in [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

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
        self.log(f"{{stage}}_loss", loss, prog_bar=True)
        self.log(f"{{stage}}_recon", recon)
        self.log(f"{{stage}}_kl", kl)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def build_loggers():
    loggers = [
{chr(10).join(logger_lines)}
    ]
    return [lg for lg in loggers if lg is not False]

def main():
    L.seed_everything(SEED, workers=True)
    datamodule = LitDataModule()
    datamodule.setup()
    model = LitVAE(flat_size=FLAT_SIZE, latent_dim=LATENT_DIM, lr=LR)
    loggers = build_loggers()
    {callback_block}
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
'''
    return train_script, prefetch_script


def _gen_custom_script(
    exp_name, ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
    logger_lines, callback_block,
    custom_dataset_path="",
):
    """Returns (train_script, None) — bare-bones stub for the user to complete."""
    train_script = f'''#!/usr/bin/env python3
"""
Generated PyTorch Lightning training script — Custom (Bare-bones).

Complete every section marked TODO to implement your model.
The structure follows PyTorch Lightning best practices.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
{_LIGHTNING_IMPORT_BLOCK}

DATA_PATH = "{_py_str(custom_dataset_path)}"

# ── Hyperparameters ────────────────────────────────────────────────────────────
LOG_DIR = "{_py_str(log_dir)}"
EXPERIMENT_NAME = "{_py_str(exp_name)}"
LR = {lr}
BATCH_SIZE = {bs}
NUM_WORKERS = {nw}
MAX_EPOCHS = {ep}
ACCELERATOR = "{_py_str(acc)}"
DEVICES = {dev}
PRECISION = {prec if str(prec).isdigit() else f'"{_py_str(prec)}"'}
LOG_EVERY_N_STEPS = {log_n}
SEED = {seed_val}


# ── Dataset ────────────────────────────────────────────────────────────────────

class MyDataset(Dataset):
    """
    TODO: Replace this stub with your actual dataset.
    The custom dataset path you selected is pre-filled as DATA_PATH.

    Options:
      - Load from CSV:   pd.read_csv(data_path)
      - Load from HDF5:  h5py.File(data_path)
      - Load .pt files:  torch.load(data_path)
      - Use torchvision: torchvision.datasets.ImageFolder(data_path)
    """

    def __init__(self, data_path=DATA_PATH, train=True):
        self.data_path = data_path
        # TODO: load your data split here
        self.length = 100  # placeholder

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # TODO: return a single (input_tensor, label) pair
        x = torch.zeros(10)   # placeholder — replace with real features
        y = torch.tensor(0)   # placeholder — replace with real label
        return x, y


class LitDataModule(L.LightningDataModule):
    """
    TODO: Wire up your dataset splits here.
    """

    def __init__(self, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = MyDataset(train=True)
        self.val_ds = MyDataset(train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size,
            num_workers=self.num_workers
        )


# ── Model ──────────────────────────────────────────────────────────────────────

class LitModel(L.LightningModule):
    """
    TODO: Define your model architecture and training logic.

    Common patterns:
      - Image classification: nn.Conv2d + nn.Linear
      - Text:                 nn.Embedding + nn.LSTM / Transformer
      - Regression:           nn.Linear layers + MSE loss
      - Custom task:          any nn.Module subclass
    """

    def __init__(self, lr=LR):
        super().__init__()
        self.save_hyperparameters()

        # TODO: define your layers, e.g.:
        # self.net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
        # self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # TODO: implement the forward pass, e.g.:
        # return self.net(x)
        raise NotImplementedError("Implement forward() in LitModel.")

    def training_step(self, batch, batch_idx):
        # TODO: compute loss and log metrics, e.g.:
        # x, y = batch
        # logits = self(x)
        # loss = self.loss_fn(logits, y)
        # acc = (logits.argmax(dim=1) == y).float().mean()
        # self.log("train_loss", loss, prog_bar=True)
        # self.log("train_acc",  acc,  prog_bar=True)
        # return loss
        raise NotImplementedError("Implement training_step() in LitModel.")

    def validation_step(self, batch, batch_idx):
        # TODO: mirror training_step without the return, e.g.:
        # x, y = batch
        # logits = self(x)
        # loss = self.loss_fn(logits, y)
        # self.log("val_loss", loss, prog_bar=True)
        pass

    def configure_optimizers(self):
        # TODO: customise optimiser / scheduler if needed
        # Example with cosine annealing:
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
        # return [optimizer], [scheduler]
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ── Loggers & callbacks ────────────────────────────────────────────────────────

def build_loggers():
    loggers = [
{chr(10).join(logger_lines)}
    ]
    return [lg for lg in loggers if lg is not False]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    L.seed_everything(SEED, workers=True)
    datamodule = LitDataModule()
    model = LitModel(lr=LR)
    loggers = build_loggers()
    {callback_block}
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
'''
    return train_script, None


# ── Main dispatcher ────────────────────────────────────────────────────────────

def generate_lightning_script_if_run(
    mode,
    modelCategory="computer_vision",
    name="lightning_run",
    datasetType="builtin",
    builtinDataset="MNIST",
    customDataPath="",
    epochs="10",
    batchSize="32",
    learningRate="1e-3",
    numWorkers="0",
    seed="42",
    gpu="",
    logDir="./lightning_logs",
    logEveryNSteps="50",
    enableTensorBoard="Yes",
    checkpointEnable="Yes",
    hfModelName="",
    hfDatasetName="",
    textDatasetType="builtin",
    textColumn="",
    labelColumn="",
    customTextPath="",
    maxSeqLen="128",
    numLabels="2",
    seqDatasetType="builtin",
    seqBuiltinDataset="mackey_glass",
    tsDataPath="",
    seqLen="50",
    predLen="1",
    targetColumn="target",
    hiddenSize="128",
    numLstmLayers="2",
    seqTaskType="regression",
    graphDatasetType="builtin",
    graphBuiltinDataset="cora",
    graphDataPath="",
    gnnHiddenDim="64",
    gnnNumLayers="2",
    genDatasetType="builtin",
    genBuiltinDataset="MNIST",
    genCustomPath="",
    latentDim="128",
    customDatasetPath="",
    job_location="",
    slurmBox="Yes",
    precision="32",
):
    if mode == "monitor":
        return ""
    return generate_lightning_script(
        modelCategory,
        name, datasetType, builtinDataset, customDataPath,
        epochs, batchSize, learningRate, numWorkers, seed, gpu,
        logDir, logEveryNSteps,
        enableTensorBoard, checkpointEnable,
        hfModelName, hfDatasetName, textDatasetType, textColumn, labelColumn, customTextPath, maxSeqLen, numLabels,
        seqDatasetType, seqBuiltinDataset, tsDataPath, seqLen, predLen, targetColumn, hiddenSize, numLstmLayers, seqTaskType,
        graphDatasetType, graphBuiltinDataset, graphDataPath, gnnHiddenDim, gnnNumLayers,
        genDatasetType, genBuiltinDataset, genCustomPath, latentDim,
        customDatasetPath,
        job_location,
        slurmBox,
        precision,
    )


def generate_lightning_script(
    modelCategory="computer_vision",
    name="lightning_run",
    datasetType="builtin",
    builtinDataset="MNIST",
    customDataPath="",
    epochs="10",
    batchSize="32",
    learningRate="1e-3",
    numWorkers="0",
    seed="42",
    gpu="",
    logDir="./lightning_logs",
    logEveryNSteps="50",
    enableTensorBoard="Yes",
    checkpointEnable="Yes",
    hfModelName="",
    hfDatasetName="",
    textDatasetType="builtin",
    textColumn="",
    labelColumn="",
    customTextPath="",
    maxSeqLen="128",
    numLabels="2",
    seqDatasetType="builtin",
    seqBuiltinDataset="mackey_glass",
    tsDataPath="",
    seqLen="50",
    predLen="1",
    targetColumn="target",
    hiddenSize="128",
    numLstmLayers="2",
    seqTaskType="regression",
    graphDatasetType="builtin",
    graphBuiltinDataset="cora",
    graphDataPath="",
    gnnHiddenDim="64",
    gnnNumLayers="2",
    genDatasetType="builtin",
    genBuiltinDataset="MNIST",
    genCustomPath="",
    latentDim="128",
    customDatasetPath="",
    job_location="",
    slurmBox="Yes",
    precision="32",
):
    # ── Parse & validate shared params ────────────────────────────────────────
    category = (modelCategory or "computer_vision").strip()
    exp_name = sanitize_job_name(name) or "lightning_run"

    lr, lr_err = _parse_learning_rate(learningRate)
    if lr_err:
        drona_add_message(lr_err, "error")
        return ""

    seed_val, seed_err = _parse_seed(seed)
    if seed_err:
        drona_add_message(seed_err, "error")
        return ""

    ep  = _resolve_int(epochs, 10)
    bs  = _resolve_int(batchSize, 32)
    nw  = _resolve_int(numWorkers, 0)
    log_n = _resolve_int(logEveryNSteps, 50)
    log_dir = _resolve_val(logDir, "./lightning_logs")

    gpu_val = _normalize_select_value(gpu)
    if gpu_val == "pvc":
        acc = "xpu"
    else:
        acc = "gpu" if _wants_gpu(gpu, slurmBox) else "cpu"
    dev = "1"
    prec = _resolve_val(precision, "32")

    tb_on   = True
    ckpt_on = _checkbox_on(checkpointEnable)

    logger_lines = []
    if tb_on:
        logger_lines.append(
            f'    TensorBoardLogger(save_dir="{_py_str(log_dir)}", name="{_py_str(exp_name)}"),'
        )
    if not logger_lines:
        logger_lines.append("    False,")

    callback_lines = []
    if ckpt_on:
        callback_lines.append(
            '        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True),'
        )
    callback_block = (
        "callbacks = [\n" + "\n".join(callback_lines) + "\n    ]"
        if callback_lines
        else "callbacks = None"
    )

    env_dir = _get_env_dir()

    # ── Dispatch to category generator ────────────────────────────────────────
    train_script = prefetch_script = None

    if category == "computer_vision":
        ds_type = datasetType or "builtin"
        builtin = builtinDataset or "MNIST"
        custom_path = customDataPath or ""
        if ds_type == "custom" and not custom_path:
            drona_add_message("Custom dataset path is required.", "error")
            return ""
        train_script, prefetch_script = _gen_computer_vision_script(
            exp_name, ds_type, builtin, custom_path,
            ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
        )

    elif category == "sequential":
        seq_ds_type = (seqDatasetType or "builtin").strip()
        seq_blt     = (seqBuiltinDataset or "mackey_glass").strip()
        ts_path     = (tsDataPath or "").strip()
        tgt_col     = (targetColumn or "target").strip()
        task_type   = (seqTaskType or "regression").strip()
        s_len       = int(seqLen)      if seqLen      else 50
        p_len       = int(predLen)     if predLen     else 1
        h_size      = int(hiddenSize)  if hiddenSize  else 128
        n_layers    = int(numLstmLayers) if numLstmLayers else 2
        
        train_script, prefetch_script = _gen_sequential_script(
            exp_name, ts_path, tgt_col, task_type,
            s_len, p_len, h_size, n_layers,
            ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
            seq_dataset_type=seq_ds_type,
            seq_builtin_dataset=seq_blt,
        )

    elif category == "gnn":
        gnn_ds_type = (graphDatasetType or "builtin").strip()
        gnn_blt     = (graphBuiltinDataset or "cora").strip()
        g_type      = gnn_blt if gnn_ds_type == "builtin" else "custom_graph"
        g_path      = (graphDataPath or "").strip()
        g_hdim      = int(gnnHiddenDim)  if gnnHiddenDim  else 64
        g_nlay      = int(gnnNumLayers)  if gnnNumLayers  else 2
        
        train_script, prefetch_script = _gen_gnn_script(
            exp_name, g_type, g_path, g_hdim, g_nlay,
            ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
        )

    elif category == "generative":
        gen_ds_type = (genDatasetType or "builtin").strip()
        gen_blt     = (genBuiltinDataset or "MNIST").strip()
        gen_type    = gen_blt if gen_ds_type == "builtin" else "custom"
        gen_path    = (genCustomPath or "").strip()
        lat_dim     = int(latentDim) if latentDim else 128
        
        train_script, prefetch_script = _gen_generative_script(
            exp_name, gen_type, gen_path, lat_dim,
            ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
        )

    elif category == "custom":
        train_script, prefetch_script = _gen_custom_script(
            exp_name, ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
            custom_dataset_path=customDatasetPath,
        )

    else:
        drona_add_message(f"Unknown model category: {category}", "error")
        return ""

    if train_script is None:
        return ""  # error already added by sub-generator

    # ── Write generated files ─────────────────────────────────────────────────
    _write_staged_file(env_dir, job_location, "train.py", train_script)
    if prefetch_script:
        _write_staged_file(env_dir, job_location, "prefetch_data.py", prefetch_script)
    else:
        # Clean up any leftover prefetch_data.py file if it exists
        for base_dir in (env_dir, job_location):
            if base_dir:
                loc = _normalize_location(base_dir)
                if loc:
                    p = Path(loc) / "prefetch_data.py"
                    if p.is_file():
                        try:
                            p.unlink()
                        except Exception:
                            pass

    port_finder = env_dir / "find_free_tb_port.sh"
    if port_finder.is_file():
        script_content = port_finder.read_text(encoding="utf-8")
    else:
        script_content = _FIND_FREE_TB_PORT_SCRIPT

    _write_staged_file(
        env_dir,
        job_location,
        "find_free_tb_port.sh",
        script_content,
        show_in_preview=True,
    )

    return ""


def setup_tensorboard_in_job(mode, enableTensorBoard, location):
    if mode == "monitor":
        return ""

    # Parse logDir from train.py if possible
    log_dir = "lightning_logs"
    if location:
        train_py_path = os.path.join(location, "train.py")
        if os.path.isfile(train_py_path):
            try:
                import re
                with open(train_py_path, "r", encoding="utf-8") as f:
                    content = f.read()
                match = re.search(r'LOG_DIR\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    val = match.group(1)
                    if val.startswith("./"):
                        val = val[2:]
                    log_dir = val
            except Exception:
                pass

    return f"""# Launch TensorBoard in the background on the compute node using srun
if [ -d "{log_dir}" ] || mkdir -p "{log_dir}"; then
    echo "Starting TensorBoard background server with srun..." >&2
    
    # Inline port finder logic: find a free port in range 6006-6050, then random high ports
    TB_PORT=""
    for port in $(seq 6006 6050); do
      if ! ss -tuln 2>/dev/null | grep -q ":${{port}} "; then
        TB_PORT=$port
        break
      fi
    done
    if [ -z "$TB_PORT" ]; then
      for i in $(seq 1 1000); do
        rand_val=$(( (RANDOM << 15) | RANDOM ))
        port=$(( 6051 + (rand_val % 59485) ))
        if ! ss -tuln 2>/dev/null | grep -q ":${{port}} "; then
          TB_PORT=$port
          break
        fi
      done
      if [ -z "$TB_PORT" ]; then
        for port in $(seq 6051 65535); do
          if ! ss -tuln 2>/dev/null | grep -q ":${{port}} "; then
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

    echo "$TB_PORT" > tb_port.txt
    NODE_NAME=$(hostname)
    
    # Isolate module loading and execution in a subshell to avoid compiler toolchain conflicts
    # Use srun to launch tensorboard concurrently within the main job's allocation
    (
        module load GCC/13.2.0 tensorboard/2.18.0
        srun --ntasks=1 --nodes=1 --cpus-per-task=1 --overlap tensorboard --logdir="{log_dir}" --port=$TB_PORT --bind_all >&2
    ) &
    TB_PID=$!
    
    # Register exit trap to ensure TensorBoard is killed once the job finishes
    cleanup() {{
        echo "Cleaning up TensorBoard server..." >&2
        if [ -n "$TB_PID" ]; then
            kill $TB_PID 2>/dev/null
            pkill -P $TB_PID 2>/dev/null
        fi
        pkill -u $USER -f "tensorboard --logdir=.*--port=$TB_PORT" 2>/dev/null
    }}
    trap cleanup EXIT
    
    echo "TensorBoard auto-started with srun on node $NODE_NAME port $TB_PORT" >&2
fi
"""

