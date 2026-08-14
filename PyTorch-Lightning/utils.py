import importlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from generators import (
    _gen_computer_vision_script,
    _gen_sequential_script,
    _gen_generative_script,
    _gen_custom_script,
    _gen_nlp_script,
    _gen_tabular_script,
    _gen_audio_script,
)



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



GEN_DATASET_MODULES = {
    "llava-onevision": "Datasets/llava-onevision/2024",
}


def setup_pytorch_modules(
    model_category="computer_vision",
    dataset_type="builtin",
    builtin_dataset="ImageNet",
    seq_dataset="mackey_glass",
    graph_dataset="cora",
    gen_dataset="llava-onevision",
    nlp_dataset_type="builtin",
    nlp_builtin_dataset="fastText",
    tab_dataset_type="builtin",
    tab_builtin_dataset="Synthetic Tabular",
    audio_dataset_type="builtin",
    audio_builtin_dataset="AISHELL",
    gpu="",
):
    """Return module load commands for the given model category and dataset."""
    cluster, cluster_module = retrieve_cluster_info()
    module_use_cmd = ""
    
    ds_t = (dataset_type or "builtin").strip()
    dataset_cmd = ""
    
    if model_category == "nlp":
        ds_t = (nlp_dataset_type or "builtin").strip()
    elif model_category == "tabular":
        ds_t = (tab_dataset_type or "builtin").strip()
    elif model_category == "audio":
        ds_t = (audio_dataset_type or "builtin").strip()

    if ds_t == "builtin":
        if model_category == "computer_vision":
            if builtin_dataset in CV_DATASET_MODULES:
                dataset_cmd = f"\nmodule load {CV_DATASET_MODULES[builtin_dataset]} 2>/dev/null || true"
        elif model_category == "sequential":
            if seq_dataset in SEQ_DATASET_MODULES:
                dataset_cmd = f"\nmodule load {SEQ_DATASET_MODULES[seq_dataset]} 2>/dev/null || true"
        elif model_category == "generative":
            if gen_dataset in GEN_DATASET_MODULES:
                dataset_cmd = f"\nmodule load {GEN_DATASET_MODULES[gen_dataset]} 2>/dev/null || true"
        elif model_category == "nlp":
            if nlp_builtin_dataset in SEQ_DATASET_MODULES:
                dataset_cmd = f"\nmodule load {SEQ_DATASET_MODULES[nlp_builtin_dataset]} 2>/dev/null || true"
        elif model_category == "audio":
            if audio_builtin_dataset in SEQ_DATASET_MODULES:
                dataset_cmd = f"\nmodule load {SEQ_DATASET_MODULES[audio_builtin_dataset]} 2>/dev/null || true"

    gpu_val = (gpu or "").strip().lower()

    base = getattr(cluster_module, "pytorch_lightning_modules", DEFAULT_PT_MODULES)

    torchvision_cmd = ""
    if model_category == "computer_vision" or (
        model_category == "generative"
        and (ds_t == "custom" or gen_dataset == "llava-onevision")
    ):
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
    graph_dataset="cora",
    gen_dataset="llava-onevision",
    nlp_dataset_type="builtin",
    nlp_builtin_dataset="fastText",
    tab_dataset_type="builtin",
    tab_builtin_dataset="Synthetic Tabular",
    audio_dataset_type="builtin",
    audio_builtin_dataset="AISHELL",
    gpu="",
):
    if mode == "monitor":
        return "# monitor mode — no training job"
    return setup_pytorch_modules(
        model_category, dataset_type, builtin_dataset, seq_dataset, graph_dataset, gen_dataset,
        nlp_dataset_type, nlp_builtin_dataset, tab_dataset_type, tab_builtin_dataset, audio_dataset_type, audio_builtin_dataset,
        gpu
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
    job_dir_str = str(job_dir) if job_dir is not None else ""
    match = re.search(r">(/[^<]+)<", job_dir_str)
    return (match.group(1) if match else job_dir_str.strip())


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


_DATA_HELPERS_BLOCK = ""


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






# Script generators are imported from the generators package.



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
    cvModelArch="cnn",
    seqModelType="lstm",
    gnnLayerType="gcn",
    genModelType="vae",
    cvDatasetFormat="tensors",
    genDatasetFormat="tensors",
    exportOnnx="No",
    nlpDatasetType="custom",
    nlpBuiltinDataset="fastText",
    nlpCustomPath="",
    nlpModelType="attention",
    nlpMaxSeqLen="128",
    tabDatasetType="custom",
    tabBuiltinDataset="synthetic_tabular",
    tabCustomPath="",
    tabTargetColumn="target",
    tabModelType="mlp",
    tabTaskType="regression",
    audioDatasetType="custom",
    audioBuiltinDataset="AISHELL",
    audioCustomPath="",
    audioModelType="classification",
    audioTransformType="mel_spectrogram",
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
        cvModelArch,
        seqModelType,
        gnnLayerType,
        genModelType,
        cvDatasetFormat,
        genDatasetFormat,
        exportOnnx,
        nlpDatasetType,
        nlpBuiltinDataset,
        nlpCustomPath,
        nlpModelType,
        nlpMaxSeqLen,
        tabDatasetType,
        tabBuiltinDataset,
        tabCustomPath,
        tabTargetColumn,
        tabModelType,
        tabTaskType,
        audioDatasetType,
        audioBuiltinDataset,
        audioCustomPath,
        audioModelType,
        audioTransformType,
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
    cvModelArch="cnn",
    seqModelType="lstm",
    gnnLayerType="gcn",
    genModelType="vae",
    cvDatasetFormat="tensors",
    genDatasetFormat="tensors",
    exportOnnx="No",
    nlpDatasetType="custom",
    nlpBuiltinDataset="fastText",
    nlpCustomPath="",
    nlpModelType="attention",
    nlpMaxSeqLen="128",
    tabDatasetType="custom",
    tabBuiltinDataset="synthetic_tabular",
    tabCustomPath="",
    tabTargetColumn="target",
    tabModelType="mlp",
    tabTaskType="regression",
    audioDatasetType="custom",
    audioBuiltinDataset="AISHELL",
    audioCustomPath="",
    audioModelType="classification",
    audioTransformType="mel_spectrogram",
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

    acc = "gpu" if _wants_gpu(gpu, slurmBox) else "cpu"
    dev = "1"
    prec = _resolve_val(precision, "32")

    tb_on   = True
    ckpt_on = _checkbox_on(checkpointEnable)

    logger_lines = []
    if tb_on:
        logger_lines.append(
            f'    TensorBoardLogger(save_dir=LOG_DIR, name="{_py_str(exp_name)}"),'
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
            cv_model_arch=cvModelArch,
            cv_dataset_format=cvDatasetFormat,
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
            seq_model_type=seqModelType,
        )

    elif category == "gnn":
        import builtins
        msg = "Graph Neural Networks (GNN) model category is no longer supported."
        if hasattr(builtins, "drona_add_message"):
            builtins.drona_add_message(msg, "error")
        else:
            print(f"[ERROR] {msg}")
        return

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
            gen_model_type=genModelType,
            gen_dataset_format=genDatasetFormat,
        )

    elif category == "nlp":
        nlp_ds_type = (nlpDatasetType or "custom").strip()
        nlp_blt     = (nlpBuiltinDataset or "fastText").strip()
        nlp_path    = (nlpCustomPath or "").strip()
        nlp_max_len = int(nlpMaxSeqLen) if nlpMaxSeqLen else 128
        
        train_script, prefetch_script = _gen_nlp_script(
            exp_name, nlp_ds_type, nlp_blt, nlp_path, nlp_max_len,
            ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
            nlp_model_type=nlpModelType,
        )

    elif category == "tabular":
        tab_ds_type = (tabDatasetType or "custom").strip()
        tab_blt     = (tabBuiltinDataset or "synthetic_tabular").strip()
        tab_path    = (tabCustomPath or "").strip()
        tab_tgt     = (tabTargetColumn or "target").strip()
        tab_task    = (tabTaskType or "regression").strip()
        
        train_script, prefetch_script = _gen_tabular_script(
            exp_name, tab_ds_type, tab_blt, tab_path, tab_tgt, tab_task,
            ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
            tab_model_type=tabModelType,
        )

    elif category == "audio":
        audio_ds_type = (audioDatasetType or "custom").strip()
        audio_blt     = (audioBuiltinDataset or "AISHELL").strip()
        audio_path    = (audioCustomPath or "").strip()
        
        train_script, prefetch_script = _gen_audio_script(
            exp_name, audio_ds_type, audio_blt, audio_path,
            ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
            logger_lines, callback_block,
            audio_model_type=audioModelType,
            audio_transform_type=audioTransformType,
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

    if _checkbox_on(exportOnnx):
        onnx_export_code = '''    trainer.fit(model, datamodule=datamodule)

    # Export final model to ONNX
    try:
        import inspect
        model.eval()
        
        # Setup datamodule and get a sample batch
        datamodule.setup("fit")
        dl = datamodule.train_dataloader()
        batch = next(iter(dl))
        
        sig = inspect.signature(model.forward)
        
        if "edge_index" in sig.parameters:
            x = getattr(batch, 'x', None)
            edge_index = getattr(batch, 'edge_index', None)
            if x is not None and edge_index is not None:
                input_sample = (x, edge_index)
            else:
                in_channels = getattr(model.hparams, 'in_channels', 64)
                input_sample = (torch.randn(10, in_channels), torch.zeros((2, 20), dtype=torch.long))
        else:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
                
            if isinstance(x, torch.Tensor):
                input_sample = x[:1]
            else:
                input_sample = x
        
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        if isinstance(input_sample, tuple):
            input_sample = tuple(
                t.to(device=device, dtype=dtype) if isinstance(t, torch.Tensor) and t.dtype.is_floating_point else t.to(device=device)
                for t in input_sample
            )
        elif isinstance(input_sample, torch.Tensor):
            if input_sample.dtype.is_floating_point:
                input_sample = input_sample.to(device=device, dtype=dtype)
            else:
                input_sample = input_sample.to(device=device)
                
        onnx_path = os.path.join(LOG_DIR, "final_model.onnx")
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        model.to_onnx(onnx_path, input_sample, export_params=True)
        print(f"Successfully exported ONNX model at: {onnx_path}")
    except Exception as e:
        print(f"Could not export model to ONNX: {e}")'''
        train_script = train_script.replace("    trainer.fit(model, datamodule=datamodule)", onnx_export_code)

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
        show_in_preview=False,
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

