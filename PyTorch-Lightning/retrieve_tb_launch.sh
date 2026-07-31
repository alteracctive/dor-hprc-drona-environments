#!/bin/bash
# Reworked TensorBoard Monitoring and Access Link Generator.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR

export RESOLVED_LOGDIR="${RESOLVED_LOGDIR:-}"
export TB_VERSION="${TB_VERSION:-tensorboard/2.18.0}"
export TB_HOST="${TB_HOST:-localhost}"
export TB_START_TRIGGER="${TB_START_TRIGGER:-}"
export TB_STOP_TRIGGER="${TB_STOP_TRIGGER:-}"

python3 <<'EOF'
import html
import os
import glob
import subprocess
import socket
import sys

FIND_FREE_TB_PORT_SCRIPT = """#!/bin/bash
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
"""


def clean_env_val(val):
    if not val:
        return ""
    val = val.strip()
    # Strip surrounding quotes
    if val.startswith('"') and val.endswith('"'):
        val = val[1:-1].strip()
    if val.startswith("'") and val.endswith("'"):
        val = val[1:-1].strip()
    # Strip literal '\n' text or newlines
    while val.endswith('\\n') or val.endswith('\n'):
        if val.endswith('\\n'):
            val = val[:-2].strip()
        else:
            val = val[:-1].strip()
    # Strip quotes again in case they were inside the newlines
    if val.startswith('"') and val.endswith('"'):
        val = val[1:-1].strip()
    return val

def get_slurm_cmd(cmd):
    path = f"/sw/local/bin/{cmd}"
    if os.path.exists(path):
        return path
    return cmd

def get_train_slurm_directives(train_job_id, job_dir):
    """Reuse partition/account from the training job so tb-session sbatch succeeds."""
    directives = []
    seen = set()
    partition = ""
    account = ""

    if train_job_id:
        try:
            res = subprocess.run(
                [get_slurm_cmd("squeue"), "-j", train_job_id, "-h", "-o", "%P|%a"],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = res.stdout.strip().split("|")
                if parts:
                    partition = parts[0].strip()
                if len(parts) > 1:
                    account = parts[1].strip()
        except Exception:
            pass

        if not partition or not account:
            try:
                res = subprocess.run(
                    [get_slurm_cmd("sacct"), "-j", train_job_id, "-X", "-n", "-P", "--format=Partition,Account"],
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0 and res.stdout.strip():
                    line = res.stdout.strip().split("\n")[0]
                    parts = line.split("|")
                    if parts and not partition:
                        partition = parts[0].strip()
                    if len(parts) > 1 and not account:
                        account = parts[1].strip()
            except Exception:
                pass

    # Directives that must not be forwarded to the TB job — they pin it to GPU
    # nodes or tie it to the training job's hardware allocation.
    SKIP_KEYS = {
        "--job-name", "--time", "--mem", "--ntasks", "--nodes", "--cpus-per-task",
        "--output", "--error", "--gres", "--nodelist", "--exclude",
        "--gpu", "--gpus", "--gpus-per-node", "--gpus-per-task",
        "--constraint",
    }

    def _sbatch_key(token):
        """Return just the flag name from a token like '--gres=gpu:a100:2' -> '--gres'."""
        return token.split("=")[0]

    for pattern in ("*.sh", "*.job", "*.sbatch"):
        for script_path in sorted(glob.glob(os.path.join(job_dir, pattern))):
            if os.path.basename(script_path) == "tb_job.sh":
                continue
            try:
                with open(script_path, "r") as f:
                    for line in f:
                        line = line.rstrip()
                        if not line.startswith("#SBATCH"):
                            continue
                        tokens = line.split()
                        # Skip lines where ANY option token matches a skip key
                        # (handles both single-option and multi-option lines)
                        if any(_sbatch_key(t) in SKIP_KEYS for t in tokens[1:]):
                            continue
                        if line not in seen:
                            seen.add(line)
                            directives.append(line)
            except Exception:
                pass

    if partition and partition not in ("", "N/A", "(null)"):
        part_line = f"#SBATCH --partition={partition}"
        if part_line not in seen:
            directives.insert(0, part_line)
    if account and account not in ("", "N/A", "(null)"):
        acct_line = f"#SBATCH --account={account}"
        if acct_line not in seen:
            directives.insert(0, acct_line)

    return "\n".join(directives)

# Retrieve environment variables
resolved_logdir = clean_env_val(os.environ.get("RESOLVED_LOGDIR", ""))
custom_logdir = clean_env_val(os.environ.get("CUSTOM_LOGDIR", ""))
logdir = custom_logdir if custom_logdir else resolved_logdir

tb_version = clean_env_val(os.environ.get("TB_VERSION", "tensorboard/2.18.0"))
tb_host = clean_env_val(os.environ.get("TB_HOST", "localhost"))
tb_start_trigger = clean_env_val(os.environ.get("TB_START_TRIGGER", ""))
tb_stop_trigger = clean_env_val(os.environ.get("TB_STOP_TRIGGER", ""))

if not logdir:
    print("<p><em>Select a training run above to resolve the log directory.</em></p>")
    sys.exit(0)

# Resolve checkpoints directory
ckpt_dir = logdir
if os.path.isdir(logdir):
    ckpt_dirs = glob.glob(os.path.join(logdir, "**", "checkpoints"), recursive=True)
    if ckpt_dirs:
        ckpt_dir = sorted(ckpt_dirs, key=os.path.getmtime, reverse=True)[0]
    else:
        ckpt_files = glob.glob(os.path.join(logdir, "**", "*.ckpt"), recursive=True)
        if ckpt_files:
            ckpt_dir = os.path.dirname(ckpt_files[0])

last_trigger = ""
job_dir = os.path.dirname(logdir)

tb_last_trigger_file = os.path.join(job_dir, "tb_last_trigger.txt")
if os.path.isfile(tb_last_trigger_file):
    try:
        with open(tb_last_trigger_file, "r") as f:
            last_trigger = f.read().strip()
    except Exception:
        pass



# Determine GCC toolchain based on selected TensorBoard version
gcc_module = "GCC/12.3.0"
if "2.18.0" in tb_version:
    gcc_module = "GCC/13.2.0"

# Auto-detect compute node if host is set to localhost
def get_first_node(nodelist_str):
    if not nodelist_str:
        return ""
    import re
    first = nodelist_str.split(",")[0].strip()
    if "[" in first:
        m = re.match(r"^([^\[]+)\[([^\]]+)\]", first)
        if m:
            prefix = m.group(1)
            num_range = m.group(2)
            first_num = num_range.split("-")[0].split(",")[0].strip()
            return f"{prefix}{first_num}"
    return first

job_dir = os.path.dirname(logdir)
separate_port_file = os.path.join(job_dir, "tb_separate_port.txt")
tb_hours_file = os.path.join(job_dir, "tb_hours.txt")

tb_hours_val = 1
if os.path.isfile(tb_hours_file):
    try:
        with open(tb_hours_file, "r") as f:
            tb_hours_val = int(f.read().strip())
    except Exception:
        pass

tb_idle_file = os.path.join(job_dir, "tb_idle.txt")
tb_idle_val = 1
if os.path.isfile(tb_idle_file):
    try:
        with open(tb_idle_file, "r") as f:
            tb_idle_val = int(f.read().strip())
    except Exception:
        pass

# Check if a separate TensorBoard job is active
tb_job_id = None
tb_job_id_file = os.path.join(job_dir, "tb_job_id.txt")
if os.path.isfile(tb_job_id_file):
    try:
        with open(tb_job_id_file, "r") as f:
            tb_job_id = f.read().strip()
    except Exception:
        pass

is_tb_job_active = False
tb_job_state = None
separate_node = tb_host

if tb_job_id:
    try:
        res = subprocess.run([get_slurm_cmd("squeue"), "-j", tb_job_id, "-h", "-o", "%T|%N"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            parts = res.stdout.strip().split('|')
            if len(parts) >= 2:
                tb_job_state = parts[0].strip()
                node = get_first_node(parts[1].strip())
                if node and not node.startswith("("):
                    separate_node = node
                if tb_job_state in ("RUNNING", "PENDING", "COMPLETING"):
                    is_tb_job_active = True
    except Exception:
        pass

# Resolve training job status and info
is_train_job_active = False
train_job_state = None
train_job_id = None
train_node = "localhost"

jobids_file = os.path.join(job_dir, "slurm_jobids.txt")
if os.path.isfile(jobids_file):
    try:
        with open(jobids_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
            if lines:
                train_job_id = lines[0]
    except Exception:
        pass
# Fallback to out.* files if jobids_file not found or empty
if not train_job_id:
    try:
        import re
        out_files = glob.glob(os.path.join(job_dir, "out.*"))
        candidates = []
        for f in out_files:
            base = os.path.basename(f)
            m = re.match(r"^out\.([0-9]+)$", base)
            if m:
                candidates.append((os.path.getmtime(f), m.group(1)))
        if candidates:
            candidates.sort(reverse=True)
            train_job_id = candidates[0][1]
    except Exception:
        pass

if train_job_id:
    try:
        res = subprocess.run([get_slurm_cmd("squeue"), "-j", train_job_id, "-h", "-o", "%T|%N"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            parts = res.stdout.strip().split('|')
            if len(parts) >= 2:
                train_job_state = parts[0].strip()
                node = get_first_node(parts[1].strip())
                if node and not node.startswith("("):
                    train_node = node
                if train_job_state in ("RUNNING", "PENDING", "COMPLETING"):
                    is_train_job_active = True
        if not is_train_job_active:
            # Fallback to sacct to find train_node if main job is finished but we just need its node (for default node lookup)
            res = subprocess.run([get_slurm_cmd("sacct"), "-j", train_job_id, "--noheader", "--format=JobID,NodeList,Submit", "--parsable2"], capture_output=True, text=True)
            if res.returncode == 0:
                ref_path = jobids_file if os.path.isfile(jobids_file) else job_dir
                ref_time = os.path.getmtime(ref_path) if os.path.exists(ref_path) else None
                best_node = None
                min_diff = float('inf')
                for line in res.stdout.strip().split('\n'):
                    if not line.strip():
                        continue
                    parts = line.split('|')
                    if len(parts) < 3:
                        continue
                    jid, raw_node, submit_str = parts[:3]
                    if jid != train_job_id:
                        continue
                    if raw_node == "None" or not raw_node:
                        continue
                    try:
                        from datetime import datetime
                        dt = datetime.strptime(submit_str.strip(), "%Y-%m-%dT%H:%M:%S")
                        submit_time = dt.timestamp()
                    except Exception:
                        submit_time = None
                    if ref_time is not None and submit_time is not None:
                        diff = abs(ref_time - submit_time)
                        if diff < min_diff:
                            min_diff = diff
                            best_node = raw_node
                    elif best_node is None:
                        best_node = raw_node
                if best_node:
                    node = get_first_node(best_node)
                    if node:
                        train_node = node
    except Exception:
        pass

if not train_node or train_node == "localhost":
    train_node = "localhost"
if not separate_node or separate_node == "localhost":
    separate_node = "localhost"

# Process stop trigger if set
if tb_stop_trigger:
    # Ensure this stop trigger is for the current run directory to avoid cross-run cancellations
    stop_parts = tb_stop_trigger.split(":")
    stop_run_dir = stop_parts[1] if len(stop_parts) > 1 else ""
    if stop_run_dir == os.path.basename(job_dir):
        last_stop_trigger = ""
        tb_last_stop_trigger_file = os.path.join(job_dir, "tb_last_stop_trigger.txt")
        if os.path.isfile(tb_last_stop_trigger_file):
            try:
                with open(tb_last_stop_trigger_file, "r") as f:
                    last_stop_trigger = f.read().strip()
            except Exception:
                pass

        if last_stop_trigger != tb_stop_trigger:
            try:
                with open(tb_last_stop_trigger_file, "w") as f:
                    f.write(tb_stop_trigger)
            except Exception:
                pass

            stop_separate_port = None
            if os.path.isfile(separate_port_file):
                try:
                    with open(separate_port_file, "r") as f:
                        port_val = f.read().strip()
                        if port_val:
                            stop_separate_port = int(port_val)
                except Exception:
                    pass

            if stop_separate_port is not None:
                pkill_pattern = f"tensorboard.*--port={stop_separate_port}"
                if separate_node == "localhost" or separate_node == "127.0.0.1":
                    stop_cmd = f'pkill -u $USER -f "{pkill_pattern}"'
                else:
                    stop_cmd = f"ssh -o StrictHostKeyChecking=no {separate_node} 'pkill -u $USER -f \"{pkill_pattern}\"'"
                try:
                    subprocess.run(stop_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass

            # If a separate TensorBoard job exists, cancel it too
            if tb_job_id:
                try:
                    subprocess.run([get_slurm_cmd("scancel"), tb_job_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass

            tb_last_active_file = os.path.join(job_dir, "tb_last_active.txt")
            for fn in [separate_port_file, tb_last_trigger_file, tb_job_id_file, tb_hours_file, tb_idle_file, tb_last_active_file]:
                try:
                    if os.path.exists(fn):
                        os.remove(fn)
                except Exception:
                    pass
            
            tb_job_id = None
            is_tb_job_active = False

# Process start trigger if set (to submit a separate TensorBoard job)
if tb_start_trigger:
    last_trigger = ""
    tb_last_trigger_file = os.path.join(job_dir, "tb_last_trigger.txt")
    if os.path.isfile(tb_last_trigger_file):
        try:
            with open(tb_last_trigger_file, "r") as f:
                last_trigger = f.read().strip()
        except Exception:
            pass

    if last_trigger != tb_start_trigger:
        try:
            with open(tb_last_trigger_file, "w") as f:
                f.write(tb_start_trigger)
        except Exception:
            pass

        # Parse duration hours and idle timeout setting from trigger value (format: timestamp:hours:idle)
        tb_hours_str = "1"
        tb_idle_str = "1"
        parts = tb_start_trigger.split(":")
        if len(parts) > 1:
            tb_hours_str = parts[1]
        if len(parts) > 2:
            tb_idle_str = parts[2]
        try:
            hours_val = int(tb_hours_str)
            if hours_val < 1:
                hours_val = 1
            if hours_val > 24:
                hours_val = 24
        except ValueError:
            hours_val = 1

        formatted_time = f"{hours_val:02d}:00:00"
        tb_logdir = logdir
        try:
            tb_logdir = os.path.relpath(logdir, job_dir)
        except Exception:
            pass
        if not tb_logdir or tb_logdir == ".":
            tb_logdir = "lightning_logs"
        slurm_directives = get_train_slurm_directives(train_job_id, job_dir)
        slurm_directives_block = f"{slurm_directives}\n" if slurm_directives else ""
        
        try:
            with open(tb_hours_file, "w") as f:
                f.write(str(hours_val))
            tb_hours_val = hours_val
        except Exception:
            pass

        try:
            with open(tb_idle_file, "w") as f:
                f.write(tb_idle_str)
            tb_idle_val = int(tb_idle_str)
        except Exception:
            pass

        # Cancel any previously running TensorBoard job to avoid leaking resources
        if tb_job_id:
            try:
                subprocess.run([get_slurm_cmd("scancel"), tb_job_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            tb_last_active_file = os.path.join(job_dir, "tb_last_active.txt")
            for fn in [separate_port_file, tb_hours_file, tb_idle_file, tb_last_active_file]:
                if os.path.exists(fn):
                    try:
                        os.remove(fn)
                    except Exception:
                        pass

        # Stage port finder script for the separate TensorBoard job
        port_finder_dst = os.path.join(job_dir, "find_free_tb_port.sh")
        try:
            with open(port_finder_dst, "w", encoding="utf-8", newline="\n") as f:
                f.write(FIND_FREE_TB_PORT_SCRIPT)
            os.chmod(port_finder_dst, 0o755)
        except Exception:
            pass

        # Generate sbatch script content for the separate TensorBoard job
        sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=tb-session
#SBATCH --time={formatted_time}
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=tb_out.%j
#SBATCH --error=tb_error.%j
{slurm_directives_block}
source /etc/profile
if [ -f /sw/lmod/lmod/init/bash ]; then
    source /sw/lmod/lmod/init/bash
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi

module purge
module load WebProxy
module load GCC/13.2.0 tensorboard/2.18.0

cd "{job_dir}"

# Find a free port inline
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

echo "$TB_PORT" > tb_separate_port.txt

echo "Starting TensorBoard on port $TB_PORT..."
tensorboard --logdir="{tb_logdir}" --port=$TB_PORT --bind_all &
TB_PID=$!

python3 -c '
import os, sys, time
port = int(sys.argv[1])
pid = int(sys.argv[2])
enable_idle = sys.argv[3] == "1"
job_dir = sys.argv[4]

last_active_file = os.path.join(job_dir, "tb_last_active.txt")

try:
    with open(last_active_file, "w") as f:
        f.write(str(int(time.time())))
except Exception:
    pass

inactive_count = 0
print(f"Watchdog started for TensorBoard on port {{port}} (PID {{pid}}), enable_idle={{enable_idle}}", flush=True)
while True:
    time.sleep(60)
    if not os.path.exists(f"/proc/{{pid}}"):
        print("Watchdog: TensorBoard process has exited. Exiting...", flush=True)
        break
    port_hex = f"{{port:04X}}"
    active = False
    for path in ["/proc/net/tcp", "/proc/net/tcp6"]:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    for line in f.readlines()[1:]:
                        parts = line.strip().split()
                        if len(parts) >= 4 and parts[1].split(":")[-1] == port_hex and parts[3] == "01":
                            remote_ip = parts[2].split(":")[0]
                            if remote_ip not in ("0100007F", "00000000000000000000000000000001"):
                                active = True
                                break
            except Exception:
                pass
    if active:
        inactive_count = 0
        try:
            with open(last_active_file, "w") as f:
                f.write(str(int(time.time())))
        except Exception:
            pass
    else:
        inactive_count += 1
        print(f"Watchdog: no active connections. Inactive count: {{inactive_count}}/10", flush=True)
    if enable_idle and inactive_count >= 10:
        print("Watchdog: TensorBoard UI webpage has not been open for 10 minutes. Ending TensorBoard job...", flush=True)
        try:
            os.kill(pid, 15)
        except Exception:
            pass
        break
' "$TB_PORT" "$TB_PID" "{tb_idle_str}" "{job_dir}" &

wait $TB_PID
"""

        tb_job_sh = os.path.join(job_dir, "tb_job.sh")
        tb_submit_error_file = os.path.join(job_dir, "tb_submit_error.txt")
        try:
            with open(tb_job_sh, "w") as f:
                f.write(sbatch_content)
            os.chmod(tb_job_sh, 0o755)
        except Exception:
            pass

        # Submit the separate TensorBoard Slurm job
        new_job_id = None
        submit_error = None
        try:
            res = subprocess.run([get_slurm_cmd("sbatch"), tb_job_sh], capture_output=True, text=True)
            if res.returncode == 0:
                import re
                match = re.search(r"Submitted batch job (\d+)", res.stdout)
                if match:
                    new_job_id = match.group(1)
                else:
                    submit_error = (res.stdout or "sbatch succeeded but no job id was returned").strip()
            else:
                submit_error = (res.stderr or res.stdout or "sbatch failed").strip()
        except Exception as exc:
            submit_error = str(exc)

        if submit_error:
            try:
                with open(tb_submit_error_file, "w") as f:
                    f.write(submit_error)
            except Exception:
                pass
            try:
                if os.path.exists(tb_last_trigger_file):
                    os.remove(tb_last_trigger_file)
            except Exception:
                pass
        else:
            try:
                if os.path.exists(tb_submit_error_file):
                    os.remove(tb_submit_error_file)
            except Exception:
                pass

        if new_job_id:
            try:
                with open(tb_job_id_file, "w") as f:
                    f.write(new_job_id)
            except Exception:
                pass
            tb_job_id = new_job_id
            is_tb_job_active = True
            tb_job_state = "PENDING"

# Re-read port values after potential script execution/auto-start
injob_port = None
injob_port_file = os.path.join(job_dir, "tb_port.txt")
if os.path.isfile(injob_port_file):
    try:
        with open(injob_port_file, "r") as f:
            port_val = f.read().strip()
            if port_val:
                injob_port = int(port_val)
    except Exception:
        pass

separate_port = None
if os.path.isfile(separate_port_file):
    try:
        with open(separate_port_file, "r") as f:
            port_val = f.read().strip()
            if port_val:
                separate_port = int(port_val)
    except Exception:
        pass

# Step 1: Check if logdir exists
logdir_exists = os.path.isdir(logdir)
if logdir_exists:
    step1_status = f'<span style="color: #16a34a; font-weight: 600;">Log directory found</span>: <code>{html.escape(logdir)}</code>'
else:
    step1_status = f'<span style="color: #ea580c; font-weight: 600;">Waiting for log directory to be generated</span>: <code>{html.escape(logdir)}</code><div class="tb-tip">Make sure your PyTorch-Lightning training job has started and has begun writing logs.</div>'

# Check if TensorBoard is already running on the target node and port
def check_port(host, port_val):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        res = s.connect_ex((host, port_val))
        s.close()
        return res == 0
    except Exception:
        return False

socket_train_host = "127.0.0.1" if train_node == "localhost" else train_node
socket_separate_host = "127.0.0.1" if separate_node == "localhost" else separate_node

is_injob_running = False
if injob_port is not None:
    is_injob_running = check_port(socket_train_host, injob_port)

is_separate_running = False
if is_tb_job_active and separate_port is not None:
    is_separate_running = check_port(socket_separate_host, separate_port)

tb_job_end_time_str = None
if is_tb_job_active and tb_job_id:
    try:
        # Query start time (%S) and end time (%e) from squeue
        res_time = subprocess.run([get_slurm_cmd("squeue"), "-j", tb_job_id, "-h", "-o", "%S|%e"], capture_output=True, text=True)
        if res_time.returncode == 0 and res_time.stdout.strip():
            parts = res_time.stdout.strip().split('|')
            if len(parts) >= 2:
                start_val = parts[0].strip()
                end_val = parts[1].strip()
                
                # Check if start_val is not "N/A"
                if start_val and start_val != "N/A":
                    import datetime
                    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                        try:
                            # Try parsing the end_val first
                            if end_val and end_val != "N/A":
                                dt_end = datetime.datetime.strptime(end_val, fmt)
                                import time
                                tz_suffix = time.tzname[time.daylight]
                                tb_job_end_time_str = f"{dt_end.strftime('%Y-%m-%d %H:%M:%S')} {tz_suffix}"
                                break
                            else:
                                # Fallback to calculating from start_val + tb_hours_val
                                dt_start = datetime.datetime.strptime(start_val, fmt)
                                dt_end = dt_start + datetime.timedelta(hours=tb_hours_val)
                                import time
                                tz_suffix = time.tzname[time.daylight]
                                tb_job_end_time_str = f"{dt_end.strftime('%Y-%m-%d %H:%M:%S')} {tz_suffix}"
                                break
                        except ValueError:
                            pass
    except Exception:
        pass

last_active_time = None
last_active_file = os.path.join(job_dir, "tb_last_active.txt")
if os.path.isfile(last_active_file):
    try:
        with open(last_active_file, "r") as f:
            last_active_time = int(f.read().strip())
    except Exception:
        pass

countdown_msg = ""
if tb_idle_val == 1 and (is_tb_job_active or is_separate_running) and last_active_time is not None:
    import time
    inactive_sec = int(time.time()) - last_active_time
    if inactive_sec >= 420:
        remaining_sec = 600 - inactive_sec
        if remaining_sec > 0:
            countdown_msg = f' <span id="tb-countdown" style="color: #ef4444; font-weight: 600; margin-left: 8px;">Warning: Webpage is inactive. Job will end in <span id="tb-countdown-secs">{remaining_sec}</span> seconds.</span>'

escaped_dir = html.escape(ckpt_dir)
link = f'<a href="/pun/sys/dashboard/files/fs{escaped_dir}" target="_blank">Open Checkpoint Directory</a>'

tb_submit_error = None
tb_submit_error_file = os.path.join(job_dir, "tb_submit_error.txt")
if os.path.isfile(tb_submit_error_file):
    try:
        with open(tb_submit_error_file, "r") as f:
            tb_submit_error = f.read().strip()
    except Exception:
        pass

still_submitting = False
if tb_start_trigger and not is_tb_job_active and not tb_submit_error:
    if os.path.isfile(tb_last_trigger_file):
        try:
            with open(tb_last_trigger_file, "r") as f:
                if f.read().strip() == tb_start_trigger:
                    still_submitting = True
        except Exception:
            pass

# Open TensorBoard Web UI button styling based on run state
# Define smart button to submit/open the separate TensorBoard job
disabled_attr = ""
disabled_style = ""
if is_tb_job_active or is_separate_running or still_submitting:
    disabled_attr = "disabled"
    disabled_style = "background-color: #f1f5f9; color: #94a3b8; cursor: not-allowed; opacity: 0.8;"

if is_separate_running:
    sep_proxy_url = f'/rnode/{separate_node}/{separate_port}/'
    action_button_html = f"""<a href="{html.escape(sep_proxy_url)}" target="_blank" class="tb-btn-primary">Open TensorBoard Web UI</a>
<button type="button" class="tb-btn-danger" style="margin-left: 8px;" onclick="this.disabled = true; this.innerHTML = '<span class=\\'tb-spinner\\'></span> Ending Session...'; this.style.opacity = '0.7'; this.style.cursor = 'not-allowed'; var input = document.querySelector('input[name=\\'tbStopTrigger\\']'); if (input) {{ var val = Date.now().toString() + \':\' + \'{os.path.basename(job_dir)}\'; var lastVal = input.value; var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; if (setter) {{ setter.call(input, val); }} else {{ input.value = val; }} var tracker = input._valueTracker; if (tracker) {{ tracker.setValue(lastVal); }} input.dispatchEvent(new Event('input', {{ bubbles: true }})); input.dispatchEvent(new Event('change', {{ bubbles: true }})); input.dispatchEvent(new Event('blur', {{ bubbles: true }})); }}">End TensorBoard Job</button>"""
    until_msg = ""
    if tb_job_end_time_str:
        until_msg = f'<div style="font-size: 0.88em; color: #475569; margin-top: 4px; margin-bottom: 12px;">This job is running until <strong>{html.escape(tb_job_end_time_str)}</strong>.{countdown_msg}</div>'
    tb_status_msg = f"""<div style="color: #16a34a; font-weight: 600; margin-bottom: 4px;">TensorBoard job {tb_job_id} is running on node <code>{html.escape(separate_node)}</code> port <code>{separate_port}</code>.</div>
{until_msg}"""
elif is_tb_job_active:
    if tb_job_state == "PENDING":
        action_button_html = f"""<button type="button" class="tb-btn-start tb-btn-disabled" style="background-color: #cbd5e1 !important; color: #64748b !important; cursor: not-allowed !important; pointer-events: none !important;" disabled><span class="tb-spinner"></span> Job Pending ({tb_job_id})...</button>
<button type="button" class="tb-btn-danger" style="margin-left: 8px;" onclick="this.disabled = true; this.innerHTML = '<span class=\\'tb-spinner\\'></span> Ending Session...'; this.style.opacity = '0.7'; this.style.cursor = 'not-allowed'; var input = document.querySelector('input[name=\\'tbStopTrigger\\']'); if (input) {{ var val = Date.now().toString() + \':\' + \'{os.path.basename(job_dir)}\'; var lastVal = input.value; var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; if (setter) {{ setter.call(input, val); }} else {{ input.value = val; }} var tracker = input._valueTracker; if (tracker) {{ tracker.setValue(lastVal); }} input.dispatchEvent(new Event('input', {{ bubbles: true }})); input.dispatchEvent(new Event('change', {{ bubbles: true }})); input.dispatchEvent(new Event('blur', {{ bubbles: true }})); }}">End TensorBoard Job</button>"""
        tb_status_msg = f'<div style="color: #ea580c; font-weight: 600; margin-bottom: 12px;">TensorBoard job {tb_job_id} is PENDING.</div>'
    else:
        action_button_html = f"""<button type="button" class="tb-btn-start tb-btn-disabled" style="background-color: #cbd5e1 !important; color: #64748b !important; cursor: not-allowed !important; pointer-events: none !important;" disabled><span class="tb-spinner"></span> Starting TensorBoard...</button>
<button type="button" class="tb-btn-danger" style="margin-left: 8px;" onclick="this.disabled = true; this.innerHTML = '<span class=\\'tb-spinner\\'></span> Ending Session...'; this.style.opacity = '0.7'; this.style.cursor = 'not-allowed'; var input = document.querySelector('input[name=\\'tbStopTrigger\\']'); if (input) {{ var val = Date.now().toString() + \':\' + \'{os.path.basename(job_dir)}\'; var lastVal = input.value; var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; if (setter) {{ setter.call(input, val); }} else {{ input.value = val; }} var tracker = input._valueTracker; if (tracker) {{ tracker.setValue(lastVal); }} input.dispatchEvent(new Event('input', {{ bubbles: true }})); input.dispatchEvent(new Event('change', {{ bubbles: true }})); input.dispatchEvent(new Event('blur', {{ bubbles: true }})); }}">End TensorBoard Job</button>"""
        until_msg = ""
        if tb_job_end_time_str:
            until_msg = f'<div style="font-size: 0.88em; color: #475569; margin-top: 4px; margin-bottom: 12px;">This job is running until <strong>{html.escape(tb_job_end_time_str)}</strong>.{countdown_msg}</div>'
        tb_status_msg = f"""<div style="color: #ea580c; font-weight: 600; margin-bottom: 4px;">TensorBoard job {tb_job_id} is running, starting server...</div>
{until_msg}"""
elif still_submitting:
    action_button_html = f"""<button type="button" class="tb-btn-start tb-btn-disabled" style="background-color: #cbd5e1 !important; color: #64748b !important; cursor: not-allowed !important; pointer-events: none !important;" disabled><span class="tb-spinner"></span> Submitting Job...</button>"""
    tb_status_msg = '<div style="color: #ea580c; font-weight: 600; margin-bottom: 12px;">Submitting TensorBoard job to Slurm...</div>'
else:
    error_msg = ""
    if tb_submit_error:
        error_msg = f'<div style="color: #dc2626; font-weight: 600; margin-bottom: 12px;">Failed to submit TensorBoard job: <code>{html.escape(tb_submit_error)}</code></div>'
    action_button_html = f"""<button type="button" class="tb-btn-start" onclick="window.__tb_submitting = true; this.disabled = true; this.innerHTML = '<span class=\\'tb-spinner\\'></span> Submitting Job...'; this.style.opacity = '0.7'; this.style.cursor = 'not-allowed'; var hoursInput = this.nextElementSibling; var hours = hoursInput ? hoursInput.value : '1'; var idleCheckbox = this.parentElement.querySelector('#tb-idle-checkbox'); var idle = (idleCheckbox && idleCheckbox.checked) ? '1' : '0'; var input = document.querySelector('input[name=\\'tbStartTrigger\\']'); if (input) {{ var val = Date.now().toString() + ':' + hours + ':' + idle; var lastVal = input.value; var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; if (setter) {{ setter.call(input, val); }} else {{ input.value = val; }} var tracker = input._valueTracker; if (tracker) {{ tracker.setValue(lastVal); }} input.dispatchEvent(new Event('input', {{ bubbles: true }})); input.dispatchEvent(new Event('change', {{ bubbles: true }})); input.dispatchEvent(new Event('blur', {{ bubbles: true }})); }}">Start TensorBoard</button>"""
    tb_status_msg = f'{error_msg}<div style="color: #64748b; font-weight: 600; margin-bottom: 12px;">TensorBoard not running currently</div>'

idle_checked_attr = "checked" if tb_idle_val == 1 else ""
step2_content = f"""
{tb_status_msg}
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
    {action_button_html}
    <input type="number" class="tb-hours-input" value="{tb_hours_val}" min="1" max="24" {disabled_attr} style="width: 70px; padding: 8px 12px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.88em; font-weight: 500; color: #1e293b; text-align: center; {disabled_style}" />
    <span style="font-size: 0.88em; color: #64748b; font-weight: 500; margin-right: 12px;">hour(s)</span>
    <label style="display: inline-flex; align-items: center; gap: 6px; font-size: 0.88em; color: #475569; font-weight: 500; cursor: pointer; user-select: none;">
        <input type="checkbox" id="tb-idle-checkbox" {idle_checked_attr} {disabled_attr} style="width: 16px; height: 16px; border: 1px solid #cbd5e1; border-radius: 4px; cursor: pointer;" />
        End job after 10 min inactivity
    </label>
</div>
<div class="tb-tip" style="color: #475569;">TensorBoard will be submitted as a separate Slurm job, and you will be charged 1 SU per hour.</div>
"""

# Render Step 1 (Active Server Status) based on training job status
if is_injob_running:
    injob_proxy_url = f'/rnode/{train_node}/{injob_port}/'
    escaped_url = html.escape(injob_proxy_url)
    tb_source_label = f"In-Job TensorBoard on <code>{html.escape(train_node)}</code> port <code>{injob_port}</code>"
    step1_content = f"""
    <div style="color: #16a34a; font-weight: 600; margin-bottom: 12px;">TensorBoard server is active and running inside the training job allocation on node <code>{html.escape(train_node)}</code> port <code>{injob_port}</code>.</div>
    <div style="margin-bottom: 16px;">
        <a href="{escaped_url}" target="_blank" class="tb-btn-primary">Open TensorBoard Web UI</a>
    </div>
    
    <div class="tb-ui-wrapper" style="margin-bottom: 16px;">
      <div class="tb-ui-iframe-container">
        <iframe src="{escaped_url}" class="tb-ui-iframe" loading="lazy" sandbox="allow-same-origin allow-scripts allow-popups allow-forms" title="TensorBoard UI"></iframe>
      </div>
    </div>
    """
elif is_train_job_active:
    if train_job_state == "PENDING":
        step1_content = f"""
        <div style="color: #ea580c; font-weight: 600; margin-bottom: 12px;">Training job {train_job_id} is PENDING.</div>
        <div class="tb-tip" style="color: #475569;">In-job TensorBoard will start up automatically once the training job begins running on the compute node.</div>
        """
    else:
        step1_content = f"""
        <div style="color: #ea580c; font-weight: 600; margin-bottom: 12px;">Training job {train_job_id} is active, but in-job TensorBoard is not detected yet...</div>
        """
elif not train_job_id:
    step1_content = f"""
    <div style="color: #ea580c; font-weight: 600; margin-bottom: 12px;">Waiting for SLURM job IDs — training job may still be starting.</div>
    """
else:
    step1_content = f"""
    <div style="color: #ef4444; font-weight: 600; margin-bottom: 12px;">Training job is not running.</div>
    """

show_injob = is_train_job_active or not train_job_id

if show_injob:
    tb_steps_html = f"""
  <!-- Step 1 -->
  <div class="tb-step-item">
    <div>
      <span class="tb-step-title">In-Job TensorBoard Session</span>
    </div>
    <div class="tb-step-content">
      {step1_content}
    </div>
  </div>
"""
else:
    tb_steps_html = f"""
  <!-- Step 2 -->
  <div class="tb-step-item">
    <div>
      <span class="tb-step-title">TensorBoard Session</span>
    </div>
    <div class="tb-step-content">
      {step2_content}
    </div>
  </div>
"""

reset_btn_html = ""
if custom_logdir:
    reset_btn_html = """<button type="button" class="tb-btn-danger" style="padding: 6px 12px; font-size: 0.82em; margin-left: 8px;" onclick="var input = document.querySelector('input[name=\\'tbLogDir\\']'); if (input) { var val = ''; var lastVal = input.value; var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; if (setter) { setter.call(input, val); } else { input.value = val; } var tracker = input._valueTracker; if (tracker) { tracker.setValue(lastVal); } input.dispatchEvent(new Event('input', { bubbles: true })); input.dispatchEvent(new Event('change', { bubbles: true })); input.dispatchEvent(new Event('blur', { bubbles: true })); }">Reset to Default</button>"""

tb_logdir_panel_html = f"""
  <div class="tb-logdir-panel" style="margin-bottom: 16px; padding: 12px; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;">
    <div style="font-size: 0.85em; font-weight: 600; color: #475569; margin-bottom: 4px;">Log Directory Address</div>
    <div style="font-family: monospace; font-size: 0.9em; word-break: break-all; color: #0f172a; background: #fff; padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 4px; margin-bottom: 8px;">{html.escape(logdir)}</div>
    <div style="display: flex; gap: 8px; align-items: center;">
      <button type="button" class="tb-btn-secondary" style="padding: 6px 12px; font-size: 0.82em; background-color: #003c71;" onclick="var input = document.querySelector('input[name=\\'tbLogDir\\']'); if (input) {{ var container = input.closest('.form-group') || input.closest('.form-row') || input.closest('[class*=\\'rowContainer\\']') || input.parentElement; var btn = container ? container.querySelector('button') : null; if (btn) btn.click(); }}">Change Directory</button>
      {reset_btn_html}
    </div>
  </div>
"""

print(f"""
<style>
  .tb-container {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px;
    margin-top: 15px;
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    clear: both;
  }}
  .tb-header {{
    font-size: 1.05em;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 16px;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 8px;
  }}
  .tb-step-item {{
    margin-bottom: 18px;
  }}
  .tb-step-item:last-child {{
    margin-bottom: 0;
  }}
  .tb-step-num {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    background-color: #003c71;
    color: #ffffff;
    border-radius: 50%;
    font-size: 0.8em;
    font-weight: 700;
    margin-right: 8px;
  }}
  .tb-step-title {{
    font-size: 0.9em;
    font-weight: 600;
    color: #1e293b;
    display: inline-block;
    vertical-align: middle;
  }}
  .tb-step-content {{
    margin-top: 6px;
    margin-left: 0;
    font-size: 0.88em;
  }}
  .tb-code-box {{
    background-color: #0f172a;
    color: #f1f5f9;
    border-radius: 6px;
    padding: 10px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.85em;
    line-height: 1.45;
    word-break: break-all;
    white-space: pre-wrap;
    margin-top: 6px;
  }}
  .tb-btn-primary {{
    display: inline-block;
    background-color: #10b981;
    color: #ffffff !important;
    font-weight: 600;
    text-decoration: none !important;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.88em;
    transition: background-color 0.2s;
    border: none;
  }}
  .tb-btn-primary:hover {{
    background-color: #059669;
  }}
  .tb-btn-disabled {{
    background-color: #cbd5e1 !important;
    color: #64748b !important;
    cursor: not-allowed !important;
    pointer-events: none !important;
  }}
  .tb-btn-secondary {{
    display: inline-block;
    background-color: #64748b;
    color: #ffffff !important;
    font-weight: 600;
    text-decoration: none !important;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.88em;
    transition: background-color 0.2s, transform 0.1s;
    border: none;
    cursor: pointer;
  }}
  .tb-btn-secondary:hover {{
    background-color: #475569;
  }}
  .tb-btn-secondary:active {{
    transform: scale(0.98);
  }}
  .tb-btn-start {{
    display: inline-block;
    background-color: #003c71;
    color: #ffffff !important;
    font-weight: 600;
    text-decoration: none !important;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.88em;
    transition: background-color 0.2s, transform 0.1s;
    border: none;
    cursor: pointer;
  }}
  .tb-btn-start:hover {{
    background-color: #002850;
  }}
  .tb-btn-start:active {{
    transform: scale(0.98);
  }}
  .tb-btn-danger {{
    display: inline-block;
    background-color: #ef4444;
    color: #ffffff !important;
    font-weight: 600;
    text-decoration: none !important;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.88em;
    transition: background-color 0.2s, transform 0.1s;
    border: none;
    cursor: pointer;
  }}
  .tb-btn-danger:hover {{
    background-color: #dc2626;
  }}
  .tb-btn-danger:active {{
    transform: scale(0.98);
  }}
  .tb-spinner {{
    display: inline-block;
    width: 12px;
    height: 12px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: tb-spin 1s ease-in-out infinite;
    margin-right: 6px;
    vertical-align: middle;
  }}
  @keyframes tb-spin {{
    to {{ transform: rotate(360deg); }}
  }}
  input[name="tbStartTrigger"],
  input[name="tbStopTrigger"] {{
    display: none !important;
  }}
  .tb-launch-wrapper {{
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    clear: both;
  }}
  .tb-tip {{
    font-size: 0.82em;
    color: #64748b;
    margin-top: 4px;
    line-height: 1.4;
  }}
  .tb-details {{
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    background-color: #f8fafc;
    margin-top: 14px;
  }}
  .tb-details-summary {{
    padding: 10px 12px;
    font-weight: 600;
    color: #475569;
    cursor: pointer;
    outline: none;
    user-select: none;
    font-size: 0.9em;
  }}
  .tb-details-summary:hover {{
    color: #0f172a;
    background-color: #f1f5f9;
    border-radius: 5px;
  }}
  .tb-details[open] .tb-details-summary {{
    border-bottom: 1px solid #e2e8f0;
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
  }}
  .tb-details-content {{
    padding: 12px;
    color: #334155;
    font-size: 0.88em;
  }}
  .ckpt-box {{
    display: inline-block;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 12px;
    font-family: monospace;
    margin-top: 5px;
    margin-bottom: 10px;
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
  }}
  .ckpt-box a {{
    color: #003c71;
    text-decoration: none;
    font-weight: 600;
  }}
  .ckpt-box a:hover {{
    text-decoration: underline;
  }}
  .tb-ui-wrapper {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    margin-top: 10px;
  }}
  .tb-ui-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .tb-ui-status {{
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.85em;
    color: #16a34a;
    font-weight: 600;
  }}
  .tb-ui-status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #16a34a;
    flex-shrink: 0;
    display: inline-block;
  }}
  .tb-ui-iframe-container {{
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    overflow: hidden;
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .tb-ui-iframe {{
    width: 100%;
    height: 600px;
    border: none;
    display: block;
  }}
  .tb-ui-source {{
    font-size: 0.8em;
    color: #94a3b8;
    margin-top: 6px;
  }}
</style>

<div class="tb-launch-wrapper">

<div class="tb-container">
  <div class="tb-header">TensorBoard Monitoring</div>
  
  {tb_logdir_panel_html}
  {tb_steps_html}
</div>

<div class="ckpt-box">{link}</div>

</div>


<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" onload="var wrapper = this.previousElementSibling; var pageText = document.body ? document.body.textContent : ''; var jobSubmitted = pageText.indexOf('Job Pending') !== -1 || pageText.indexOf('is PENDING') !== -1 || pageText.indexOf('is running on node') !== -1 || pageText.indexOf('starting server') !== -1; if (jobSubmitted) {{ window.__tb_submitting = false; var input = document.querySelector('input[name=\\'tbStartTrigger\\']'); if (input && input.value !== '') {{ var val = ''; var lastVal = input.value; var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; if (setter) {{ setter.call(input, val); }} else {{ input.value = val; }} var tracker = input._valueTracker; if (tracker) {{ tracker.setValue(lastVal); }} input.dispatchEvent(new Event('input', {{ bubbles: true }})); input.dispatchEvent(new Event('change', {{ bubbles: true }})); input.dispatchEvent(new Event('blur', {{ bubbles: true }})); }} }} else if (window.__tb_submitting && wrapper) {{ var startBtn = wrapper.querySelector('.tb-btn-start'); if (startBtn && startBtn.textContent.indexOf('Start TensorBoard') !== -1) {{ startBtn.disabled = true; startBtn.innerHTML = '<span class=\\'tb-spinner\\'></span> Submitting Job...'; startBtn.style.opacity = '0.7'; startBtn.style.cursor = 'not-allowed'; startBtn.classList.add('tb-btn-disabled'); }} }} var stopInput = document.querySelector('input[name=\\'tbStopTrigger\\']'); if (stopInput && stopInput.value !== '') {{ var val = ''; var lastVal = stopInput.value; var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; if (setter) {{ setter.call(stopInput, val); }} else {{ stopInput.value = val; }} var tracker = stopInput._valueTracker; if (tracker) {{ tracker.setValue(lastVal); }} stopInput.dispatchEvent(new Event('input', {{ bubbles: true }})); stopInput.dispatchEvent(new Event('change', {{ bubbles: true }})); stopInput.dispatchEvent(new Event('blur', {{ bubbles: true }})); }} ['tbStartTrigger', 'tbStopTrigger', 'tbLogDir'].forEach(function(name) {{ var el = document.querySelector('input[name=\\'' + name + '\\']'); if (el) {{ var container = el.closest('.form-group') || el.closest('.form-row') || el.closest('[class*=\\'rowContainer\\']') || el.parentElement; if (container) {{ container.style.setProperty('display', 'none', 'important'); }} }} }}); if (window.__tb_countdown_interval) {{ clearInterval(window.__tb_countdown_interval); window.__tb_countdown_interval = null; }} var cdSecs = document.getElementById('tb-countdown-secs'); if (cdSecs) {{ window.__tb_countdown_interval = setInterval(function() {{ var el = document.getElementById('tb-countdown-secs'); if (el) {{ var sec = parseInt(el.textContent, 10); if (sec > 0) {{ el.textContent = sec - 1; }} else {{ clearInterval(window.__tb_countdown_interval); window.__tb_countdown_interval = null; }} }} else {{ clearInterval(window.__tb_countdown_interval); window.__tb_countdown_interval = null; }} }}, 1000); }}" style="display: none;" />

""")
EOF
