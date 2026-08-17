#!/bin/bash
# Combined monitor dashboard: Slurm summary + output/error logs + shared UI scripts.

export LOCATION="${LOCATION:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

html_escape() {
    sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g'
}

emit_slurm_summary() {
    if [ -z "$LOCATION" ] || [ ! -d "$LOCATION" ]; then
        echo "<div style='font-size:0.83em;color:#6c757d;padding:4px 2px'>Workflow directory not available yet.</div>"
        return 0
    fi

    local JOBIDS_FILE="${LOCATION}/slurm_jobids.txt"
    local -a JOBIDS=()

    if [ -f "$JOBIDS_FILE" ]; then
        mapfile -t JOBIDS < <(grep -v '^\s*$' "$JOBIDS_FILE")
    fi

    if [ ${#JOBIDS[@]} -eq 0 ]; then
        shopt -s nullglob
        local f base
        for f in "$LOCATION"/out.*; do
            base="$(basename "$f")"
            if [[ "$base" =~ ^out\.([0-9]+)$ ]]; then
                JOBIDS+=("${BASH_REMATCH[1]}")
            fi
        done
        shopt -u nullglob
    fi

    if [ ${#JOBIDS[@]} -eq 0 ]; then
        echo "<div style='font-size:0.83em;color:#6c757d;padding:4px 2px'>Waiting for SLURM job IDs — training job may still be starting.</div>"
        return 0
    fi

    local rows=""
    local has_running=false
    local has_failed=false
    local has_pending=false
    local JID ROW STATUS ROW_CLASS
    local C1 C2 C3 C4 C5

    for JID in "${JOBIDS[@]}"; do
        JID="${JID//$'\r'/}"
        JID="$(echo "$JID" | xargs)"
        [ -z "$JID" ] && continue

        ROW=$(squeue -j "$JID" -h -o "%i|%j|%T|%M|%N" 2>/dev/null | head -1)

        if [ -n "$ROW" ]; then
            STATUS=$(echo "$ROW" | cut -d'|' -f3)
            case "$STATUS" in
                RUNNING) ROW_CLASS="table-success"; has_running=true ;;
                PENDING) ROW_CLASS="table-warning"; has_pending=true ;;
                *)       ROW_CLASS="" ;;
            esac
        else
            RAW_SACCT=$(sacct -j "$JID" --noheader --format=JobID,JobName,State,Elapsed,NodeList,Submit --parsable2 2>/dev/null)
            ROW=$(python3 -c "
import sys, os, datetime
raw_sacct = sys.argv[1]
location = sys.argv[2]
target_jid = sys.argv[3]
ref_path = os.path.join(location, 'slurm_jobids.txt')
if not os.path.exists(ref_path):
    ref_path = location
ref_time = os.path.getmtime(ref_path) if os.path.exists(ref_path) else None
best_row = ''
min_diff = float('inf')
for line in raw_sacct.strip().split('\n'):
    if not line.strip():
        continue
    parts = line.split('|')
    if len(parts) < 6:
        continue
    jid, name, state, elapsed, nodelist, submit_str = parts[:6]
    if jid != target_jid:
        continue
    try:
        dt = datetime.datetime.strptime(submit_str.strip(), '%Y-%m-%dT%H:%M:%S')
        submit_time = dt.timestamp()
    except Exception:
        submit_time = None
    if ref_time is not None and submit_time is not None:
        diff = abs(ref_time - submit_time)
        if diff < min_diff:
            min_diff = diff
            best_row = f'{jid}|{name}|{state}|{elapsed}|{nodelist}'
    elif not best_row:
        best_row = f'{jid}|{name}|{state}|{elapsed}|{nodelist}'
if best_row:
    print(best_row)
" "$RAW_SACCT" "$LOCATION" "$JID")
            STATUS=$(echo "$ROW" | cut -d'|' -f3 | xargs)
            case "$STATUS" in
                COMPLETED)   ROW_CLASS="table-success" ;;
                PENDING)     ROW_CLASS="table-warning"; has_pending=true ;;
                FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                             ROW_CLASS="table-danger"; has_failed=true ;;
                *)           ROW_CLASS="" ;;
            esac
        fi

        C1=$(echo "$ROW" | cut -d'|' -f1 | html_escape)
        C2=$(echo "$ROW" | cut -d'|' -f2 | html_escape)
        C3=$(echo "$ROW" | cut -d'|' -f3 | html_escape)
        C4=$(echo "$ROW" | cut -d'|' -f4 | html_escape)
        C5=$(echo "$ROW" | cut -d'|' -f5 | html_escape)
        JID_ESC=$(echo "$JID" | html_escape)

        rows+="<tr class='${ROW_CLASS}'><td>${C1:-$JID_ESC}</td><td>${C2:-—}</td><td>${C3:-unknown}</td><td>${C4:-—}</td><td>${C5:-—}</td></tr>"
    done

    local dot_color status_label
    if $has_failed; then
        dot_color="#dc3545"
        status_label="Failed — check output/error logs below."
    elif $has_running; then
        dot_color="#0d6efd"
        status_label="Running — dashboard auto-refreshes every 10 seconds."
    elif $has_pending; then
        dot_color="#ffc107"
        status_label="Pending — queued, waiting to start."
    else
        dot_color="#198754"
        status_label="Completed — see logs below."
    fi

    echo "<div style='display:flex;align-items:center;gap:7px;margin-bottom:8px;font-size:0.83em;color:#495057'>"
    echo "  <span style='width:8px;height:8px;border-radius:50%;background:${dot_color};flex-shrink:0;display:inline-block'></span>"
    echo "  ${status_label}"
    echo "</div>"

    echo "<table class='table table-sm table-bordered mb-0' style='font-size:0.85em'>"
    echo "<thead class='table-light'><tr><th>Job ID</th><th>Name</th><th>Status</th><th>Elapsed</th><th>Node(s)</th></tr></thead>"
    echo "<tbody>${rows}</tbody></table>"
}

emit_epoch_progress() {
    if [ -z "$LOCATION" ] || [ ! -d "$LOCATION" ]; then
        return 0
    fi

    local JOBIDS_FILE="${LOCATION}/slurm_jobids.txt"
    local -a JOBIDS=()

    if [ -f "$JOBIDS_FILE" ]; then
        mapfile -t JOBIDS < <(grep -v '^\s*$' "$JOBIDS_FILE")
    fi

    if [ ${#JOBIDS[@]} -eq 0 ]; then
        shopt -s nullglob
        local f base
        for f in "$LOCATION"/out.*; do
            base="$(basename "$f")"
            if [[ "$base" =~ ^out\.([0-9]+)$ ]]; then
                JOBIDS+=("${BASH_REMATCH[1]}")
            fi
        done
        shopt -u nullglob
    fi

    if [ ${#JOBIDS[@]} -eq 0 ]; then
        return 0
    fi

    local LATEST_JID="${JOBIDS[-1]}"
    LATEST_JID="${LATEST_JID//$'\r'/}"
    LATEST_JID="$(echo "$LATEST_JID" | xargs)"
    
    local STATUS="unknown"
    local ROW
    ROW=$(squeue -j "$LATEST_JID" -h -o "%T" 2>/dev/null | head -1 | xargs)
    if [ -n "$ROW" ]; then
        STATUS="$ROW"
    else
        local RAW_SACCT
        RAW_SACCT=$(sacct -j "$LATEST_JID" --noheader --format=State 2>/dev/null | head -1 | xargs)
        if [ -n "$RAW_SACCT" ]; then
            STATUS="$RAW_SACCT"
        fi
    fi

    if [ "$STATUS" = "PENDING" ]; then
        echo "<div class='pt-epoch-section' style='margin-top:14px'>"
        echo "  <div style='font-size:0.9em;font-weight:600;color:#495057;margin-bottom:8px'>Training Progress</div>"
        echo "  <div class='card' style='border: 1px solid #dee2e6; border-radius: 8px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); overflow: hidden;'>"
        echo "    <div class='card-body' style='padding: 15px; display: flex; flex-direction: column; gap: 8px;'>"
        echo "      <div style='display:flex; justify-content:space-between; align-items:center; font-size:0.9em; font-weight:600;'>"
        echo "        <span>Status: <span style='color:#ea580c;'>Pending for running</span></span>"
        echo "        <span style='color:#868e96;'>0%</span>"
        echo "      </div>"
        echo "      <div style='background: #e9ecef; border-radius: 4px; height: 10px; overflow: hidden;'>"
        echo "        <div style='background: #cbd5e1; width: 0%; height: 100%;'></div>"
        echo "      </div>"
        echo "    </div>"
        echo "  </div>"
        echo "</div>"
        return 0
    fi

    local LATEST_LOG=""
    if [ -f "$LOCATION/out.$LATEST_JID" ]; then
        LATEST_LOG="$LOCATION/out.$LATEST_JID"
    fi
    if [ -z "$LATEST_LOG" ]; then
        LATEST_LOG=$(ls -t "$LOCATION"/out.* 2>/dev/null | head -n 1)
    fi

    if [ -z "$LATEST_LOG" ] || [ ! -f "$LATEST_LOG" ]; then
        return 0
    fi

    local max_epochs="10"
    if [ -f "$LOCATION/train.py" ]; then
        local parsed_max
        parsed_max=$(grep -E '^\s*MAX_EPOCHS\s*=' "$LOCATION/train.py" | grep -o -E '[0-9]+' | head -n 1)
        if [ -n "$parsed_max" ]; then
            max_epochs="$parsed_max"
        fi
    fi

    local PARSED
    PARSED=$(python3 - <<EOF "$LATEST_LOG" "$max_epochs" "$STATUS"
import sys
import os
import re

log_path = sys.argv[1]
max_epochs = int(sys.argv[2])
status = sys.argv[3].upper()

if status in ("COMPLETED", "COMPLETE"):
    print(f"{max_epochs}|{max_epochs}|100|100|—")
    sys.exit(0)

try:
    with open(log_path, "rb") as f:
        try:
            f.seek(-102400, os.SEEK_END)
        except OSError:
            f.seek(0)
        content = f.read().decode("utf-8", errors="ignore")
except Exception:
    content = ""

matches = re.findall(r'Epoch\s+(\d+)(?:/\d+)?:?\s*(\d+)%', content)

curr_ep = 0
ep_percent = 0

if matches:
    curr_ep_str, ep_percent_str = matches[-1]
    try:
        curr_ep = int(curr_ep_str)
        ep_percent = int(ep_percent_str)
    except ValueError:
        pass
else:
    simple_epochs = re.findall(r'Epoch\s+(\d+)', content)
    if simple_epochs:
        try:
            curr_ep = int(simple_epochs[-1])
        except ValueError:
            pass

display_epoch = curr_ep + 1
if display_epoch > max_epochs:
    display_epoch = max_epochs

if status in ("COMPLETED", "COMPLETE") or display_epoch == max_epochs and ep_percent == 100:
    overall_percent = 100
    display_epoch = max_epochs
    ep_percent = 100
else:
    overall_percent = int(((curr_ep + (ep_percent / 100.0)) / max_epochs) * 100)

speed_matches = re.findall(r'(\d+(?:\.\d+)?\s*(?:it/s|s/it))', content)
speed = speed_matches[-1] if speed_matches else "—"

print(f"{display_epoch}|{max_epochs}|{ep_percent}|{overall_percent}|{speed}")
EOF
)

    local display_epoch max_ep ep_percent overall_percent speed
    display_epoch=$(echo "$PARSED" | cut -d'|' -f1)
    max_ep=$(echo "$PARSED" | cut -d'|' -f2)
    ep_percent=$(echo "$PARSED" | cut -d'|' -f3)
    overall_percent=$(echo "$PARSED" | cut -d'|' -f4)
    speed=$(echo "$PARSED" | cut -d'|' -f5)

    if [ -z "$display_epoch" ] || [ -z "$max_ep" ]; then
        return 0
    fi

    echo "<div class='pt-epoch-section' style='margin-top:14px'>"
    echo "  <div style='font-size:0.9em;font-weight:600;color:#495057;margin-bottom:8px'>Training Progress</div>"
    echo "  <div class='card' style='border: 1px solid #dee2e6; border-radius: 8px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); overflow: hidden;'>"
    echo "    <div class='card-body' style='padding: 15px; display: flex; flex-direction: column; gap: 8px;'>"
    
    echo "      <div style='display:flex; justify-content:space-between; align-items:center; font-size:0.9em; font-weight:600;'>"
    echo "        <span>Epoch: <span style='color:#0d6efd;'>${display_epoch}</span> / ${max_ep}</span>"
    if [ -n "$speed" ] && [ "$speed" != "—" ]; then
        echo "        <span style='font-size:0.85em; font-weight:normal; color:#6c757d;'>Speed: <span style='font-weight:600; color:#495057;'>${speed}</span></span>"
    fi
    echo "        <span style='color:#198754;'>${ep_percent}%</span>"
    echo "      </div>"

    echo "      <div style='background: #e9ecef; border-radius: 4px; height: 10px; overflow: hidden;'>"
    echo "        <div style='background: #198754; width: ${ep_percent}%; height: 100%; transition: width 0.4s ease;'></div>"
    echo "      </div>"

    echo "    </div>"
    echo "  </div>"
    echo "</div>"
}

emit_gpu_summary() {
    if [ -z "$LOCATION" ] || [ ! -d "$LOCATION" ]; then
        return 0
    fi

    local JOBIDS_FILE="${LOCATION}/slurm_jobids.txt"
    local -a JOBIDS=()

    if [ -f "$JOBIDS_FILE" ]; then
        mapfile -t JOBIDS < <(grep -v '^\s*$' "$JOBIDS_FILE")
    fi

    if [ ${#JOBIDS[@]} -eq 0 ]; then
        shopt -s nullglob
        local f base
        for f in "$LOCATION"/out.*; do
            base="$(basename "$f")"
            if [[ "$base" =~ ^out\.([0-9]+)$ ]]; then
                JOBIDS+=("${BASH_REMATCH[1]}")
            fi
        done
        shopt -u nullglob
    fi

    if [ ${#JOBIDS[@]} -eq 0 ]; then
        return 0
    fi

    local has_running_gpu=false
    local gpu_html=""

    for JID in "${JOBIDS[@]}"; do
        JID="${JID//$'\r'/}"
        JID="$(echo "$JID" | xargs)"
        [ -z "$JID" ] && continue

        # Get status and node name from squeue
        local ROW
        ROW=$(squeue -j "$JID" -h -o "%T|%N" 2>/dev/null | head -1)
        [ -z "$ROW" ] && continue

        local STATUS NODE_LIST
        STATUS=$(echo "$ROW" | cut -d'|' -f1)
        NODE_LIST=$(echo "$ROW" | cut -d'|' -f2)

        if [ "$STATUS" = "RUNNING" ] && [ -n "$NODE_LIST" ] && [ "$NODE_LIST" != "(null)" ]; then
            # Expand node list using scontrol if possible
            local NODE
            NODE=$(scontrol show hostnames "$NODE_LIST" 2>/dev/null | head -1)
            [ -z "$NODE" ] && NODE="$NODE_LIST"

            # Query GPU stats on the compute node using the existing job allocation
            local RAW_GPU
            RAW_GPU=$(timeout 5 srun --jobid="${JID}" --nodelist="${NODE}" --nodes=1 --ntasks=1 --overlap -t 0:03 python3 - <<'EOF' 2>/dev/null
import subprocess
import shutil
import re
import json

def get_nvidia_gpus():
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]).decode("utf-8", errors="ignore")
        gpus = []
        for line in out.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 6:
                    gpus.append({
                        "type": "NVIDIA",
                        "index": parts[0],
                        "name": parts[1],
                        "gpu_util": parts[2],
                        "mem_util": parts[3],
                        "mem_used": parts[4],
                        "mem_total": parts[5]
                    })
        return gpus
    except Exception as e:
        return []

def get_intel_gpus():
    if shutil.which("xpumcli"):
        try:
            try:
                out = subprocess.check_output(["xpumcli", "stats", "--json"]).decode("utf-8", errors="ignore")
                data = json.loads(out)
                gpus = []
                if isinstance(data, list):
                    for dev in data:
                        dev_id = dev.get("device_id", 0)
                        dev_name = dev.get("device_name", "Intel PVC")
                        telemetry = dev.get("telemetry", {})
                        gpu_util = telemetry.get("gpu_utilization", {}).get("value", 0)
                        mem_util = telemetry.get("memory_utilization", {}).get("value", 0)
                        mem_used = telemetry.get("memory_used", {}).get("value", 0)
                        mem_total = telemetry.get("memory_total", {}).get("value", 131072)
                        gpus.append({
                            "type": "Intel",
                            "index": str(dev_id),
                            "name": dev_name,
                            "gpu_util": str(gpu_util),
                            "mem_util": str(mem_util),
                            "mem_used": str(int(mem_used)),
                            "mem_total": str(int(mem_total))
                        })
                return gpus
            except Exception:
                out = subprocess.check_output(["xpumcli", "stats"]).decode("utf-8", errors="ignore")
                gpus = []
                dev_blocks = re.split(r'Device ID:', out)
                for block in dev_blocks[1:]:
                    lines = block.split('\n')
                    dev_id = lines[0].strip()
                    gpu_util = "0"
                    mem_util = "0"
                    mem_used = "0"
                    mem_total = "131072"
                    for line in lines:
                        if "GPU Utilization" in line:
                            m = re.search(r'(\d+)', line)
                            if m: gpu_util = m.group(1)
                        elif "Memory Utilization" in line:
                            m = re.search(r'(\d+)', line)
                            if m: mem_util = m.group(1)
                        elif "Memory Used" in line:
                            m = re.search(r'(\d+)', line)
                            if m: mem_used = m.group(1)
                        elif "Memory Total" in line:
                            m = re.search(r'(\d+)', line)
                            if m: mem_total = m.group(1)
                    gpus.append({
                        "type": "Intel",
                        "index": dev_id,
                        "name": "Intel Data Center GPU Max",
                        "gpu_util": gpu_util,
                        "mem_util": mem_util,
                        "mem_used": mem_used,
                        "mem_total": mem_total
                    })
                return gpus
        except Exception as e:
            return []
    
    if shutil.which("xpu-smi"):
        try:
            out = subprocess.check_output(["xpu-smi", "stats", "-j"]).decode("utf-8", errors="ignore")
            data = json.loads(out)
            gpus = []
            devices = data.get("devices", [])
            for dev in devices:
                dev_id = dev.get("device_id", 0)
                dev_name = dev.get("device_name", "Intel PVC")
                gpu_util = dev.get("gpu_utilization", 0)
                mem_util = dev.get("memory_utilization", 0)
                mem_used = dev.get("memory_used", 0)
                mem_total = dev.get("memory_total", 131072)
                gpus.append({
                    "type": "Intel",
                    "index": str(dev_id),
                    "name": dev_name,
                    "gpu_util": str(gpu_util),
                    "mem_util": str(mem_util),
                    "mem_used": str(int(mem_used)),
                    "mem_total": str(int(mem_total))
                })
            return gpus
        except Exception as e:
            return []
    return []

def get_amd_gpus():
    if not shutil.which("rocm-smi"):
        return []
    try:
        out = subprocess.check_output(["rocm-smi", "--showuse", "--showmemuse", "--json"]).decode("utf-8", errors="ignore")
        data = json.loads(out)
        gpus = []
        for card, metrics in data.items():
            if card.startswith("card"):
                idx = card.replace("card", "")
                gpu_util = metrics.get("GPU use (%)", "0")
                mem_util = metrics.get("GPU memory use (%)", "0")
                gpus.append({
                    "type": "AMD",
                    "index": idx,
                    "name": "AMD Instinct GPU",
                    "gpu_util": str(gpu_util),
                    "mem_util": str(mem_util),
                    "mem_used": "0",
                    "mem_total": "0"
                })
        return gpus
    except Exception as e:
        return []

def main():
    gpus = get_nvidia_gpus()
    if not gpus:
        gpus = get_intel_gpus()
    if not gpus:
        gpus = get_amd_gpus()
    
    if not gpus:
        print("NONE")
        return
        
    for g in gpus:
        if g["type"] == "ERROR":
            print(f"ERROR|{g['msg']}")
        else:
            print(f"{g['type']}|{g['index']}|{g['name']}|{g['gpu_util']}|{g['mem_util']}|{g['mem_used']}|{g['mem_total']}")

if __name__ == '__main__':
    main()
EOF
)

            if [ -n "$RAW_GPU" ] && [ "$RAW_GPU" != "NONE" ]; then
                has_running_gpu=true
                
                while IFS='|' read -r type idx name gpu_util mem_util mem_used mem_total; do
                    [ -z "$type" ] && continue
                    
                    if [ "$type" = "ERROR" ]; then
                        gpu_html+="<div style='color:#dc3545;font-size:0.85em;padding:4px 2px'>Error fetching GPU stats: ${idx}</div>"
                        continue
                    fi
                    
                    # Sanitize utilization values to ensure valid progress bar widths
                    if [ -z "$gpu_util" ] || ! [[ "$gpu_util" =~ ^[0-9]+$ ]]; then
                        gpu_util=0
                    fi
                    if [ -z "$mem_util" ] || ! [[ "$mem_util" =~ ^[0-9]+$ ]]; then
                        mem_util=0
                    fi
                    if [ -z "$mem_used" ] || ! [[ "$mem_used" =~ ^[0-9]+$ ]]; then
                        mem_used=0
                    fi
                    if [ -z "$mem_total" ] || ! [[ "$mem_total" =~ ^[0-9]+$ ]]; then
                        mem_total=0
                    fi
                    
                    local mem_text=""
                    if [ "$mem_total" -gt 0 ]; then
                        local used_gb total_gb
                        used_gb=$(python3 -c "print(f'{float($mem_used)/1024:.1f}')")
                        total_gb=$(python3 -c "print(f'{float($mem_total)/1024:.1f}')")
                        mem_text=" &nbsp;|&nbsp; Memory: ${used_gb} GB / ${total_gb} GB"
                        # Recalculate capacity utilization percentage to match capacity instead of bandwidth
                        mem_util=$(python3 -c "print(int(round((float($mem_used)/float($mem_total))*100)))")
                    fi
                    
                    local gpu_color="#198754"
                    if [ "$gpu_util" -ge 80 ]; then
                        gpu_color="#dc3545"
                    elif [ "$gpu_util" -ge 50 ]; then
                        gpu_color="#ffc107"
                    fi

                    local mem_color="#0d6efd"
                    if [ "$mem_util" -ge 80 ]; then
                        mem_color="#dc3545"
                    elif [ "$mem_util" -ge 50 ]; then
                        mem_color="#ffc107"
                    fi

                    gpu_html+="<div class='card mb-3' style='border: 1px solid #dee2e6; border-radius: 8px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); overflow: hidden; margin-bottom: 15px;'>"
                    gpu_html+="  <div class='card-header' style='background: #f8f9fa; padding: 10px 15px; font-weight: 600; font-size: 0.9em; border-bottom: 1px solid #dee2e6; display: flex; justify-content: space-between; align-items: center;'>"
                    gpu_html+="    <span>${type} GPU #${idx}: <span style='font-weight: normal; color: #495057;'>${name}</span></span>"
                    gpu_html+="    <span style='font-size: 0.85em; color: #6c757d; font-weight: normal;'>Node: ${NODE}</span>"
                    gpu_html+="  </div>"
                    gpu_html+="  <div class='card-body' style='padding: 15px; display: flex; flex-direction: column; gap: 12px;'>"
                    
                    gpu_html+="    <div>"
                    gpu_html+="      <div style='display: flex; justify-content: space-between; font-size: 0.85em; font-weight: 500; margin-bottom: 4px;'>"
                    gpu_html+="        <span>GPU Core Utilization</span>"
                    gpu_html+="        <span style='color: ${gpu_color}; font-weight: 600;'>${gpu_util}%</span>"
                    gpu_html+="      </div>"
                    gpu_html+="      <div style='background: #e9ecef; border-radius: 4px; height: 10px; overflow: hidden;'>"
                    gpu_html+="        <div style='background: ${gpu_color}; width: ${gpu_util}%; height: 100%; transition: width 0.3s;'></div>"
                    gpu_html+="      </div>"
                    gpu_html+="    </div>"
                    
                    if [ -n "$mem_util" ] && [ "$mem_util" -ne 0 ] || [ -n "$mem_text" ]; then
                        gpu_html+="    <div>"
                        gpu_html+="      <div style='display: flex; justify-content: space-between; font-size: 0.85em; font-weight: 500; margin-bottom: 4px;'>"
                        gpu_html+="        <span>GPU Memory Usage${mem_text}</span>"
                        gpu_html+="        <span style='color: ${mem_color}; font-weight: 600;'>${mem_util}%</span>"
                        gpu_html+="      </div>"
                        gpu_html+="      <div style='background: #e9ecef; border-radius: 4px; height: 10px; overflow: hidden;'>"
                        gpu_html+="        <div style='background: ${mem_color}; width: ${mem_util}%; height: 100%; transition: width 0.3s;'></div>"
                        gpu_html+="      </div>"
                        gpu_html+="    </div>"
                    fi
                    
                    gpu_html+="  </div>"
                    gpu_html+="</div>"
                done <<< "$RAW_GPU"
            fi
        fi
    done

    if $has_running_gpu; then
        echo "<div class='pt-gpu-section' style='margin-top:14px'>"
        echo "<div style='font-size:0.9em;font-weight:600;color:#495057;margin-bottom:8px'>GPU Utilization</div>"
        echo "${gpu_html}"
        echo "</div>"
    fi
}

emit_logs() {
  if [ -f "$SCRIPT_DIR/retrieve_pt_logs.sh" ]; then
      bash "$SCRIPT_DIR/retrieve_pt_logs.sh"
  else
      echo "<div style='font-size:0.85em;color:#6c757d'>Log viewer unavailable.</div>"
  fi
}

LOCATION_ESC=$(echo "$LOCATION" | html_escape | sed 's/"/\&quot;/g')
echo "<div id='pt-monitor-dashboard' data-pt-location=\"$LOCATION_ESC\">"

echo "<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap'>"
echo "  <span style='font-size:0.82em;color:#868e96'>Auto-refreshes every 10 seconds.</span>"
echo "  <span id='pt-refresh-cooldown-hint' style='font-size:0.82em;color:#e67700'></span>"
echo "</div>"

echo "<div class='pt-slurm-section'>"
echo "<div style='font-size:0.9em;font-weight:600;color:#495057;margin-bottom:8px'>Slurm Jobs</div>"
emit_slurm_summary
echo "</div>"

emit_epoch_progress

emit_gpu_summary

echo "<div class='pt-logs-section' style='margin-top:14px'>"
echo "<div style='font-size:0.9em;font-weight:600;color:#495057;margin-bottom:8px'>Output/Error logs</div>"
emit_logs
echo "</div>"

echo "</div>"

cat << 'SCRIPT'
<script>
(function () {
  var COOLDOWN_MS = 5000;

  function updateDashboard(existing, newDash) {
    // 1. Update the slurm section
    var existingSlurm = existing.querySelector('.pt-slurm-section');
    var newSlurm = newDash.querySelector('.pt-slurm-section');
    if (existingSlurm && newSlurm) {
      existingSlurm.innerHTML = newSlurm.innerHTML;
    }

    // 1.2. Update the epoch progress section
    var existingEpoch = existing.querySelector('.pt-epoch-section');
    var newEpoch = newDash.querySelector('.pt-epoch-section');
    if (existingEpoch && newEpoch) {
      existingEpoch.innerHTML = newEpoch.innerHTML;
    } else if (newEpoch) {
      var slurmSec = existing.querySelector('.pt-slurm-section');
      if (slurmSec) {
        var div = document.createElement('div');
        div.className = 'pt-epoch-section';
        div.style.marginTop = '14px';
        div.innerHTML = newEpoch.innerHTML;
        slurmSec.parentNode.insertBefore(div, slurmSec.nextSibling);
      }
    } else if (existingEpoch) {
      existingEpoch.parentNode.removeChild(existingEpoch);
    }

    // 1.5. Update the gpu section
    // Skip GPU update for 30s after any TensorBoard interaction so the bars
    // don't reset to 0% while the TB-triggered refresh re-runs the srun query.
    var tbGuardActive = window.__tb_interaction_ts &&
      (Date.now() - window.__tb_interaction_ts < 30000);
    var existingGpu = existing.querySelector('.pt-gpu-section');
    var newGpu = newDash.querySelector('.pt-gpu-section');
    if (!tbGuardActive) {
      if (existingGpu && newGpu) {
        existingGpu.innerHTML = newGpu.innerHTML;
      } else if (newGpu) {
        var logsSec = existing.querySelector('.pt-logs-section');
        if (logsSec) {
          var div = document.createElement('div');
          div.className = 'pt-gpu-section';
          div.style.marginTop = '14px';
          div.innerHTML = newGpu.innerHTML;
          logsSec.parentNode.insertBefore(div, logsSec);
        } else {
          var div = document.createElement('div');
          div.className = 'pt-gpu-section';
          div.style.marginTop = '14px';
          div.innerHTML = newGpu.innerHTML;
          existing.appendChild(div);
        }
      } else if (existingGpu) {
        existingGpu.parentNode.removeChild(existingGpu);
      }
    }

    // 2. Update the logs section
    var existingLogsRoot = existing.querySelector('.pt-logs-root');
    var newLogsRoot = newDash.querySelector('.pt-logs-root');
    if (existingLogsRoot && newLogsRoot) {
      var newDetails = newLogsRoot.querySelectorAll('.pt-log-container');
      var existingDetails = existingLogsRoot.querySelectorAll('.pt-log-container');

      var existingMap = {};
      existingDetails.forEach(function (d) {
        var logfile = d.getAttribute('data-pt-logfile');
        if (logfile) {
          existingMap[logfile] = d;
        }
      });

      var presentKeys = {};
      newDetails.forEach(function (newD) {
        var logfile = newD.getAttribute('data-pt-logfile');
        if (!logfile) return;
        presentKeys[logfile] = true;

        var existingD = existingMap[logfile];
        if (existingD) {
          // Update header text
          var existingHeader = existingD.querySelector('.pt-log-header');
          var newHeader = newD.querySelector('.pt-log-header');
          if (existingHeader && newHeader) {
            if (existingHeader.innerHTML !== newHeader.innerHTML) {
              existingHeader.innerHTML = newHeader.innerHTML;
            }
          }
        } else {
          // New file stream, append it!
          var importedD = document.importNode(newD, true);
          existingLogsRoot.appendChild(importedD);
        }
      });

      // Remove removed file streams
      existingDetails.forEach(function (d) {
        var logfile = d.getAttribute('data-pt-logfile');
        if (logfile && !presentKeys[logfile]) {
          d.parentNode.removeChild(d);
        }
      });
    }

    // 3. Update cooldown hint
    var existingHint = existing.querySelector('#pt-refresh-cooldown-hint');
    var newHint = newDash.querySelector('#pt-refresh-cooldown-hint');
    if (existingHint && newHint) {
      existingHint.innerHTML = newHint.innerHTML;
    }
  }

  // Monkeypatch Element.prototype.innerHTML setter once to intercept dashboard updates
  if (!window.__pt_inner_html_patched) {
    window.__pt_inner_html_patched = true;
    var descriptor = Object.getOwnPropertyDescriptor(Element.prototype, 'innerHTML');
    if (descriptor && descriptor.set) {
      var originalSet = descriptor.set;
      Object.defineProperty(Element.prototype, 'innerHTML', {
        set: function (html) {
          if (typeof html === 'string' && html.indexOf('pt-monitor-dashboard') !== -1) {
            var existingDashboard = document.getElementById('pt-monitor-dashboard');
            if (existingDashboard) {
              try {
                var parser = new DOMParser();
                var newDoc = parser.parseFromString(html, 'text/html');
                var newDashboard = newDoc.getElementById('pt-monitor-dashboard');
                if (newDashboard) {
                  var oldLoc = existingDashboard.getAttribute('data-pt-location');
                  var newLoc = newDashboard.getAttribute('data-pt-location');
                  if (oldLoc === newLoc) {
                    updateDashboard(existingDashboard, newDashboard);
                    return; // Skip replacing innerHTML of the container
                  }
                }
              } catch (err) {
                console.error('Error doing smart dashboard update:', err);
              }
            }
          }
          originalSet.call(this, html);
        },
        configurable: true,
        enumerable: true
      });
    }
  }

  function findRefreshButton(root) {
    var el = root;
    while (el && el !== document.body) {
      var parent = el.parentElement;
      if (parent) {
        var buttons = parent.querySelectorAll('button');
        for (var i = 0; i < buttons.length; i++) {
          var b = buttons[i];
          if (b.dataset.ptRefreshBtn) return b;
          var label = ((b.textContent || '') + ' ' + (b.title || '') + ' ' + (b.getAttribute('aria-label') || '')).toLowerCase();
          if (label.indexOf('refresh') !== -1) {
            b.dataset.ptRefreshBtn = '1';
            return b;
          }
        }
      }
      el = parent;
    }
    return null;
  }

  // Stamp window.__tb_interaction_ts whenever a TensorBoard trigger input
  // changes value, so updateDashboard() can skip the GPU section update
  // for the next 30 seconds and avoid a false 0% flash.
  function bindTbInteractionGuard() {
    if (window.__ptTbGuardBound) return;
    window.__ptTbGuardBound = true;

    function markInteraction(e) {
      if (e.target && (e.target.name === 'tbStartTrigger' || e.target.name === 'tbStopTrigger')) {
        if (e.target.value !== '') {
          window.__tb_interaction_ts = Date.now();
        }
      }
    }
    document.addEventListener('input', markInteraction, true);
    document.addEventListener('change', markInteraction, true);
  }

  function bindRefreshCooldown() {
    var root = document.getElementById('pt-monitor-dashboard');
    if (!root) return;

    var btn = findRefreshButton(root);
    if (!btn || btn.dataset.ptCooldownBound) return;
    btn.dataset.ptCooldownBound = '1';

    var hint = document.getElementById('pt-refresh-cooldown-hint');
    var last = 0;
    var timer = null;

    function setHint(msg) {
      if (hint) hint.textContent = msg || '';
    }

    function startCooldown(remaining) {
      btn.disabled = true;
      btn.style.opacity = '0.65';
      btn.style.cursor = 'not-allowed';
      if (timer) clearInterval(timer);
      var left = Math.ceil(remaining / 1000);
      setHint('Refresh available in ' + left + 's');
      timer = setInterval(function () {
        left -= 1;
        if (left <= 0) {
          clearInterval(timer);
          timer = null;
          btn.disabled = false;
          btn.style.opacity = '';
          btn.style.cursor = '';
          setHint('');
          return;
        }
        setHint('Refresh available in ' + left + 's');
      }, 1000);
    }

    btn.addEventListener('click', function (e) {
      var now = Date.now();
      if (last && now - last < COOLDOWN_MS) {
        e.preventDefault();
        e.stopPropagation();
        startCooldown(COOLDOWN_MS - (now - last));
        return false;
      }
      last = now;
      startCooldown(COOLDOWN_MS);
    }, true);
  }

  function init() {
    var root = document.getElementById('pt-monitor-dashboard');
    if (!root) return;
    bindRefreshCooldown();
    bindTbInteractionGuard();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  setTimeout(init, 200);
  setTimeout(init, 800);
 })();
</script>
SCRIPT
