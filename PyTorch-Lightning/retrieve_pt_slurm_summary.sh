#!/bin/bash
# Given $LOCATION, emits an HTML table summarising the SLURM jobs for this run.
export PYTHONDONTWRITEBYTECODE=1

LOCATION="${LOCATION:-}"
if [ -z "$LOCATION" ] || [ ! -d "$LOCATION" ]; then
    echo "<div style='font-size:0.83em;color:#6c757d;padding:4px 2px'>Workflow directory not available yet.</div>"
    exit 0
fi

html_escape() {
    sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g'
}

# Estimate remaining training time from Slurm elapsed + Lightning progress in out.<jid>.
estimate_job_eta() {
    local location="$1"
    local jid="$2"
    local status="$3"
    local elapsed="$4"
    local log_path=""
    local max_epochs="10"
    local parsed_max

    if [ -f "${location}/out.${jid}" ]; then
        log_path="${location}/out.${jid}"
    fi
    if [ -f "${location}/train.py" ]; then
        parsed_max=$(grep -E '^\s*MAX_EPOCHS\s*=' "${location}/train.py" | grep -o -E '[0-9]+' | head -n 1)
        if [ -n "$parsed_max" ]; then
            max_epochs="$parsed_max"
        fi
    fi

    python3 - <<'PY' "$status" "$elapsed" "$log_path" "$max_epochs"
import os
import re
import sys

status = (sys.argv[1] or "").strip().upper()
elapsed_str = (sys.argv[2] or "").strip()
log_path = sys.argv[3] or ""
try:
    max_epochs = max(1, int(sys.argv[4]))
except ValueError:
    max_epochs = 10

def parse_elapsed(s):
    s = (s or "").strip()
    if not s or s in ("—", "-", "Unknown", "INVALID"):
        return None
    days = 0
    if "-" in s:
        day_part, s = s.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return None
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, parts[0], parts[1]
    elif len(parts) == 1:
        h, m, sec = 0, 0, parts[0]
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + sec

def format_elapsed(total):
    if total < 0:
        total = 0
    days, rem = divmod(int(total), 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    if days > 0:
        return f"{days}-{h:02d}:{m:02d}:{sec:02d}"
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"

terminal_done = status in ("COMPLETED", "COMPLETE")
terminal_other = status.startswith("CANCELLED") or status in (
    "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL",
)

if terminal_done:
    print("0:00")
    sys.exit(0)
if status in ("PENDING", "CONFIGURING") or terminal_other or not status:
    print("—")
    sys.exit(0)

elapsed_s = parse_elapsed(elapsed_str)
if elapsed_s is None or elapsed_s <= 0 or not log_path or not os.path.isfile(log_path):
    print("—")
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

matches = re.findall(r"Epoch\s+(\d+)(?:/\d+)?:?\s*(\d+)%", content)
curr_ep = 0
ep_percent = 0
if matches:
    try:
        curr_ep = int(matches[-1][0])
        ep_percent = int(matches[-1][1])
    except ValueError:
        pass
else:
    simple = re.findall(r"Epoch\s+(\d+)", content)
    if simple:
        try:
            curr_ep = int(simple[-1])
        except ValueError:
            pass

frac = (curr_ep + (ep_percent / 100.0)) / float(max_epochs)
if frac >= 0.999:
    print("0:00")
    sys.exit(0)
if frac < 0.01:
    print("—")
    sys.exit(0)

remaining_s = int(round(elapsed_s * (1.0 - frac) / frac))
print(format_elapsed(remaining_s))
PY
}

JOBIDS_FILE="${LOCATION}/slurm_jobids.txt"
JOBIDS=()

if [ -f "$JOBIDS_FILE" ]; then
    mapfile -t JOBIDS < <(grep -v '^\s*$' "$JOBIDS_FILE")
fi

if [ ${#JOBIDS[@]} -eq 0 ]; then
    shopt -s nullglob
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
    exit 0
fi

rows=""
has_running=false
has_failed=false
has_pending=false

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
        # Fall back to sacct (completed jobs)
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
            FAILED|CANCELLED*|TIMEOUT|NODE_FAIL)
                         ROW_CLASS="table-danger"; has_failed=true ;;
            *)           ROW_CLASS=""; all_done=false ;;
        esac
    fi

    RAW_STATUS=$(echo "$ROW" | cut -d'|' -f3)
    RAW_ELAPSED=$(echo "$ROW" | cut -d'|' -f4)
    ETA=$(estimate_job_eta "$LOCATION" "$JID" "$RAW_STATUS" "$RAW_ELAPSED")
    ETA=$(echo "${ETA:-—}" | html_escape)

    C1=$(echo "$ROW" | cut -d'|' -f1 | html_escape)
    C2=$(echo "$ROW" | cut -d'|' -f2 | html_escape)
    C3=$(echo "$RAW_STATUS" | html_escape)
    C4=$(echo "$RAW_ELAPSED" | html_escape)
    C5=$(echo "$ROW" | cut -d'|' -f5 | html_escape)
    JID_ESC=$(echo "$JID" | html_escape)

    rows+="<tr class='$ROW_CLASS'><td>${C1:-$JID_ESC}</td><td>${C2:-—}</td><td>${C3:-unknown}</td><td>${C4:-—}</td><td title='Estimated time remaining based on training progress'>${ETA}</td><td>${C5:-—}</td></tr>"
done

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
echo "<thead class='table-light'><tr><th>Job ID</th><th>Name</th><th>Status</th><th>Elapsed</th><th title='Estimated time remaining based on training progress'>ETA</th><th>Node(s)</th></tr></thead>"
echo "<tbody>${rows}</tbody></table>"
