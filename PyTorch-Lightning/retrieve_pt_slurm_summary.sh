#!/bin/bash
# Given $LOCATION, emits an HTML table summarising the SLURM jobs for this run.

LOCATION="${LOCATION:-}"
if [ -z "$LOCATION" ] || [ ! -d "$LOCATION" ]; then
    echo "<div style='font-size:0.83em;color:#6c757d;padding:4px 2px'>Workflow directory not available yet.</div>"
    exit 0
fi

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

    C1=$(echo "$ROW" | cut -d'|' -f1 | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
    C2=$(echo "$ROW" | cut -d'|' -f2 | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
    C3=$(echo "$ROW" | cut -d'|' -f3 | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
    C4=$(echo "$ROW" | cut -d'|' -f4 | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
    C5=$(echo "$ROW" | cut -d'|' -f5 | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
    JID_ESC=$(echo "$JID" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')

    rows+="<tr class='$ROW_CLASS'><td>${C1:-$JID_ESC}</td><td>${C2:-—}</td><td>${C3:-unknown}</td><td>${C4:-—}</td><td>${C5:-—}</td></tr>"
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
echo "<thead class='table-light'><tr><th>Job ID</th><th>Name</th><th>Status</th><th>Elapsed</th><th>Node(s)</th></tr></thead>"
echo "<tbody>${rows}</tbody></table>"
