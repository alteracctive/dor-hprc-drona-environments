#!/bin/bash
# Start TensorBoard inside the current Slurm allocation.
# Sourced by the training job after it cds into the job directory:
#   . ./launch_tensorboard.sh <logdir>
# Writes tb_port.txt and stops TensorBoard when the job script exits.

LOG_DIR="${1:-lightning_logs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR" || {
        echo "ERROR: Could not create TensorBoard log directory: $LOG_DIR" >&2
        exit 1
    }
fi

echo "Starting TensorBoard background server with srun..." >&2

if [ -f "$SCRIPT_DIR/find_free_tb_port.sh" ]; then
    TB_PORT="$(bash "$SCRIPT_DIR/find_free_tb_port.sh")"
else
    echo "ERROR: find_free_tb_port.sh not found next to launch_tensorboard.sh" >&2
    exit 1
fi

if [ -z "$TB_PORT" ]; then
    echo "ERROR: No free port found on node $(hostname)" >&2
    exit 1
fi

echo "$TB_PORT" > tb_port.txt
export TB_PORT
export TB_LOGDIR="$LOG_DIR"
NODE_NAME=$(hostname)

(
    srun --ntasks=1 --nodes=1 --cpus-per-task=1 --overlap bash -c '
        source /etc/profile
        if [ -f /sw/lmod/lmod/init/bash ]; then
            source /sw/lmod/lmod/init/bash
        elif [ -f /usr/share/lmod/lmod/init/bash ]; then
            source /usr/share/lmod/lmod/init/bash
        fi
        module load GCC/13.2.0 tensorboard/2.18.0 2>/dev/null || \
        module load GCC/12.3.0 tensorboard/2.17.0 2>/dev/null || \
        module load GCC/12.3.0 tensorboard/2.15.1-CUDA-12.1.1 2>/dev/null || \
        module load tensorboard 2>/dev/null || true

        if command -v tensorboard >/dev/null 2>&1; then
            tensorboard --logdir="$TB_LOGDIR" --port=$TB_PORT --bind_all
        elif python3 -m tensorboard.main --help >/dev/null 2>&1; then
            python3 -m tensorboard.main --logdir="$TB_LOGDIR" --port=$TB_PORT --bind_all
        else
            echo "ERROR: TensorBoard binary or python module not found on $(hostname)" >&2
            exit 1
        fi
    ' >&2
) &
TB_PID=$!

_tb_cleanup() {
    echo "Cleaning up TensorBoard server..." >&2
    if [ -n "$TB_PID" ]; then
        kill "$TB_PID" 2>/dev/null
        pkill -P "$TB_PID" 2>/dev/null
    fi
    if [ -n "$TB_PORT" ]; then
        pkill -u "$USER" -f "tensorboard.*--port=$TB_PORT" 2>/dev/null
    fi
}
trap _tb_cleanup EXIT

echo "TensorBoard auto-started with srun on node $NODE_NAME port $TB_PORT" >&2
