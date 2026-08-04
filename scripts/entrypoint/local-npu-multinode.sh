
if [ -n "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

_LOCAL_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# ── delegate to ray-job.sh when inside an existing Ray cluster ─────────────
# When RAY_ADDRESS is set AND `ray status` succeeds, we're already part of an
# externally-managed Ray cluster. Skip local Ray startup / process cleanup and
# fall through to ray-job.sh (source mode) for env setup.
if [ -n "${RAY_ADDRESS:-}" ] && timeout 5 ray status >/dev/null 2>&1; then
    echo "=== Detected existing Ray cluster (RAY_ADDRESS=$RAY_ADDRESS); delegating to ray-job.sh ==="
    # shellcheck source=./ray-job.sh
    source "${_LOCAL_SH_DIR}/ray-job-npu.sh"
    return 0 2>/dev/null || exit 0
fi

set -eo pipefail

# ── process cleanup ─────────────────────────────────────────────────────────
echo "=== Cleaning up stale processes ==="
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
# pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
# pkill -9 python 2>/dev/null || true

set -x

# ── environment setup ───────────────────────────────────────────────────────
unset MASTER_ADDR 2>/dev/null || true
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON=${MEGATRON:-/root/Megatron-LM/}
export MEGATRON_BRIDGE_SRC=${MEGATRON_BRIDGE_SRC:-/root/Megatron-Bridge/src/}
export MINDSPEED=${MINDSPEED:-/root/MindSpeed/}
export RELAX=${RELAX:-${_LOCAL_SH_DIR}/../../}
export PYTHONPATH=${RELAX}:${MEGATRON_BRIDGE_SRC}:${MINDSPEED}:$MEGATRON:$RELAX:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${_LOCAL_SH_DIR}/../models"

# ── Ray cluster startup (single node) ──────────────────────────────────────
export MASTER_ADDR=${MASTER_ADDR:-"xx.xx.xx.xx"}
export SOCKET_IFNAME="xxxxx"
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
NUM_GPUS="${NUM_GPUS:-16}"
NNODES="${WORLD_SIZE:-2}"

if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    ray start --head \
        --node-ip-address ${MASTER_ADDR} \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265

    sleep 5

    while true; do
        ray_status_output=$(ray status)
        gpu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        echo "Current GPU count: $gpu_count"
        gpu_count_int=$(echo "$gpu_count" | awk '{print int($1)}')
        device_count=$((gpu_count_int / ${NUM_GPUS}))

        if [ "$device_count" -eq "$NNODES" ]; then
            echo "Ray cluster is ready with $device_count devices (from $gpu_count GPU resources)."
            ray status
            break
        else
            echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
            sleep 5
        fi
    done

    # ── set entrypoint mode ────────────────────────────────────────────────────
    export RELAX_ENTRYPOINT_MODE="local"

    # Runtime env for single-node (empty, env inherited from Ray cluster)
    export RUNTIME_ENV_JSON="{
    \"env_vars\": {
        \"PYTHONUNBUFFERED\": \"1\",
        \"PYTHONPATH\": \"${PYTHONPATH}\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
        \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\"
    }
    }"

    echo "=== Local environment ready ==="
else
    # ── WORKER NODE ─────────────────────────────────────────────────────────
    echo "=== Worker node: joining Ray cluster at ${MASTER_ADDR}:6379 ==="
    while true; do
        ray start \
            --address="${MASTER_ADDR}:6379" \
            --node-ip-address "${CURRENT_IP}" \
            --disable-usage-stats \
            --dashboard-host=0.0.0.0 \
            --dashboard-port=8265

        sleep 5
        ray status
        if [ $? -eq 0 ]; then
            echo "Successfully connected to the Ray cluster!"
            break
        else
            echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
        fi
    done

    # Worker nodes block indefinitely (training runs on head node)
    echo "=== Worker node ready, waiting for training to complete ==="
    sleep inf
fi

