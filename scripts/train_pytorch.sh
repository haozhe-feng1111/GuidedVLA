#!/bin/bash
# =============================================================================
# train_pytorch.sh - Multi-Node Multi-GPU PyTorch Training Script
# =============================================================================
#
# Usage:
#   Single node (standalone):
#     STANDALONE=1 ./scripts/train_pytorch.sh <config_name> [options]
#
#   Multi-node (torchrun elastic launch):
#     ./scripts/train_pytorch.sh <config_name> [options]
#
# Environment variables:
#   STANDALONE=1          Force single-node mode (default: auto-detect)
#   PET_NNODES            Number of nodes (set by cluster scheduler)
#   PET_NPROC_PER_NODE    GPUs per node (default: all visible GPUs)
#   PET_NODE_RANK         Node rank (default: 0)
#   MASTER_ADDR           Master node address (default: localhost)
#   MASTER_PORT           Master node port (default: 29500)
#
# Examples:
#   # Single node, all GPUs
#   STANDALONE=1 ./scripts/train_pytorch.sh pi0_libero --exp_name=my_run
#
#   # Single node, 4 GPUs explicitly
#   STANDALONE=1 PET_NPROC_PER_NODE=4 ./scripts/train_pytorch.sh pi0_libero --exp_name=my_run
#
#   # Multi-node (set PET_* variables from your cluster scheduler)
#   ./scripts/train_pytorch.sh pi0_libero --exp_name=multi_node_run
# =============================================================================

# Get script directory and switch to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# =============================================================================
# Parse Arguments
# =============================================================================

CONFIG_NAME="${1:-}"
if [[ -z "$CONFIG_NAME" ]]; then
    echo "Error: Config name is required as the first argument"
    echo "Usage: $0 <config_name> [--exp_name=<name>] [other options]"
    exit 1
fi
shift

DATASET_MIRROR_SOURCE="${LIBERO_DATASET_SOURCE:-}"
DATASET_MIRROR_DEST="${LIBERO_DATASET_DEST:-./data/libero_mirror}"

if [[ -n "$DATASET_MIRROR_SOURCE" ]]; then
    mkdir -p "$DATASET_MIRROR_DEST"
    cp -r "$DATASET_MIRROR_SOURCE" "$DATASET_MIRROR_DEST"
fi

source .venv/bin/activate

# =============================================================================
# Environment Setup
# =============================================================================

# NCCL settings (tune for your network fabric)
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

# IB is only needed for multi-node. Disable it in standalone/debug mode so NCCL
# doesn't attempt IB probes on machines without IB hardware.
# In multi-node, the cluster scheduler provides the IB vars (NCCL_IB_HCA,
# NCCL_IB_GID_INDEX, NCCL_GDR_LEVEL, etc.) — don't override them here.
_NNODES_EARLY="${PET_NNODES:-1}"
_STANDALONE_EARLY="${STANDALONE:-0}"
if [[ "$_STANDALONE_EARLY" == "1" ]] || [[ "$_NNODES_EARLY" -eq 1 ]]; then
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"  # no IB needed on single node
else
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"  # let cluster IB vars take effect
fi

# WandB settings
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

# PyTorch settings
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

# =============================================================================
# Distributed Training Configuration
# =============================================================================

NNODES="${PET_NNODES:-1}"
NPROC_PER_NODE="${PET_NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
NODE_RANK="${PET_NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
STANDALONE="${STANDALONE:-0}"

# =============================================================================
# Print Configuration
# =============================================================================

current_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "============================================================"
echo "PyTorch Distributed Training"
echo "============================================================"
echo "Start Time:         $current_time"
echo "Num Machines:       $NNODES"
echo "Num GPUs per Node:  $NPROC_PER_NODE"
echo "Node Rank:          $NODE_RANK"
echo "Master Address:     $MASTER_ADDR"
echo "Master Port:        $MASTER_PORT"
echo "Standalone Mode:    $STANDALONE"
echo "Config Name:        $CONFIG_NAME"
echo "Additional Args:    $*"
echo "============================================================"

# =============================================================================
# Launch torchrun
# =============================================================================

if [[ "$STANDALONE" == "1" ]] || [[ "$NNODES" -eq 1 ]]; then
    echo "Running in single-node mode with $NPROC_PER_NODE GPUs"

    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="$NPROC_PER_NODE" \
        scripts/train_pytorch.py \
        "$CONFIG_NAME" \
        "$@"
else
    echo "Running in multi-node mode: Node $NODE_RANK of $NNODES"

    torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$NPROC_PER_NODE" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        scripts/train_pytorch.py \
        "$CONFIG_NAME" \
        "$@"
fi
