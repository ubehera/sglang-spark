#!/bin/bash
# Archetype recipe: DENSE bf16 + trained MTP head, TP=2 cross-node, NEXTN k=3.
# For dense (non-MoE) Qwen3.5/3.6-class bf16 checkpoints that ship a trained MTP
# (nextn) head. No quantization flag; flashinfer attention; SPEC_V2 required so
# NEXTN can coexist with the radix cache under the mamba scheduler.
#   Example:  MODEL_REPO=<a dense bf16 Qwen3.5/3.6 checkpoint with an MTP head>
#
# GB10 unified-memory mitigations baked in (same wall as the MoE recipes):
# mem-frac 0.78, mamba extra_buffer + 0.7 mem ratio, cuda-graph-bs {1,2,4,8},
# --disable-piecewise-cuda-graph, stream-interval 32, NEXTN k=3, multimodal OFF.
#
# Usage:
#   MODEL_REPO=<hf-repo-id> ./tp2-dense-bf16-mtp.sh <node_rank>
#     head   (rank 0):  ./tp2-dense-bf16-mtp.sh 0
#     worker (rank 1):  ./tp2-dense-bf16-mtp.sh 1
set -euo pipefail

NODE_RANK="${1:?must pass node_rank: 0 for head (rank 0), 1 for worker (rank 1)}"
NEXTN_STEPS="${2:-${NEXTN_STEPS:-3}}"
NEXTN_TOPK="${NEXTN_TOPK:-1}"
NUM_DRAFT_TOKENS="${NUM_DRAFT_TOKENS:-4}"

# Load deployment config (paths, fabric, secrets). Copy spark-fabric.env.example
# to spark-fabric.env and fill it in; override its location with SGLANG_SPARK_ENV.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${SGLANG_SPARK_ENV:-$SCRIPT_DIR/../env/spark-fabric.env}"
set +a

export LD_LIBRARY_PATH="${NCCL_HOME:?set NCCL_HOME in spark-fabric.env}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$NCCL_HOME/lib/libnccl.so.2"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:?set NCCL_SOCKET_IFNAME in spark-fabric.env}"
export TP_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
export NCCL_NET_GDR_LEVEL=5
export CUDA_HOME=/usr/local/cuda-13
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PATH=/usr/local/cuda-13/bin:$PATH
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in spark-fabric.env or environment}"
# Qwen3_5ForConditionalGeneration is the registered class for dense AND hybrid
# Qwen3.5/3.6; the dense variant still needs the mamba scheduler config to let
# NEXTN coexist with the radix cache. SPEC_V2 is required for that path:
#   ValueError: Speculative decoding for Qwen3_5ForConditionalGeneration is not
#   compatible with radix cache when using --mamba-scheduler-strategy no_buffer
export SGLANG_ENABLE_SPEC_V2=1

source "${SGLANG_VENV:?set SGLANG_VENV in spark-fabric.env}/bin/activate"

KEY="${SGLANG_API_KEY:?set SGLANG_API_KEY in spark-fabric.env or environment}"

MODEL_REPO="${MODEL_REPO:?set MODEL_REPO (HuggingFace repo id) — a dense bf16 model with a trained MTP head}"
# Prefer a local snapshot under HF_HUB_CACHE (offline / NFS-backed); else pass the
# repo id straight to sglang (it resolves/downloads via HF_HUB_CACHE).
_snap_dir="${HF_HUB_CACHE:?set HF_HUB_CACHE in spark-fabric.env}/models--${MODEL_REPO//\//--}/snapshots"
if compgen -G "$_snap_dir/*" >/dev/null 2>&1; then MODEL_PATH=$(echo "$_snap_dir"/*); else MODEL_PATH="$MODEL_REPO"; fi

echo "[dense-bf16-mtp TP=2 NEXTN k=$NEXTN_STEPS rank=$NODE_RANK] $(date '+%Y-%m-%d %H:%M:%S') launching"
echo "  model: $MODEL_REPO  (path: $MODEL_PATH)"
echo "  num_steps=$NEXTN_STEPS num_draft_tokens=$NUM_DRAFT_TOKENS eagle_topk=$NEXTN_TOPK"

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "${SERVED_NAME:-$MODEL_REPO}" \
  --host 0.0.0.0 --port "${SGLANG_PORT:-30000}" \
  --api-key "$KEY" \
  --tp 2 --nnodes 2 --node-rank "$NODE_RANK" \
  --dist-init-addr "${SGLANG_HEAD_IP:?set SGLANG_HEAD_IP in spark-fabric.env}:${SGLANG_DIST_PORT:-25000}" \
  --context-length "${SGLANG_CONTEXT_LEN:-262144}" \
  --mem-fraction-static 0.78 \
  --max-running-requests 8 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --schedule-conservativeness 0.8 \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-full-memory-ratio 0.7 \
  --kv-cache-dtype bf16 \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}" \
  --attention-backend flashinfer \
  --load-format fastsafetensors \
  --cuda-graph-bs 1 2 4 8 \
  --disable-piecewise-cuda-graph \
  --speculative-algorithm NEXTN \
  --speculative-num-steps "$NEXTN_STEPS" \
  --speculative-num-draft-tokens "$NUM_DRAFT_TOKENS" \
  --speculative-eagle-topk "$NEXTN_TOPK" \
  --enforce-disable-flashinfer-allreduce-fusion \
  --enable-metrics \
  --trust-remote-code \
  --stream-interval 32 \
  --incremental-streaming-output \
  --soft-watchdog-timeout 600 \
  --gc-warning-threshold-secs 0.5
