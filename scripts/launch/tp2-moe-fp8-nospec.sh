#!/bin/bash
# Archetype recipe: MoE + FP8 weights, NO speculative decoding, TP=2 cross-node,
# multimodal ON. For Qwen3.5/3.6-class MoE FP8 checkpoints that do NOT ship a
# trained MTP head (num_nextn_predict_layers=0) — so NEXTN is off.
#   Validated example:  MODEL_REPO=Qwen/Qwen3.6-35B-A3B-FP8
#
# Useful as a quant drift-check sibling to the NVFP4 MoE recipe (same TP=2
# plumbing, different quant). GB10 mitigations: mem-frac 0.78, cuda-graph-bs
# {1,2,4,8}, --disable-piecewise-cuda-graph. Default context 16K (raise via
# SGLANG_CONTEXT_LEN if your memory budget allows).
#
# Usage:
#   MODEL_REPO=<hf-repo-id> ./tp2-moe-fp8-nospec.sh <node_rank>
#     head   (rank 0):  MODEL_REPO=Qwen/Qwen3.6-35B-A3B-FP8 ./tp2-moe-fp8-nospec.sh 0
#     worker (rank 1):  MODEL_REPO=Qwen/Qwen3.6-35B-A3B-FP8 ./tp2-moe-fp8-nospec.sh 1
set -uo pipefail

NODE_RANK="${1:?must pass node_rank: 0 for head (rank 0), 1 for worker (rank 1)}"

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
# NCCL_IB_GID_INDEX intentionally unset: GID indices are boot-race dependent
# across nodes; let NCCL auto-pick per HCA.
export NCCL_NET_GDR_LEVEL=5
export CUDA_HOME=/usr/local/cuda-13
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PATH=/usr/local/cuda-13/bin:$PATH
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in spark-fabric.env or environment}"

source "${SGLANG_VENV:?set SGLANG_VENV in spark-fabric.env}/bin/activate"

KEY="${SGLANG_API_KEY:?set SGLANG_API_KEY in spark-fabric.env or environment}"

MODEL_REPO="${MODEL_REPO:?set MODEL_REPO (HuggingFace repo id), e.g. Qwen/Qwen3.6-35B-A3B-FP8}"
# Prefer a local snapshot under HF_HUB_CACHE (offline / NFS-backed); else pass the
# repo id straight to sglang (it resolves/downloads via HF_HUB_CACHE).
_snap_dir="${HF_HUB_CACHE:?set HF_HUB_CACHE in spark-fabric.env}/models--${MODEL_REPO//\//--}/snapshots"
if compgen -G "$_snap_dir/*" >/dev/null 2>&1; then MODEL_PATH=$(echo "$_snap_dir"/*); else MODEL_PATH="$MODEL_REPO"; fi

echo "[moe-fp8-nospec TP=2 rank=$NODE_RANK] $(date '+%Y-%m-%d %H:%M:%S') launching"
echo "  model: $MODEL_REPO  (path: $MODEL_PATH)"
echo "  NEXTN=disabled (no MTP head in FP8 sibling), cudagraphs=ON bs={1,2,4,8}, multimodal=ON"

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "${SERVED_NAME:-$MODEL_REPO}" \
  --host 0.0.0.0 --port "${SGLANG_PORT:-30000}" \
  --api-key "$KEY" \
  --tp 2 --nnodes 2 --node-rank "$NODE_RANK" \
  --dist-init-addr "${SGLANG_HEAD_IP:?set SGLANG_HEAD_IP in spark-fabric.env}:${SGLANG_DIST_PORT:-25000}" \
  --context-length "${SGLANG_CONTEXT_LEN:-16384}" \
  --mem-fraction-static 0.78 \
  --max-running-requests 8 \
  --cuda-graph-bs 1 2 4 8 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --schedule-conservativeness 0.8 \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-full-memory-ratio 0.5 \
  --kv-cache-dtype bf16 \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}" \
  --attention-backend triton \
  --load-format fastsafetensors \
  --disable-piecewise-cuda-graph \
  --enable-multimodal \
  --keep-mm-feature-on-device \
  --enforce-disable-flashinfer-allreduce-fusion \
  --enable-metrics \
  --trust-remote-code
