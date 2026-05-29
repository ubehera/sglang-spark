#!/bin/bash
# Build the sm_121a sgl-kernel wheel from a PATCHED SGLang source checkout into an
# output dir. ~15-20 min on a single GB10 (sm_121a-only, FA3/FlashMLA disabled by
# patches/sgl-kernel-cmakelists-sm121a-only.patch).
#   Required: SGLANG_SRC (your sglang checkout with the patch applied),
#             SGLANG_VENV (target venv, from spark-fabric.env).
#   Optional: SGLANG_WHEEL_DIR (default $HOME/sglang-spark-artifacts),
#             SGL_KERNEL_COMPILE_THREADS (default 1), CMAKE_BUILD_PARALLEL_LEVEL (default 4).
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SGLANG_SPARK_ENV:-$SCRIPT_DIR/../env/spark-fabric.env}"
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }

SGLANG_SRC="${SGLANG_SRC:?set SGLANG_SRC to your patched sglang source checkout}"
VENV="${SGLANG_VENV:?set SGLANG_VENV in spark-fabric.env}"
OUT_DIR="${SGLANG_WHEEL_DIR:-$HOME/sglang-spark-artifacts}"
LOG="${LOG:-/tmp/sgl-kernel-wheel.log}"
exec > >(tee -a "$LOG") 2>&1

echo "=== sgl-kernel wheel build start: $(date -Iseconds) on $(hostname) ==="
mkdir -p "$OUT_DIR"
cd "$SGLANG_SRC/sgl-kernel"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
export TORCH_CUDA_ARCH_LIST="12.1a"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
export SKBUILD_CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=${SGL_KERNEL_COMPILE_THREADS:-1}"

echo "Torch: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo "nvcc:  $(nvcc --version | tail -1)"
echo "ARCH:  $TORCH_CUDA_ARCH_LIST   OUT_DIR: $OUT_DIR"

UV="${UV:-$HOME/.local/bin/uv}"
VENV_PY="$VENV/bin/python"
"$UV" pip install --python "$VENV_PY" "scikit-build-core>=0.10" ninja cmake wheel
"$UV" build --python "$VENV_PY" --wheel --no-build-isolation --verbose --out-dir "$OUT_DIR" .

echo "=== wheel build done: $(date -Iseconds) ==="
shopt -s nullglob
WHEELS=( "$OUT_DIR"/sgl_kernel-*.whl "$OUT_DIR"/sglang_kernel-*.whl )
shopt -u nullglob
(( ${#WHEELS[@]} == 0 )) && { echo "WARN: no *.whl produced under $OUT_DIR" >&2; exit 1; }
ls -lh "${WHEELS[@]}" | tail -3
