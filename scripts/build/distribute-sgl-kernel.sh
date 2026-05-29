#!/bin/bash
# Distribute a freshly-built sgl-kernel from this (head) node to the TP=2 peer node
# via rsync, then snapshot it to an artifact dir. Run after the wheel build installs
# into SGLANG_VENV. Config from spark-fabric.env (SGLANG_VENV, SGLANG_PEER_HOST).
#   Optional: SGLANG_ARTIFACT_DIR (default $HOME/sglang-spark-artifacts),
#             SGLANG_SRC (patched sglang checkout, only for the commit id in the name),
#             CUOBJDUMP (default /usr/local/cuda-13.0/bin/cuobjdump).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "${SGLANG_SPARK_ENV:-$SCRIPT_DIR/../env/spark-fabric.env}"; set +a

VENV="${SGLANG_VENV:?set SGLANG_VENV in spark-fabric.env}"
PEER="${SGLANG_PEER_HOST:?set SGLANG_PEER_HOST in spark-fabric.env (the other Spark)}"
ARTIFACT_DIR="${SGLANG_ARTIFACT_DIR:-$HOME/sglang-spark-artifacts}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda-13.0/bin/cuobjdump}"

LOCAL_PKG=$(echo "$VENV"/lib/python*/site-packages)

# Upstream renamed the distribution sgl_kernel -> sglang_kernel; match the new name.
DIST_INFO=$(ls -d "$LOCAL_PKG"/sglang_kernel-*.dist-info 2>/dev/null | head -1)
[[ -z "$DIST_INFO" ]] && { echo "ERROR: no sglang_kernel-*.dist-info under $LOCAL_PKG" >&2; exit 1; }
DIST_INFO_NAME=$(basename "$DIST_INFO")
VERSION=${DIST_INFO_NAME#sglang_kernel-}; VERSION=${VERSION%.dist-info}
COMMIT=$( [ -n "${SGLANG_SRC:-}" ] && git -C "$SGLANG_SRC" rev-parse --short HEAD 2>/dev/null || echo unknown )
DATE=$(date +%Y%m%d)
ARTIFACT="$ARTIFACT_DIR/sglang_kernel-${VERSION}-${COMMIT}-sm121a-${DATE}.tar.zst"

echo "=== distribute sgl-kernel ===  version=$VERSION commit=$COMMIT peer=$PEER"

# Sanity: confirm sm_121a is actually inside common_ops before shipping a fallback build.
COMMON_OPS="$LOCAL_PKG/sgl_kernel/sm90/common_ops.abi3.so"
[[ -f "$COMMON_OPS" ]] || { echo "ERROR: $COMMON_OPS missing" >&2; exit 1; }
ARCHES=$("$CUOBJDUMP" --list-elf "$COMMON_OPS" 2>/dev/null | grep -oE 'sm_[0-9a-z]+' | sort -u | tr '\n' ' ')
echo "common_ops arches: $ARCHES"
echo "$ARCHES" | grep -q 'sm_121a' || { echo "WARN: sm_121a NOT present; build fell back. Aborting." >&2; exit 1; }
echo "[OK] sm_121a confirmed"

echo "=== rsync sgl_kernel/ -> $PEER ==="
rsync -a --delete "$LOCAL_PKG/sgl_kernel/" "$PEER:$LOCAL_PKG/sgl_kernel/"
echo "=== rotate dist-info on $PEER (clear stale sgl_kernel-* + sglang_kernel-*) ==="
ssh "$PEER" "rm -rf $LOCAL_PKG/sglang_kernel-*.dist-info $LOCAL_PKG/sgl_kernel-*.dist-info"
rsync -a "$DIST_INFO/" "$PEER:$LOCAL_PKG/$DIST_INFO_NAME/"
echo "=== verify on $PEER ==="
ssh "$PEER" "$VENV/bin/python -P -c 'import sgl_kernel; print(\"OK\", sgl_kernel.__version__)'"

echo "=== snapshot -> $ARTIFACT ==="
mkdir -p "$ARTIFACT_DIR"
tar --zstd -C "$LOCAL_PKG" -cf "$ARTIFACT" sgl_kernel "$DIST_INFO_NAME"
ls -lh "$ARTIFACT"
echo "=== done ==="
