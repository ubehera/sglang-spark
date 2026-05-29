# Installation (from a clean DGX Spark)

Prerequisites: Ubuntu 24.04, CUDA 13.0 toolkit at `/usr/local/cuda-13.0`, Python 3.12, [`uv`](https://github.com/astral-sh/uv), and a verified `sm_121` GPU:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader   # -> NVIDIA GB10, 12.1, 124610 MiB
```

For the exact per-release pinned wheel + checksums, see [releases/](../releases/) (that `INSTALL.md` is the authoritative step list; this page is the from-scratch overview).

## 1. Custom sm_121 NCCL (required for cross-node TP=2)

The PyTorch-bundled `libnccl.so.2` has no `sm_121` cubin; cross-node TP collectives deadlock after ~30 min. Build NCCL ≥ 2.30 from source (it auto-detects `sm_121` from the CUDA 13 toolchain) and point `NCCL_HOME` at it:

```bash
git clone https://github.com/NVIDIA/nccl ~/nccl && cd ~/nccl && make -j src.build
# spark-fabric.env:  NCCL_HOME=/home/<you>/nccl/build
```

The launch scripts `LD_PRELOAD` `$NCCL_HOME/lib/libnccl.so.2`. (A turnkey `scripts/build/build-nccl-sm121.sh` is planned.)

## 2. venv + SGLang + the sm_121a kernel

```bash
uv venv --python 3.12 ~/sglang/.venv && source ~/sglang/.venv/bin/activate
uv pip install --torch-backend=auto sglang          # pulls torch 2.12+cu130, flashinfer, etc.
# Install the sm_121a-native kernel OVER upstream's (which lacks sm_121):
uv pip install --no-deps --force-reinstall \
  https://github.com/ubehera/sglang-spark/releases/download/v0.4.3+sm121a-2026-05-28/sglang_kernel-0.4.3+sm121a-cp310-abi3-linux_aarch64.whl
```

Then set `SGLANG_VENV=~/sglang/.venv` in `spark-fabric.env`.

## 3. Multinode patch (single-GPU-per-node TP only)

Until [sgl-project/sglang#26597](https://github.com/sgl-project/sglang/pull/26597) merges, apply the fastsafetensors fix into your installed sglang (see the exact `patch -p2` command in [releases/.../INSTALL.md](../releases/v0.4.3-sm121a-2026-05-28/INSTALL.md) step 4), then set `SGLANG_FASTSAFETENSORS_NOGDS=1` if your model snapshots live on NFS/SMB/FUSE.

## 4. Configure + launch

```bash
cp scripts/env/spark-fabric.env.example scripts/env/spark-fabric.env && $EDITOR scripts/env/spark-fabric.env
export MODEL_REPO=RedHatAI/Qwen3.6-35B-A3B-NVFP4
./scripts/launch/tp2-moe-nvfp4-mtp.sh 0                                   # head (rank 0)
ssh "$SGLANG_PEER_HOST" "cd $PWD && MODEL_REPO=$MODEL_REPO ./scripts/launch/tp2-moe-nvfp4-mtp.sh 1"
```

See [recipes/](recipes/) for which archetype fits your model, and [wedge-handling.md](wedge-handling.md) to install the watcher.
