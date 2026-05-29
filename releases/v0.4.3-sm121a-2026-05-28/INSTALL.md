# Install — `v0.4.3+sm121a-2026-05-28`

## Prerequisites

- NVIDIA DGX Spark (GB10) hardware, `sm_121` compute capability — verify: `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
- Ubuntu 24.04.4 LTS (other modern Linux likely works, untested)
- CUDA 13.0 toolkit installed at `/usr/local/cuda-13.0`
- Python 3.12
- `uv` (recommended) or `pip` 24.0+

## Step 1 — Verify hardware

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
```

Expected:
```
NVIDIA GB10, 12.1, 124610 MiB
```

If `compute_cap` is not `12.1`, this wheel won't load — the upstream pypi `sgl-kernel-0.4.3` is what you want instead.

## Step 2 — Download the wheel

```bash
WHEEL=sglang_kernel-0.4.3+sm121a-cp310-abi3-linux_aarch64.whl
curl -sLO https://github.com/ubehera/sglang-spark/releases/download/v0.4.3+sm121a-2026-05-28/$WHEEL
curl -sLO https://github.com/ubehera/sglang-spark/releases/download/v0.4.3+sm121a-2026-05-28/SHA256SUMS
sha256sum -c SHA256SUMS
```

The `sha256sum -c` should output `OK` for the wheel filename. If it fails, do not install — open an issue with the mismatched hash.

## Step 3 — Install into your venv

If you have an existing SGLang venv, install the wheel into it:

```bash
uv pip install --no-deps --force-reinstall $WHEEL
```

The `--no-deps` is important — you already have torch, CUDA toolkit, etc. installed, and this wheel doesn't need to re-pull them.

If you don't have an SGLang venv yet:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --torch-backend=auto sglang
uv pip install --no-deps --force-reinstall $WHEEL
```

## Step 4 — Apply the multinode-fastsafetensors patch (multinode TP only)

If you're running multinode TP=2 with one GPU per node, also apply the local patch until [sgl-project/sglang#26597](https://github.com/sgl-project/sglang/pull/26597) merges:

```bash
PATCH=/path/to/sglang-spark/patches/weight_utils-multinode-fastsafetensors.patch
VENV_SGL=$(.venv/bin/python -c 'import sglang, os; print(os.path.dirname(sglang.__file__))')
patch -d "$VENV_SGL/.." -p2 < $PATCH
```

(Single-node TP=1 deployments can skip this step.)

After applying, set `SGLANG_FASTSAFETENSORS_NOGDS=1` in the launch environment if your model snapshot lives on NFS / SMB / FUSE.

## Step 5 — Verify the install

```bash
# CWD-pollution defense: cd to /tmp to avoid Python loading from a source tree
cd /tmp
python -c "
import sgl_kernel
print('sgl_kernel:', sgl_kernel.__version__)
print('  module:', sgl_kernel.__file__)
"
```

Expected:
```
sgl_kernel: 0.4.3+sm121a
  module: /.../.venv/lib/python3.12/site-packages/sgl_kernel/__init__.py
```

If you see `ImportError: CRITICAL: Could not load any common_ops library!`, the wheel didn't install correctly — typically because the previous `sgl-kernel` install wasn't fully removed. Re-run with `--force-reinstall`.

## Step 6 — Launch

See the archetype recipes in [scripts/launch/](../../scripts/launch/). Pick the one matching your model and pass `MODEL_REPO`. Example (hybrid-MoE NVFP4):

```bash
cd /path/to/sglang-spark
export HF_TOKEN=hf_... SGLANG_API_KEY=... MODEL_REPO=RedHatAI/Qwen3.6-35B-A3B-NVFP4
./scripts/launch/tp2-moe-nvfp4-mtp.sh 0   # head node, rank 0
```

For multi-Spark, run rank-1 on the worker node (`SGLANG_PEER_HOST` is set in spark-fabric.env):

```bash
ssh "$SGLANG_PEER_HOST" "cd /path/to/sglang-spark && export HF_TOKEN=... SGLANG_API_KEY=... MODEL_REPO=$MODEL_REPO && ./scripts/launch/tp2-moe-nvfp4-mtp.sh 1"
```

The endpoint comes up on `http://<head-node-ip>:30000/v1/*` once both ranks load.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: Could not load any common_ops library!` | Wrong arch directory, or wheel not properly installed | Reinstall with `--force-reinstall`; verify with the cuobjdump check from RELEASE_NOTES.md |
| Weight load hangs mid-shard | NFS-backed model + GDS path failing | Set `SGLANG_FASTSAFETENSORS_NOGDS=1`, ensure the multinode-fastsafetensors patch is applied |
| Cross-node TP collective deadlocks after ~30 min | Wheel NCCL lacks sm_121 cubins | Symlink-swap to a locally-built NCCL 2.30+ (see `scripts/build/build-nccl-sm121.sh`) |
| Detokenizer heartbeat stops, scheduler wedges | Userspace stall under concurrent decode | Add `--stream-interval 32` to the launch flags; enable the wedge watcher service |
| `RuntimeError: flashinfer-cubin version (X) does not match flashinfer version (Y)` | Sibling-package version drift | Reinstall both in lockstep: `uv pip install --no-deps --upgrade 'flashinfer-python==X' 'flashinfer-cubin==X'` |
