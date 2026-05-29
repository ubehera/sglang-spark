# Release v0.4.3+sm121a-2026-05-28

**First public release.** Built and validated on NVIDIA DGX Spark (GB10) hardware.

## What's in this release

| Artifact | Description |
|---|---|
| `sglang_kernel-0.4.3+sm121a-cp310-abi3-linux_aarch64.whl` | `sgl-kernel` wheel built for `sm_121a` (abi3 — loads on Python 3.10+, incl. 3.12) |
| `SHA256SUMS` | Cryptographic checksums for verification |
| `cuobjdump-arches.txt` | Output of `cuobjdump --list-elf` showing the wheel's gencode coverage |
| `RELEASE_NOTES.md` | This document |
| `INSTALL.md` | Step-by-step install instructions |

## Build provenance

| Field | Value |
|---|---|
| Source | [sgl-project/sglang](https://github.com/sgl-project/sglang) @ `34ea682a0` (main HEAD on 2026-05-28T16:30:03Z) |
| sgl-kernel version | 0.4.3+sm121a |
| Local patch | `patches/sgl-kernel-cmakelists-sm121a-only.patch` (sm_121a-only gencode; FA3 + FlashMLA targets disabled) |
| Compiler | nvcc 13.0 / gcc 13.3.0 |
| Target | `aarch64-linux-gnu`, Python 3.12 ABI3 |
| torch ABI | 2.12.0+cu130 |
| Build host | NVIDIA DGX Spark (GB10), 121 GiB unified memory, Ubuntu 24.04.4 LTS |
| Build duration | ~15-20 min single-node (sm_121a-only, FA3/FlashMLA disabled), `SGL_KERNEL_COMPILE_THREADS=1`, `CMAKE_BUILD_PARALLEL_LEVEL=4` |

## Architectural coverage

The wheel ships `sm_121a` cubins. Per the upstream `sgl_kernel/load_utils.py` dispatcher, on a `compute_capability=121` device (GB10), the loader picks `sgl_kernel/sm100/common_ops.abi3.so` — which in this build contains the `sm_121a`-targeted cubin embedded via the gencode flag in CMakeLists.

Verify with:

```bash
cuobjdump --list-elf $(python -c 'import sgl_kernel; import os; print(os.path.join(os.path.dirname(sgl_kernel.__file__), "sm100", "common_ops.abi3.so"))') | grep -oE 'sm_[0-9a-z]+' | sort -u
```

Expected output: `sm_121a` only (the patch also drops the always-on `sm_90` gencode, so nothing else remains).

## Verified workloads

Smoke-tested on a two-node cross-node TP=2 GB10 cluster across all three archetypes:

- **hybrid-MoE NVFP4 + MTP** (`tp2-moe-nvfp4-mtp.sh`) — e.g. `RedHatAI/Qwen3.6-35B-A3B-NVFP4`, NEXTN k=3, 256K context
- **dense bf16 + MTP** (`tp2-dense-bf16-mtp.sh`) — any dense Qwen3.5/3.6 bf16 with a trained MTP head, NEXTN k=3
- **MoE FP8, no MTP** (`tp2-moe-fp8-nospec.sh`) — e.g. `Qwen/Qwen3.6-35B-A3B-FP8`
- Also exercised on Qwen3.5-122B-A10B-FP8 (MoE + built-in MTP) via the MoE archetype with NEXTN re-enabled.

See [scripts/launch/](../../scripts/launch/) for the exact commands (each takes `MODEL_REPO`).

## What's NOT validated

- Single-node deployments (the recipes are TP=2 multinode — they will likely work single-node with `--tp 1`, but untested in this release)
- Other Blackwell variants (`sm_120` RTX 5090, `sm_100` data-center) — should work via the family-compat path but untested
- CUDA < 13.0 (the wheel pulls in CUDA 13 dynamic deps)
- torch != 2.12.0+cu130 (ABI3 means it likely works with newer 2.x, but untested)

## Known issues

- `flashinfer-cubin` must match `flashinfer-python` exactly, or import fails with a version-mismatch `RuntimeError` — install both in lockstep: `uv pip install --no-deps --upgrade 'flashinfer-python==X' 'flashinfer-cubin==X'`
- The pypi `sgl-kernel-0.4.3` wheel does NOT have an `sm_121` subdirectory; this release fills that gap
- Until [#26597](https://github.com/sgl-project/sglang/pull/26597) merges, multinode-single-GPU-per-node deployments must also apply `patches/weight_utils-multinode-fastsafetensors.patch`

## Install

See [INSTALL.md](INSTALL.md) in this directory.

## Reproducing from source

```bash
git clone --recursive https://github.com/sgl-project/sglang.git sglang-src
cd sglang-src
git checkout 34ea682a0
git apply /path/to/sglang-spark/patches/sgl-kernel-cmakelists-sm121a-only.patch
cd sgl-kernel
TORCH_CUDA_ARCH_LIST="12.1a" SKBUILD_CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1" CMAKE_BUILD_PARALLEL_LEVEL=4 \
    uv build --wheel --no-build-isolation .
```

Compare your `SHA256` against the published `SHA256SUMS` — they should match (up to filesystem metadata).
