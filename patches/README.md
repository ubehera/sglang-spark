# Patches

Local patches applied on top of upstream SGLang / sgl-kernel. Each patch is documented below with what it changes, why, and the upstream status.

## `sgl-kernel-cmakelists-sm121a-only.patch`

**What it changes:** `sgl-kernel/CMakeLists.txt` — (1) narrows the `SGL_KERNEL_CUDA_FLAGS` gencode list to only `-gencode=arch=compute_121a,code=sm_121a` (dropping the always-on `sm_90` plus `sm_100a`/`sm_120a`/`sm_103a`/`sm_110a`/`sm_101a`), adds `--compress-mode=size`, and sets `CUDA_ARCHITECTURES OFF` on the `common_ops` targets; (2) disables the `SGL_KERNEL_ENABLE_FA3` (`flash_ops`) target — 252 Hopper `sm_90a` instantiations that cannot run on `sm_121a`; (3) skips the FlashMLA include (`flashmla_ops`, `sm_100a/103a/90a`). Result: the package ships `common_ops` + `spatial_ops`, `sm_121a` only.

**Why:** A vanilla upstream sgl-kernel build emits 6+ gencodes per CUDA file plus the full FA3 (252 Hopper instantiations) and FlashMLA targets. With the default `--threads=32` per nvcc and a high `CMAKE_BUILD_PARALLEL_LEVEL`, concurrent `cicc` processes exceed GB10's 121 GiB unified memory (GPU + CPU share one pool) and OOM-kill the build mid-compile. FA3 (`sm_90a` Hopper) and FlashMLA (`sm_100a/103a`) can't execute on `sm_121a` and aren't used by the triton/flashinfer backends, so dropping them is free. Net: ~15-20 min build at `SGL_KERNEL_COMPILE_THREADS=1` and a much smaller `.so` set.

**Apply when:** Building `sgl-kernel` from source on a GB10. Skip if building on a host with abundant RAM, or if you want fat binaries that also run on H100 / B200 / RTX 5090.

**Upstream status:** Not intended to be upstreamed. Local build optimization only.

**Apply command:**
```bash
cd /path/to/sgl-project-sglang
git apply /path/to/sglang-spark/patches/sgl-kernel-cmakelists-sm121a-only.patch
```

## `sgl-kernel-version-sm121a.patch`

**What it changes:** `sgl-kernel/pyproject.toml` + `python/sgl_kernel/version.py` — sets the package version to `0.4.3+sm121a` (a PEP 440 local-version tag) so the built wheel is distinguishable from upstream `sgl-kernel 0.4.3` in `pip list` / `importlib.metadata` and isn't silently shadowed by the pypi build. The runtime code is identical to `0.4.3`; only the version label differs.

**Apply when:** building the wheel yourself, to reproduce the published `0.4.3+sm121a` artifact.

**Upstream status:** Not for upstream — a local distribution tag only.

## `weight_utils-multinode-fastsafetensors.patch`

**What it changes:** `python/sglang/srt/model_loader/weight_utils.py` — two fixes in `fastsafetensors_weights_iterator`:

1. `device = torch.device(f"cuda:{rank}")` → `torch.device(f"cuda:{torch.cuda.current_device()}")` — fixes single-GPU-per-node multinode TP where the global rank exceeds the local GPU count
2. New `SGLANG_FASTSAFETENSORS_NOGDS` env var (declared in `Envs`) to opt out of GPU Direct Storage for NFS / SMB / FUSE-backed model loads

Also touches `python/sglang/srt/environ.py` to register the new env var.

**Why:** On a multinode TP=2 setup with one GPU per node (e.g., NVIDIA DGX Spark), the global rank-1 process tries to bind `cuda:1`, which doesn't exist on the second node — `fastsafetensors` then fails mid-load with an obscure CUDA error. Separately, `fastsafetensors` defaults to GDS (cuFile), which fails on NFS-mounted models because most NFS servers don't expose cuFile.

**Apply when:** Running SGLang multinode TP with one GPU per node, OR running on a node where the model snapshot lives on a shared filesystem (NFS, SMB, FUSE).

**Upstream status:** Submitted as [sgl-project/sglang#26597](https://github.com/sgl-project/sglang/pull/26597). Once merged, this patch will be removed from this repo.

**Apply command:**
```bash
cd /path/to/sgl-project-sglang
git apply /path/to/sglang-spark/patches/weight_utils-multinode-fastsafetensors.patch
```

After applying, set `SGLANG_FASTSAFETENSORS_NOGDS=1` in the environment to enable the non-GDS path.

## Verifying patches apply cleanly

After updating the upstream SGLang source, dry-run each patch:

```bash
cd /path/to/sgl-project-sglang
for p in /path/to/sglang-spark/patches/*.patch; do
  echo "--- $p ---"
  git apply --check "$p" && echo "  OK" || echo "  FAILS — needs rebase"
done
```

If a patch fails to apply against a newer upstream HEAD, the patch needs rebasing. Open an issue with the failure output.
