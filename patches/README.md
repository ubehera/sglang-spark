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

**What it changes:** `sgl-kernel/pyproject.toml` + `python/sgl_kernel/version.py` — sets the package version to `0.4.4+sm121a` (a PEP 440 local-version tag) so the built wheel is distinguishable from upstream `sgl-kernel 0.4.4` in `pip list` / `importlib.metadata` and isn't silently shadowed by the pypi build. The runtime code is identical to `0.4.4`; only the version label differs.

**Apply when:** building the wheel yourself, to reproduce the published `0.4.4+sm121a` artifact.

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

## `eagle-draft-extend-cudagraph-unanimous-multinode.patch`

**What it changes:** `python/sglang/srt/speculative/eagle_worker_v2.py` — in `EagleDraftWorker.init_cuda_graphs`, the decision to build the EAGLE draft-extend cudagraph runner is reconciled to a logical-AND across the TP group (`all_gather_object` on the TP `cpu_group`) before the build. The runner is now built on **all** TP ranks or **none**.

**Why:** The draft-extend cudagraph runner runs a collective `tp_group.barrier()` per batch size during capture, so every TP rank must enter it together. But the per-rank build condition keys on the draft-extend attention backend *type*, which is resolved per rank from a current-device capability probe. On multinode single-GPU-per-node TP (e.g. NVIDIA DGX Spark / GB10), the two ranks can resolve different backends (one flashinfer, one triton) — so one rank builds the runner and enters the collective barrier while the peer skips it and proceeds to the scheduler event loop. Result: a permanent init deadlock (rank-0 idle in `recv_requests` broadcast, rank-1 blocked in `_capture_init` barrier); the server never becomes healthy. Pinning a single attention backend (e.g. `--attention-backend triton` for hybrid-GDN models) avoids the divergence, but this AND-reconcile makes the capture deadlock-proof regardless of how the per-rank backend resolves.

**Apply when:** Running SGLang multinode TP with NEXTN/EAGLE speculative decoding on single-GPU-per-node hosts. Harmless otherwise (single rank, or already-agreeing ranks → no-op, no warning).

**Upstream status:** GB10-discovered; candidate for upstream (the "collective capture must be unanimous across ranks" invariant is general). Not yet submitted.

**Apply command:**
```bash
cd /path/to/sgl-project-sglang
git apply /path/to/sglang-spark/patches/eagle-draft-extend-cudagraph-unanimous-multinode.patch
```

## `weight-loader-drop-cache-fastsafetensors.patch`

**What it changes:** `python/sglang/srt/model_loader/weight_utils.py` + `python/sglang/srt/model_loader/loader.py` — threads the existing `--weight-loader-drop-cache-after-load` flag through `fastsafetensors_weights_iterator`, calling `posix_fadvise(POSIX_FADV_DONTNEED)` on each shard after `copy_files_to_device`, matching the safetensors / multi-thread iterators that already honored it. Off by default; no behavior change unless the flag is set.

**Why:** The `drop_cache_after_load` param and its `posix_fadvise(DONTNEED)` call existed only on the safetensors and multi-thread iterators — the fastsafetensors path silently ignored the flag, so a fastsafetensors load left every shard resident in the page cache. On a host where GPU and CPU share one memory pool (GB10 unified memory), evicting the just-read shard pages after the copy to device reduces post-load page-cache pressure. (It does not change KV-pool sizing: `get_available_gpu_memory` reads `MemAvailable`, which already counts reclaimable page cache as available.)

**Apply when:** Running SGLang with `--load-format fastsafetensors` and `--weight-loader-drop-cache-after-load` on a unified-memory host, to get the same post-load cache eviction the other iterators already provide. Harmless otherwise (flag off → no-op).

**Upstream status:** GB10-discovered; candidate for upstream (parity fix across the weight iterators). Not yet submitted.

**Apply command:**
```bash
cd /path/to/sgl-project-sglang
git apply /path/to/sglang-spark/patches/weight-loader-drop-cache-fastsafetensors.patch
```

## `eagle-verify-sync-broadcast-fusion.patch`

**What it changes:** `python/sglang/srt/speculative/eagle_utils.py` — in the EAGLE verify-sync path (`EagleVerifyInput.sample()`, the non-greedy / `temp>0` branch), fuses the three small back-to-back `broadcast` collectives over the TP device group into a SINGLE `broadcast` over one contiguous int32 buffer. The packed buffer length derives only from `bs` + `spec_steps` (byte-identical across ranks), so the single collective is issued unanimously.

**Why:** Three small back-to-back broadcasts are the documented NCCL desync hazard ("order of collectives not the same / a scheduled collective didnt run"). On cross-node single-GPU-per-node TP=2 with NEXTN under non-greedy (thinking-on) decoding, the inter-broadcast window is a stall locus: the rank-0 NCCL watchdog pins the hang at this broadcast trio, the downstream `process_batch_result_decode → copy_done.synchronize()` drain never returns, and the `/health` detokenizer heartbeat trips (it is a scheduler-iteration block, not a slow detok thread). The greedy branch has no broadcast and does not wedge — matching the thinking-on-only symptom. Fusing 3→1 removes the inter-broadcast desync window and cuts the cross-node launch cost 3x. Feature-preserving: only the wire format of the existing sync values changes (NEXTN/cudagraph/mamba/multimodal untouched).

**Apply when:** Running SGLang multinode TP with NEXTN/EAGLE speculative decoding under non-greedy sampling on single-GPU-per-node hosts. Harmless otherwise (greedy path unaffected; replicated TP → unanimous single broadcast).

**Upstream status:** GB10-discovered; the broadcast-fusion is a general NCCL-desync-window reduction and is an upstream candidate. Whether fusion is fully curative vs a strong mitigation should be confirmed with a live dual-rank py-spy (`diagnostics/capture-nextn-stall-dualrank.sh`) at a reproduced stall before declaring closed. Not yet submitted.

**Apply command:**
```bash
cd /path/to/sgl-project-sglang
git apply /path/to/sglang-spark/patches/eagle-verify-sync-broadcast-fusion.patch
```

## `nextn-decode-sync-stall-journal-selfclassify.patch`

**What it changes:** `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` + `python/sglang/srt/environ.py` — adds an opt-in diagnostic to the NEXTN decode-sync path. When `SGLANG_DEBUG_SPEC_DECODE_SYNC_WARN_SECS > 0`, the decode-result `copy_done` event is polled non-blockingly and, if the sync exceeds the threshold, a WARN is logged (with `tp_rank`, `bs`, `forward_ct`); a `RECOVERED` line is logged if it later completes. This distinguishes a SLOW sync from a hard cross-rank deadlock (event never completes) directly in the journal. Registers the new env var (`DEBUG_` prefix) in `environ.py`. Default `0.0` = unchanged blocking `synchronize()`.

**Why:** On cross-node single-GPU-per-node TP=2 with NEXTN, the `/health` detokenizer heartbeat is refreshed only after `process_batch_result_decode → copy_done.synchronize()`, which is a node-local CPU wait on a CUDA event recorded after the whole verify/draft/draft-extend forward. If a TP collective baked into that forward stalls on one rank, the wait never returns and `/health` trips — but the journal alone could not tell a slow-but-progressing sync from a hard deadlock, nor which rank/batch was involved. This self-classifying probe captures `tp_rank`/`bs`/`forward_ct` at the stall and reports SLOW-vs-deadlock without a manual py-spy. It is cross-rank SYMMETRIC (identical per-rank code; event + poll are node-local; no collective added/moved/gated) so it cannot itself desync.

**Apply when:** Diagnosing a NEXTN decode-sync stall / detokenizer-heartbeat trip on multinode TP. Leave `SGLANG_DEBUG_SPEC_DECODE_SYNC_WARN_SECS` unset (0.0) in production for the unchanged blocking path; set it (e.g. `5`) to arm the in-journal classifier.

**Upstream status:** Diagnostic instrumentation (GB10-discovered). Pairs with `diagnostics/capture-nextn-stall-dualrank.sh`. Not intended for upstream as-is; informs the root-cause patch.

**Apply command:**
```bash
cd /path/to/sgl-project-sglang
git apply /path/to/sglang-spark/patches/nextn-decode-sync-stall-journal-selfclassify.patch
```

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
