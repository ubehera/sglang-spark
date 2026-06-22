# Changelog

All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning follows the convention:

```
v<sgl-kernel-version>+sm121a-<YYYY-MM-DD>
```

## [v0.4.4+sm121a-2026-06-21] — 2026-06-21

Refresh against current upstream. Rebases the GB10 patch stack onto a newer SGLang `main` and the upstream `sgl-kernel 0.4.4` line; adds three speculative-decoding / diagnostics patches surfaced while chasing the cross-node NEXTN decode-sync stall, plus the dual-rank capture script. No wheel rebuilt in this entry — rebuild from these patches to produce `0.4.4+sm121a`.

### Provenance

- Refreshed against SGLang `main` HEAD `73448b0d7` (June 2026).
- Upstream `sgl-kernel` is now `0.4.4` (was `0.4.3` at the initial release). The version patch tags the local build `0.4.4+sm121a`.
- All `patches/*.patch` re-verified to `git apply --check` cleanly, both independently and stacked, against `73448b0d7`.

### Added

- `patches/weight-loader-drop-cache-fastsafetensors.patch` — honors `--weight-loader-drop-cache-after-load` on the `fastsafetensors` iterator (`posix_fadvise(DONTNEED)` per shard after copy-to-device), matching the safetensors / multi-thread iterators. Off by default.
- `patches/eagle-verify-sync-broadcast-fusion.patch` — fuses the three back-to-back EAGLE verify-sync broadcasts (non-greedy branch) into one contiguous-int32-buffer broadcast: removes the inter-broadcast NCCL desync window (a cross-node NEXTN stall locus) and cuts the cross-node launch cost 3x. Cross-rank symmetric; wire-format-only change.
- `patches/nextn-decode-sync-stall-journal-selfclassify.patch` — opt-in in-journal classifier (`SGLANG_DEBUG_SPEC_DECODE_SYNC_WARN_SECS > 0`) that non-blockingly polls the decode `copy_done` event and logs SLOW-vs-deadlock with `tp_rank`/`bs`/`forward_ct`. Default `0.0` = unchanged blocking synchronize.
- `diagnostics/capture-nextn-stall-dualrank.sh` — dual-rank py-spy capture for the cross-node NEXTN detokenizer-stall; snapshots both TP ranks at the stall before logs roll. Operator tool (not an upstream source patch).

### Changed

- `patches/sgl-kernel-version-sm121a.patch` — bumped `0.4.3` → `0.4.4` (now tags the wheel `0.4.4+sm121a` against upstream `sgl-kernel 0.4.4`).
- `patches/weight_utils-multinode-fastsafetensors.patch`, `patches/sgl-kernel-cmakelists-sm121a-only.patch`, `patches/eagle-draft-extend-cudagraph-unanimous-multinode.patch` — regenerated from the rebased commits so they apply cleanly to HEAD `73448b0d7`.
- `scripts/launch/tp2-moe-fp8-nospec.sh`, `tp2-moe-nvfp4-mtp.sh`, `tp2-dense-bf16-mtp.sh` — `--tp 2` → `--tp-size 2`. The abbreviated `--tp` relies on argparse prefix-matching and would become ambiguous (and crash at launch) if upstream adds another `--tp*` option; `--tp-size` is the unambiguous canonical flag.

### Known limitations

- `scripts/build/build-nccl-sm121.sh` is still NOT included (carried over from the initial release). For now, build NCCL 2.30+ from source against `sm_121` and symlink-swap it into the venv (see the NCCL row in INSTALL.md troubleshooting). Still to land in a later release.
- Single-Spark (one box) deployments still arent covered by the shipped TP=2 two-Spark recipes; adapt a script to `--tp-size 1 --nnodes 1` for a single box.
- The `0.4.4+sm121a` wheel is not rebuilt in this entry — these are the source-side artifacts (patches + recipes + diagnostics); run `scripts/build/build-sgl-kernel-wheel.sh` after applying the patches to produce it.

## [v0.4.3+sm121a-2026-05-28] — 2026-05-28

### Added

- Initial release.
- `sglang_kernel-0.4.3+sm121a-cp310-abi3-linux_aarch64.whl` (sm_121a-native; abi3, loads on Python 3.10+) built against:
  - SGLang main HEAD `34ea682a`
  - torch 2.12.0+cu130
  - CUDA 13.0
  - Python 3.12, aarch64
  - With `patches/sgl-kernel-cmakelists-sm121a-only.patch` (sm_121a-only gencode + FA3/FlashMLA disabled)
- `patches/sgl-kernel-cmakelists-sm121a-only.patch` — narrows gencodes to `sm_121a` only and disables the unused FA3 + FlashMLA targets (Hopper/non-sm_121a kernels that can't run on GB10); cuts build time and peak memory on GB10 unified memory.
- `patches/sgl-kernel-version-sm121a.patch` — tags the wheel `0.4.3+sm121a` (PEP 440 local version) so it's distinguishable from upstream `sgl-kernel 0.4.3`.
- `patches/weight_utils-multinode-fastsafetensors.patch` — extracted from PR [sgl-project/sglang#26597](https://github.com/sgl-project/sglang/pull/26597). Fixes `cuda:{rank}` → `cuda:{torch.cuda.current_device()}` for single-GPU-per-node multinode TP, and adds `SGLANG_FASTSAFETENSORS_NOGDS` env var for NFS-backed model loads.
- Model-agnostic archetype launch recipes (model supplied via `MODEL_REPO`):
  - `scripts/launch/tp2-moe-nvfp4-mtp.sh` — hybrid-MoE NVFP4 + trained MTP head, TP=2 cross-node, NEXTN k=3, 256K context.
  - `scripts/launch/tp2-dense-bf16-mtp.sh` — dense bf16 + trained MTP head, TP=2 cross-node, NEXTN k=3.
  - `scripts/launch/tp2-moe-fp8-nospec.sh` — MoE FP8 (no MTP head), TP=2 cross-node, multimodal.
- `scripts/build/build-sgl-kernel-wheel.sh` — builds the wheel locally (defaults `SGL_KERNEL_COMPILE_THREADS=1` + `CMAKE_BUILD_PARALLEL_LEVEL=4`; tunable for your RAM).
- `scripts/build/distribute-sgl-kernel.sh` — rsyncs the built `sgl_kernel/` package between cluster nodes and archives to NAS.
- `systemd/sglang-wedge-watcher.service` + `sglang-wedge-watcher.sh` — auto-stops a wedged SGLang TP=2 unit on early-warning signals (detokenizer heartbeat miss, NVRM OOM, hung_task).
- Documentation:
  - `README.md`, `DISCLAIMER.md`, `CHANGELOG.md`
  - `docs/architecture.md`, `docs/installation.md`, `docs/failure-modes.md`, `docs/wedge-handling.md`, `docs/recipes/README.md`

### Known limitations

- Single-Spark (one box) deployments aren't covered by the shipped recipes — they're TP=2 two-Spark templates; adapt a script to `--tp 1 --nnodes 1` for a single box.
- The NCCL custom-build script (`scripts/build/build-nccl-sm121.sh`) is not yet included; for now, build NCCL 2.30+ from source against `sm_121` and symlink-swap it into the venv (see the NCCL row in INSTALL.md troubleshooting). Will land in a later release.

### Compatibility

| Component | Required version |
|---|---|
| Hardware | NVIDIA DGX Spark (GB10), `sm_121` |
| OS | Ubuntu 24.04.4 LTS (other modern Linux likely works, untested) |
| CUDA | 13.0 |
| Python | 3.12 |
| torch | 2.12.0+cu130 (binary-pinned via wheel ABI) |
| SGLang | main HEAD around `34ea682a` (May 2026) — older versions may work with manual sgl-kernel patching |
