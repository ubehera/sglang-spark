# Architecture

## Hardware: NVIDIA DGX Spark (GB10)

Consumer-class Grace-Blackwell. Per box:

- **GPU: GB10, compute capability `sm_121`** (`sm_121a` PTX) — distinct from `sm_120` (RTX 5090) and `sm_100`/`sm_103` (datacenter Blackwell). Native NVFP4 / FP8 tensor cores.
- **121 GiB unified memory** (LPDDR5X) shared between CPU and GPU — there is no separate VRAM. This is the single most important constraint: GPU allocations count toward kernel/cgroup memory accounting, and oversubscription deadlocks rather than OOM-killing cleanly (see [failure-modes.md](failure-modes.md)).
- ARM64 (aarch64) CPU. Everything here is `linux/arm64`.

## Multi-Spark fabric (TP=2 and up)

Two Sparks join over a 200G ConnectX-7 QSFP link. GB10 exposes one cable as **two PCIe-x4 RoCE "twins"**; both HCAs must be used (`NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, `NCCL_IB_MERGE_NICS=1`) or NCCL silently caps at ~100 Gbps PHY / ~12 GB/s usable. Cross-node TP=2 also needs a custom **sm_121 NCCL build** — the PyTorch-bundled `libnccl` has no sm_121 cubin and deadlocks after ~30 min of sustained collectives ([installation.md](installation.md) step 1).

## Where the kernel fits

Upstream `sgl-kernel-0.4.3` ships `sm_90` / `sm_100` cubins but **no `sm_121`** — so `import sgl_kernel` fails the loader's per-arch dispatch (`sgl_kernel/load_utils.py`) on GB10. This repo's wheel adds the `sm_121a` cubin (and drops the Hopper FA3 + FlashMLA targets, which can't run on sm_121a and aren't used by the triton/flashinfer backends). On a `sm_121` device the loader selects `sgl_kernel/sm100/common_ops.abi3.so`, which in this build contains the `sm_121a` cubin.

## What this repo is (and isn't)

`sglang-spark` is the **runtime layer**: the sm_121a-native kernel wheel, the patches upstream hasn't merged, model-agnostic launch recipes with the GB10 mitigations baked in, and the wedge watcher. It is *not* an orchestration UI, an eval harness, or a model hub — see the README "What's NOT in this repo".

## Coexistence

One big-model TP=2 server runs at a time per box. SGLang can share the 121 GiB pool with smaller sidecars (e.g. TEI embedders) only within budget; `--mem-fraction-static ≤ 0.78` leaves the headroom that keeps the box off the unified-memory deadlock threshold.
