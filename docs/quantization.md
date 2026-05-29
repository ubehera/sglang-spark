# Quantization (block-FP8)

`scripts/quantize-fp8-blockwise.py` — a hand-rolled 128×128 block-FP8 (e4m3)
quantizer that produces the DeepSeek-style format SGLang already loads
(`quant_method: fp8`, `weight_block_size: [128,128]`, dynamic activations).

## Why it exists

As of mid-2026, `llm-compressor` (the usual path to SGLang-loadable compressed
quants) is pinned to **transformers 4.x** and fails to import on **transformers
5.x** (`ImportError: TORCH_INIT_FUNCTIONS`). Recent architectures — e.g. Qwen3.5 /
Qwen3.6 (`model_type: qwen3_5`), the hybrid-GDN + MTP models this repo targets —
**require transformers ≥ 5.6**. So the standard quant path is blocked until
llm-compressor ships transformers-5 support. This script sidesteps it: pure
`torch` + `safetensors`, no llm-compressor.

NVFP4 via llm-compressor is blocked for the same reason; it's deferred until the
tool catches up. FP8 (this script) is the available win today.

## What it does

- **Data-free:** weight-only block scales (`blockamax / 448` stored bf16 as
  `weight_scale_inv`), activations dynamic. No calibration set, no forward passes.
- **CPU, deterministic, streaming:** runs in seconds-to-minutes on CPU
  (`CUDA_VISIBLE_DEVICES=''`), one shard at a time (low memory). Run it on each
  serving node — outputs are byte-identical, so no need to copy the result around.
- **Scope is the correctness bit.** Use `--reference <official-FP8-of-same-arch>`
  to replicate *exactly* which tensors a known-good FP8 build quantizes (and graft
  its `quantization_config`) — guarantees SGLang loads the result. Without a
  reference it falls back to pattern matching (quantize projection Linears whose
  dims are ÷ block size; preserve norms / gates / embeddings / lm_head / conv1d /
  small SSM-GDN projections / vision / MTP fc).

## Example

```bash
python scripts/quantize-fp8-blockwise.py \
    --input  /models/MyModel-bf16 \
    --output /models/MyModel-FP8 \
    --reference /models/MyModel-FP8-official    # same arch, official FP8 release
# then point SGLang --model-path at /models/MyModel-FP8 ; run on each node
```

## Result (reference deployment)

Qwen3.6-27B (hybrid-GDN, abliterated bf16) → block-FP8 on a 2-node GB10 cluster:
**+58% single-stream throughput (10.7 → 16.9 tok/s)** at no measurable quality
cost — NEXTN accept-length preserved (~2.3), refusal behavior unchanged, output
voice indistinguishable; 54 GB → 31 GB. The MTP draft head's MLP is quantized
(matches official FP8 + keeps NEXTN working); `lm_head`, embeddings, norms, the
GDN small projections, and the vision tower stay bf16.

**Don't FP8 the KV cache on hybrid-GDN models** unless you've measured it: the
full-attention KV is real (tens of GB) but quantizing it injects attention noise
that accumulates over long context — and the token pool is usually already huge
because FP8 weights free static-pool budget. Keep `--kv-cache-dtype bf16` unless a
specific workload proves otherwise.
