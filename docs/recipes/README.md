# Launch recipes

Model-agnostic **archetype** templates in [`../../scripts/launch/`](../../scripts/launch/). Each encodes the GB10 + model-family tuning; you supply the model via `MODEL_REPO`. Pick by your checkpoint's family / quant / MTP head:

| Script | Family / quant | NEXTN (MTP) | Notes |
|---|---|---|---|
| `tp2-moe-nvfp4-mtp.sh` | hybrid-MoE, NVFP4 | yes, k=3 | mamba flags, 256K ctx, triton attn; multimodal OFF (NEXTN+image wedges flashinfer.prefill.plan) |
| `tp2-dense-bf16-mtp.sh` | dense, bf16 | yes, k=3 | flashinfer attn; `SGLANG_ENABLE_SPEC_V2=1` (needed for NEXTN + radix cache under the mamba scheduler) |
| `tp2-moe-fp8-nospec.sh` | MoE, FP8 | no | NEXTN off (FP8 sibling has no trained MTP head); multimodal ON |

```bash
MODEL_REPO=<hf-repo-id> ./scripts/launch/<script> <rank>      # rank 0 = head, 1 = worker
```

Deployment config (paths, fabric, secrets) comes from `scripts/env/spark-fabric.env`. Validated on Qwen3.5/3.6-class checkpoints; the MoE-FP8 archetype also runs Qwen3.5-122B-A10B-FP8 (which *does* have a trained MTP head — re-enable NEXTN there).

## Shared GB10 tuning — why these flags

- `--mem-fraction-static 0.78` — the unified-memory wall; higher risks the memcg deadlock ([../failure-modes.md](../failure-modes.md)).
- `--cuda-graph-bs 1 2 4 8` + `--disable-piecewise-cuda-graph` — an explicit short capture list. The sm_121 piecewise compiler hard-fails, and a full default capture list OOMs the KV pool. (`--cuda-graph-max-bs N` does *not* filter the list — only the explicit form does.)
- `--stream-interval 32` — keeps the detokenizer heartbeat alive under concurrent decode (default 8 stalls and the watcher SIGTERMs the unit). Do NOT add `--detokenizer-worker-num 2` (NVRM OOM).
- `--kv-cache-dtype bf16` — fp8 KV silently corrupts on the Qwen family (sglang #19603).
- NEXTN k=3 = `--speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-eagle-topk 1` — only for checkpoints that ship a trained MTP head.

## Measuring NEXTN acceptance honestly

`spec_accept_length` is meaningless without stating workload + sampling + surface. Measure per-request via `/generate` `meta_info.spec_accept_length`, N≥10, discard warmups, temperature-matched, single-stream. A deterministic-repeat prompt at temp 0 approaches the `num_draft_tokens` ceiling (~4.0); real reasoning workloads sit lower (~2.5-3.2) — that's workload, not regression.

## Overridable env knobs

`SGLANG_PORT` (30000), `SGLANG_CONTEXT_LEN`, `SERVED_NAME`, `REASONING_PARSER` (qwen3), `TOOL_CALL_PARSER` (qwen3_coder), and `NEXTN_STEPS` / `NEXTN_TOPK` / `NUM_DRAFT_TOKENS`.

## Single Spark

These are TP=2 (two-box) recipes. For one Spark, set `--tp 1 --nnodes 1` and drop the `--dist-init-addr` / `--node-rank` block.
