#!/usr/bin/env python
"""Hand-rolled 128x128 block-FP8 (e4m3) quantizer for SGLang serving on GB10 / sm_121a.

Why this exists: as of mid-2026, llm-compressor (the usual path to SGLang-loadable
compressed quants) is pinned to transformers 4.x and fails to import on transformers
5.x, which recent architectures (e.g. Qwen3.5/3.6, model_type qwen3_5) require. This
script produces the *same* DeepSeek-style block-FP8 format that SGLang already loads
(`quant_method: fp8`, `weight_block_size: [128,128]`, dynamic activations) without
llm-compressor — pure torch + safetensors.

It is DATA-FREE (weight-only block scales, dynamic activations): no calibration set,
no forward passes, runs on CPU in seconds-to-minutes, deterministic (run it on each
node and the outputs are byte-identical — no need to copy the result between nodes).

Scope (which tensors get quantized) is the critical correctness bit. Two modes:

  --reference <fp8-model>   RECOMMENDED. Replicate exactly which tensors an existing,
                            known-good FP8 build of the SAME architecture stores as
                            F8_E4M3 (e.g. an official ...-FP8 release), and graft its
                            quantization_config. Guarantees SGLang loads it.

  (pattern mode, default)   Quantize 2D Linear .weight tensors whose name matches a
                            projection pattern AND whose dims are both divisible by the
                            block size; preserve norms/gates/embeddings/lm_head/conv1d/
                            small-SSM-projections/vision. Use --ignore to add patterns.

Preserved-in-bf16 by default (pattern mode): layernorms, router gates, lm_head,
embed_tokens, conv1d, SSM/GDN small params (A_log, dt_bias, in_proj_a/b/ba), the MTP
fc, and any vision tower — matching what official FP8 builds keep in higher precision.

Example:
  python quantize-fp8-blockwise.py \
      --input  /models/MyModel-bf16 \
      --output /models/MyModel-FP8 \
      --reference /models/MyModel-FP8-official   # same arch, official FP8

Then point SGLang at --output. Run on each serving node (deterministic).
"""
import argparse, glob, json, os, re, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

E4M3_MAX = 448.0

# Pattern-mode defaults (only used without --reference).
QUANT_PATTERNS = [
    r"\.mlp\.(gate|up|down)_proj\.weight$",
    r"\.(self_attn|attn)\..*(q|k|v|o|qkv)_proj\.weight$",
    r"\.linear_attn\.in_proj_qkv\.weight$",
    r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$",
    r"\.mtp\.layers\.\d+\.mlp\.(gate|up|down)_proj\.weight$",
]
IGNORE_PATTERNS = [
    r"lm_head", r"embed_tokens", r"\.visual\.", r"norm", r"\.gate\.weight$",
    r"shared_expert_gate", r"conv1d", r"A_log", r"dt_bias",
    r"\.in_proj_(a|b|ba)\.", r"mtp\.fc\.",
]


def block_quant_fp8(W, block):
    W = W.float()
    O, I = W.shape
    if O % block or I % block:
        return None, None
    Wb = W.reshape(O // block, block, I // block, block)
    amax = Wb.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale = amax / E4M3_MAX
    Wq = (Wb / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).reshape(O, I)
    scale_inv = scale.reshape(O // block, I // block).to(torch.bfloat16)
    return Wq.contiguous(), scale_inv.contiguous()


def build_quant_set(ref_dir):
    s = set()
    for f in sorted(glob.glob(os.path.join(ref_dir, "*.safetensors"))):
        with safe_open(f, framework="pt") as st:
            for k in st.keys():
                if k.endswith(".weight") and st.get_slice(k).get_dtype() == "F8_E4M3":
                    s.add(k)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="bf16/fp16 source model dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--reference", help="known-good FP8 build of the SAME arch to replicate")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--ignore", action="append", default=[], help="extra ignore regex (repeatable)")
    args = ap.parse_args()

    src = args.input.rstrip("/") + "/"
    os.makedirs(args.output, exist_ok=True)
    quant_pat = [re.compile(p) for p in QUANT_PATTERNS]
    ignore_pat = [re.compile(p) for p in IGNORE_PATTERNS + args.ignore]
    ref_set = build_quant_set(args.reference) if args.reference else None
    if ref_set is not None:
        print(f"reference mode: replicating {len(ref_set)} quantized tensors", flush=True)

    def should_quant(name, shape):
        if len(shape) != 2 or not name.endswith(".weight"):
            return False
        if ref_set is not None:
            return name in ref_set
        if any(p.search(name) for p in ignore_pat):
            return False
        return any(p.search(name) for p in quant_pat)

    weight_map, total, nq, ncopy = {}, 0, 0, 0
    shards = sorted(glob.glob(src + "*.safetensors"))
    for i, s in enumerate(shards):
        fn = os.path.basename(s)
        out = {}
        with safe_open(s, framework="pt") as st:
            for k in st.keys():
                v = st.get_tensor(k)
                if should_quant(k, tuple(v.shape)):
                    Wq, sc = block_quant_fp8(v, args.block)
                    if Wq is None:
                        out[k] = v; ncopy += 1
                    else:
                        out[k] = Wq
                        out[k[:-len(".weight")] + ".weight_scale_inv"] = sc
                        nq += 1
                else:
                    out[k] = v; ncopy += 1
        save_file(out, os.path.join(args.output, fn), metadata={"format": "pt"})
        for k, t in out.items():
            weight_map[k] = fn
            total += t.numel() * t.element_size()
        print(f"  [{i+1}/{len(shards)}] {fn}: q={nq} copy={ncopy}", flush=True)

    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(args.output, "model.safetensors.index.json"), "w"), indent=2)

    cfg = json.load(open(src + "config.json"))
    if args.reference:
        cfg["quantization_config"] = json.load(open(os.path.join(args.reference, "config.json")))["quantization_config"]
    else:
        cfg["quantization_config"] = {
            "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
            "weight_block_size": [args.block, args.block],
        }
    json.dump(cfg, open(os.path.join(args.output, "config.json"), "w"), indent=2)
    for f in glob.glob(src + "*"):
        b = os.path.basename(f)
        if b.endswith(".safetensors") or b in ("model.safetensors.index.json", "config.json"):
            continue
        if os.path.isfile(f):
            shutil.copy2(f, os.path.join(args.output, b))
    print(f"DONE: quantized={nq} copied_bf16={ncopy} size={total/1e9:.1f}GB -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
