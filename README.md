# sglang-spark

**Production-grade SGLang inference on consumer Blackwell (NVIDIA DGX Spark / GB10) — native install, sm_121a-native kernels, validated recipes, wedge handling.**

> **DISCLAIMER:** This project is NOT affiliated with, endorsed by, or supported by NVIDIA, sgl-project, LMSYS, or any other organization. It is a community-built downstream collection of build artifacts, patches, and recipes for running [SGLang](https://github.com/sgl-project/sglang) on consumer Blackwell hardware. Use at your own risk. See [DISCLAIMER.md](DISCLAIMER.md).

## What this is

NVIDIA's DGX Spark (GB10) is consumer-class Blackwell silicon — `sm_121` compute capability with `sm_121a`-specific PTX instructions for NVFP4 MMA, custom NCCL collectives, and unified memory architecture. As of 2026-05-28, the upstream SGLang and `sgl-kernel` pypi wheels **do not ship `sm_121a`-native binaries** — the published `sgl-kernel-0.4.3` wheel only has `sm_90`, `sm_100` per-arch directories, with no `sm_121` variant. This collides with the loader's per-arch dispatch in `sgl_kernel/load_utils.py` and prevents `import sgl_kernel` from succeeding on GB10 hardware.

`sglang-spark` ships the missing pieces:

- **`sm_121a`-native `sgl-kernel` wheel** built against current SGLang main, with a `CMakeLists.txt` patch that narrows gencodes to `sm_121a` only and disables the unused FA3 + FlashMLA targets (which can't run on `sm_121a` anyway) — slashing build time and peak memory on GB10's unified memory.
- **`weight_utils.py` patches** for `fastsafetensors`: multinode-device fix + `nogds` opt-out for NFS-backed model loads. Upstream PR: [sgl-project/sglang#26597](https://github.com/sgl-project/sglang/pull/26597).
- **Model-agnostic launch recipes** — archetype templates (hybrid-MoE-NVFP4, dense-bf16, MoE-FP8) parameterized by `MODEL_REPO`, with NEXTN k=3 speculative decoding (where the model ships a trained MTP head) and the full set of GB10 unified-memory mitigations baked in. Validated on Qwen3.5/3.6-class checkpoints.
- **Wedge watcher** systemd service for detecting detokenizer thread-stalls, NVRM OOM warnings, and `hung_task` events before they cascade into cross-rank NCCL deadlocks.
- **Failure-mode documentation** — the three distinct GB10 unified-memory deadlock signatures, how to distinguish them, and where each mitigation applies.
- **Custom NCCL 2.30.4** build instructions — the PyTorch-bundled NCCL wheel (`libnccl.so.2.28.9`) lacks `sm_121` cubins, which causes cross-node TP collectives to deadlock after ~30 min. Symlink-swap to a locally-built NCCL fixes it.

## Quick start (2-Spark TP=2)

```bash
git clone https://github.com/ubehera/sglang-spark.git
cd sglang-spark

# 1. Install the sm_121a-native sgl-kernel wheel (release asset)
WHEEL_URL=https://github.com/ubehera/sglang-spark/releases/download/v0.4.3+sm121a-2026-05-28/sglang_kernel-0.4.3+sm121a-cp310-abi3-linux_aarch64.whl
uv pip install --no-deps "$WHEEL_URL"

# 2. Configure your deployment (paths, fabric, secrets)
cp scripts/env/spark-fabric.env.example scripts/env/spark-fabric.env
$EDITOR scripts/env/spark-fabric.env   # NCCL_HOME, SGLANG_VENV, HF_HUB_CACHE, SGLANG_HEAD_IP, HF_TOKEN, SGLANG_API_KEY, ...

# 3. Launch a recipe. Pick the archetype for your model and pass MODEL_REPO.
#    Example: a hybrid-MoE NVFP4 checkpoint with a trained MTP head:
export MODEL_REPO=RedHatAI/Qwen3.6-35B-A3B-NVFP4
./scripts/launch/tp2-moe-nvfp4-mtp.sh 0                                                   # head   (rank 0)
ssh "$SGLANG_PEER_HOST" "cd $PWD && MODEL_REPO=$MODEL_REPO ./scripts/launch/tp2-moe-nvfp4-mtp.sh 1"   # worker (rank 1)
```

Recipes are **model-agnostic archetype templates** — the launch flags encode the GB10 + model-family tuning; you supply the model via `MODEL_REPO`:

| Script | For | Example model |
|---|---|---|
| `tp2-moe-nvfp4-mtp.sh` | hybrid-MoE, NVFP4, trained MTP head (NEXTN k=3, 256K ctx) | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` |
| `tp2-dense-bf16-mtp.sh` | dense bf16, trained MTP head (NEXTN k=3) | any dense Qwen3.5/3.6 bf16 with an MTP head |
| `tp2-moe-fp8-nospec.sh` | MoE, FP8, no MTP (NEXTN off), multimodal | `Qwen/Qwen3.6-35B-A3B-FP8` |

These are TP=2 (two-Spark) recipes. For a single Spark, adapt a script to `--tp 1 --nnodes 1`.

## Where this fits

`sglang-spark` is the **runtime layer** for SGLang on consumer Blackwell — native install, `sm_121a`-native kernels, validated per-model launch recipes with the GB10 unified-memory mitigations baked in, and wedge handling. Adjacent community efforts cover different concerns: [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) covers the vLLM stack on the same hardware. Orchestration / model-management UIs are handled by separate projects (see [spark-arena/sparkrun](https://github.com/spark-arena/sparkrun), [calico88x/DGX-Model-Manager](https://github.com/calico88x/DGX-Model-Manager)).

## What's NOT in this repo

- Inference orchestration / model management UIs — see [spark-arena/sparkrun](https://github.com/spark-arena/sparkrun) or [calico88x/DGX-Model-Manager](https://github.com/calico88x/DGX-Model-Manager)
- Evaluation harnesses — separate concern, lives in [ubehera/rhumb](https://github.com/ubehera/rhumb)
- Abliterated / fine-tuned models — see the Hugging Face Hub
- vLLM or TRT-LLM stacks — different inference engines (eugr covers vLLM)
- Sidecar services (TEI embedders, STT, etc.) — separate scope

The line: **SGLang runtime on consumer Blackwell, nothing else.**

## Repo layout

```
docs/
├── architecture.md         # cluster topology, multi-stack coexistence
├── installation.md         # step-by-step from a clean DGX Spark
├── failure-modes.md        # the three GB10 unified-mem deadlocks
├── wedge-handling.md       # watcher design + diagnosis playbook
└── recipes/                # per-model validated launch recipes
patches/                    # CMakeLists sm_121a-only + weight_utils (until #26597 merges)
scripts/
├── env/                    # fabric env file template
├── launch/                 # archetype launch recipes (model via MODEL_REPO)
├── build/                  # sgl-kernel + NCCL build recipes
└── ops/                    # cache drop, systemd reset, etc.
systemd/                    # wedge watcher service + script
releases/                   # per-release notes + checksums
```

## Releases

| Version | SGLang | sgl-kernel | torch | CUDA | Date | Notes |
|---|---|---|---|---|---|---|
| `v0.4.3+sm121a-2026-05-28` | main `34ea682a` | 0.4.3+sm121a | 2.12.0+cu130 | 13.0 | 2026-05-28 | Initial release |

See [releases/](releases/) for per-release notes, install instructions, SHA256 sums, and cuobjdump arch verification.

## Contributing

Open an issue first — this project is intentionally narrow in scope (SGLang runtime on consumer Blackwell). PRs welcome for:

- New launch recipes for additional Qwen / LFM / Granite / Gemma hybrid models tested on GB10
- Bug fixes in the build / patch / wedge-watcher pieces
- Documentation improvements

Not in scope:

- Model orchestration UI
- Eval harnesses
- Non-`sm_121` hardware support (other Blackwell variants may work but aren't tested)

## License

[Apache 2.0](LICENSE) — matches upstream SGLang. Patches and recipes are original work; the `sgl-kernel` wheel is a derivative work of [sgl-project/sglang](https://github.com/sgl-project/sglang), redistributed under the same license.

## Status

Active, on a small two-node DGX Spark cluster as the production daily driver. Built and maintained by [@ubehera](https://github.com/ubehera). Updates roughly track upstream SGLang minor releases — see [CHANGELOG.md](CHANGELOG.md).
