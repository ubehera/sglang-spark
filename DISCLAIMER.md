# Disclaimer

`sglang-spark` is an independent community project. It is **not** affiliated with, endorsed by, or supported by:

- NVIDIA Corporation
- The [sgl-project](https://github.com/sgl-project) maintainers
- LMSYS / LMSYS Org
- Any other organization

## What this project is

A collection of:

- Pre-built `sgl-kernel` wheel binaries targeting `sm_121` (NVIDIA DGX Spark / GB10 consumer Blackwell)
- Patches against [sgl-project/sglang](https://github.com/sgl-project/sglang) source for multi-node single-GPU-per-node deployments
- Bash launch scripts wrapping `sglang.launch_server` with validated flag combinations for GB10 unified memory
- Systemd service definitions for a wedge-detection watcher
- Documentation of failure modes empirically observed on the maintainer's two-node cluster

## What this project is not

- Not a fork of SGLang with diverging behavior — the upstream binary semantics are preserved; we only ship build artifacts and config templates
- Not an officially maintained product — there is no SLA, no roadmap commitment, no guarantee of compatibility with future upstream versions
- Not validated on any hardware other than NVIDIA DGX Spark with GB10 silicon
- Not a substitute for reading the upstream SGLang [documentation](https://docs.sglang.ai/)

## Software warranty

The software is provided **"AS IS"**, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the maintainer be liable for any claim, damages, or other liability arising from the use of this software. See [LICENSE](LICENSE) for the full Apache 2.0 terms.

## Upstream relationship

We carry patches in `patches/` that are intended to be upstreamed. As of the initial release:

- `weight_utils-multinode-fastsafetensors.patch` — submitted as [sgl-project/sglang#26597](https://github.com/sgl-project/sglang/pull/26597)
- `sgl-kernel-cmakelists-sm121a-only.patch` — local-only build optimization, not intended to be upstreamed (strips non-target gencodes for faster GB10 builds)

When upstream lands a fix, the corresponding patch will be removed from this repo and the wheel rebuilt against the new upstream HEAD.

## Trust and provenance

- All wheel binaries are built reproducibly from the included build scripts on the maintainer's hardware
- Each release ships `SHA256SUMS` and `cuobjdump`-verified architecture listings
- Build provenance is documented in [docs/build-from-source.md](docs/build-from-source.md) so users can independently verify the binaries

If you find a discrepancy between the published wheel and a fresh build, please open an issue.
