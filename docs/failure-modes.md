# GB10 unified-memory failure modes

GB10 / DGX Spark shares one 121 GiB pool between CPU and GPU. Under SGLang TP=2
that produces three distinct stall/deadlock signatures. The wedge watcher
(`systemd/sglang-wedge-watcher.{service,sh}`) fires on the earliest actionable
signal and stops the TP=2 unit on **both** ranks (TP=2 is one logical job —
leaving one rank up causes an NCCL hang) before a hard reset becomes necessary.

## 1. Detokenizer thread-stall — earliest, userspace-only

- **Signal:** `Health check failed ... detokenizer for last N seconds`.
- **Cause:** under concurrent decode the detokenizer queue starves and scheduler
  iterations exceed the health heartbeat. No kernel involvement — distinct from #2/#3.
- **Mitigation:** `--stream-interval 32` (up from the default 8) plus
  `--max-running-requests 8`. Do **not** add `--detokenizer-worker-num 2` — it
  triggers NVRM OOM on unified memory.

## 2. NVRM out-of-memory — driver descriptor pool

- **Signal:** `NVRM: Out of memory ... _memdescAllocInternal` in `dmesg`, often
  minutes after the detokenizer signal.
- **Cause:** sustained mixed-size NCCL collectives plus KV / cudagraph growth
  exhaust the driver's internal descriptor pool. Frequently correlates with the
  wheel-NCCL-lacks-`sm_121` deadlock — use a custom `sm_121` NCCL build
  (`NCCL_HOME` in `spark-fabric.env`).
- **Mitigation:** keep `--mem-fraction-static` ≤ 0.78. The watcher **gates** NVRM
  signals for the first ~180 s after the unit goes `active`, because init
  transients (KV pool, cudagraph capture, NEXTN draft load) emit the same
  signature and retry successfully.

## 3. memcg rw-semaphore deadlock — terminal

- **Signal:** `hung_task ... blocked for more than N seconds`,
  `memcg_rstat_updated`, then the journal cuts off — hard reset required.
- **Cause:** cgroup memory accounting deadlocks when the GPU allocation (visible
  to the kernel via unified memory) collides with reclaim. This is the state the
  watcher exists to pre-empt.
- **Mitigation:** the watcher stops the unit at signal #1 or #3 so the kernel can
  reclaim cleanly. A `MemoryMax=` cap on the systemd unit is a secondary guard.

## Distinguishing them

| | dmesg NVRM? | process responsive? | recoverable without reset? |
|---|---|---|---|
| #1 detokenizer-stall | no | yes (py-spy works) | yes — restart the unit |
| #2 NVRM-OOM | yes | limping | usually — restart |
| #3 memcg deadlock | (then cutoff) | no | no — hard reset |

The watcher treats #1 and #3 as fire-immediately; #2 is gated past the init
window to avoid killing healthy cold starts.
