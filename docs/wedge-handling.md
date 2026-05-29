# Wedge handling

GB10 unified memory makes SGLang TP=2 vulnerable to three deadlock signatures (see [failure-modes.md](failure-modes.md)). The wedge watcher pre-empts the terminal hard-reset state by stopping the TP=2 unit on the earliest actionable signal.

## Watcher design — `systemd/sglang-wedge-watcher.{service,sh}`

- Tails `journalctl -f --since=now` piped to a `grep -E` of the early-warning patterns: detokenizer heartbeat miss, `NVRM: Out of memory`, `hung_task`, `memcg_rstat_updated`.
- On a match it finds the active `sglang-*-tp2` unit and `systemctl stop`s it **locally and on the peer** (`SGLANG_PEER_HOST`, via `runuser SGLANG_SSH_USER` for cross-node ssh+sudo). TP=2 is one logical job — a half-stopped pair NCCL-hangs.
- **NVRM gating:** NVRM-OOM also fires during a healthy cold start (weight load, Mamba/KV alloc, cudagraph capture, draft-model load), so it is gated for the first ~600 s after the unit goes `active`. The grace must cover the *full* cold load, or a slower-loading model false-positive-kills (observed 2026-05-28: a bf16 27B's Mamba alloc at ~190 s, past a 180 s grace, killed a healthy boot). Detokenizer / hung_task / memcg signals fire immediately — by their nature they cannot false-positive during init.
- Runs as root; **exits after firing** so systemd restarts it with a fresh `--since=now`, which (plus carefully not echoing matched lines) avoids self-triggering on its own journal output.

## Install

```bash
sudo install -d /opt/sglang-spark/systemd
sudo cp systemd/sglang-wedge-watcher.sh /opt/sglang-spark/systemd/      # match ExecStart in the .service
sudo cp systemd/sglang-wedge-watcher.service /etc/systemd/system/
sudo install -d /etc/sglang-spark && sudo cp scripts/env/spark-fabric.env /etc/sglang-spark/   # SGLANG_PEER_HOST / SGLANG_SSH_USER
sudo systemctl daemon-reload && sudo systemctl enable --now sglang-wedge-watcher
```

On a single Spark, leave `SGLANG_PEER_HOST` blank — the watcher just stops the local unit (no peer step).

## Diagnosis playbook

1. **What fired:** `journalctl -u sglang-wedge-watcher` and `/var/lib/sglang-watcher/triggers.log` — the signal, the unit, the timestamp.
2. **Detokenizer-stall** (userspace, no dmesg): raise `--stream-interval`, lower `--max-running-requests`. Do **not** add `--detokenizer-worker-num 2` (NVRM OOM on unified memory).
3. **Repeated NVRM-OOM under load:** confirm the custom sm_121 NCCL is actually mapped — `grep nccl /proc/<scheduler-pid>/maps` should show your `$NCCL_HOME` build, not the wheel's. Then lower `--mem-fraction-static`.
4. **Capture a live hang:** `py-spy dump --pid <scheduler-pid>` (needs `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope`). A stuck NCCL collective shows `sched_yield → ncclProxyProgress`; the two ranks sitting in *different* collectives confirms a mismatched-collective deadlock (usually the wheel-NCCL-lacks-sm_121 bug).
5. **Recover:** the watcher already stopped both ranks; just `systemctl start` the unit again (cold load ~6-9 min).
6. **Startup fails with `RuntimeError: memory capacity is unbalanced ... occupied by other processes`**: page cache from the model's NAS/NFS reads counts against unified GPU memory and unbalances the cross-rank check. First `systemctl stop` the unit (break any `Restart=on-failure` loop so attempts don't pile up), then `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` on **every** node, then start once. Distinct from a wedge — this is an init-time memory-accounting artifact, not a deadlock.
