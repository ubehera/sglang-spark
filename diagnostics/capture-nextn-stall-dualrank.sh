#!/usr/bin/env bash
# capture-nextn-stall-dualrank.sh
#
# Dual-rank py-spy capture for the cross-node NEXTN detokenizer-stall (the
# "Server couldn't get a response from detokenizer for last N seconds" 503).
#
# WHY THIS EXISTS (source-grounded, NOT yet py-spy-confirmed):
#   The health-check 503 is emitted by http_server.py:572 when
#   tokenizer_manager.last_receive_tstamp goes stale for HEALTH_CHECK_TIMEOUT
#   (default 20s; SGLANG_HEALTH_CHECK_TIMEOUT can raise it to 60). The ONLY
#   heartbeat-bearing emit in a busy server is scheduler.maybe_send_health_check_signal()
#   (scheduler.py:3103), which runs at the END of process_batch_result(). In the
#   overlap loop (scheduler.py:1404) every iteration calls pop_and_process() ->
#   process_batch_result -> batch_result_processor.process_batch_result_decode,
#   whose FIRST statement is result.copy_done.synchronize() (batch_result_processor.py:594).
#   copy_done (recorded at utils.py:123 after the spec forward is enqueued on
#   forward_stream) transitively gates the entire NEXTN forward: draft k=3 loop
#   (eagle_worker_v2.py:494 draft_runner.forward x3) + verify
#   (eagle_worker_v2.py:1105 target forward) + draft_extend
#   (eagle_worker_v2.py:897), every layer of which issues
#   tensor_model_parallel_all_reduce (layers/linear.py) across the two nodes.
#
#   HYPOTHESIS (confirm/refute with this capture): if one rank's spec path
#   participates in a different *count/shape* of NCCL collectives than its peer
#   (the SIBLING of d3433cb09, which was the init-time analog), the peer blocks
#   inside the collective -> copy_done never fires -> copy_done.synchronize()
#   hangs -> maybe_send_health_check_signal() never runs -> 503. This is a
#   SCHEDULER-ITERATION block, NOT detok batch_decode slowness.
#
# DECISION RULE (which stack proves which hypothesis):
#   * SCHEDULER-ITERATION BLOCK (upstream #22511 family) -- the bug we expect:
#       BOTH ranks' MAIN scheduler thread (named "sglang::scheduler" /
#       "Scheduler.event_loop_overlap") is parked in ONE of:
#         - batch_result_processor.py:594  copy_done.synchronize()   (cuda event wait)
#         - eagle_worker_v2.py:1105  target verify forward            (in a c10d/nccl frame)
#         - eagle_worker_v2.py:494/517 draft_forward                 (in a c10d/nccl frame)
#         - eagle_worker_v2.py:897  _draft_extend_for_decode          (in a c10d/nccl frame)
#         - scheduler.py:1412  schedule_stream.wait_stream            (stream-order wait)
#       The DETOKENIZER process ("sglang::detokenizer") is IDLE in
#       detokenizer_manager.py:149 recv_pyobj() (blocked on an EMPTY zmq PULL) --
#       i.e. it has NO work; it is starved, not slow. THIS confirms scheduler block.
#       BONUS (the sibling signature): the two ranks are in DIFFERENT frames
#       (e.g. rank0 in copy_done.synchronize while rank1 is deep in a nccl
#       allReduce) == cross-rank collective desync. Capture BOTH-rank stacks to see it.
#
#   * GENUINE DETOK THROUGHPUT (the alternative to RULE OUT):
#       The DETOKENIZER process is ACTIVELY RUNNING (state R, not in recv_pyobj)
#       inside detokenizer_manager.py _grouped_batch_decode -> tokenizer.batch_decode
#       (HF/tokenizers Rust frame) for many consecutive samples, WHILE the scheduler
#       main thread is progressing (different stacks across snapshots, sending output).
#       In that world the zmq PULL queue from scheduler->detok is BACKED UP. Source
#       makes this UNLIKELY for the 503: the detok never updates last_receive_tstamp
#       directly; the heartbeat is scheduler-emitted and only this path is checked.
#       But capture it to be honest -- if detok is pegged AND scheduler is healthy,
#       the mechanism is different and the d3433cb09-sibling patch is the wrong fix.
#
# This script does NOT trigger a stall. Run it AGAINST an already-stalled service
# (watcher about to fire, or /health returning 503) on BOTH nodes simultaneously.
# It takes 3 snapshots 5s apart per process so you can confirm the stack is STUCK
# (identical across snapshots) vs merely slow (advancing).
#
# Usage (run on BOTH node-1 and node-2 within the same ~10s window):
#   sudo bash capture-nextn-stall-dualrank.sh
# Then copy /tmp/nextn-stall-*/ off both nodes and diff the rank0 vs rank1 stacks.

set -u
OUT="/tmp/nextn-stall-$(hostname -s)-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
echo "[capture] writing to $OUT on $(hostname -s)"

# py-spy must be on PATH (pip install py-spy in a sys venv, or use the sglang .venv).
PYSPY="${PYSPY:-py-spy}"
if ! command -v "$PYSPY" >/dev/null 2>&1; then
  if [ -x /home/umankb/projects/sglang-server/.venv/bin/py-spy ]; then
    PYSPY=/home/umankb/projects/sglang-server/.venv/bin/py-spy
  else
    echo "[capture] ERROR: py-spy not found. pip install py-spy first." >&2
    exit 1
  fi
fi

# Identify the relevant processes by setproctitle names set in SGLang:
#   scheduler -> "sglang::scheduler"   (run_scheduler_process / setproctitle)
#   detok     -> "sglang::detokenizer" (detokenizer_manager.run_detokenizer_process:432)
mapfile -t SCHED_PIDS < <(pgrep -f "sglang::scheduler" || true)
mapfile -t DETOK_PIDS < <(pgrep -f "sglang::detokenizer" || true)

# Fallback: some builds keep the generic python proctitle. Grab launch_server PIDs too.
if [ "${#SCHED_PIDS[@]}" -eq 0 ]; then
  mapfile -t SCHED_PIDS < <(pgrep -f "sglang.launch_server" || true)
fi

{
  echo "host=$(hostname -s) date=$(date -Is)"
  echo "scheduler_pids=${SCHED_PIDS[*]:-NONE}"
  echo "detok_pids=${DETOK_PIDS[*]:-NONE}"
  echo "--- /health (rank0 only serves /v1; rank1 is health-dummy) ---"
  curl -s -m 5 -o /dev/null -w "health_http=%{http_code}\n" http://127.0.0.1:30000/health || echo "health_http=curl_failed"
  echo "--- process states ---"
  for p in "${SCHED_PIDS[@]}" "${DETOK_PIDS[@]}"; do
    [ -n "$p" ] && ps -o pid,stat,pcpu,etimes,comm -p "$p" 2>/dev/null
  done
} > "$OUT/context.txt" 2>&1
cat "$OUT/context.txt"

snap() {  # $1=pid $2=label
  local pid="$1" label="$2"
  for i in 1 2 3; do
    echo "[capture] $label pid=$pid snapshot $i/3"
    # --native shows the C/NCCL frames so we can SEE the allReduce; needs root.
    "$PYSPY" dump --pid "$pid" --native > "$OUT/${label}-pid${pid}-snap${i}.txt" 2>&1 \
      || "$PYSPY" dump --pid "$pid" > "$OUT/${label}-pid${pid}-snap${i}.txt" 2>&1
    sleep 5
  done
}

for p in "${SCHED_PIDS[@]}"; do [ -n "$p" ] && snap "$p" "scheduler"; done
for p in "${DETOK_PIDS[@]}"; do [ -n "$p" ] && snap "$p" "detokenizer"; done

echo "[capture] DONE -> $OUT"
echo "[capture] Next: diff scheduler stacks across the two NODES (rank0 vs rank1)."
echo "          If they sit in DIFFERENT spec frames while detok is idle in recv_pyobj,"
echo "          it is the d3433cb09-SIBLING cross-rank collective desync (scheduler block)."
echo "          If detok is pegged in batch_decode while scheduler advances, it is the"
echo "          (refuted-by-source-but-verify) genuine-detok-throughput alternative."
