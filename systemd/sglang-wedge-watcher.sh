#!/usr/bin/env bash
# Tail the journal for early-warning signals that SGLang TP=2 is about to wedge
# the GB10 box (a memcg rw-semaphore deadlock under unified-memory pressure; see
# docs/failure-modes.md). On first match, stop the currently
# active sglang-*-tp2 service so the kernel can reclaim memory cleanly instead
# of locking up and forcing a hard reset. Also stops the peer node's service
# because TP=2 is one logical job — leaving one rank running causes NCCL hang.
#
# Signal ordering observed on 2026-05-17:
#   1. SGLang "Health check failed ... detokenizer for last N seconds"   <- earliest
#   2. NVRM "Out of memory _memdescAllocInternal"                         <- minutes later
#   3. hung_task / blocked-for-more-than / memcg_rstat                    <- deadlock active
#   4. journal cutoff -> hard reset required                              <- terminal
#
# We fire on signal 1 (earliest actionable). Signal 3 is belt-and-braces in
# case detokenizer health-check format changes.
#
# Signal 2 (NVRM) is gated: kernel emits this during transient large allocs
# in NVFP4/TP=2 init (KV pool, cudagraph capture, NEXTN draft load) which
# retry successfully. Three consecutive cold starts were false-positive-killed
# on 2026-05-17 PM. The gate requires the active service to be in `active`
# state (not `activating`) AND to have been active > 60s before NVRM fires.
# ActiveEnterTimestampMonotonic returns 0 during `activating`, which the gate
# handles via the state-string check (NOT arithmetic on the 0 timestamp, which
# would fail open).
#
# Runs as root (needs systemctl-stop permission + journal read + ssh peer).
set -uo pipefail

# Load deployment config (SGLANG_PEER_HOST + SGLANG_SSH_USER for the multi-Spark
# peer-stop). Optional: on a single Spark, leave both unset.
SGLANG_SPARK_ENV="${SGLANG_SPARK_ENV:-/etc/sglang-spark/spark-fabric.env}"
[ -f "$SGLANG_SPARK_ENV" ] && { set -a; . "$SGLANG_SPARK_ENV"; set +a; }

PATTERNS='detokenizer for last [0-9]+ seconds|NVRM:.*Out of memory|hung_task.*sglang|blocked for more than [0-9]+ seconds.*sglang|memcg_rstat_updated'

# Grace period after service enters `active` state before NVRM can fire.
# Init transients have already happened by the time state==active, but a brief
# settling window catches any post-active spikes during warmup.
# Bumped from 60s on 2026-05-26: Mamba KV cache allocation for 35B-A3B-NVFP4
# triggers NVRM-OOM-like signature at ~102s during boot.
NVRM_GRACE_S=180

STATE_DIR=/var/lib/sglang-watcher
TRIGGER_LOG="$STATE_DIR/triggers.log"
mkdir -p "$STATE_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Find the currently-active sglang TP=2 service (only one at a time per
# HAProxy-frontend convention). Returns empty if none active.
current_tp2_service() {
  for svc in $(systemctl list-unit-files --no-legend 'sglang-*-tp2.service' 2>/dev/null | awk '{print $1}'); do
    if systemctl is-active --quiet "$svc"; then
      echo "$svc"
      return
    fi
  done
}

# TP=2 peer node: hostname/alias that resolves (e.g. via /etc/hosts).
# Set SGLANG_PEER_HOST in spark-fabric.env. Empty on a single Spark (no peer-stop).
peer_node() {
  echo "${SGLANG_PEER_HOST:-}"
}

# Returns 0 if the NVRM gate is satisfied (OK to fire), 1 if NVRM should be
# skipped because the service is still settling. svc passed in as $1.
nvrm_gate_open() {
  local svc="$1"
  local state active_us uptime_us age_s
  state=$(systemctl is-active "$svc" 2>/dev/null)
  if [ "$state" != "active" ]; then
    # `activating` or transient state — kernel NVRM here is init transient.
    log "init-phase kernel mem trigger (skipped): $svc state=$state"
    return 1
  fi
  active_us=$(systemctl show "$svc" -p ActiveEnterTimestampMonotonic --value 2>/dev/null)
  if [ -z "$active_us" ] || [ "$active_us" = "0" ]; then
    log "init-phase kernel mem trigger (skipped): $svc has no active-enter timestamp yet"
    return 1
  fi
  # /proc/uptime is "<float-seconds> <float-idle>"; awk to microseconds (int).
  uptime_us=$(awk '{printf "%.0f", $1 * 1000000}' /proc/uptime)
  age_s=$(( (uptime_us - active_us) / 1000000 ))
  if [ "$age_s" -lt "$NVRM_GRACE_S" ]; then
    log "init-phase kernel mem trigger (skipped): $svc active for ${age_s}s (<${NVRM_GRACE_S}s grace)"
    return 1
  fi
  return 0
}

log "watcher starting on $(hostname) (peer=$(peer_node))"
# NOTE: do NOT echo $PATTERNS or any matched $line to stdout outside of the
# exit-0 branch — our own log lines feed back through journalctl and would
# self-trigger, creating an infinite match loop. The gate-skip log lines above
# deliberately avoid the substrings "NVRM:" and "Out of memory" (they say
# "kernel mem trigger" instead) so they cannot self-match PATTERNS.

# --since=now: don't replay history (avoids re-firing on past trigger entries)
# --line-buffered: per-line latency from grep
# --no-pager: prevent block buffering / paging
journalctl -f --no-pager --since=now 2>/dev/null \
  | grep -E --line-buffered "$PATTERNS" \
  | while IFS= read -r line; do
      svc=$(current_tp2_service)
      if [ -z "$svc" ]; then
        # Suppress $line in this log path — see self-trigger NOTE above.
        log "trigger pattern observed but no active sglang-*-tp2 service; not stopping"
        continue
      fi
      # NVRM gate: skip transient init-phase emissions. Other patterns
      # (detokenizer heartbeat / hung_task / blocked-for-more-than / memcg)
      # cannot false-positive during init by their nature, so they go through.
      case "$line" in
        *"NVRM"*"Out of memory"*)
          if ! nvrm_gate_open "$svc"; then
            continue
          fi
          ;;
      esac
      # Safe to include $line here because we exit immediately after stopping;
      # systemd restarts the watcher with a fresh --since=now ~10s later,
      # past our own log entry.
      log "MATCH on $svc: $line"
      printf '%s\t%s\t%s\n' "$(date -Iseconds)" "$svc" "$line" >> "$TRIGGER_LOG"
      log "stopping $svc locally"
      systemctl stop "$svc" || log "WARN: systemctl stop $svc returned non-zero"
      peer=$(peer_node)
      if [ -n "$peer" ]; then
        log "stopping $svc on peer $peer (TP=2 is one logical job)"
        # Watcher runs as root, but the cross-node SSH keys + NOPASSWD sudo are
        # set up for an unprivileged user. runuser switches to that user
        # (SGLANG_SSH_USER) for the ssh + sudo chain.
        runuser -u "${SGLANG_SSH_USER:?set SGLANG_SSH_USER in spark-fabric.env}" -- ssh -o ConnectTimeout=5 -o BatchMode=yes "$peer" "sudo -n systemctl stop $svc" \
          || log "WARN: peer stop on $peer failed; TP=2 partner may hang"
      else
        log "WARN: unknown local hostname, cannot stop peer; TP=2 partner may hang"
      fi
      log "stopped $svc (local+peer); exiting (systemd will restart watcher with fresh --since=now)"
      exit 0
    done

log "journal tail pipeline ended unexpectedly; exiting for systemd restart"
exit 1
