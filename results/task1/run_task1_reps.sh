#!/usr/bin/env bash
# Repeats for the task-1 transfer test (GPU non-determinism gives independent samples).
# Usage: run_task1_reps.sh <arm: base|rule50> <proto_a|proto_b> <base_port> <reps>
set -uo pipefail
ARM="${1:?arm}"; PROTO="${2:?proto}"; BP="${3:?port}"; REPS="${4:-5}"
extra=(); [ "$ARM" = rule50 ] && extra=(--min-closed-steps 50)
for r in $(seq 1 "$REPS"); do
  tag="rep_${ARM}_${PROTO##*_}_r${r}"
  port=$((BP + r))
  echo "### $tag port $port"
  bash /root/autodl-tmp/run_task1.sh "$PROTO" "$tag" "$port" "${extra[@]}"
done
echo "### REPS_DONE $ARM $PROTO"
