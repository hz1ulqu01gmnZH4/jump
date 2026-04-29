#!/bin/bash
# Monitors for E4 completion, then runs aggregation and writes task output
set -euo pipefail
cd /home/ak/tmux-agents/projects/jump/repo

echo "=== E4 Completion Monitor started $(date -u) ==="

# Kill any existing monitor
# Wait for completion flag (poll every 60s)
while [ ! -f /tmp/e4_complete ]; do
    COUNT=$(ls arm_a/results_v3_e4_*.json 2>/dev/null | grep -v sanity | grep -v summary | wc -l)
    echo "[$(date -u)] waiting... ${COUNT}/170 done"
    sleep 60
done

echo "=== E4 COMPLETE detected $(date -u) ==="
COUNT=$(ls arm_a/results_v3_e4_*.json 2>/dev/null | grep -v sanity | grep -v summary | wc -l)
echo "Total result files: $COUNT"

# Run aggregation script
uv run python arm_a/e4_aggregate.py

echo "=== Task output file written ==="
echo "Monitor complete $(date -u)"
touch /tmp/e4_monitor_done
