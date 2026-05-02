#!/bin/bash
# E4 resume script — idempotent, skips completed runs.
# Usage: bash arm_a/resume_e4.sh
# Safe to launch if run_e4_full.sh dies or system reboots.

set -uo pipefail  # NOT -e: tolerate individual failures
cd /home/ak/projects/jump

mkdir -p arm_a/e4_logs

RESTART_SERVER() {
    echo "=== Restarting llama-server ===" >&2
    SERVER_PID=$(ps aux | grep "llama-server.*Qwen3.6-35B" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        sleep 8
    fi
    /home/ak/llama.cpp/build/bin/llama-server \
      -m /home/ak/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
      --host 127.0.0.1 --port 8080 \
      --ctx-size 65536 --n-gpu-layers 99 --jinja \
      --chat-template-file /home/ak/llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja \
      --parallel 1 > /tmp/llama_server_e4.log 2>&1 &
    sleep 25
    echo "Server restarted" >&2
}

CHECK_SERVER() {
    curl -s --max-time 10 http://localhost:8080/v1/models > /dev/null 2>&1
}

run_instance() {
    local INSTANCE=$1
    local FAMILY=$2
    for i in $(seq 1 10); do
        local OUT="arm_a/results_v3_e4_${FAMILY}_${INSTANCE}_run${i}.json"
        if [ -f "$OUT" ]; then
            echo "[skip] $OUT exists" | tee -a arm_a/e4_logs/resume.log
            continue
        fi
        echo "=== E4 ${INSTANCE} run ${i}/10 ===" | tee -a arm_a/e4_logs/resume.log

        if ! CHECK_SERVER; then
            echo "[restart] Server down" | tee -a arm_a/e4_logs/resume.log
            RESTART_SERVER
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) restart ${INSTANCE} run${i}" >> arm_a/e4_logs/restarts.log
        fi

        uv run python arm_a/run_arm_a_v3.py \
            --only "${INSTANCE}" \
            --output "$OUT" \
            2>&1 | tee "arm_a/run_log_v3_e4_${FAMILY}_${INSTANCE}_run${i}.txt" || true

        if [ -f "arm_a/v3_progress_${INSTANCE}.jsonl" ]; then
            cp "arm_a/v3_progress_${INSTANCE}.jsonl" \
               "arm_a/e4_logs/v3_progress_${FAMILY}_${INSTANCE}_run${i}.jsonl"
        fi
        echo "--- done ${INSTANCE} run ${i} ---" | tee -a arm_a/e4_logs/resume.log
    done
}

echo "=== E4 RESUME START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a arm_a/e4_logs/resume.log

# Same instance order as run_e4_full.sh
for inst in world_seq_001 world_seq_002 world_seq_003 world_seq_004 world_seq_005 world_seq_006; do
    run_instance "$inst" seq
done

for inst in world_pt_001 world_pt_002 world_pt_003 world_pt_004 world_pt_005; do
    run_instance "$inst" pt
done

for inst in world_ca_001 world_ca_002 world_ca_003 world_ca_004 world_ca_005 world_ca_006; do
    run_instance "$inst" ca
done

# Final count
DONE=$(ls arm_a/results_v3_e4_*.json 2>/dev/null | wc -l)
echo "=== E4 RESUME COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) — ${DONE}/170 results exist ===" | tee -a arm_a/e4_logs/resume.log
if [ "$DONE" -ge 170 ]; then
    touch arm_a/e4_logs/e4_complete.flag
    echo "[flag] arm_a/e4_logs/e4_complete.flag created" | tee -a arm_a/e4_logs/resume.log
fi
