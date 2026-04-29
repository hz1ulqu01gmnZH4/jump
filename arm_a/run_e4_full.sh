#!/bin/bash
set -euo pipefail
cd /home/ak/tmux-agents/projects/jump/repo

mkdir -p arm_a/e4_logs

RESTART_SERVER() {
    echo "=== Restarting llama-server ==="
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
    echo "Server restarted"
}

CHECK_SERVER() {
    curl -s --max-time 10 http://localhost:8080/v1/models > /dev/null 2>&1
}

run_instance() {
    local INSTANCE=$1
    local FAMILY=$2
    for i in $(seq 1 10); do
        echo "=== E4 ${INSTANCE} run ${i}/10 ===" | tee -a /tmp/e4_progress.log

        # Check server health before each run
        if ! CHECK_SERVER; then
            echo "Server down, restarting..." | tee -a /tmp/e4_progress.log
            RESTART_SERVER
            echo "e4_restart ${INSTANCE} run${i} $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /tmp/e4_restarts.log
        fi

        uv run python arm_a/run_arm_a_v3.py \
            --only "${INSTANCE}" \
            --output "arm_a/results_v3_e4_${FAMILY}_${INSTANCE}_run${i}.json" \
            2>&1 | tee "arm_a/run_log_v3_e4_${FAMILY}_${INSTANCE}_run${i}.txt"

        if [ -f "arm_a/v3_progress_${INSTANCE}.jsonl" ]; then
            cp "arm_a/v3_progress_${INSTANCE}.jsonl" \
               "arm_a/e4_logs/v3_progress_${FAMILY}_${INSTANCE}_run${i}.jsonl"
        fi
        echo "--- done ${INSTANCE} run ${i} ---" | tee -a /tmp/e4_progress.log
    done
}

echo "=== E4 FULL RUN START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee /tmp/e4_progress.log

# SEQ family
for inst in world_seq_001 world_seq_002 world_seq_003 world_seq_004 world_seq_005 world_seq_006; do
    run_instance "$inst" seq
done

# PT family
for inst in world_pt_001 world_pt_002 world_pt_003 world_pt_004 world_pt_005; do
    run_instance "$inst" pt
done

# CA family
for inst in world_ca_001 world_ca_002 world_ca_003 world_ca_004 world_ca_005 world_ca_006; do
    run_instance "$inst" ca
done

echo "=== E4 FULL RUN COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a /tmp/e4_progress.log
touch /tmp/e4_complete
