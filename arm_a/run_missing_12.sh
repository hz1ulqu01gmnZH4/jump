#!/bin/bash
# Run the 12 missing E4 CA runs that failed with 503 during the main resume.
# Missing: ca_002 runs 7-10, ca_003 runs 1-8
set -uo pipefail
cd /home/ak/tmux-agents/projects/jump/repo

LOG=/tmp/e4_missing12.log
echo "=== MISSING-12 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

CHECK_SERVER() {
    curl -s --max-time 10 "http://localhost:8080/v1/models" | grep -q "model" 2>/dev/null
}

RESTART_SERVER() {
    echo "[restart] llama-server down — restarting" | tee -a "$LOG"
    SERVER_PID=$(ps aux | grep "llama-server.*Qwen3.6-35B" | grep -v grep | awk '{print $2}' | head -1)
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
    sleep 8
    /home/ak/llama.cpp/build/bin/llama-server \
      -m /home/ak/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
      --host 127.0.0.1 --port 8080 \
      --ctx-size 65536 --n-gpu-layers 99 --jinja \
      --chat-template-file /home/ak/llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja \
      --parallel 1 > /tmp/llama_server_e4.log 2>&1 &
    # Wait until server returns valid models response
    for attempt in $(seq 1 30); do
        sleep 5
        if CHECK_SERVER; then
            echo "[restart] Server ready after ${attempt} attempts" | tee -a "$LOG"
            return 0
        fi
    done
    echo "[restart] Server failed to start after 150s" | tee -a "$LOG"
    return 1
}

run_one() {
    local INST=$1 RUN=$2
    local OUT="arm_a/results_v3_e4_ca_${INST}_run${RUN}.json"
    if [ -f "$OUT" ]; then
        echo "[skip] $OUT exists" | tee -a "$LOG"
        return 0
    fi
    echo "=== MISSING12 ${INST} run ${RUN} ===" | tee -a "$LOG"

    # Verify server responds with valid JSON (stricter than resume_e4.sh)
    if ! CHECK_SERVER; then
        RESTART_SERVER
    fi

    uv run python arm_a/run_arm_a_v3.py \
        --only "${INST}" \
        --output "$OUT" \
        2>&1 | tee "arm_a/run_log_v3_e4_ca_${INST}_run${RUN}.txt" || true

    if [ -f "$OUT" ]; then
        echo "--- done ${INST} run ${RUN} (file written) ---" | tee -a "$LOG"
    else
        echo "--- FAIL ${INST} run ${RUN} (no file written) ---" | tee -a "$LOG"
    fi
}

# ca_002 runs 7-10
for r in 7 8 9 10; do
    run_one "world_ca_002" "$r"
done

# ca_003 runs 1-8
for r in 1 2 3 4 5 6 7 8; do
    run_one "world_ca_003" "$r"
done

DONE=$(ls arm_a/results_v3_e4_*.json 2>/dev/null | grep -v sanity | grep -v summary | wc -l)
echo "=== MISSING-12 COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) — ${DONE}/170 results exist ===" | tee -a "$LOG"
if [ "$DONE" -ge 170 ]; then
    touch arm_a/e4_logs/e4_complete.flag
    echo "[flag] e4_complete.flag created" | tee -a "$LOG"
fi
