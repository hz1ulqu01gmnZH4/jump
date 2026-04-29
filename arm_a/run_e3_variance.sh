#!/usr/bin/env bash
# E3 variance baseline: 10xCA + 5xSEQ + 5xPT
set -euo pipefail

cd /home/ak/tmux-agents/projects/jump/repo

LOG_DIR="arm_a/e3_logs"
mkdir -p "$LOG_DIR"

run_instance() {
    local world=$1
    local idx=$2
    local total=$3
    local label=$4

    echo "=== ${label} run ${idx}/${total} ===" | tee -a "$LOG_DIR/master.log"
    local out="arm_a/results_v3_e3_${label}_run${idx}.json"
    local log="arm_a/run_log_v3_e3_${label}_run${idx}.txt"
    local progress_src="arm_a/v3_progress_${world}.jsonl"
    local progress_bak="$LOG_DIR/v3_progress_${label}_run${idx}.jsonl"

    uv run python arm_a/run_arm_a_v3.py \
        --only "${world}" \
        --output "${out}" \
        2>&1 | tee "${log}"

    # Backup JSONL before next run overwrites it
    if [ -f "${progress_src}" ]; then
        cp "${progress_src}" "${progress_bak}"
        echo "  -> JSONL backed up: ${progress_bak}" | tee -a "$LOG_DIR/master.log"
    else
        echo "  -> WARNING: ${progress_src} not found" | tee -a "$LOG_DIR/master.log"
    fi

    echo "--- done ${label} run ${idx} ---" | tee -a "$LOG_DIR/master.log"
    echo "" | tee -a "$LOG_DIR/master.log"
}

echo "=== E3 variance baseline start: $(date) ===" | tee "$LOG_DIR/master.log"

# Track 1: CA (10 runs)
for i in $(seq 1 10); do
    run_instance "world_ca_001" "$i" 10 "ca"
done

# Track 2: SEQ (5 runs)
for i in $(seq 1 5); do
    run_instance "world_seq_001" "$i" 5 "seq"
done

# Track 3: PT (5 runs)
for i in $(seq 1 5); do
    run_instance "world_pt_001" "$i" 5 "pt"
done

echo "=== E3 variance baseline complete: $(date) ===" | tee -a "$LOG_DIR/master.log"
