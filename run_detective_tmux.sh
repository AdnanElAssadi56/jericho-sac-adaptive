#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to project root (where the script is located)
cd "${SCRIPT_DIR}"

echo "Running from: $(pwd)"

# Name of tmux session
SESSION="final_detective"

# Base output directory
BASE_OUT="logging/final"

# ROM and spm - use absolute paths
ROM="${SCRIPT_DIR}/dependencies/z-machine-games-master/jericho-game-suite/detective.z5"
SPM="${SCRIPT_DIR}/models/spm_models/unigram_8k.model"


# Common args (shared by all experiments)
COMMON="--rom_path ${ROM} --spm_path ${SPM} --max_steps 50000"

# Original experiment suffixes (these match your provided names)
NAMES=(
  "baseline_no_shaping"
  "baseline_static_shaping"
  "adaptive_time_decay_exp"
  "adaptive_time_decay_linear"
  "adaptive_time_decay_cosine"
  "adaptive_sparsity_triggered"
  "adaptive_sparsity_sensitive"
  "adaptive_uncertainty_informed"
)

# Extra per-experiment args (kept as you provided)
EXTRA_ARGS=(
  "--reward_shaping False --adaptive_shaping False"
  "--reward_shaping True --adaptive_shaping False"
  "--reward_shaping True --adaptive_shaping True --scheduler_type time_decay --initial_alpha 1.0 --decay_rate 0.0002 --decay_type exponential"
  "--reward_shaping True --adaptive_shaping True --scheduler_type time_decay --initial_alpha 1.0 --decay_rate 0.001 --decay_type linear"
  "--reward_shaping True --adaptive_shaping True --scheduler_type time_decay --initial_alpha 1.0 --decay_rate 0.001 --decay_type cosine"
  "--reward_shaping True --adaptive_shaping True --scheduler_type sparsity_triggered --initial_alpha 1.0 --sparsity_threshold 50"
  "--reward_shaping True --adaptive_shaping True --scheduler_type sparsity_triggered --initial_alpha 1.0 --sparsity_threshold 25"
  "--reward_shaping True --adaptive_shaping True --scheduler_type uncertainty_informed --initial_alpha 1.0 --entropy_threshold 0.5 --uncertainty_window 50 --min_alpha 0.1 --max_alpha 2.0"
)

# Create detached tmux session
tmux new-session -d -s "${SESSION}" -n "launcher"
echo "Created tmux session '${SESSION}'. Adding windows for experiments..."

SEEDS=(0)

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  extra="${EXTRA_ARGS[$i]}"

  for seed in "${SEEDS[@]}"; do
    outdir="${SCRIPT_DIR}/${BASE_OUT}/detective/seed${seed}/${name}"
    
    # Create output directory
    mkdir -p "${outdir}"
    
    cmd="cd ${SCRIPT_DIR} && python3 src/train.py --output_dir ${outdir} ${COMMON} ${extra} --seed ${seed}"

    win_name="det_${name}_s${seed}"
    tmux new-window -t "${SESSION}" -n "${win_name}"
    tmux send-keys -t "${SESSION}:${win_name}" "${cmd}" C-m
    echo " -> Window '${win_name}': seed=${seed}, out=${outdir}"
  done
done


# Optionally kill the launcher window so first real window is index 0
tmux kill-window -t "${SESSION}:launcher" || true

echo ""
echo "All experiments started inside tmux session '${SESSION}'."
echo "Attach with: tmux attach -t ${SESSION}"
echo "List sessions: tmux ls"
echo "To view a specific window: tmux select-window -t ${SESSION}:exp0_baseline_no_shaping (or use tmux a and then prefix + w)"
