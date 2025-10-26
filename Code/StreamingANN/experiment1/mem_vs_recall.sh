#!/bin/bash
set -euo pipefail

# ============================
# S-ANN : Memory vs Recall
# ============================

# --- Default values (can be overridden by arguments) ---
FILES="data_csvs/points.csv"
EPSILON_VALUES=(0.5 0.6 0.7 0.8 0.9 1.0)
ETA_VALUES=(0.1 0.2 0.3 0.4 0.5 0.7 0.9)
R_VALUES=(0.5)
K=50
N=50000
D=32
N_QUERIES=5000
LOGDIR="logs/default_run"
RESULTS="results/mem_vs_recall_default.csv"
MAX_PARALLEL=4

# --- Parse command-line arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --files) FILES="$2"; shift 2 ;;
    --epsilons) IFS=',' read -r -a EPSILON_VALUES <<< "$2"; shift 2 ;;
    --etas) IFS=',' read -r -a ETA_VALUES <<< "$2"; shift 2 ;;
    --r_values) IFS=',' read -r -a R_VALUES <<< "$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --n_insert) N="$2"; shift 2 ;;
    --d) D="$2"; shift 2 ;;
    --n_queries) N_QUERIES="$2"; shift 2 ;;
    --logdir) LOGDIR="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    --max_parallel) MAX_PARALLEL="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash experiment1/mem_vs_recall.sh [options]"
      echo ""
      echo "Options:"
      echo "  --files <path>          Input CSV file"
      echo "  --epsilons <list>       Comma-separated epsilon values (default: 0.5,0.6,0.7,0.8,0.9,1.0)"
      echo "  --etas <list>           Comma-separated eta values (default: 0.1,0.2,0.3,0.4,0.5,0.7,0.9)"
      echo "  --r_values <list>       Comma-separated r values (default: 1.1)"
      echo "  --k <int>               Recall parameter (default: 50)"
      echo "  --n_insert <int>        Total number of points (default: 50000)"
      echo "  --d <int>               Dimension (default: 128)"
      echo "  --n_queries <int>       Number of queries (default: 5000)"
      echo "  --logdir <path>         Directory to store logs (default: logs/default_run)"
      echo "  --results <path>        Output results CSV (default: results/mem_vs_recall_default.csv)"
      echo "  --max_parallel <int>    Max parallel jobs (default: 4)"
      echo ""
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$(dirname "$RESULTS")"
mkdir -p "$LOGDIR"

echo "r,epsilon,eta,recall,cr_ann_accuracy,memory_MB" > "$RESULTS"

running_pids=()
declare -A start_times
declare -A jobs_info

script_start=$(date +%s)

# ============================
# Launch jobs in parallel
# ============================
for R in "${R_VALUES[@]}"; do
  for EPS in "${EPSILON_VALUES[@]}"; do
    for ETA in "${ETA_VALUES[@]}"; do

      LOGFILE="$LOGDIR/r${R}_eps${EPS}_eta${ETA}.log"
      echo "[INFO] Launching r=$R eps=$EPS eta=$ETA (log: $LOGFILE)"

      python3 -u experiment1/mem_vs_recall.py \
        --files "$FILES" \
        --epsilon "$EPS" \
        --r "$R" \
        --K "$K" \
        --n "$N" \
        --n_queries "$N_QUERIES" \
        --eta "$ETA" \
        > "$LOGFILE" 2>&1 &

      pid=$!
      running_pids+=($pid)
      start_times[$pid]=$(date +%s)
      jobs_info[$pid]="$R,$EPS,$ETA"

      # throttle
      if [[ ${#running_pids[@]} -ge $MAX_PARALLEL ]]; then
        wait -n
        finished_pid=$!
        end=$(date +%s)
        elapsed=$((end - start_times[$finished_pid]))
        echo "[INFO] Finished job (${jobs_info[$finished_pid]}) in ${elapsed}s"
        # cleanup
        new_pids=()
        for pid in "${running_pids[@]}"; do
          if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
          fi
        done
        running_pids=("${new_pids[@]}")
      fi

    done
  done
done

# --- Wait for all ---
for pid in "${running_pids[@]}"; do
  wait "$pid"
  end=$(date +%s)
  elapsed=$((end - start_times[$pid]))
  echo "[INFO] Job (${jobs_info[$pid]}) finished in ${elapsed}s"
done

# ============================
# Parse Results
# ============================
echo "[INFO] Collecting results..."
for f in "$LOGDIR"/*.log; do
  line=$(grep "\[SUMMARY\]" "$f" || true)
  if [[ -n "$line" ]]; then
    r_val=$(echo "$line" | sed -E 's/.*r=([0-9.]+).*/\1/')
    eps=$(echo "$line" | sed -E 's/.*eps=([0-9.]+).*/\1/')
    eta=$(echo "$line" | sed -E 's/.*eta=([0-9.]+).*/\1/')
    recall=$(echo "$line" | sed -E 's/.*recall=([0-9.]+).*/\1/')
    cr_acc=$(echo "$line" | sed -E 's/.*\(c,r\)-ANN accuracy=([0-9.]+|None).*/\1/')
    mem=$(echo "$line" | sed -E 's/.*memory=([0-9.]+).*/\1/')
    if [[ "$cr_acc" == "None" || -z "$cr_acc" ]]; then cr_acc=0; fi
    echo "$r_val,$eps,$eta,$recall,$cr_acc,$mem" >> "$RESULTS"
  else
    echo "[WARN] No SUMMARY found in $f"
  fi
done

total_elapsed=$(( $(date +%s) - script_start ))
echo "[INFO] All jobs done in ${total_elapsed}s"

