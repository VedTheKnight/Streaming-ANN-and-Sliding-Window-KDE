#!/bin/bash
set -euo pipefail

# ============================
# JL Baseline: Memory vs Recall
# ============================

# --- Default values (override with args) ---
FILE="data_csvs/points.csv"
K=50
R=0.5
D=32
N_INSERT=50000
N_QUERIES=5000
LOGDIR="logs/jl_default"
RESULTS="results/jl_results_default.csv"
MAX_PARALLEL=4

# Parameter sweeps
K_VALUES=(2 4 8 16 32 )
C_VALUES=(1.5 1.6 1.7 1.8 1.9 2)

# --- Parse command-line arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --file) FILE="$2"; shift 2 ;;
    --k_values) IFS=',' read -r -a K_VALUES <<< "$2"; shift 2 ;;
    --c_values) IFS=',' read -r -a C_VALUES <<< "$2"; shift 2 ;;
    --K) K="$2"; shift 2 ;;
    --r) R="$2"; shift 2 ;;
    --d) D="$2"; shift 2 ;;
    --n_insert) N_INSERT="$2"; shift 2 ;;
    --n_queries) N_QUERIES="$2"; shift 2 ;;
    --logdir) LOGDIR="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    --max_parallel) MAX_PARALLEL="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash experiment1/jl_memory_vs_recall.sh [options]"
      echo ""
      echo "Options:"
      echo "  --file <path>           Input dataset path (default: data/sift_base.fvecs)"
      echo "  --k_values <list>       Comma-separated k values (default: 2,4,8,16,32,64,128)"
      echo "  --c_values <list>       Comma-separated c values (default: 1.6,1.7,1.8,1.9)"
      echo "  --K <int>               Recall parameter (default: 50)"
      echo "  --r <float>             Radius parameter (default: 0.5)"
      echo "  --d <int>               Dimension (default: 128)"
      echo "  --n_insert <int>        Number of insertions (default: 50000)"
      echo "  --n_queries <int>       Number of queries (default: 5000)"
      echo "  --logdir <path>         Directory for logs (default: logs/jl_default)"
      echo "  --results <path>        Output results CSV (default: results/jl_results_default.csv)"
      echo "  --max_parallel <int>    Max concurrent jobs (default: 4)"
      echo ""
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"; exit 1 ;;
  esac
done

# --- Setup directories ---
mkdir -p "$LOGDIR" "$(dirname "$RESULTS")"

echo "k,c,r,K,n_insert,n_queries,recall,cr_ann_accuracy,memory_MB" > "$RESULTS"

running_pids=()
declare -A start_times
declare -A jobs_info

script_start=$(date +%s)

# ============================
# Launch jobs
# ============================
for k in "${K_VALUES[@]}"; do
  for c in "${C_VALUES[@]}"; do
    LOGFILE="$LOGDIR/k${k}_c${c}.log"
    echo "[INFO] Launching k=$k c=$c (log: $LOGFILE)"

    python3 -u experiment1/jl_memory_vs_recall.py \
      --file "$FILE" \
      --K "$K" \
      --c "$c" \
      --r "$R" \
      --n_insert "$N_INSERT" \
      --n_queries "$N_QUERIES" \
      --k "$k" > "$LOGFILE" 2>&1 &

    pid=$!
    running_pids+=($pid)
    start_times[$pid]=$(date +%s)
    jobs_info[$pid]="$k,$c"

    # Throttle parallel jobs
    while [[ ${#running_pids[@]} -ge $MAX_PARALLEL ]]; do
      for i in "${!running_pids[@]}"; do
        pid="${running_pids[i]}"
        if ! kill -0 "$pid" 2>/dev/null; then
          wait "$pid"
          end=$(date +%s)
          elapsed=$((end - start_times[$pid]))
          echo "[INFO] Job k,c=${jobs_info[$pid]} finished in ${elapsed}s"
          unset 'running_pids[i]'
          break
        fi
      done
    done
  done
done

# --- Wait for all remaining ---
for pid in "${running_pids[@]}"; do
  wait "$pid"
  end=$(date +%s)
  elapsed=$((end - start_times[$pid]))
  echo "[INFO] Job k,c=${jobs_info[$pid]} finished in ${elapsed}s"
done

# ============================
# Collect Results
# ============================
echo "[INFO] Collecting results..."
for f in "$LOGDIR"/*.log; do
  line=$(grep "\[SUMMARY\]" "$f" || true)
  if [[ -n "$line" ]]; then
    k=$(echo "$line" | sed -E 's/.*k=([0-9.]+).*/\1/')
    c=$(echo "$line" | sed -E 's/.*c=([0-9.]+).*/\1/')
    r=$(echo "$line" | sed -E 's/.*r=([0-9.]+).*/\1/')
    K_val=$(echo "$line" | sed -E 's/.*K=([0-9.]+).*/\1/')
    n_insert=$(echo "$line" | sed -E 's/.*n_insert=([0-9.]+).*/\1/')
    n_queries=$(echo "$line" | sed -E 's/.*n_queries=([0-9.]+).*/\1/')
    recall=$(echo "$line" | sed -E 's/.*recall=([0-9.]+).*/\1/')
    cr_acc=$(echo "$line" | sed -E 's/.*\(c,r\)-ANN accuracy=([0-9.]+|None).*/\1/')
    mem=$(echo "$line" | sed -E 's/.*memory=([0-9.]+).*/\1/')
    [[ "$cr_acc" == "None" || -z "$cr_acc" ]] && cr_acc=0
    echo "$k,$c,$r,$K_val,$n_insert,$n_queries,$recall,$cr_acc,$mem" >> "$RESULTS"
  else
    echo "[WARN] No SUMMARY found in $f"
  fi
done

total_elapsed=$(( $(date +%s) - script_start ))
echo "[INFO] All jobs done in ${total_elapsed}s"

