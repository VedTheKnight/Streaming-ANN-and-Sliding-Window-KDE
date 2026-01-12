#!/bin/bash
set -euo pipefail

# ----------------------------
# Default values
# ----------------------------
FILES="data_csvs/points.csv"
EPSILON_VALUES=(0.5)
ETA_VALUES=(0.2 0.3 0.4 0.5 0.6 0.7 0.8)
R_VALUES=(1)
N_VALUES=(64000 96000 128000 160000)
K=50
D=128
N_QUERIES=1
LOGDIR="logs/sann_mem_variation"
RESULTS="results_mem_variation.csv"
MAX_PARALLEL=4

# ----------------------------
# Parse command-line args
# ----------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --files)
      FILES="$2"; shift 2 ;;
    --epsilons)
      IFS=',' read -ra EPSILON_VALUES <<< "$2"; shift 2 ;;
    --etas)
      IFS=',' read -ra ETA_VALUES <<< "$2"; shift 2 ;;
    --r_values)
      IFS=',' read -ra R_VALUES <<< "$2"; shift 2 ;;
    --N_values)
      IFS=',' read -ra N_VALUES <<< "$2"; shift 2 ;;
    --K)
      K="$2"; shift 2 ;;
    --d)
      D="$2"; shift 2 ;;
    --n_queries)
      N_QUERIES="$2"; shift 2 ;;
    --logdir)
      LOGDIR="$2"; shift 2 ;;
    --results)
      RESULTS="$2"; shift 2 ;;
    --max_parallel)
      MAX_PARALLEL="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ----------------------------
# Main script below 
# ----------------------------
mkdir -p "$LOGDIR"
echo "N,r,epsilon,eta,recall,cr_ann_acc,memory_MB" > "$RESULTS"

running_pids=()
declare -A start_times
declare -A jobs_info
script_start=$(date +%s)

for R in "${R_VALUES[@]}"; do
  for N in "${N_VALUES[@]}"; do
    for EPS in "${EPSILON_VALUES[@]}"; do
      for ETA in "${ETA_VALUES[@]}"; do
        LOGFILE="$LOGDIR/N${N}_r${R}_eps${EPS}_eta${ETA}.log"
        echo "[INFO] Launching N=$N r=$R eps=$EPS eta=$ETA (log: $LOGFILE)"

        python3 -u mem_vs_recall.py \
            --files "$FILES" \
            --epsilon "$EPS" \
            --r "$R" \
            --K "$K" \
            --n "$N" \
            --n_queries "$N_QUERIES" \
            --eta "$ETA" > "$LOGFILE" 2>&1 &

        pid=$!
        running_pids+=($pid)
        start_times[$pid]=$(date +%s)
        jobs_info[$pid]="$N $EPS $ETA"

        # throttle parallel jobs
        if [[ ${#running_pids[@]} -ge $MAX_PARALLEL ]]; then
          wait -n
          finished_pid=$!
          end=$(date +%s)
          elapsed=$((end - start_times[$finished_pid]))
          echo "[INFO] Job ${jobs_info[$finished_pid]} finished in ${elapsed}s"

          # remove finished pid from list
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
done

# Wait for remaining jobs
for pid in "${running_pids[@]}"; do
  wait "$pid"
  end=$(date +%s)
  elapsed=$((end - start_times[$pid]))
  echo "[INFO] Job ${jobs_info[$pid]} finished in ${elapsed}s"
done

echo "[INFO] All runs finished. Collecting results..."

# ----------------------------
# Parse [SUMMARY] lines safely
# ----------------------------
for f in "$LOGDIR"/*.log; do
    line=$(grep "\[SUMMARY\]" "$f" || true)
    if [[ -n "$line" ]]; then
        N_val=$(echo "$f" | sed -E 's/.*N([0-9]+)_r.*/\1/')
        r_val=$(echo "$line" | sed -E 's/.*r=([0-9.]+).*/\1/')
        eps=$(echo "$line" | sed -E 's/.*eps=([0-9.]+).*/\1/')
        eta=$(echo "$line" | sed -E 's/.*eta=([0-9.]+).*/\1/')
        recall=$(echo "$line" | sed -E 's/.*recall=([0-9.]+).*/\1/')
        cr_acc=$(echo "$line" | sed -E 's/.*\(c,r\)-ANN accuracy=([0-9.]+|None).*/\1/')
        mem=$(echo "$line" | sed -E 's/.*memory=([0-9.]+).*/\1/')

        if [[ "$cr_acc" == "None" || -z "$cr_acc" ]]; then
            cr_acc=0
        fi

        echo "$N_val,$r_val,$eps,$eta,$recall,$cr_acc,$mem" >> "$RESULTS"
    else
        echo "[WARN] No SUMMARY found in $f"
    fi
done

total_elapsed=$(( $(date +%s) - script_start ))
echo "[INFO] All jobs done in ${total_elapsed}s"
