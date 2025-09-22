#!/bin/bash
set -euo pipefail

FILES="data/sift_base.fvecs"
# FILES="data/encodings_combined.npy"
EPSILON_VALUES=(0.5 1 0.1)
ETA_VALUES=(0.1 0.2 0.3 0.4 0.5 0.7 0.9)
R_VALUES=(1)
# EPSILON_VALUES=(0.2)
# ETA_VALUES=(0.1)
# R_VALUES=(0.5)
K=50
N=50000
D=128
N_QUERIES=5000
LOGDIR="logs/sann"
RESULTS="results.csv"

# mkdir -p "$LOGDIR"
# echo "r,epsilon,eta,recall,cr_ann_acc,memory_MB" > "$RESULTS"

# MAX_PARALLEL=4
# running_pids=()
# declare -A start_times   # pid → start time
# declare -A jobs_info     # pid → "r eps eta"

script_start=$(date +%s)

# # ----------------------------
# # Launch jobs in parallel
# # ----------------------------
# for R in "${R_VALUES[@]}"; do
#   for EPS in "${EPSILON_VALUES[@]}"; do
#     for ETA in "${ETA_VALUES[@]}"; do

#       LOGFILE="$LOGDIR/r${R}_eps${EPS}_eta${ETA}.log"
#       echo "[INFO] Launching r=$R eps=$EPS eta=$ETA (log: $LOGFILE)"

#       python3 -u mem_vs_recall.py \
#           --files "$FILES" \
#           --epsilon "$EPS" \
#           --r "$R" \
#           --K "$K" \
#           --n "$N" \
#           --n_queries "$N_QUERIES" \
#           --eta "$ETA" > "$LOGFILE" 2>&1 &

#       pid=$!
#       running_pids+=($pid)
#       start_times[$pid]=$(date +%s)
#       jobs_info[$pid]="$EPS $ETA"

#       # throttle to MAX_PARALLEL
#       if [[ ${#running_pids[@]} -ge $MAX_PARALLEL ]]; then
#         wait -n
#         finished_pid=$!
#         end=$(date +%s)
#         elapsed=$((end - start_times[$finished_pid]))
#         echo "[INFO] Job eps=${jobs_info[$finished_pid]} finished in ${elapsed}s"
#         # clean running list
#         new_pids=()
#         for pid in "${running_pids[@]}"; do
#           if kill -0 "$pid" 2>/dev/null; then
#             new_pids+=("$pid")
#           fi
#         done
#         running_pids=("${new_pids[@]}")
#       fi

#     done
#   done
# done



# # Wait for remaining jobs
# for pid in "${running_pids[@]}"; do
#   wait "$pid"
#   end=$(date +%s)
#   elapsed=$((end - start_times[$pid]))
#   echo "[INFO] Job r=${jobs_info[$pid]} finished in ${elapsed}s"
# done

# echo "[INFO] All runs finished. Collecting results..."

# ----------------------------
# Parse [SUMMARY] lines safely
# ----------------------------
for f in "$LOGDIR"/*.log; do
    line=$(grep "\[SUMMARY\]" "$f" || true)
    if [[ -n "$line" ]]; then
        r_val=$(echo "$line" | sed -E 's/.*r=([0-9.]+).*/\1/')
        eps=$(echo "$line" | sed -E 's/.*eps=([0-9.]+).*/\1/')
        eta=$(echo "$line" | sed -E 's/.*eta=([0-9.]+).*/\1/')
        recall=$(echo "$line" | sed -E 's/.*recall=([0-9.]+).*/\1/')
        cr_acc=$(echo "$line" | sed -E 's/.*\(c,r\)-ANN accuracy=([0-9.]+|None).*/\1/')
        mem=$(echo "$line" | sed -E 's/.*memory=([0-9.]+).*/\1/')

        # Replace None with 0
        if [[ "$cr_acc" == "None" || -z "$cr_acc" ]]; then
            cr_acc=0
        fi

        echo "$r_val,$eps,$eta,$recall,$cr_acc,$mem" >> "$RESULTS"
    else
        echo "[WARN] No SUMMARY found in $f"
    fi
done


total_elapsed=$(( $(date +%s) - script_start ))
echo "[INFO] All jobs done in ${total_elapsed}s"

# ----------------------------
# Plotting step
# ----------------------------
echo "[INFO] Generating plots..."
python3 mem_vs_recall_plot.py --N "$N" --d "$D" --csv "$RESULTS"
echo "[INFO] Plots saved."
