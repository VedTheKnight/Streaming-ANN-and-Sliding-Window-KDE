#!/bin/bash
set -euo pipefail

FILE="data/fashion_mnist/fashion-mnist_train.csv"
K=50
R=0.5
d=784
N_INSERT=50000
N_QUERIES=5000
LOGDIR="logs/jl_fm"
PLOT_DIR="plots/jl_fm"
RESULTS="jl_results_fm.csv"
MAX_PARALLEL=8

# # Sweep parameters
K_VALUES=(2 4 8 16 32 64 128 256 512)
C_VALUES=(1.5 1.6 1.7 1.8 1.9 2)

mkdir -p "$LOGDIR" "$PLOT_DIR"
echo "k,c,r,K,n_insert,n_queries,recall,cr_ann_accuracy,memory_MB" > "$RESULTS"

running_pids=()
declare -A start_times
declare -A jobs_info

script_start=$(date +%s)

for k in "${K_VALUES[@]}"; do
    for c in "${C_VALUES[@]}"; do
        LOGFILE="$LOGDIR/k${k}_c${c}.log"
        echo "[INFO] Launching k=$k c=$c (log: $LOGFILE)"
        
        python3 -u jl_memory_vs_recall.py \
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
        jobs_info[$pid]="$k $c"

        # throttle to MAX_PARALLEL
        while [[ ${#running_pids[@]} -ge $MAX_PARALLEL ]]; do
            for i in "${!running_pids[@]}"; do
                pid="${running_pids[i]}"
                if ! kill -0 "$pid" 2>/dev/null; then
                    wait "$pid"
                    end=$(date +%s)
                    elapsed=$((end - start_times[$pid]))
                    echo "[INFO] Job k=${jobs_info[$pid]} finished in ${elapsed}s"
                    # remove pid from running_pids
                    unset 'running_pids[i]'
                    break  # exit for-loop to check array length again
                fi
            done
        done
    done
done

# wait for remaining
for pid in "${running_pids[@]}"; do
    wait "$pid"
    end=$(date +%s)
    elapsed=$((end - start_times[$pid]))
    echo "[INFO] Job k=${jobs_info[$pid]} finished in ${elapsed}s"
done

# -------------------------
# Collect results into CSV
# -------------------------
echo "[INFO] Collecting results..."
for f in "$LOGDIR"/*.log; do
    line=$(grep "\[SUMMARY\]" "$f" || true)
    if [[ -n "$line" ]]; then
        k=$(echo "$line" | sed -E 's/.*k=([0-9.]+),.*/\1/')
        c=$(echo "$line" | sed -E 's/.*c=([0-9.]+),.*/\1/')
        r=$(echo "$line" | sed -E 's/.*r=([0-9.]+),.*/\1/')
        K_val=$(echo "$line" | sed -E 's/.*K=([0-9.]+),.*/\1/')
        n_insert=$(echo "$line" | sed -E 's/.*n_insert=([0-9.]+),.*/\1/')
        n_queries=$(echo "$line" | sed -E 's/.*n_queries=([0-9.]+),.*/\1/')
        recall=$(echo "$line" | sed -E 's/.*recall=([0-9.]+),.*/\1/')
        cr_ann=$(echo "$line" | sed -E 's/.*\(c,r\)-ANN=([0-9.]+),.*/\1/')
        mem=$(echo "$line" | sed -E 's/.*memory=([0-9.]+) MB.*/\1/')
        echo "$k,$c,$r,$K_val,$n_insert,$n_queries,$recall,$cr_ann,$mem" >> "$RESULTS"
    else
        echo "[WARN] No SUMMARY found in $f"
    fi
done

total_elapsed=$(( $(date +%s) - script_start ))
echo "[INFO] All jobs done in ${total_elapsed}s"

# -------------------------
# Generate plots
# -------------------------
# echo "[INFO] Generating plots..."
# python3 jl_plot.py --csv "$RESULTS" --N "$N_INSERT" --d "$d" --outdir "$PLOT_DIR"
# echo "[INFO] Plots saved in $PLOT_DIR"
