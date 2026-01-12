#!/bin/bash
# qps_script.sh
# Runs JL and ANN experiments in parallel

# --- Config (with defaults) ---
N_POINTS=${1:-10000}       # total points
N_QUERIES=${2:-100}        # number of queries
EPSILON=${3:-0.5}          # epsilon for ANN/JL
LOG_FILE=${4:-qps_log.txt}       # log file
RESULTS_FILE=${5:-qps_results.csv}  # results CSV
MAX_JOBS=${6:-4}           # max parallel jobs

echo "[INFO] Config: N_POINTS=$N_POINTS, N_QUERIES=$N_QUERIES, EPSILON=$EPSILON, LOG_FILE=$LOG_FILE, RESULTS_FILE=$RESULTS_FILE, MAX_JOBS=$MAX_JOBS"

# Clean old logs
rm -f "$LOG_FILE" "$RESULTS_FILE"

# Helper to run one experiment
run_experiment() {
    dataset=$1
    method=$2
    epsilon=$3
    eta=$4
    k=$5

    if [[ "$method" == "ann" ]]; then
        echo "[INFO] Running ANN on $(basename "$dataset") (eta=$eta)"
        python qps_vs_recall.py \
            --files "$dataset" \
            --method "ann" \
            --epsilon "$epsilon" \
            --n_points "$N_POINTS" \
            --n_queries "$N_QUERIES" \
            --eta "$eta" \
            >> "$LOG_FILE" 2>&1
    else
        echo "[INFO] Running JL on $(basename "$dataset") (k=$k)"
        python qps_vs_recall.py \
            --files "$dataset" \
            --method "jl" \
            --epsilon "$epsilon" \
            --n_points "$N_POINTS" \
            --n_queries "$N_QUERIES" \
            --k "$k" \
            >> "$LOG_FILE" 2>&1
    fi
}

# Iterate over datasets
for dataset in data_csvs/fashion_mnist.csv data_csvs/sift-1m.csv data_csvs/synthetic_points.csv; do
    case $(basename "$dataset") in
        fashion_mnist.csv) ks=(4 8 16 32 64 128 256 512 784) ;;
        sift-1m.csv)       ks=(4 8 16 32 64 128) ;;
        synthetic_points.csv) ks=(4 8 16 32) ;;
    esac

    # --- JL: vary over k only ---
    for k in "${ks[@]}"; do
        while (( $(jobs | wc -l) >= MAX_JOBS )); do sleep 1; done
        run_experiment "$dataset" "jl" "$EPSILON" "0" "$k" &
    done

    # --- ANN: vary over eta only ---
    etas=(0 0.1 0.2 0.3 0.5 0.7 0.9)
    for eta in "${etas[@]}"; do
        while (( $(jobs | wc -l) >= MAX_JOBS )); do sleep 1; done
        run_experiment "$dataset" "ann" "$EPSILON" "$eta" "0" &
    done
done

# Wait for all jobs to finish
wait

# --- Parse results into CSV ---
echo "method,dataset,eta,k,recall,QPS,memory_MB" > "$RESULTS_FILE"
grep "\[Result\]" "$LOG_FILE" | awk '
{
    for (i=1; i<=NF; i++) {
        if ($i ~ /^method=/)  { split($i, a, "="); method=a[2]; }
        else if ($i ~ /^dataset=/) { split($i, a, "="); dataset=a[2]; }
        else if ($i ~ /^eta=/) { split($i, a, "="); eta=a[2]; }
        else if ($i ~ /^k=/) { split($i, a, "="); k=a[2]; }
        else if ($i ~ /^recall=/) { split($i, a, "="); recall=a[2]; gsub(",", "", recall); }
        else if ($i ~ /^QPS=/) { split($i, a, "="); qps=a[2]; gsub(",", "", qps); }
        else if ($i ~ /^memory=/) { split($i, a, "="); mem=a[2]; }
    }
    print method","dataset","eta","k","recall","qps","mem
}' >> "$RESULTS_FILE"

echo "[DONE] Results written to $RESULTS_FILE"
