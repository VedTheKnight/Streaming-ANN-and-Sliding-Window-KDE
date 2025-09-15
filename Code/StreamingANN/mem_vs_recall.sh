#!/bin/bash

# Paths to your dataset shards
FILES="data/encodings_combined.npy"

# ANN parameters
EPSILON=0.1
R=1.0
K=10
N=10000
N_QUERIES=500

# Number of eta values
NUM_ETA=7

# Directory for individual logs
LOGDIR="logs"
mkdir -p $LOGDIR

# Clear old logs
rm -f $LOGDIR/*.log

echo "Launching sequential ANN runs..."

# Generate evenly spaced eta values from 0 to 1 using awk
ETAS=($(awk -v n=$NUM_ETA 'BEGIN{for(i=0;i<n;i++){printf "%.3f ", i/(n-1)}}'))

# Run each eta sequentially
for ETA in "${ETAS[@]}"; do
    LOGFILE="$LOGDIR/eta_${ETA}.log"
    echo "Running eta=$ETA -> $LOGFILE"
    
    python mem_vs_recall.py \
        --files $FILES \
        --epsilon $EPSILON \
        --r $R \
        --K $K \
        --n $N \
        --n_queries $N_QUERIES \
        --eta $ETA \
        > "$LOGFILE" 2>&1

done

echo "All runs completed."

# Combine filtered logs into a single file
COMBINED_LOG="combined_results.log"
> "$COMBINED_LOG"  # empty file first

for FILE in $LOGDIR/*.log; do
    # Only include lines from "Total dataset points:" onward
    awk '/Total dataset points:/ {flag=1} flag' "$FILE" >> "$COMBINED_LOG"
done

echo "Filtered combined log saved to $COMBINED_LOG"
