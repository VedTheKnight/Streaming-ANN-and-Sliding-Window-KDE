#!/bin/bash
set -eu

LOGDIR="jl"   # change to your actual logs directory
RESULTS="final_jl_sift.csv"

echo "[INFO] Collecting results..."
echo "k,c,r,K,n_insert,n_queries,recall,cr_ann_accuracy,memory_MB" > "$RESULTS"

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

        # Handle both variants of (c,r)-ANN or (c,r)-ANN accuracy
        cr_ann=$(echo "$line" | sed -E 's/.*\(c,r\)-ANN( accuracy)?=([0-9.]+),.*/\2/')

        # Handle memory with or without 'MB' suffix
        mem=$(echo "$line" | sed -E 's/.*memory=([0-9.]+).*/\1/')

        echo "$k,$c,$r,$K_val,$n_insert,$n_queries,$recall,$cr_ann,$mem" >> "$RESULTS"
    else
        echo "[WARN] No SUMMARY found in $f"
    fi
done

echo "[INFO] Results saved to $RESULTS"
