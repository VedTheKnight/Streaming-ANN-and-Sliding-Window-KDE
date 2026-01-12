#!/bin/bash
set -eu

LOGDIR="sann_fm"     # change to your actual log directory
RESULTS="final_sann_fm.csv"

echo "[INFO] Collecting SANN results..."
echo "r,epsilon,eta,recall,cr_ann_accuracy,memory_MB" > "$RESULTS"

for f in "$LOGDIR"/*.log; do
    line=$(grep "\[SUMMARY\]" "$f" || true)
    if [[ -n "$line" ]]; then
        r=$(echo "$line" | sed -E 's/.*r=([0-9.]+),.*/\1/')
        epsilon=$(echo "$line" | sed -E 's/.*eps=([0-9.]+),.*/\1/')
        eta=$(echo "$line" | sed -E 's/.*eta=([0-9.]+),.*/\1/')
        recall=$(echo "$line" | sed -E 's/.*recall=([0-9.]+),.*/\1/')
        cr_ann=$(echo "$line" | sed -E 's/.*\(c,r\)-ANN( accuracy)?=([0-9.]+),.*/\2/')
        mem=$(echo "$line" | sed -E 's/.*memory=([0-9.]+).*/\1/')
        echo "$r,$epsilon,$eta,$recall,$cr_ann,$mem" >> "$RESULTS"
    else
        echo "[WARN] No SUMMARY found in $f"
    fi
done

echo "[INFO] Results saved to $RESULTS"
