#!/bin/bash
# What the cluster runs are doing: the queue, jobs that ended in the last day, and each
# PESO run's finished stages and latest progress bar. Read-only.
#
#   bash /mnt/scratch/${USER:0:1}/$USER/peso/cp4285/replication/peso/slurm/status.sh
W=${PESO_WORKSPACE:-/mnt/scratch/${USER:0:1}/$USER/peso}
echo "== Queue"
squeue -u "$USER" -o "%.8i %.9P %.8T %.10M %.22b %R"
echo "== Ended in the last 24 h"
sacct -X -n -S now-24hours -o JobID,JobName%12,State%12,Elapsed,NodeList \
    | grep -v -E "RUNNING|PENDING"
echo "== PESO runs (finished stages | latest log)"
for run in "$W"/peso-runs/runs/*/; do
    stages=$(ls "$run/done" 2>/dev/null | sed 's/\.json$//' | tr '\n' ' ')
    log=$(ls -t "$run"/logs/*.log 2>/dev/null | head -1)
    bar=""
    if [ -n "$log" ]; then
        bar="$(basename "$log" .log): $(tail -c 3000 "$log" | tr '\r' '\n' \
            | grep -a -E '[0-9]+/[0-9]+ \[' | tail -1 | sed 's/^ *//' | cut -c1-60)"
    fi
    echo "$(basename "$run") | ${stages:-none} | $bar"
done
echo "== SASRec blocks runs"
ls "$W/blocks-runs" 2>/dev/null
