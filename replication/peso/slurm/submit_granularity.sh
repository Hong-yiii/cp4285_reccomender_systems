#!/bin/bash
# Submit one `cp4285 adapter-blocks` job per line of a granularity plan (see
# granularity_plans.py). Prints the sbatch commands unless SUBMIT=1, so the default is a
# dry run. Branch jobs wait for their level/seed's init job with --dependency=afterok.
# A run whose metrics.json already exists is skipped by the job script.
#
#   bash submit_granularity.sh plans/runs.tsv
#   SUBMIT=1 SRC=/mnt/scratch/$USER/peso/cp4285-main/src bash submit_granularity.sh plans/runs.tsv
#
# TIME may override the 3-hour partition limit.
set -euo pipefail
PLAN=${1:?usage: submit_granularity.sh PLAN.tsv}
TIME=${TIME:-03:00:00}
HERE=$(cd "$(dirname "$0")" && pwd)
declare -A INIT_JOBS
while IFS=$'\t' read -r run data categories names args gres; do
    [ -z "$run" ] && continue
    init_run=""
    if [[ $run =~ ^(.+-s[0-9]+)- ]]; then
        init_run="${BASH_REMATCH[1]}-init"
    fi
    dependency=()
    if [ -n "$init_run" ] && [ -n "${INIT_JOBS[$init_run]:-}" ]; then
        dependency=(--dependency="afterok:${INIT_JOBS[$init_run]}")
    fi
    if [ "${SUBMIT:-0}" = 1 ]; then
        job=$(RUN="$run" DATA="$data" CATEGORIES="$categories" NAMES="$names" ARGS="$args" \
              sbatch --parsable --time="$TIME" --gres="gpu:${gres:-h100-47}:1" \
              "${dependency[@]}" --export=ALL "$HERE/granularity_run.sbatch")
        echo "$job $run"
        if [ "$run" = "$init_run" ]; then
            INIT_JOBS[$init_run]=$job
        fi
    else
        printf 'RUN=%q DATA=%q CATEGORIES=%q NAMES=%q ARGS=%q sbatch --time=%q --gres=%q ' \
            "$run" "$data" "$categories" "$names" "$args" "$TIME" "gpu:${gres:-h100-47}:1"
        printf '%s ' "${dependency[@]}"
        printf -- '--export=ALL %q\n' "$HERE/granularity_run.sbatch"
    fi
done < "$PLAN"
