#!/bin/bash
# Queue one seed's run as short jobs that fit the 3-hour gpu partition. Each pod_run.py
# stage resumes and refuses to overwrite, so a repeat job only finishes what a time limit
# cut off, and returns at once if nothing is left.
#
#   bash replication/peso/slurm/submit_seed.sh 42 [gres]    # category run cat-seed42
#   RUN=item-seed42 PESO_DATASET=Instruments_cl_ori PESO_INDEX=.index.epoch20000_edim32_beta0.0.json \
#       REPLICATE_JOBS=3 MATRIX_ARGS=--subset EXTRA=--mem=128G bash replication/peso/slurm/submit_seed.sh 42
#
# Run from $PESO_WORKSPACE on a login node; prints the job ids.
set -euo pipefail
SEED=${1:?seed}
GRES=${2:-gpu:h100-47:1}
RUN=${RUN:-cat-seed$SEED}
DATA=""
[ -n "${PESO_DATASET:-}" ] && DATA+=",PESO_DATASET=$PESO_DATASET"
[ -n "${PESO_INDEX:-}" ] && DATA+=",PESO_INDEX=$PESO_INDEX"
SB=cp4285/replication/peso/slurm/peso.sbatch
# shellcheck disable=SC2086  # EXTRA holds extra sbatch options
run() { sbatch --parsable --gres="$GRES" ${EXTRA:-} "$@"; }
ids=()
after=""
for _ in $(seq "${REPLICATE_JOBS:-2}"); do
    ids+=("$(run ${after:+--dependency=afterany:$after} --export="ALL$DATA,STAGE=replicate,SEED=$SEED,RUN=$RUN" "$SB")")
    after=${ids[-1]}
done
dep="afterok:$after"
for _ in $(seq "${MATRIX_JOBS:-2}"); do
    ids+=("$(run --dependency="$dep" --export="ALL$DATA,STAGE=matrix,SEED=$SEED,RUN=$RUN,ARGS=${MATRIX_ARGS:-}" "$SB")")
    dep="afterany:${ids[-1]}"
done
echo "$RUN: ${ids[*]}"
