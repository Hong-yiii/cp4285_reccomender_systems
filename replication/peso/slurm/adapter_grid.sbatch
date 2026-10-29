#!/bin/bash
# Several `cp4285 adapter-blocks` runs in sequence on one GPU, from a plan file whose lines
# are "RUN<TAB>ARGS". DATASET names a PESO-format folder under $W/peso/data; its category
# files come from `cp4285 block-categories` in $W/categories/$DATASET-cat2. A run whose
# metrics.json exists is skipped, so a resubmitted job resumes; a folder an interrupted run
# left without metrics.json is moved aside, never deleted. A failed run is reported and the
# plan goes on.
#
#   sbatch --export=ALL,PLAN=$W/plans/core-ins-s42.tsv,DATASET=Instruments_cl_ori,SRC=$W/cp4285-main/src \
#       replication/peso/slurm/adapter_grid.sbatch
#SBATCH --job-name=lm-grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# xgpj0 is aarch64; xgpi17 trained at a third of the normal speed on 24 October.
#SBATCH --exclude=xgpj0,xgpi17
set -uo pipefail
W=${PESO_WORKSPACE:-/mnt/scratch/${USER:0:1}/$USER/peso}
ENV=/tmp/$USER-cp4285-llm
healthy() {
    "$ENV/repo/.venv/bin/python" -c "import cp4285, torch, transformers, peft" 2>/dev/null &&
        "$ENV/repo/.venv/bin/cp4285" --help > /dev/null 2>&1
}
(
    flock 9  # jobs sharing a node unpack once; only a copy that cannot run is replaced
    if ! healthy; then
        rm -rf "$ENV"
        tar -xf "$W/cp4285-llm-env.tar" -C /tmp
        healthy || { echo "Environment still broken after unpacking"; exit 1; }
    fi
) 9> "/tmp/$USER-cp4285-llm.lock" || exit 1
export HF_HUB_OFFLINE=1 HF_HOME=/tmp/$USER-hf  # the base model is a local directory
if [ -n "${SRC:-}" ]; then export PYTHONPATH=$SRC; fi
C=$W/categories/${DATASET:?set DATASET}-cat2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
while IFS=$'\t' read -r run args; do
    [ -z "$run" ] && continue
    out=$W/lm-runs/$run
    if [ -f "$out/metrics.json" ]; then echo "skip $run (done)"; continue; fi
    if [ -e "$out" ]; then mv "$out" "$out.interrupted-${SLURM_JOB_ID:-manual}"; fi
    echo "start $run $(date +%T)"
    # shellcheck disable=SC2086  # args holds the run's flags
    "$ENV/repo/.venv/bin/cp4285" adapter-blocks --data "$W/peso/data/$DATASET" \
        --categories "$C/$DATASET.index.cat2.json" --names "$C/category_stats.json" \
        --model "$W/models/Llama-3.2-1B-4e20de362430" \
        --revision 4e20de362430cd3b72f300e6b0f18e50e7166e08 --output "$out" $args ||
        echo "FAILED $run"
done < "${PLAN:?set PLAN}"
