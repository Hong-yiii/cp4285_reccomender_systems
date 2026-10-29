#!/bin/bash
# One `cp4285 blocks` run (main checkout: SASRec, with or without a category channel, on
# PESO's Instruments blocks) on a CPU node, from build_main_env.sbatch's tar.
#
#   sbatch --export=ALL,ARM=category,SEED=42 replication/peso/slurm/blocks.sbatch
#
# ARGS passes extra flags, e.g. ARGS="--continuation-lr 3e-4"; RUN names the output.
#SBATCH --job-name=blocks
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# xcng0 and xcng1 are aarch64; the environment is built for x86-64.
#SBATCH --exclude=xcng0,xcng1
set -euo pipefail
W=${PESO_WORKSPACE:-/mnt/scratch/${USER:0:1}/$USER/peso}
ENV=/tmp/$USER-cp4285-main
# As in peso.sbatch: only a copy that fails to import is cleared and unpacked again.
healthy() {
    "$ENV/repo/.venv/bin/python" -c "import cp4285, torch" 2>/dev/null &&
        "$ENV/repo/.venv/bin/cp4285" --help > /dev/null 2>&1
}
(
    flock 9
    if ! healthy; then
        rm -rf "$ENV"
        tar -xf "$W/cp4285-main-env.tar" -C /tmp
        healthy || { echo "Environment still broken after unpacking"; exit 1; }
    fi
) 9> "/tmp/$USER-cp4285-main.lock"
# SRC runs a newer source tree on the same environment, e.g. SRC=$W/cp4285-main/src after
# rsyncing the main checkout again; the packed copy stays as built.
if [ -n "${SRC:-}" ]; then export PYTHONPATH=$SRC; fi
D=$W/peso/data
RUN=${RUN:-blocks-${ARM:?set ARM}-seed${SEED:?set SEED}}
# shellcheck disable=SC2086  # ARGS holds extra flags
"$ENV/repo/.venv/bin/cp4285" blocks --data "$D/Instruments_cl_ori" \
    --categories "$D/Instruments_cl_cat/Instruments_cl_cat.index.cat2.json" \
    --arm "$ARM" --seed "$SEED" --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --output "$W/blocks-runs/$RUN" ${ARGS:-}
