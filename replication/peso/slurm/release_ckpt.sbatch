#!/bin/bash
# Download a GitHub weight release and rebuild it as a pod_run.py run (release_ckpt.py),
# on a CPU node with the PESO environment. Item-level data, so PESO_DATASET stays unset.
#
#   sbatch --export=ALL,RELEASE=https://github.com/<repo>/releases/download/<tag>,RUN=<name> \
#       replication/peso/slurm/release_ckpt.sbatch
#SBATCH --job-name=peso-release
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# xcng0 and xcng1 are aarch64; the environment is built for x86-64.
#SBATCH --exclude=xcng0,xcng1
set -euo pipefail
W=${PESO_WORKSPACE:-/mnt/scratch/${USER:0:1}/$USER/peso}
ENV=/tmp/$USER-peso-env
# Only a copy that cannot run (an unpack cut short) is cleared and unpacked again, so a copy
# another job on this node is using is never touched. The check runs what jobs run.
healthy() {
    "$ENV/venv310/bin/python" -c "import torch, transformers, flash_attn" 2>/dev/null &&
        "$ENV/venv310/bin/torchrun" --help > /dev/null 2>&1
}
(
    flock 9  # jobs sharing a node unpack once
    if ! healthy; then
        rm -rf "$ENV"
        tar -xf "$W/peso-env.tar" -C /tmp
        healthy || { echo "Environment still broken after unpacking"; exit 1; }
    fi
) 9> "/tmp/$USER-peso-env.lock"
unset PESO_DATASET PESO_INDEX
export PESO_WORKSPACE=$W
cd "$W/cp4285/replication/peso"
"$ENV/venv310/bin/python" release_ckpt.py "${RELEASE:?set RELEASE}" "${RUN:?set RUN}"
