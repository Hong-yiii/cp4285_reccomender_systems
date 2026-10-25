#!/bin/bash
# One pod_run.py stage as a Slurm job on the NUS SoC cluster, on the category-target data
# (category_index.py) unless PESO_DATASET/PESO_INDEX say otherwise. Stages resume and
# refuse to overwrite as on the pod, so a resubmitted job continues where it stopped.
# The environment comes from build_env.sbatch's tar, unpacked to node-local /tmp, and
# checkpoints train on local disk before each finished run is published to scratch.
#
#   sbatch --export=ALL,STAGE=smoke,RUN=cat-smoke-seed42 replication/peso/slurm/peso.sbatch
#   sbatch -p gpu-long -t 3-00:00:00 --export=ALL,STAGE=replicate,SEED=42,RUN=cat-seed42 \
#       replication/peso/slurm/peso.sbatch
#   sbatch -p gpu-long -t 1-00:00:00 --export=ALL,STAGE=matrix,SEED=42,RUN=cat-seed42 \
#       replication/peso/slurm/peso.sbatch
#
# ARGS passes extra pod_run.py flags, e.g. ARGS="--sequential".
#SBATCH --job-name=peso-cat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# xgpj0 is aarch64; the environment is built for x86-64.
#SBATCH --exclude=xgpj0
set -euo pipefail
W=${PESO_WORKSPACE:-/mnt/scratch/${USER:0:1}/$USER/peso}
ENV=/tmp/$USER-peso-env
# Only a copy that cannot run (for example an unpack cut short) is cleared and unpacked again,
# so a copy another job on this node is using is never touched. The check runs what jobs run.
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
export PESO_WORKSPACE=$W PESO_VENV=$ENV/venv310
export PESO_HOST="SoC cluster ${SLURMD_NODENAME:-?} job ${SLURM_JOB_ID:-?}"
export PESO_DATASET=${PESO_DATASET:-Instruments_cl_cat}
export PESO_INDEX=${PESO_INDEX:-.index.cat2.json}
# Same weights; generation_config.json sets do_sample=false (setup.sh). The released config
# samples inside beam search (T=0.6, top_p=0.9), which overflows upstream's position table
# when one-token category targets leave fewer valid beams than requested.
export PESO_MODEL=${PESO_MODEL:-$W/models/Llama-3.2-1B-4e20de362430-beam}
export PESO_LOCAL=${PESO_LOCAL:-/tmp/$USER-peso-work/${SLURM_JOB_ID:-manual}}
export PESO_PORT_OFFSET=$(( (${SLURM_JOB_ID:-0} % 100) * 100 ))  # jobs can share a node
export HF_HUB_OFFLINE=1 HF_HOME=/tmp/$USER-hf  # the base model is a local directory
# A stage cut off by the time limit restarts from scratch, so its local checkpoints are junk.
trap 'rm -rf "$PESO_LOCAL"' EXIT
trap 'exit 143' TERM
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$W/boot/bin/python" "$W/cp4285/replication/peso/pod_run.py" "${STAGE:?set STAGE}" \
    --seed "${SEED:-42}" ${RUN:+--run "$RUN"} ${ARGS:-}
