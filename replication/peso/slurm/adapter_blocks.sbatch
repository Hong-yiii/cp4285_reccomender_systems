#!/bin/bash
# One `cp4285 adapter-blocks` stage (main checkout: a category language model with LoRA
# adapters on PESO's Instruments blocks, optionally with a shill stream) on a GPU node, from
# build_main_env.sbatch's llm tar (NAME=cp4285-llm, EXTRAS=llm).
#
#   sbatch --export=ALL,RUN=lm-init-seed42,ARGS="--stage init" replication/peso/slurm/adapter_blocks.sbatch
#   sbatch --export=ALL,RUN=lm-shared-seed42,ARGS="--stage branch --init $W/lm-runs/lm-init-seed42" ...
#
# ARGS holds the stage and condition flags; RUN names the output under $W/lm-runs.
#SBATCH --job-name=lm-blocks
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# xgpj0 is aarch64; xgpi17 trained at a third of the normal speed on 24 October.
#SBATCH --exclude=xgpj0,xgpi17
set -euo pipefail
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
) 9> "/tmp/$USER-cp4285-llm.lock"
export HF_HUB_OFFLINE=1 HF_HOME=/tmp/$USER-hf  # the base model is a local directory
# SRC runs a newer source tree on the same environment, e.g. SRC=$W/cp4285-main/src after an rsync.
if [ -n "${SRC:-}" ]; then export PYTHONPATH=$SRC; fi
D=$W/peso/data
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# shellcheck disable=SC2086  # ARGS holds extra flags
"$ENV/repo/.venv/bin/cp4285" adapter-blocks --data "$D/Instruments_cl_ori" \
    --categories "$D/Instruments_cl_cat/Instruments_cl_cat.index.cat2.json" \
    --names "$W/cp4285/replication/peso/results/category/category_stats.json" \
    --model "$W/models/Llama-3.2-1B-4e20de362430" \
    --revision 4e20de362430cd3b72f300e6b0f18e50e7166e08 --seed "${SEED:-42}" \
    --output "$W/lm-runs/${RUN:?set RUN}" ${ARGS:?set ARGS}
