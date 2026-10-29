#!/bin/bash
# Builds the pinned PESO environment (pod_run.py setup) on a compute node's local disk and
# packs it into one tar file on scratch. Cluster NFS creates only ~20 files/s, so writing
# a venv there takes hours; local /tmp manages ~17,000/s. peso.sbatch unpacks the tar to
# the same local path, which keeps the venv's absolute interpreter links valid.
#
#   sbatch replication/peso/slurm/build_env.sbatch    # after slurm/setup.sh made boot/
#SBATCH --job-name=peso-env
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# xcng0 and xcng1 are aarch64; the environment is built for x86-64.
#SBATCH --exclude=xcng0,xcng1
set -euo pipefail
W=${PESO_WORKSPACE:-/mnt/scratch/${USER:0:1}/$USER/peso}
ENV=/tmp/$USER-peso-env
TAR=$W/peso-env.tar
[ -e "$TAR" ] && { echo "Refusing to overwrite $TAR"; exit 1; }
rm -rf "$ENV"
mkdir -p "$ENV"
export UV_PYTHON_INSTALL_DIR=$ENV/python UV_CACHE_DIR=$ENV/uv-cache PESO_VENV=$ENV/venv310
export PESO_WORKSPACE=$W PESO_HOST="SoC cluster $(hostname)"
"$W/boot/bin/python" "$W/cp4285/replication/peso/pod_run.py" setup
"$PESO_VENV/bin/python" -c \
    "import torch, transformers, peft, flash_attn; print(torch.__version__, torch.version.cuda, transformers.__version__, peft.__version__, flash_attn.__version__)"
tar -cf "$TAR.partial" -C /tmp "$USER-peso-env/python" "$USER-peso-env/venv310"
mv "$TAR.partial" "$TAR"
rm -rf "$ENV"
ls -l "$TAR"
