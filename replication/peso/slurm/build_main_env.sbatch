#!/bin/bash
# Builds the main checkout's locked environment (uv sync --locked) on a compute node's
# local disk and packs checkout and venv into one tar on scratch, as build_env.sbatch does
# for PESO. blocks.sbatch unpacks it to the same local path. Copy the checkout first:
#
#   rsync -a --exclude .venv --exclude data --exclude runs <main checkout>/ $W/cp4285-main/
#   sbatch replication/peso/slurm/build_main_env.sbatch
#   sbatch --export=ALL,NAME=cp4285-llm,EXTRAS=llm replication/peso/slurm/build_main_env.sbatch
#
# NAME names the local folder and the tar (default cp4285-main); EXTRAS adds a uv extra.
#SBATCH --job-name=main-env
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# xcng0 and xcng1 are aarch64; the environment is built for x86-64.
#SBATCH --exclude=xcng0,xcng1
set -euo pipefail
W=${PESO_WORKSPACE:-/mnt/scratch/${USER:0:1}/$USER/peso}
NAME=${NAME:-cp4285-main}
ENV=/tmp/$USER-$NAME
TAR=$W/$NAME-env.tar
[ -e "$TAR" ] && { echo "Refusing to overwrite $TAR"; exit 1; }
rm -rf "$ENV"
mkdir -p "$ENV"
cp -r "$W/cp4285-main" "$ENV/repo"
export UV_PYTHON_INSTALL_DIR=$ENV/python UV_CACHE_DIR=$ENV/uv-cache
cd "$ENV/repo"
"$W/boot/bin/uv" sync --locked ${EXTRAS:+--extra "$EXTRAS"}
.venv/bin/cp4285 doctor
cd /tmp
# uv may use the system Python 3.12, which every node has; then there is no python/.
tar -cf "$TAR.partial" --exclude "$USER-$NAME/uv-cache" "$USER-$NAME"
mv "$TAR.partial" "$TAR"
rm -rf "$ENV"
ls -l "$TAR"
