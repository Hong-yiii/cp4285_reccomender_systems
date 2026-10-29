#!/bin/bash
# One-time preparation for PESO on the NUS SoC cluster; run on a login node.
# Makes the small bootstrap venv that runs pod_run.py, then submits build_env.sbatch,
# which builds the pinned Python 3.10 environment on a compute node's local disk.
# The gated base model is not downloaded here: copy it from a machine with Hugging Face
# access to $PESO_WORKSPACE/models/Llama-3.2-1B-4e20de362430 (no token on the cluster),
# and copy the category data to $PESO_WORKSPACE/peso/data/Instruments_cl_cat.
#
#   PESO_WORKSPACE=/mnt/scratch/${USER:0:1}/$USER/peso bash replication/peso/slurm/setup.sh
#
# Expects this checkout's replication/peso at $PESO_WORKSPACE/cp4285/replication/peso.
set -euo pipefail
W=${PESO_WORKSPACE:?set PESO_WORKSPACE}
cd "$W"
[ -x boot/bin/pip ] || python3 -m venv boot
boot/bin/pip install -q uv  # pod_run.py calls `python -m uv`
# Deterministic beam search for the category runs: hard links to the pinned weights, and a
# generation config without the released sampling settings (see peso.sbatch).
M=models/Llama-3.2-1B-4e20de362430
if [ ! -e "$M-beam" ]; then
    mkdir "$M-beam.partial"
    for f in "$M"/*; do [ "$(basename "$f")" = generation_config.json ] || ln "$f" "$M-beam.partial/" 2>/dev/null || cp -r "$f" "$M-beam.partial/"; done
    python3 -c "import json, sys; c = json.load(open(sys.argv[1])); [c.pop(k, None) for k in ('temperature', 'top_p')]; c['do_sample'] = False; json.dump(c, open(sys.argv[2], 'w'), indent=2)" \
        "$M/generation_config.json" "$M-beam.partial/generation_config.json"
    mv "$M-beam.partial" "$M-beam"
fi
sbatch --export=ALL,PESO_WORKSPACE="$W" cp4285/replication/peso/slurm/build_env.sbatch
