"""PESO replication (Yoo et al., ICLR 2026) on Modal: one A100-80GB per stage.

The upstream repository has no licence, so it is cloned at a pinned commit inside
the image and never vendored here; this launcher and peso_eval.py are our code.
Deliberate deviations, both forced by the environment:
- Python 3.10 instead of 3.9 (Modal images need >= 3.10); other pins are kept.
- One GPU with gradient accumulation 4 instead of two with accumulation 2: the
  same micro-batch (8) and effective batch (32). Two-rank runs hit races in upstream
  block-0 copying and in transformers 4.52.4 best-checkpoint tracking, where a rank
  that does not yet see the newest checkpoint reloads one rank 0 has deleted.
Run from the repository root. The smoke check can run attached; long stages must be
spawned on the deployed app (calls.py), because a detached `modal run` loses its
in-flight calls when this client disconnects:

    modal run replication/peso/modal_app.py --stage smoke --run smoke-<name>
    modal deploy replication/peso/modal_app.py
    python replication/peso/calls.py spawn replicate 42 43 44
    python replication/peso/calls.py spawn matrix 42

Each stage writes a create-only completion record; rerunning a finished stage, or
one that left partial output, is refused. Choose a new --run name instead.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

UPSTREAM = "https://github.com/hsyoo32/peso"
UPSTREAM_COMMIT = "edbec5f68ddc83e39f22a041f1f2eaf1c4ad172e"
BASE_MODEL = "meta-llama/Llama-3.2-1B"
BASE_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"  # resolved 19 Oct 2026
DATASET = "Instruments_cl_ori"
GPU = "A100-80GB"
NPROC = "--nproc_per_node=1"
FLASH_ATTN = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)

# Upstream launcher settings (Fine-tuning/quick_test_run2.py, README manual commands).
TAG = "QUICK_ws20.msl-1.temp0.8.sf{flag}.blocks5"
METHODS = {
    "pretrain": {"flag": "lora_pretrain", "weight": 0.0, "tag": "lora_pretrain"},
    "lora": {"flag": "lora", "weight": 0.0, "tag": "lora_cl_0.0"},
    "peso": {"flag": "lora_kldiv_latest", "weight": 2.0, "tag": "lora_kldiv_latest_cl_2.0"},
}
COMMON = [
    "--dataset", DATASET, "--data_path", "../data",
    "--index_file", ".index.epoch20000_edim32_beta0.0.json",
    "--temperature", "0.8", "--max_seq_len", "-1", "--max_his_len", "20",
    "--num_blocks", "5",
    "--lora_target_modules", "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
    "--lora_modules_to_save", "embed_tokens,lm_head",
]  # fmt: skip
TRAIN = [
    "--per_device_batch_size", "8", "--gradient_accumulation_steps", "4",
    "--learning_rate", "2e-4", "--epochs", "4",
    "--tasks", "seqrec", "--train_prompt_sample_num", "1", "--train_data_sample_num", "0",
    "--only_train_response",
]  # fmt: skip

UPSTREAM_DIR = "/opt/peso"
WORK = f"{UPSTREAM_DIR}/Fine-tuning"
VOL = Path("/vol")
LOCAL = Path("/root/work")  # container-local disk for training checkpoints

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu121")
    .run_commands(
        f"git clone {UPSTREAM} {UPSTREAM_DIR}",
        f"git -C {UPSTREAM_DIR} checkout {UPSTREAM_COMMIT}",
        f"pip install -r {UPSTREAM_DIR}/requirements.txt",
        f"pip install {FLASH_ATTN}",
    )
    .env(
        {
            "WANDB_MODE": "disabled",
            # Container-local: hf_xet keeps logs open, which blocks Volume.reload().
            "HF_HOME": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_file(Path(__file__).with_name("peso_eval.py"), f"{WORK}/cp4285_peso_eval.py")
)
app = modal.App("cp4285-peso-replication", image=image)
volume = modal.Volume.from_name("cp4285-peso", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


def _paths(run, method):
    root = VOL / "runs" / run
    ckpt = root / "ckpt" / DATASET / "lama3-1b_edim32_beta0.0" / "data_v2"
    tag = TAG.format(flag=METHODS[method]["tag"])
    return root, ckpt / tag, root / "results" / f"{tag}.json"


def _hf_token():
    """Expose the workspace secret under HF_TOKEN; report key names, never values."""
    names = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
    present = [n for n in names if os.environ.get(n)]
    if not present:
        raise RuntimeError(f"Secret 'huggingface' defines none of {names}")
    os.environ.setdefault("HF_TOKEN", os.environ[present[0]])
    return present


def _base_model():
    """One pinned copy of the gated base model on the volume, shared by every stage."""
    from huggingface_hub import snapshot_download

    path = VOL / "models" / f"Llama-3.2-1B-{BASE_REVISION[:12]}"
    if not (path / "config.json").exists():
        snapshot_download(
            BASE_MODEL,
            revision=BASE_REVISION,
            local_dir=path,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
        )
        volume.commit()
    return str(path)


def _stream(cmd, log):
    """Run a command from the upstream directory, mirroring output to Modal and the log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log.open("a") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )  # fmt: skip
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        code = proc.wait()
    if code:
        raise RuntimeError(f"Command failed ({code}); see {log}")
    return time.time() - start


def _claim(root, stage):
    """Refuse finished or partially written stages; publish the record only on success."""
    done = root / "done" / f"{stage}.json"
    if done.exists():
        raise FileExistsError(f"Stage already completed: {done}; choose a new --run")
    return done


def _publish(done, record):
    done.parent.mkdir(parents=True, exist_ok=True)
    with done.open("x") as f:
        json.dump(record, f, indent=2)
    volume.commit()


def _environment(root, stage):
    import torch

    report = root / "environment" / f"{stage}.txt"  # one file per parallel container
    if not report.exists():
        freeze, smi, nvcc = (
            subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
            for cmd in (["pip", "freeze"], ["nvidia-smi"], ["nvcc", "--version"])
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            f"python {sys.version}\ntorch {torch.__version__} cuda {torch.version.cuda}\n"
            f"upstream {UPSTREAM_COMMIT}\n\n{nvcc}\n{smi}\n{freeze}"
        )


@app.function(gpu=GPU, volumes={VOL: volume}, secrets=secrets, timeout=12 * 3600)
def train(run: str, method: str, seed: int, debug: bool = False):
    """Train on fast local disk, then publish the finished run directory to the volume."""
    volume.reload()
    root, out, _ = _paths(run, method)
    done = _claim(root, f"train-{method}")
    if out.exists():
        raise FileExistsError(f"Partial training output exists: {out}; choose a new --run")
    local = LOCAL / out.relative_to(VOL)
    keys = _hf_token()
    _environment(root, f"train-{method}")
    model = _base_model()
    spec = METHODS[method]
    cmd = [
        "torchrun", NPROC, "--master_port=1234", "continual_train.py",
        "--base_model", model, "--output_dir", str(local), "--seed", str(seed),
        *COMMON, *TRAIN,
        "--shift_flag", spec["flag"], "--continual_loss_weight", str(spec["weight"]),
        "--wandb_project", "data_v2", "--wandb_run_name", out.name,
    ]  # fmt: skip
    if method != "pretrain":
        shared = out.parent / TAG.format(flag="lora_pretrain") / "block_0"
        if not shared.exists():
            raise FileNotFoundError("Train the shared pretrain stage for this run first")
        # Copy before launch so upstream takes its "block 0 already exists" path.
        shutil.copytree(shared, local / "block_0")
        cmd.append("--skip_block0")
    if debug:
        cmd.append("--debug")
    seconds = _stream(cmd, root / "logs" / f"train-{method}.log")
    if method != "pretrain":
        # Keep full per-block weights (the flag disables upstream's default deletion).
        cmd = [
            "python",
            "extract_modules_subset.py",
            str(local),
            "5",
            "--delete_full_after_extract",
        ]
        seconds += _stream(cmd, root / "logs" / f"train-{method}.log")
    start = time.time()
    staging = out.with_name(out.name + ".partial")  # left by a preempted publish
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(local, staging)
    staging.rename(out)  # the run directory appears complete or not at all
    volume.commit()
    record = {
        "stage": f"train-{method}", "seed": seed, "debug": debug, "seconds": seconds,
        "publish_seconds": time.time() - start, "gpu": GPU, "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION, "upstream_commit": UPSTREAM_COMMIT,
        "hf_secret_keys": keys,
    }  # fmt: skip
    _publish(done, record)
    return record


@app.function(gpu=GPU, volumes={VOL: volume}, secrets=secrets, timeout=12 * 3600)
def test(run: str, method: str, seed: int, block: int | None = None):
    """Upstream continual_test.py, unchanged; the paper's reported numbers come from here."""
    volume.reload()
    root, out, results = _paths(run, method)
    stage = f"test-{method}" + ("" if block is None else f"-block{block}")
    done = _claim(root, stage)
    if block is not None:
        results = results.with_name(f"{results.stem}.block{block}.json")
    if results.exists():
        raise FileExistsError(f"Partial results exist: {results}; choose a new --run")
    results.parent.mkdir(parents=True, exist_ok=True)
    _hf_token()
    model = _base_model()
    cmd = [
        "torchrun", NPROC, "--master_port=1234", "continual_test.py",
        "--ckpt_path", str(out), "--base_model", model, "--seed", str(seed),
        "--results_file", str(results), "--test_batch_size", "1", "--num_beams", "10",
        "--shift_flag", METHODS[method]["flag"], *COMMON,
    ]  # fmt: skip
    cmd += ["--test_all_blocks"] if block is None else ["--test_block", str(block)]
    if method != "pretrain":
        cmd.append("--use_final_model")  # pretrain evaluates block_0 for every block
    seconds = _stream(cmd, root / "logs" / f"{stage}.log")
    record = {"stage": stage, "seed": seed, "seconds": seconds, "results": str(results)}
    _publish(done, {**record, "metrics": json.loads(results.read_text())})
    return record


@app.function(gpu=GPU, volumes={VOL: volume}, secrets=secrets, timeout=12 * 3600)
def matrix(run: str, method: str, seed: int, states: str, blocks: str):
    """Our evaluator: every saved block state on every requested test block, with dumps."""
    volume.reload()
    root, out, _ = _paths(run, method)
    stage = f"matrix-{method}-s{states.replace(',', '')}-b{blocks.replace(',', '')}"
    done = _claim(root, stage)
    _hf_token()
    model = _base_model()
    cmd = [
        "torchrun", NPROC, "--master_port=1235", "cp4285_peso_eval.py",
        "--ckpt_path", str(out), "--base_model", model, "--seed", str(seed),
        "--test_batch_size", "1", "--num_beams", "10", "--shift_flag", METHODS[method]["flag"],
        "--states", states, "--test_blocks", blocks,
        "--dump_dir", str(root / "predictions" / method), *COMMON,
    ]  # fmt: skip
    seconds = _stream(cmd, root / "logs" / f"{stage}.log")
    record = {"stage": stage, "seed": seed, "seconds": seconds}
    _publish(done, record)
    return record


@app.function(volumes={VOL: volume}, timeout=24 * 3600)
def pipeline(stage: str, run: str, seed: int):
    """Remote orchestration so detached runs survive a local disconnect."""
    if stage == "smoke":
        # Pipeline check on 100 sampled train/validation examples per block; tests use
        # one incremental block and must match between upstream and our evaluator.
        steps = [train.remote(run, "pretrain", seed, debug=True)]
        steps += [
            c.get() for c in [train.spawn(run, m, seed, debug=True) for m in ("lora", "peso")]
        ]
        calls = [
            test.spawn(run, "peso", seed, block=1),
            matrix.spawn(run, "peso", seed, states="1", blocks="1"),
        ]
        return steps + [c.get() for c in calls]
    if stage == "replicate":
        steps = [train.remote(run, "pretrain", seed)]
        steps += [c.get() for c in [train.spawn(run, m, seed) for m in ("lora", "peso")]]
        return steps + [c.get() for c in [test.spawn(run, m, seed) for m in METHODS]]
    if stage == "matrix":
        # Every saved state on every block's test users; one container per state.
        calls = [
            matrix.spawn(run, m, seed, states=str(i), blocks="0,1,2,3,4")
            for m in ("lora", "peso")
            for i in range(5)
        ]
        return [c.get() for c in calls]
    raise ValueError(f"Unknown pipeline stage {stage!r}")


@app.local_entrypoint()
def main(stage: str = "smoke", seed: int = 42, run: str = ""):
    run = run or (f"smoke-seed{seed}" if stage == "smoke" else f"seed{seed}")
    print(json.dumps(pipeline.remote(stage, run, seed), indent=2))
