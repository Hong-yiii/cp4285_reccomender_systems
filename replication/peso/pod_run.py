"""PESO replication runner for a plain GPU machine (a Runpod on-demand pod).

Mirrors modal_app.py: the same upstream commit, base-model revision, released data,
upstream settings and documented deviations (Python 3.10; one process per stage with
gradient accumulation 4). Modal GPU containers are preemptible and upstream cannot
resume mid-block, so long runs moved to a non-preemptible pod. Training writes to
container-local disk and publishes each finished run directory atomically to the
network volume. Every stage writes a create-only completion record; a finished stage
is refused, never overwritten. Standard library only, so it runs before setup.

    python3 replication/peso/pod_run.py setup
    setsid python3 replication/peso/pod_run.py replicate --seed 42 \\
        > /workspace/peso-runs/seed42.out 2>&1 < /dev/null &
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UPSTREAM = "https://github.com/hsyoo32/peso"
UPSTREAM_COMMIT = "edbec5f68ddc83e39f22a041f1f2eaf1c4ad172e"
BASE_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
DATASET = "Instruments_cl_ori"
FLASH_ATTN = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)

WORKSPACE = Path("/workspace")  # network volume: survives the pod
VOL = WORKSPACE / "peso-runs"
UPSTREAM_DIR = WORKSPACE / "peso"
WORK = UPSTREAM_DIR / "Fine-tuning"
VENV = WORKSPACE / "venv310"
MODEL = WORKSPACE / "models" / f"Llama-3.2-1B-{BASE_REVISION[:12]}"
LOCAL = Path("/root/work")  # container disk: fast checkpoints, lost on stop
EVAL = Path(__file__).with_name("peso_eval.py")

# Upstream launcher settings (Fine-tuning/quick_test_run2.py, README manual commands);
# keep identical to modal_app.py.
TAG = "QUICK_ws20.msl-1.temp0.8.sf{flag}.blocks5"
METHODS = {
    "pretrain": {"flag": "lora_pretrain", "weight": 0.0, "tag": "lora_pretrain"},
    "lora": {"flag": "lora", "weight": 0.0, "tag": "lora_cl_0.0"},
    "peso": {"flag": "lora_kldiv_latest", "weight": 2.0, "tag": "lora_kldiv_latest_cl_2.0"},
}
# Penalty variants for direction O8, named peso-<variant>-w<continual_loss_weight>:
# "sum" is the paper's Eq. 12 with upstream code (which averages over the 112 modules, so
# pass 112 * lambda); "size" scales each matrix's KL by its entry count (penalty_variants.py).
VARIANT_SCRIPTS = {"sum": "continual_train.py", "size": "cp4285_penalty_train.py"}
# Our scripts the upstream directory needs, installed under these names.
SCRIPTS = {
    "peso_eval.py": "cp4285_peso_eval.py", "jlens.py": "cp4285_jlens.py",
    "jlens_items.py": "cp4285_jlens_items.py", "penalty_size.py": "cp4285_penalty_size.py",
    "penalty_calibrate.py": "cp4285_penalty_calibrate.py",
    "penalty_variants.py": "cp4285_penalty_train.py",
}  # fmt: skip


def spec(method):
    """A replicated method, or a penalty variant parsed from its name."""
    if method in METHODS:
        return {**METHODS[method], "script": "continual_train.py", "args": []}
    prefix, variant, weight = method.split("-")
    if prefix != "peso" or variant not in VARIANT_SCRIPTS or not weight.startswith("w"):
        raise ValueError(f"Unknown method {method}; expected peso-<sum|size>-w<weight>")
    return {
        "flag": "lora_kldiv_latest", "weight": float(weight[1:]),
        "tag": f"lora_kldiv_latest_{variant}_cl_{float(weight[1:]):g}",
        "script": VARIANT_SCRIPTS[variant],
        "args": ["--penalty_variant", variant] if variant == "size" else [],
    }  # fmt: skip


def install_scripts():
    """Copy our scripts (siblings of this file) into the upstream directory."""
    for source, target in SCRIPTS.items():
        shutil.copyfile(Path(__file__).with_name(source), WORK / target)


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
ENV = {"WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}


def sh(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kw)


def setup():
    """Idempotent: Python 3.10 venv with upstream pins, pinned upstream clone, eval script."""
    if not UPSTREAM_DIR.exists():
        sh(["git", "clone", UPSTREAM, UPSTREAM_DIR])
    sh(["git", "-C", UPSTREAM_DIR, "checkout", UPSTREAM_COMMIT])
    if not (VENV / "bin" / "python").exists():
        sh([sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", "uv"])
        sh([sys.executable, "-m", "uv", "venv", "--python", "3.10", "--seed", VENV])
    pip = [sys.executable, "-m", "uv", "pip", "install", "--python", VENV / "bin" / "python"]
    sh([*pip, "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121"])
    sh([*pip, "-r", UPSTREAM_DIR / "requirements.txt"])
    sh([*pip, FLASH_ATTN])
    install_scripts()
    if not (MODEL / "model.safetensors").exists():
        raise FileNotFoundError(f"Environment ready; copy the pinned base model to {MODEL}")


def paths(run, method):
    root = VOL / "runs" / run
    tag = TAG.format(flag=spec(method)["tag"])
    out = root / "ckpt" / DATASET / "lama3-1b_edim32_beta0.0" / "data_v2" / tag
    return root, out, root / "results" / f"{tag}.json"


def claim(root, stage):
    done = root / "done" / f"{stage}.json"
    if done.exists():
        raise FileExistsError(f"Stage already completed: {done}; choose a new --run")
    return done


def completed(run, stage):
    """Resume support: a finished stage is reported from its record, never rerun."""
    done = VOL / "runs" / run / "done" / f"{stage}.json"
    return {**json.loads(done.read_text()), "resumed": True} if done.exists() else None


def publish(done, record):
    done.parent.mkdir(parents=True, exist_ok=True)
    with done.open("x") as f:
        json.dump(record, f, indent=2)


def stream(cmd, log):
    """Run from the upstream directory in the venv, appending output to the stage log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **ENV, "PATH": f"{VENV / 'bin'}:{os.environ['PATH']}"}
    start = time.time()
    with log.open("a") as f:
        f.write(f"$ {' '.join(map(str, cmd))}\n")
        f.flush()
        code = subprocess.run(
            list(map(str, cmd)), cwd=WORK, env=env, stdout=f, stderr=subprocess.STDOUT, check=False
        ).returncode
    if code:
        raise RuntimeError(f"Command failed ({code}); see {log}")
    return time.time() - start


def environment(root, stage):
    report = root / "environment" / f"{stage}.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    python = VENV / "bin" / "python"
    parts = [
        subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
        for cmd in (
            [
                python,
                "-c",
                "import sys,torch;print(sys.version,torch.__version__,torch.version.cuda)",
            ],
            ["nvidia-smi"],
            [sys.executable, "-m", "uv", "pip", "freeze", "--python", python],
        )
    ]
    report.write_text(f"upstream {UPSTREAM_COMMIT}\nhost pod\n\n" + "\n".join(parts))


def train(run, method, seed, port, debug=False):
    root, out, _ = paths(run, method)
    done = claim(root, f"train-{method}")
    if out.exists():
        raise FileExistsError(f"Partial training output exists: {out}; choose a new --run")
    local = LOCAL / out.relative_to(VOL)
    if local.exists():
        shutil.rmtree(local)  # container-local leftovers of an interrupted attempt
    environment(root, f"train-{method}")
    method_spec = spec(method)
    cmd = [
        "torchrun", "--nproc_per_node=1", f"--master_port={port}", method_spec["script"],
        "--base_model", MODEL, "--output_dir", local, "--seed", seed,
        *COMMON, *TRAIN,
        "--shift_flag", method_spec["flag"], "--continual_loss_weight", method_spec["weight"],
        "--wandb_project", "data_v2", "--wandb_run_name", out.name, *method_spec["args"],
    ]  # fmt: skip
    if method != "pretrain":
        shared = out.parent / TAG.format(flag="lora_pretrain") / "block_0"
        if not shared.exists():
            raise FileNotFoundError("Train the shared pretrain stage for this run first")
        shutil.copytree(shared, local / "block_0")  # upstream then skips block 0
        cmd.append("--skip_block0")
    if debug:
        cmd.append("--debug")
    log = root / "logs" / f"train-{method}.log"
    seconds = stream(cmd, log)
    if method != "pretrain":
        # Keep full per-block weights (the flag disables upstream's default deletion).
        cmd = ["python", "extract_modules_subset.py", local, "5", "--delete_full_after_extract"]
        seconds += stream(cmd, log)
    start = time.time()
    staging = out.with_name(out.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(local, staging)
    staging.rename(out)  # appears complete or not at all
    shutil.rmtree(local)
    record = {
        "stage": f"train-{method}", "seed": seed, "debug": debug, "seconds": seconds,
        "publish_seconds": time.time() - start, "gpu": gpu_name(), "host": "runpod pod",
        "base_revision": BASE_REVISION, "upstream_commit": UPSTREAM_COMMIT,
    }  # fmt: skip
    publish(done, record)
    return record


def test(run, method, seed, port, block=None):
    """Upstream continual_test.py, unchanged; the paper's reported numbers come from here."""
    root, out, results = paths(run, method)
    stage = f"test-{method}" + ("" if block is None else f"-block{block}")
    done = claim(root, stage)
    if block is not None:
        results = results.with_name(f"{results.stem}.block{block}.json")
    if results.exists():
        raise FileExistsError(f"Partial results exist: {results}; choose a new --run")
    results.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun", "--nproc_per_node=1", f"--master_port={port}", "continual_test.py",
        "--ckpt_path", out, "--base_model", MODEL, "--seed", seed,
        "--results_file", results, "--test_batch_size", "1", "--num_beams", "10",
        "--shift_flag", spec(method)["flag"], *COMMON,
    ]  # fmt: skip
    cmd += ["--test_all_blocks"] if block is None else ["--test_block", block]
    if method != "pretrain":
        cmd.append("--use_final_model")  # pretrain evaluates block_0 for every block
    seconds = stream(cmd, root / "logs" / f"{stage}.log")
    record = {"stage": stage, "seed": seed, "seconds": seconds, "results": str(results)}
    publish(done, {**record, "metrics": json.loads(results.read_text())})
    return record


METRICS_20 = "hit@1,hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20"


def matrix(run, method, seed, port, states, blocks="0,1,2,3,4", beams=10, batch=1,
           max_users=0, tag="predictions"):  # fmt: skip
    """Our evaluator: saved block states on requested test blocks, resumable per pair."""
    root, out, _ = paths(run, method)
    stage = f"{tag}-{method}-s{states.replace(',', '')}-b{blocks.replace(',', '')}"
    if completed(run, stage):  # resume: a finished stage is reported, never redone
        return completed(run, stage)
    done = claim(root, stage)
    cmd = [
        "torchrun", "--nproc_per_node=1", f"--master_port={port}", "cp4285_peso_eval.py",
        "--ckpt_path", out, "--base_model", MODEL, "--seed", seed,
        "--test_batch_size", batch, "--num_beams", beams, "--max_users", max_users,
        "--metrics", METRICS_20 if beams >= 20 else "hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
        "--shift_flag", spec(method)["flag"], "--states", states, "--test_blocks", blocks,
        "--dump_dir", root / tag / method, *COMMON,
    ]  # fmt: skip
    seconds = stream(cmd, root / "logs" / f"{stage}.log")
    record = {"stage": stage, "seed": seed, "seconds": seconds, "beams": beams, "batch": batch}
    publish(done, record)
    return record


def validate(run, seed, batch):
    """Our evaluator vs upstream, and batched vs one-at-a-time generation, on block 1.

    Upstream's saved block-1 result is state 1 on block 1 with 10 beams and batch 1.
    """
    jobs = [(matrix, run, m, seed, 29520 + i, "1", "1") for i, m in enumerate(("lora", "peso"))]
    concurrently(*[(*job, 10, 1, 0, "validate-b1") for job in jobs])
    concurrently(*[(*job, 10, batch, 0, f"validate-b{batch}") for job in jobs])
    root = VOL / "runs" / run
    report = {}
    for method in ("lora", "peso"):
        upstream = json.loads(paths(run, method)[2].read_text())["all_block_results"]["block_1"]
        upstream = upstream["all_prompt_results"][0]
        ours = {}
        for b in (1, batch):
            ours[b] = json.loads((root / f"validate-b{b}" / method / "state1_block1.metrics.json")
                                 .read_text())["metrics"]  # fmt: skip
        one = [json.loads(x) for x in (root / "validate-b1" / method / "state1_block1.jsonl")
               .read_text().splitlines()]  # fmt: skip
        many = {r["index"]: r for r in (json.loads(x) for x in (root / f"validate-b{batch}" / method
               / "state1_block1.jsonl").read_text().splitlines())}  # fmt: skip
        same = sum(r["ranked"] == many[r["index"]]["ranked"] for r in one)
        report[method] = {
            "upstream_a100": upstream, "ours_batch1": ours[1], f"ours_batch{batch}": ours[batch],
            f"identical_top10_lists_batch1_vs_{batch}": f"{same}/{len(one)}",
        }  # fmt: skip
    publish(root / "done" / f"validate-batch{batch}.json", report)
    return report


def jlens(run, method, seed, port, states, drift_from=None):
    """Jacobian-lens probe (replication/peso/jlens.py) on saved block states, resumable.

    With drift_from (a reference lens.pt), computes only each state's item-head drift.
    """
    root, out, _ = paths(run, method)
    stage = f"jlens{'-drift' if drift_from else ''}-{method}-s{states.replace(',', '')}"
    if completed(run, stage):
        return completed(run, stage)
    done = claim(root, stage)
    cmd = [
        "torchrun", "--nproc_per_node=1", f"--master_port={port}", "cp4285_jlens.py",
        "--ckpt_path", out, "--base_model", MODEL, "--seed", seed,
        "--shift_flag", spec(method)["flag"], "--states", states,
        "--out_dir", root / "jlens" / method, *COMMON,
    ]  # fmt: skip
    if drift_from:
        cmd += ["--head_drift_from", drift_from]
    seconds = stream(cmd, root / "logs" / f"{stage}.log")
    record = {"stage": stage, "seed": seed, "seconds": seconds}
    publish(done, record)
    return record


def calibrate(run, seed, port=29550):
    """Penalty-variant strength against the task gradient (penalty_calibrate.py)."""
    root, lora, _ = paths(run, "lora")
    _, peso, _ = paths(run, "peso")
    if completed(run, "calibrate"):
        return completed(run, "calibrate")
    done = claim(root, "calibrate")
    install_scripts()
    out = root / "reports" / "penalty_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun", "--nproc_per_node=1", f"--master_port={port}", "cp4285_penalty_calibrate.py",
        "--ckpt_path", lora, "--base_model", MODEL, "--seed", seed,
        "--shift_flag", spec("lora")["flag"], "--peso_ckpt", peso, "--out", out, *COMMON,
    ]  # fmt: skip
    seconds = stream(cmd, root / "logs" / "calibrate.log")
    record = {"stage": "calibrate", "seed": seed, "seconds": seconds,
              "report": json.loads(out.read_text())}  # fmt: skip
    publish(done, record)
    return record


def gpu_name():
    query = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    return subprocess.run(query, capture_output=True, text=True, check=False).stdout.strip()


def concurrently(*jobs):
    """Independent evaluation stages share the GPU; each owns its process, port, log, outputs.

    Training stays sequential: one training process holds ~48 GB of the A100's 80 GB.
    """
    if not jobs:
        return []
    with ThreadPoolExecutor(len(jobs)) as pool:
        futures = [pool.submit(fn, *args) for fn, *args in jobs]
        return [f.result() for f in futures]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "setup",
            "smoke",
            "replicate",
            "validate",
            "matrix",
            "jlens",
            "jlens-drift",
            "calibrate",
            "variant",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", default="")
    parser.add_argument("--batch", type=int, default=1, help="Generation batch for evaluation")
    parser.add_argument("--beams", type=int, default=20)
    parser.add_argument("--max_users", type=int, default=2500, help="Per-block subsample cap")
    parser.add_argument("--sequential", action="store_true", help="One evaluator at a time")
    parser.add_argument("--method", help="variant stage: peso-<sum|size>-w<weight>")
    parser.add_argument("--with_test", action="store_true", help="variant: also Table 2 test")
    parser.add_argument("--full_matrix", action="store_true", help="variant: all 20 pairs")
    args = parser.parse_args()
    if args.stage == "setup":
        return setup()
    run, seed = args.run or f"seed{args.seed}", args.seed
    steps = []
    if args.stage == "smoke":
        run = args.run or f"smoke-seed{seed}"
        steps.append(train(run, "pretrain", seed, 29500, debug=True))
        steps += [train(run, m, seed, 29501, debug=True) for m in ("lora", "peso")]
        steps += concurrently(
            (test, run, "peso", seed, 29503, 1), (matrix, run, "peso", seed, 29504, "1", "1")
        )
    elif args.stage == "replicate":
        # Rerunning resumes: finished stages are skipped, interrupted ones start over.
        for method in METHODS:
            steps.append(completed(run, f"train-{method}") or train(run, method, seed, 29500))
        steps += [completed(run, f"test-{m}") for m in METHODS if completed(run, f"test-{m}")]
        pending = [m for m in METHODS if not completed(run, f"test-{m}")]
        steps += concurrently(*[(test, run, m, seed, 29503 + i) for i, m in enumerate(pending)])
    elif args.stage == "jlens":
        # Sequential: each state's backward passes use most of a 24 GB GPU.
        steps.append(jlens(run, "lora", seed, 29530, "0,1,2,3,4"))
        steps.append(jlens(run, "peso", seed, 29531, "1,2,3,4"))
    elif args.stage == "jlens-drift":
        # Item-head part of the drift from the shared block-0 state: (U_i - U_0) J_i.
        reference = VOL / "runs" / run / "jlens" / "lora" / "state0" / "lens.pt"
        steps.append(jlens(run, "lora", seed, 29530, "1,2,3,4", reference))
        steps.append(jlens(run, "peso", seed, 29531, "1,2,3,4", reference))
    elif args.stage == "validate":
        steps.append(validate(run, seed, args.batch))
    elif args.stage == "calibrate":
        steps.append(calibrate(run, seed))
    elif args.stage == "variant":
        # One penalty variant: train blocks 1-4 from the shared block 0, then evaluate on the
        # same users and settings as single LoRA's transfer matrix. By default only the
        # pairs backward transfer and new-block accuracy need: each block's own state, and
        # the last state on every block (--full_matrix for all 20; --with_test for Table 2).
        method = args.method
        spec(method)  # fail fast on a malformed name
        install_scripts()
        steps.append(completed(run, f"train-{method}") or train(run, method, seed, 29500))
        if args.with_test:
            steps.append(completed(run, f"test-{method}") or test(run, method, seed, 29503))
        settings = (args.beams, args.batch, args.max_users, f"matrix-beam{args.beams}")
        pairs = [("1,2,3,4", "0,1,2,3,4")] if args.full_matrix else [
            ("4", "0,1,2,3,4"), ("1", "1"), ("2", "2"), ("3", "3"),
        ]  # fmt: skip
        for states, blocks in pairs:
            steps.append(matrix(run, method, seed, 29510, states, blocks, *settings))
    else:
        # State 0 is the shared block-0 checkpoint, so PESO reuses single LoRA's state 0.
        settings = (args.beams, args.batch, args.max_users, f"matrix-beam{args.beams}")
        jobs = [
            (matrix, run, "lora", seed, 29510, "0,1,2,3,4", "0,1,2,3,4", *settings),
            (matrix, run, "peso", seed, 29511, "1,2,3,4", "0,1,2,3,4", *settings),
        ]
        if args.sequential:  # a 20-beam evaluator holds ~8 GB; small GPUs fit one at a time
            steps += [job[0](*job[1:]) for job in jobs]
        else:
            steps += concurrently(*jobs)
    print(json.dumps(steps, indent=2))


if __name__ == "__main__":
    main()
