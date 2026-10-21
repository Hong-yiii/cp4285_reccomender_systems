# PESO replication results (committed copies)

These are our own measurements: the metric files written by upstream `continual_test.py` at commit `edbec5f`, three per seed. They contain metrics only, with no data, items, users or model weights. They are copied from the ignored run folders so that [the notebook](../../../notebooks/02_peso_replication.ipynb) and [`summarize.py`](../summarize.py) work from a clean clone. Protocol, deviations and interpretation are in [`docs/PESO-REPLICATION.md`](../../../docs/PESO-REPLICATION.md).

| Seed | Where it ran | Finished |
| --- | --- | --- |
| 42 | Runpod, 1× A100-SXM4-80GB | 20 Oct 2026 |
| 43 | Modal, 1× A100-80GB | 20 Oct 2026 |
| 44 | Modal, 1× A100-80GB | 20 Oct 2026 |

`summary-3seeds.json` was regenerated from these files with:

```sh
python replication/peso/summarize.py replication/peso/results/seed42 \
    replication/peso/results/seed43 replication/peso/results/seed44 \
    --json replication/peso/results/summary-3seeds.json
```

It is identical to the summary produced when the runs finished.

SHA-256:

```text
84df573834e3127f3a2edbd57f24a2fa330319170a287cefe5e00e9430f6ab14  seed42/results/QUICK_ws20.msl-1.temp0.8.sflora_cl_0.0.blocks5.json
6e7b9b99d78aa987e4cf3456af2f92973f13520ff3bc2f1fc420e8a9aff8d844  seed42/results/QUICK_ws20.msl-1.temp0.8.sflora_kldiv_latest_cl_2.0.blocks5.json
9e334cdbe05fb4932b822fa097e47d675a31b543554eca7952544570b45a5a1d  seed42/results/QUICK_ws20.msl-1.temp0.8.sflora_pretrain.blocks5.json
b60432e26e3cce3173f73f65abdb8afb9bc1e51064a2ca4858514a1b52ddf13e  seed43/results/QUICK_ws20.msl-1.temp0.8.sflora_cl_0.0.blocks5.json
139a89406638bad864389b24475cbb06679dd8e9158c61ef88672e953e6a71fb  seed43/results/QUICK_ws20.msl-1.temp0.8.sflora_kldiv_latest_cl_2.0.blocks5.json
c91eff7735029b91b02f4affed3c96c6339204fe46e6da550cdc77b340b074d6  seed43/results/QUICK_ws20.msl-1.temp0.8.sflora_pretrain.blocks5.json
1d486e75ebb610601f45834e502c18d517588831ea7708a1c6d65682b68a541e  seed44/results/QUICK_ws20.msl-1.temp0.8.sflora_cl_0.0.blocks5.json
ad4c27dfbfc01e7d46fea7ef5e7d6373f4f5c000b4bbe934b67dee5136c34fb0  seed44/results/QUICK_ws20.msl-1.temp0.8.sflora_kldiv_latest_cl_2.0.blocks5.json
54b1a909b5f5f77b3560d7efc7200bae443965379c9b9e5ad403c1237303db3b  seed44/results/QUICK_ws20.msl-1.temp0.8.sflora_pretrain.blocks5.json
50faf8c107b18c7b38bf27946e54fb6b0a77dd69fbb2e2129f4bc6cf3bf1e2cc  summary-3seeds.json
```

## Seed-42 analysis reports (`seed42/reports/`)

Aggregate counts and statistics only, with no user IDs, items or weights. Written on the RTX 4090 pod on 20 October 2026 from the seed-42 checkpoints:

| File | Script | What it holds |
| --- | --- | --- |
| `transfer.json` | [`transfer_report.py`](../transfer_report.py) | State × block NDCG/Hit matrices (20 beams, at most 2,500 users per block), backward transfer, per-step kept/lost/gained counts by user group, and PESO-vs-single-LoRA McNemar tests |
| `jlens.json` | [`jlens_report.py`](../jlens_report.py) | J-lens and logit-lens readout quality per state, internal vs output rank change, and the drift split |
| `penalty.json` | [`penalty_size.py`](../penalty_size.py) | PESO's KL penalty and its gradient at the end of each block, in float32 and bfloat16 |
| `jlens_items.json` | [`jlens_items_report.py`](../jlens_items_report.py) | Item-level J-lens (all four codes, 15 layers): readout quality, forgetting by code, internal vs output with a permutation baseline, drift split by code and layer, and a level-0 check against `jlens.json` |
| `penalty_calibration.json` | [`penalty_calibrate.py`](../penalty_calibrate.py) | Task gradient on the LoRA adapters vs each penalty variant's gradient (released, summed Eq. 12, size-normalized) |
| `o8_peso-size-w400.json` | [`variant_report.py`](../variant_report.py) | O8: PESO with the size-normalized penalty at λ = 400 against single LoRA and released PESO: NDCG/Hit per state–block pair, backward transfer, own-state averages, head-to-head flips with McNemar tests |
| `o8_jlens_peso-size-w400.json` | [`variant_jlens_compare.py`](../variant_jlens_compare.py) | O8 at state 4 against single LoRA: item-level drift split by code and layer, per-code readout ranks on old and new blocks |

```text
d8c0277824fa6e9202403cd69e61c75c221dca71fff44973ada55142f8c0b19b  seed42/reports/transfer.json
95fd838a6298ae2aa2fab7db4996d757fe40c3a8e503414a2c21cfbc984f0568  seed42/reports/jlens.json
184866ac4ef08c8ee9ca28ecb7c3f62cde28b86c9ba0923fefd0fc4b41e661ae  seed42/reports/penalty.json
b0e078d9fb254934e59359ca53d8b1676247ff05a4588a46255cfa491b1219a2  seed42/reports/jlens_items.json
777f389f8955f2cc40e6f442e1a96c80d9f6508748c79bf002cd96f9bdc04990  seed42/reports/penalty_calibration.json
7248d70bc37b7660787714111fbf4e433e5c080f427f3120060442355975d013  seed42/reports/o8_peso-size-w400.json
19c308a5c0c577935068304e40a306d2a9100d1b4f42c6112229e17d63925928  seed42/reports/o8_jlens_peso-size-w400.json
```

Per-user prediction dumps, J-lens tensors and the user-group file are not committed. The groups can be regenerated from PESO's released blocks with [`user_groups.py`](../user_groups.py).
