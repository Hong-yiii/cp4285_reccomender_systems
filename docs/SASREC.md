# Original SASRec architecture

**Decision, 28 September 2026:** Hongyi selected [kang205/SASRec](https://github.com/kang205/SASRec) rather than the earlier generic Transformer approximation. The implemented model is a **PyTorch port of that original architecture**, not the original TensorFlow executable and not a reproduction of the paper's benchmark numbers.

## Source and runtime

Pinned upstream commit: `e3738967fddab206d6eeb4fda433e7a7034dd8b1`.

- [model.py](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/model.py): model assembly, prediction, loss and optimizer.
- [modules.py](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/modules.py): normalization, embeddings, attention and pointwise feedforward.
- [sampler.py](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/sampler.py): left-padded sequences, shifted targets and negative sampling.
- [main.py](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/main.py): reference defaults and training schedule.
- [util.py](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/util.py): original leave-one-out and sampled-candidate evaluation.

Upstream requires Python 2 / TensorFlow 1.12, incompatible with this project's Python 3.12 environment. We translated the small model into the existing PyTorch dependency rather than introduce a legacy runtime or substitute a newer SASRec variant. No upstream datasets are vendored. Attribution and Apache-2.0 terms are retained in [NOTICE](../NOTICE) and [the license](../licenses/SASRec-Apache-2.0.txt).

## Model fidelity

| Original behavior | Port in `src/cp4285/model.py` |
| --- | --- |
| Learned item embeddings scaled by sqrt(hidden), plus unscaled learned positions | Preserved; item 0 is zero padding |
| Fixed-length, left-padded sequences; latest item at the last position | Preserved for every batch, including evaluation |
| Layer normalization, epsilon 1e-8 | Preserved |
| Linear biased Q/K/V; Q from normalized sequence, K/V from raw sequence | Preserved |
| Causal/key masks, query mask, attention-probability dropout | Preserved, including upstream finite masking sentinel |
| Concatenate attention heads without output projection | Preserved; no generic `TransformerEncoderLayer` |
| Residual against the normalized attention queries | Preserved, rather than the standard raw-input Transformer residual |
| Pointwise hidden→hidden→hidden, ReLU then linear, dropout after both | Preserved using Linear layers equivalent to width-1 convolutions |
| Feedforward residual against normalized input; remask after each block | Preserved, followed by final normalization |
| Tied input/output item table and sampled binary next-item loss | Preserved; loss averaged over active targets only |
| Embedding-only L2; Adam beta2=0.98 | Preserved; no blanket weight decay or gradient clipping |

Default architecture matches upstream `main.py`: hidden 50, two blocks, one head, dropout 0.5, sequence length 50, embedding L2 0. Initial learning rate is 0.001 and batch size 128. The demo uses this same architecture—not a one-block substitute—while bounding examples and update count.

Numerical/runtime differences are explicit: PyTorch Glorot initialization and RNG do not reproduce TensorFlow draws; stable softplus replaces epsilon-clipped logarithms; the unused raw padding embedding row is fixed at zero (rather than an independently regularized, forward-inaccessible TF variable). Framework Adam arithmetic may differ despite matching betas. There is no cross-framework bitwise-parity claim or TF checkpoint converter.

## Training versus our experiment

Initial training now samples users uniformly with replacement, constructs their final training windows, and supervises every nonpadding next-item position. Negatives exclude the user's **entire initial training item set**, including items outside the truncated window. Earlier stored prefix rows reconstruct that set; they are not independently re-supervised as overlapping windows. Each epoch has `floor(initial_users / batch_size)` batches, with one minimum batch so small synthetic cohorts still train.

Continued training intentionally supervises **only the newly arriving final target** in each example. Earlier positions supply context but no repeated labels. Applying all-position loss here would silently replay old targets, violating the fresh-event continuation condition. Padding and unused targets have zero loss weight.

Our study retains these deliberate differences from upstream's benchmark:

- Familiar-item vocabulary from initial training only, bounded cohort and explicit temporal cutoff.
- Validation and retention events excluded from every training target/prefix. Both currently use the initial training prefix; unlike original SASRec test evaluation, retention does **not** append the validation event. Their retained cohorts can differ after OOV filtering.
- Validation each epoch selects the initial checkpoint; retention never selects it.
- Fixed full initial catalogue for retention instead of one positive plus 100 sampled negatives; repeated items remain eligible under the review-event task.
- Three initial epochs in the default **feasibility pilot**, not the authors' 201-epoch schedule or a convergence claim. Review the budget before drawing real-data conclusions.
- A frozen control and chronological A-only continuation, with Adam reset once at the phase boundary and retained across cycles. B/mixed branches remain unimplemented.
- Continued negatives exclude the supplied history and fresh target, not future observations or held-out labels.

Model fidelity does not justify comparing these scores directly with paper results. Dataset choice, training budget and two-domain design still need review.

## Verification and compatibility

```sh
uv run pytest -q
uv run cp4285 demo --output runs/sasrec-check-01
```

`tests/test_sasrec.py` checks one- and two-head forward states, full-catalogue scores and masked/regularized loss against a separate NumPy translation of the upstream equations. It also checks causal isolation, finite gradients, zero padding gradient, initial sequence sampling and fresh-target masks. Existing tests cover batch-invariant positions, checkpoint reload, fixed frozen control and holdout exclusion. These are equation/lifecycle checks, **not a live TensorFlow run or an empirical benchmark reproduction**.

Checkpoints and `metrics.json` record `implementation = kang205-sasrec-pytorch-v1` and the upstream commit. Earlier `SequentialRecommender` checkpoints are incompatible; retain them as historical artifacts and start a new output directory. Prepared event JSON remains usable; initial-training weighting and model behavior have changed, so do not mix old and new run results as the same baseline.
