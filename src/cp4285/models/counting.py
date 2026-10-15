"""An isolation control for contract checks, not the project's classical comparator."""

from collections import Counter

from cp4285.experiment import Example, Query, Recommender


class CountingRecommender(Recommender):
    """Count supervised targets independently per domain; ignore history deliberately.

    B cannot change A scores. This demonstrates a meaningful null/control outcome,
    not a mechanism for studying shared-parameter interference or model capacity.
    """

    def __init__(self) -> None:
        self.counts: Counter[tuple[str, str]] = Counter()

    def train(self, examples: tuple[Example, ...]) -> None:
        self.counts.update((e.target.domain, e.target.item) for e in examples)

    def score(self, query: Query, candidates: tuple[str, ...]) -> tuple[float, ...]:
        return tuple(float(self.counts[query.domain, item]) for item in candidates)

    def begin_continuation(self) -> None:
        """No optimizer to reset; keep the learned sufficient statistics."""

    def describe(self) -> dict[str, object]:
        return {
            "family": "independent-domain target counts (plumbing/isolation control)",
            "config": {},
            "seed": None,
            "randomness": "none",
            "update_policy": "increment counts for supplied targets only; no optimizer or refit",
            "parameter_count": None,
            "trainable_parameter_count": None,
            "capacity_note": "dynamic count table, not a neural parameterization",
        }
