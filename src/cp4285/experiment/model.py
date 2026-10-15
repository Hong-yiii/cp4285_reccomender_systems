"""Common learner lifecycle; native batch/query types do not imply a common experiment."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from operator import index
from typing import Self


class Recommender[Batch, Queries, Scores](ABC):
    """Implement stateful updates and pure scoring; experiment code owns the protocol.

    Each train call consumes one declared native batch, without hidden resampling.
    Declare its supervision/update unit in describe(). Never silently refit during
    continuation. Candidate indices are zero-based score columns, NOT source IDs.
    """

    @abstractmethod
    def train(self, batch: Batch) -> float | None:
        """Update the existing learner once; optionally return its scalar loss."""

    @abstractmethod
    def begin_continuation(self) -> None:
        """Transition once without discarding learned parameters; reject a second call."""

    @abstractmethod
    def score(self, queries: Queries, /, candidates: Sequence[int] | None = None) -> Scores:
        """Return batch × candidates scores in requested order, without changing state.

        None selects the full fixed catalogue. Masking, ties and metrics belong
        to the evaluator, not this method. Scoring must not consume training RNG.
        """

    @abstractmethod
    def describe(self) -> dict:
        """JSON-compatible model, update-unit, phase and state-policy metadata."""

    def fork(self) -> Self:
        """Copy parameters, optimizer and owned RNG state into an independent learner."""
        # ponytail: in-memory copy for sequential pilots; override for large/device backends.
        return deepcopy(self)


def candidate_indices(candidates: Sequence[int] | None, count: int) -> tuple[int, ...] | None:
    """Validate explicit catalogue columns before NumPy/PyTorch can wrap negative indices."""
    if candidates is None:
        return None
    try:
        indices = tuple(index(value) for value in candidates)
    except TypeError as exc:
        raise ValueError("Candidates must be integer catalogue columns") from exc
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(i < 0 or i >= count for i in indices)
    ):
        raise ValueError("Candidates must be nonempty, unique and within the fixed catalogue")
    return indices
