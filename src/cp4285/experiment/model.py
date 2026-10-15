"""The only model dependency of the experiment runner: a shared ABC, not a backend."""

from abc import ABC, abstractmethod
from copy import deepcopy

from .data import Example, Query


class Recommender(ABC):
    """One training example = one supervised target; histories are context only.

    Models own representations, losses and optimizers, never partitions, metrics,
    candidates or retention labels. Reject unsupported inputs; do not drop them.
    Constructor/configuration owns seeds and training vocabulary (not test labels).
    """

    @abstractmethod
    def train(self, examples: tuple[Example, ...]) -> None:
        """One declared update over exactly these targets, mutating the current state.

        No hidden epochs, resampling, replay or refits. If a backend needs a
        different exposure contract, propose it before comparing its results.
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, query: Query, candidates: tuple[str, ...]) -> tuple[float, ...]:
        """Finite scores in supplied candidate order, higher is better, no mutation.

        Disable dropout/gradients here; restore training mode in train(). Target
        labels are deliberately absent. Domains namespace otherwise identical IDs.
        """
        raise NotImplementedError

    @abstractmethod
    def begin_continuation(self) -> None:
        """Keep learned state; apply declared optimizer policy once per branch.

        Neural default proposal: reset the optimizer once, retain it thereafter.
        Incremental statistical models may explicitly do nothing here.
        """
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict[str, object]:
        """JSON metadata: family/configuration/seed/update policy/parameter counts.

        State unsupported or inapplicable fields explicitly. Include all knobs
        needed to reconstruct the model, not just its display name.
        """
        raise NotImplementedError

    def fork(self) -> "Recommender":
        """Independent full state, including optimizer and model-owned RNG.

        No process-global random state: own RNG objects inside the learner.
        """
        # ponytail: deepcopy suits a small in-memory pilot; override for large/device-bound state.
        return deepcopy(self)
