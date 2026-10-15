"""Immutable, model-independent inputs. Real-data preparation belongs in this package."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    domain: str
    user: str
    event_id: str
    item: str
    timestamp: int

    @property
    def key(self) -> tuple[str, str, str]:
        return self.domain, self.user, self.event_id


@dataclass(frozen=True)
class Query:
    domain: str
    user: str
    history: tuple[str, ...]


@dataclass(frozen=True)
class Example:
    history: tuple[Event, ...]
    target: Event

    @property
    def query(self) -> Query:
        return Query(self.target.domain, self.target.user, tuple(e.item for e in self.history))


@dataclass(frozen=True)
class Evaluation:
    name: str
    domain: str
    examples: tuple[Example, ...]
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class Condition:
    name: str
    blocks: tuple[tuple[Example, ...], ...]


@dataclass(frozen=True)
class PreparedExperiment:
    provenance: str
    initial: tuple[Example, ...]
    validation: Evaluation
    tests: tuple[Evaluation, ...]
    conditions: tuple[Condition, ...]

    def validate(self) -> None:
        """Reject inconsistent identities, held-out leakage and recycled update targets."""
        if not self.provenance.strip() or not self.initial or not self.tests or not self.conditions:
            raise ValueError("Provenance, initial training, tests and conditions are required")
        evaluations = (self.validation, *self.tests)
        _unique_names(tuple(e.name for e in evaluations))
        _unique_names(tuple(c.name for c in self.conditions))
        if any(c.name == "frozen" for c in self.conditions):
            raise ValueError("'frozen' is reserved for the no-update control")
        events: dict[tuple[str, str, str], Event] = {}
        for example in self.initial:
            _validate_example(example, events)
        held_out = set()
        for evaluation in evaluations:
            if not evaluation.examples or not evaluation.candidates:
                raise ValueError("Evaluation examples and candidates cannot be empty")
            if len(set(evaluation.candidates)) != len(evaluation.candidates):
                raise ValueError("Duplicate evaluation candidates")
            targets = set()
            for example in evaluation.examples:
                _validate_example(example, events)
                if example.target.domain != evaluation.domain:
                    raise ValueError("Evaluation domain differs from target domain")
                if example.target.item not in evaluation.candidates:
                    raise ValueError("Evaluation target is missing from candidates")
                if example.target.key in targets or example.target.key in held_out:
                    raise ValueError("Evaluation target is reused across rows or splits")
                targets.add(example.target.key)
            held_out.update(targets)
        test_targets = {e.target.key for task in self.tests for e in task.examples}
        if any(
            e.key in test_targets for example in self.validation.examples for e in example.history
        ):
            raise ValueError("Test target leaked into a validation prefix")
        initial_targets = {e.target.key for e in self.initial}
        if len(initial_targets) != len(self.initial):
            raise ValueError("Duplicate initial training target")
        training = list(self.initial)
        latest_initial: dict[tuple[str, str], int] = {}
        for example in self.initial:
            event = example.target
            identity = event.domain, event.user
            latest_initial[identity] = max(latest_initial.get(identity, -1), event.timestamp)
        for condition in self.conditions:
            if not condition.blocks or any(not block for block in condition.blocks):
                raise ValueError("Each condition needs nonempty update blocks")
            used = set(initial_targets)
            latest = dict(latest_initial)
            for block in condition.blocks:
                for example in block:
                    _validate_example(example, events)
                    event = example.target
                    if event.key in used:
                        raise ValueError("Updates must be fresh; replay needs a separate protocol")
                    identity = event.domain, event.user
                    if event.timestamp <= latest.get(identity, -1):
                        raise ValueError("Updates must be chronological within each domain/user")
                    used.add(event.key)
                    latest[identity] = event.timestamp
                    training.append(example)
        for example in training:
            if any(e.key in held_out for e in (*example.history, example.target)):
                raise ValueError("Held-out event leaked into a training target or prefix")


class DataSource(ABC):
    @abstractmethod
    def load_data(self) -> PreparedExperiment:
        """Prepare auditable partitions/schedules; never silently acquire remote data."""
        raise NotImplementedError


def _unique_names(names: tuple[str, ...]) -> None:
    if any(not n.strip() for n in names) or len(names) != len(set(names)):
        raise ValueError("Names must be nonempty and unique")


def _validate_example(example: Example, events: dict[tuple[str, str, str], Event]) -> None:
    previous = -1
    for event in (*example.history, example.target):
        if any(not isinstance(s, str) or not s for s in (*event.key, event.item)):
            raise ValueError("Original domain/user/event/item identities must be nonempty strings")
        if type(event.timestamp) is not int or event.timestamp <= previous:
            raise ValueError("Example timestamps must be nonnegative and strictly increasing")
        if (event.domain, event.user) != (example.target.domain, example.target.user):
            raise ValueError("Do not splice histories across domains or users")
        if event.key in events and events[event.key] != event:
            raise ValueError("An event identity has conflicting contents")
        events[event.key] = event
        previous = event.timestamp


def interleave(
    a: tuple[Example, ...],
    b: tuple[Example, ...],
    *,
    b_shares: tuple[float, ...],
    block_size: int,
    name: str,
) -> Condition:
    """B / (A+B) supervised targets per block, preserving order inside each stream.

    Round B counts half-up and report actual counts in runner output. Streams are
    not recycled. Calling again starts independent cursors for another condition.
    """
    if type(block_size) is not int or block_size <= 0 or not b_shares:
        raise ValueError("Positive integer block_size and at least one B share required")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in b_shares):
        raise ValueError("B shares must be finite fractions in [0, 1]")
    if any(e.target.domain != "A" for e in a) or any(e.target.domain != "B" for e in b):
        raise ValueError("interleave expects explicitly labelled A and B streams")
    a_cursor = b_cursor = 0
    blocks = []
    for share in b_shares:
        n_b = math.floor(share * block_size + 0.5)
        n_a = block_size - n_b
        if a_cursor + n_a > len(a) or b_cursor + n_b > len(b):
            raise ValueError("Insufficient fresh stream examples; reduce the common budget")
        # ponytail: in-memory blocks; introduce streaming only when a real cohort exceeds memory.
        blocks.append(a[a_cursor : a_cursor + n_a] + b[b_cursor : b_cursor + n_b])
        a_cursor += n_a
        b_cursor += n_b
    return Condition(name, tuple(blocks))


class SyntheticSource(DataSource):
    """Tiny invented data for wiring checks, never an Amazon result or model benchmark."""

    def load_data(self) -> PreparedExperiment:
        def example(domain: str, index: int, item: str) -> Example:
            return Example((), Event(domain, "synthetic-user", str(index), item, index))

        initial = (example("A", 1, "a1"), example("A", 2, "a1"), example("A", 3, "a2"))
        a = tuple(example("A", i, "a2") for i in range(10, 18))
        b = tuple(example("B", i, "b2") for i in range(10, 18))
        data = PreparedExperiment(
            provenance="SYNTHETIC plumbing fixture; no real users, no purchase or review claims",
            initial=initial,
            validation=Evaluation("validation-A", "A", (example("A", 4, "a1"),), ("a1", "a2")),
            tests=(
                Evaluation("retention-A", "A", (example("A", 5, "a1"),), ("a1", "a2")),
                Evaluation("adaptation-B", "B", (example("B", 5, "b2"),), ("b1", "b2")),
            ),
            conditions=tuple(
                interleave(a, b, b_shares=shares, block_size=4, name=name)
                for name, shares in (
                    ("A-only", (0.0, 0.0)),
                    ("mixed", (0.5, 0.5)),
                    ("B-only", (1.0, 1.0)),
                    ("changing", (0.25, 0.75)),
                )
            ),
        )
        data.validate()
        return data
