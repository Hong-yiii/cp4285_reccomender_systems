"""Build contaminated training streams.

Item ids are laid out as [base items | contaminating items]. The injected volume is a
fraction of the base training interactions, filled by sampling whole contaminating
users at random.

designs
  disjoint : injected histories become brand-new users (design D in the plan)
  shared   : only users who also have base history; their contaminating interactions
             from before their training cutoff are merged into their own timeline
             (design A), so a user's "last item" can become an off-domain item
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Domain, Split


@dataclass
class Stream:
    """Training interactions sorted by (user, time)."""

    user: np.ndarray
    item: np.ndarray
    ts: np.ndarray
    n_users: int
    n_items: int
    info: dict


def build_stream(
    split: Split,
    base_n_items: int,
    contam: Domain,
    n_users: int,
    frac: float,
    design: str = "disjoint",
    seed: int = 0,
) -> Stream:
    base_nnz = len(split.train_user)
    target = int(frac * base_nnz)
    rng = np.random.default_rng(seed)

    cu, ci, ct = contam.user, contam.item, contam.ts
    if design == "shared":
        cutoff = np.full(n_users, -1, np.int64)
        cutoff[split.cutoff[:, 0]] = split.cutoff[:, 1]
        m = ct < cutoff[cu]  # only users with base history, and only their past
        cu, ci, ct = cu[m], ci[m], ct[m]
    elif design != "disjoint":
        raise ValueError(f"unknown design {design!r}")

    # sample whole users until the target volume is reached
    users, counts = np.unique(cu, return_counts=True)
    perm = rng.permutation(len(users))
    cum = np.cumsum(counts[perm])
    n_take = min(int(np.searchsorted(cum, target) + 1), len(users)) if target > 0 else 0
    m = np.isin(cu, users[perm[:n_take]])
    cu, ci, ct = cu[m], ci[m] + base_n_items, ct[m]

    total_users = n_users
    if design == "disjoint":  # fresh user ids after all real users
        _, cu = np.unique(cu, return_inverse=True)
        cu = cu + n_users
        total_users += n_take

    u = np.r_[split.train_user, cu]
    i = np.r_[split.train_item, ci]
    t = np.r_[split.train_ts, ct]
    order = np.lexsort((t, u))
    info = {
        "design": design,
        "frac_target": frac,
        "injected": len(cu),
        "frac_actual": len(cu) / base_nnz,
        "injected_users": int(n_take),
        "capped": bool(target > 0 and len(cu) < target),
    }
    return Stream(u[order], i[order], t[order], total_users, base_n_items + contam.n_items, info)
