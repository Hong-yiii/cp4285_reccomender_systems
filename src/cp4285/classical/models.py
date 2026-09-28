"""Classical recommenders built on QR factorisation.

QRSVD             batch PureSVD (Cremonesi et al. 2010). The item subspace comes from a
                  randomized range finder (Halko et al. 2011): repeated QR of X^T X Q.
IncrementalQRSVD  the same model updated chunk by chunk in the spirit of Brand (2006):
                  new rows are projected on the current basis, the residual is
                  orthonormalised with a QR step, and a small (k+r) x (k+r) core is
                  re-diagonalised. Only the item side (V, S) is kept, so the cost of an
                  update does not grow with the number of users seen so far.

All models score a batch of user rows as  x_u V V^T  (PureSVD fold-in), so users never
need their own stored factors.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def orth(a: np.ndarray) -> np.ndarray:
    """Orthonormal basis for the columns of a (reduced QR)."""
    q, _ = np.linalg.qr(a)
    return q


class MostPop:
    def fit(self, x: sp.csr_matrix) -> MostPop:
        self.pop = np.asarray(x.sum(axis=0), dtype=np.float32).ravel()
        return self

    def score(self, rows: sp.csr_matrix) -> np.ndarray:
        return np.tile(self.pop, (rows.shape[0], 1))


class QRSVD:
    def __init__(self, k: int = 64, oversample: int = 16, n_iter: int = 3, seed: int = 0):
        self.k, self.oversample, self.n_iter, self.seed = k, oversample, n_iter, seed

    def fit(self, x: sp.csr_matrix) -> QRSVD:
        rng = np.random.default_rng(self.seed)
        x = x.astype(np.float32)
        l = self.k + self.oversample
        q = orth(x.T @ (x @ rng.standard_normal((x.shape[1], l), dtype=np.float32)))
        for _ in range(self.n_iter):  # power iterations sharpen the spectrum
            q = orth(x.T @ (x @ q))
        b = (x @ q).astype(np.float64)
        lam, w = np.linalg.eigh(b.T @ b)  # right singular vectors of X restricted to span(q)
        top = np.argsort(lam)[::-1][: self.k]
        self.V = (q @ w[:, top]).astype(np.float32)
        self.S = np.sqrt(np.clip(lam[top], 0, None))
        return self

    def score(self, rows: sp.csr_matrix) -> np.ndarray:
        return np.asarray((rows @ self.V) @ self.V.T)

    def base_share(self, n_base_items: int) -> np.ndarray:
        """Fraction of each component's mass that sits on base-domain items."""
        v2 = self.V.astype(np.float64) ** 2
        return v2[:n_base_items].sum(0) / v2.sum(0)


class IncrementalQRSVD(QRSVD):
    def __init__(self, k: int = 64, residual_rank: int | None = None, forget: float = 1.0, **kw):
        super().__init__(k=k, **kw)
        self.r = residual_rank or k
        self.forget = forget  # < 1 shrinks old singular values: more plastic, less memory
        self._rng = np.random.default_rng(self.seed)

    def partial_fit(self, c: sp.csr_matrix) -> IncrementalQRSVD:
        c = c.astype(np.float32)
        if not hasattr(self, "V"):
            return self.fit(c)
        v = self.V
        r = min(self.r, c.shape[0])
        p = np.asarray(c @ v)  # coordinates of the new rows in the current basis
        omega = self._rng.standard_normal((c.shape[0], r), dtype=np.float32)
        y = np.asarray(c.T @ omega) - v @ (p.T @ omega)  # range of the residual C - P V^T
        y -= v @ (v.T @ y)  # re-orthogonalise against V for stability
        j = orth(y)
        b = np.hstack([p, np.asarray(c @ j)]).astype(np.float64)
        g = b.T @ b
        g[: self.k, : self.k] += np.diag((self.forget * self.S) ** 2)
        lam, w = np.linalg.eigh(g)
        top = np.argsort(lam)[::-1][: self.k]
        self.V = (np.hstack([v, j]) @ w[:, top]).astype(np.float32)
        self.S = np.sqrt(np.clip(lam[top], 0, None))
        return self


# --- sequential (Markov-chain) models -------------------------------------------------


def transitions(
    user: np.ndarray, item: np.ndarray, n_items: int, window: int = 1, back: float = 0.5
) -> sp.csr_matrix:
    """Item -> next-item counts from timelines sorted by (user, time).

    Pairs up to `window` steps apart count 1/step. `back` adds the reverse direction
    at that weight, which helps on sparse data where order is noisy.
    """
    rows, cols, w = [], [], []
    for d in range(1, window + 1):
        same = np.flatnonzero(user[d:] == user[:-d])
        rows.append(item[same])
        cols.append(item[same + d])
        w.append(np.full(len(same), 1.0 / d, np.float32))
    t = sp.csr_matrix(
        (np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n_items, n_items)
    )
    return (t + back * t.T).tocsr() if back else t


def last_items(user: np.ndarray, item: np.ndarray, n_users: int, n_items: int) -> sp.csr_matrix:
    """One-hot row per user holding their most recent item (timelines sorted by time)."""
    end = np.r_[user[1:] != user[:-1], True]
    return sp.csr_matrix(
        (np.ones(end.sum(), np.float32), (user[end], item[end])), shape=(n_users, n_items)
    )


class Markov:
    """Unfactorised first-order Markov chain: score = transitions out of the last item."""

    def fit(self, t: sp.csr_matrix) -> Markov:
        self.T = t
        return self

    def score(self, rows: sp.csr_matrix) -> np.ndarray:
        return (rows @ self.T).toarray()


class MarkovQRSVD:
    """Low-rank Markov chain: the transition matrix compressed to rank k with QRSVD.

    This is the factorised-transition idea of FPMC (Rendle et al. 2010) without the
    user term. Columns are scaled by popularity^-beta before factorising so the top
    components are not all spent on bestsellers.
    """

    def __init__(self, k: int = 64, beta: float = 0.25, **kw):
        self.svd = QRSVD(k=k, **kw)
        self.beta = beta

    def fit(self, t: sp.csr_matrix, pop: np.ndarray) -> MarkovQRSVD:
        self.T = t
        self.D = sp.diags(((pop + 1.0) ** -self.beta).astype(np.float32))
        self.svd.fit((t @ self.D).tocsr())
        return self

    def score(self, rows: sp.csr_matrix) -> np.ndarray:
        return self.svd.score((rows @ self.T @ self.D).tocsr())

    def base_share(self, n_base_items: int) -> np.ndarray:
        return self.svd.base_share(n_base_items)


# --- SLIST (Choi et al., WWW 2021) ---------------------------------------------------


def recency_query(
    user: np.ndarray, item: np.ndarray, n_users: int, n_items: int, decay: float
) -> sp.csr_matrix:
    """Row per user weighting each past item by exp(-steps_from_end / decay)."""
    end = np.flatnonzero(np.r_[user[1:] != user[:-1], True])
    lens = np.diff(np.r_[-1, end])
    steps = np.repeat(end, lens) - np.arange(len(user))
    w = np.exp(-steps / decay).astype(np.float32) if decay > 0 else np.ones(len(user), np.float32)
    return sp.csr_matrix((w, (user, item)), shape=(n_users, n_items))


class SLIST:
    """Session-aware Linear Item-Item model (Choi et al. 2021), closed form.

    Follows the authors' implementation (github.com/jin530/SLIST): the SLIS objective
    ||X - XB|| (items that co-occur in a timeline) and the SLIT objective ||T - SB||
    (each item predicts the items after it, weighted exp(-distance / train_weight),
    direction "sr") are stacked with weights alpha and 1 - alpha, and solved together:

        B = (alpha Xn'W Xn + (1-alpha) S'W S + reg I)^-1 (alpha Xn'W X + (1-alpha) S'W T)

    Xn is X with rows L1-normalised, and W optionally down-weights timelines by age
    (exp(-age / session_weight_days); the paper uses alpha=0.8 and 256 days on Diginetica, but
    on Amazon timelines alpha=0.5 without age weighting scored best). Each user's timeline is
    treated as one session.

    B is dense N x N, so only the max_items most popular items are modelled (20K needs
    about 3.2 GB; 50K needs about 20 GB). The other items score 0.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        reg: float = 10.0,
        train_weight: float = 1.0,
        session_weight_days: float = -1.0,
        max_items: int = 20000,
        window: int = 8,
    ):
        self.alpha, self.reg, self.train_weight = alpha, reg, train_weight
        self.session_weight_days, self.max_items, self.window = (
            session_weight_days,
            max_items,
            window,
        )

    def fit(
        self,
        user: np.ndarray,
        item: np.ndarray,
        ts: np.ndarray,
        n_users: int,
        n_items: int,
        chunk: int = 200_000,
    ) -> SLIST:
        """Timelines must be sorted by (user, time)."""
        import scipy.linalg

        pop = np.bincount(item, minlength=n_items)
        self.items = np.argsort(-pop, kind="stable")[: min(self.max_items, int((pop > 0).sum()))]
        self.n_items = n_items
        n = len(self.items)
        col = np.full(n_items, -1)
        col[self.items] = np.arange(n)
        c = col[item]

        end = np.r_[user[1:] != user[:-1], True]
        w_user = np.ones(n_users, np.float32)
        if self.session_weight_days > 0:
            age_days = (ts.max() - ts[end]) / 86_400_000  # timestamps are in milliseconds
            w_user[user[end]] = np.exp(-age_days / self.session_weight_days)

        g = np.zeros((n, n), np.float32)
        r = np.zeros((n, n), np.float32)

        # SLIS: whole timelines, input rows L1-normalised, targets binary
        m = c >= 0
        x = sp.csr_matrix((np.ones(m.sum(), np.float32), (user[m], c[m])), shape=(n_users, n))
        x.data[:] = 1.0
        cnt = np.diff(x.indptr).astype(np.float32)
        safe = np.maximum(cnt, 1)
        for lo in range(0, n_users, chunk):
            xc = x[lo : lo + chunk]
            if xc.nnz == 0:
                continue
            wc, sc = w_user[lo : lo + chunk], safe[lo : lo + chunk]
            for target, scale in ((g, wc / sc**2), (r, wc / sc)):
                prod = (xc.T @ sp.diags(self.alpha * scale) @ xc).tocoo()
                target[prod.row, prod.col] += prod.data

        # SLIT ("sr"): each non-final position predicts the next `window` positions
        has_next = ~end & m
        np.add.at(g, (c[has_next], c[has_next]), (1 - self.alpha) * w_user[user[has_next]])
        rows, cols, vals = [], [], []
        for d in range(1, self.window + 1):
            p = np.flatnonzero((user[d:] == user[:-d]) & m[:-d] & m[d:])
            rows.append(c[p])
            cols.append(c[p + d])
            vals.append((1 - self.alpha) * w_user[user[p]] * np.exp(-(d - 1) / self.train_weight))
        t = sp.csr_matrix(
            (np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ).tocoo()
        r[t.row, t.col] += t.data

        g[np.diag_indices(n)] += self.reg
        # in-place Cholesky: g is symmetric positive definite; avoids solve()'s extra copies
        factor = scipy.linalg.cho_factor(g, overwrite_a=True, check_finite=False)
        self.B = scipy.linalg.cho_solve(factor, r, overwrite_b=True, check_finite=False)
        return self

    def score(self, rows: sp.csr_matrix) -> np.ndarray:
        out = np.zeros((rows.shape[0], self.n_items), np.float32)
        out[:, self.items] = np.asarray(rows[:, self.items] @ self.B)
        return out

    def base_share(self, n_base_items: int) -> float:
        """Fraction of the modelled item slots that are base-domain items."""
        return float((self.items < n_base_items).mean())
