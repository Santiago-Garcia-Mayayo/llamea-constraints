"""L-SHADE (Tanabe & Fukunaga, CEC 2014) with a pluggable comparison.

Success-history adaptation of F and CR (H memories), current-to-pbest/1/bin with an
external archive, linear population size reduction. Integer variables are optimised as
continuous and rounded by the harness (the standard treatment in DE). Initial population
18*d, capped at 200 and at budget/25 (our budgets are far below CEC 2014's 10,000*d).
One generation is evaluated as one batch.
"""
from __future__ import annotations

import numpy as np

from .common import DebComparison, EpsilonComparison, EpsilonComparisonLong, PenaltyComparison, improvement, leq, lt, order


class _LSHADE:
    comparison = None
    N_INIT_PER_DIM = 18
    N_MAX = 200
    N_MIN = 4
    H = 6
    P_BEST = 0.11
    ARC_RATE = 2.6

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        rng = np.random.default_rng(np.random.randint(2**31))
        cmp = self.comparison()
        lb, ub = np.asarray(func.bounds.lb, float), np.asarray(func.bounds.ub, float)
        d = self.dim
        n_init = int(min(self.N_MAX, max(self.N_MIN, self.N_INIT_PER_DIM * d, 10)))
        # small budgets (2,000-11,000 here, vs 10,000*d in CEC 2014): keep >= ~25 generations
        n_init = min(n_init, max(self.N_MIN, self.budget // 25))
        NP = n_init
        X = rng.uniform(lb, ub, size=(NP, d))
        K = cmp.evaluate(func, X)
        nfe = NP
        MF, MCR = np.full(self.H, 0.5), np.full(self.H, 0.5)
        k_mem = 0
        archive = np.empty((0, d))
        while True:
            if hasattr(cmp, "rekey"):                   # epsilon level changes with the budget used
                K = cmp.rekey(K, nfe / self.budget)
            idx = order(K)
            p_num = max(2, int(round(self.P_BEST * NP)))
            r = rng.integers(self.H, size=NP)
            CR = np.where(MCR[r] < 0, 0.0, np.clip(rng.normal(MCR[r], 0.1), 0, 1))
            F = np.empty(NP)
            for i in range(NP):
                f = -1.0
                while f <= 0:
                    f = MF[r[i]] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
                F[i] = min(f, 1.0)
            pbest = idx[rng.integers(p_num, size=NP)]
            r1 = np.array([rng.choice([j for j in range(NP) if j != i]) for i in range(NP)])
            pool = np.vstack([X, archive]) if archive.size else X
            r2 = np.empty(NP, int)
            for i in range(NP):
                while True:
                    c = rng.integers(len(pool))
                    if c != i and c != r1[i]:
                        r2[i] = c
                        break
            V = X + F[:, None] * (X[pbest] - X) + F[:, None] * (X[r1] - pool[r2])
            V = np.where(V < lb, (lb + X) / 2, V)       # midpoint bound repair
            V = np.where(V > ub, (ub + X) / 2, V)
            jrand = rng.integers(d, size=NP)
            mask = rng.random((NP, d)) <= CR[:, None]
            mask[np.arange(NP), jrand] = True
            U = np.where(mask, V, X)
            KU = cmp.evaluate(func, U)                  # raises BudgetExhausted at the end
            nfe += NP
            better = lt(KU, K)
            keep = leq(KU, K)
            if better.any():
                archive = np.vstack([archive, X[better]]) if archive.size else X[better].copy()
                w = improvement(K[better], KU[better])
                w = w / w.sum()
                sf, scr = F[better], CR[better]
                MF[k_mem] = np.sum(w * sf ** 2) / max(np.sum(w * sf), 1e-300)
                MCR[k_mem] = -1.0 if (MCR[k_mem] < 0 or scr.max() == 0) else \
                    np.sum(w * scr ** 2) / max(np.sum(w * scr), 1e-300)
                k_mem = (k_mem + 1) % self.H
            X = np.where(keep[:, None], U, X)
            K = np.where(keep[:, None], KU, K)
            # linear population size reduction
            new_np = int(round((self.N_MIN - n_init) / self.budget * nfe + n_init))
            new_np = max(self.N_MIN, new_np)
            if new_np < NP:
                sel = order(K)[:new_np]
                X, K, NP = X[sel], K[sel], new_np
            max_arc = int(round(self.ARC_RATE * NP))
            if len(archive) > max_arc:
                archive = archive[rng.choice(len(archive), max_arc, replace=False)]


class LSHADEPenalty(_LSHADE):
    comparison = PenaltyComparison


class LSHADEDeb(_LSHADE):
    comparison = DebComparison


class LSHADEEps(_LSHADE):
    """Same L-SHADE, epsilon-constrained comparison instead of Deb's rules (the only difference)."""
    comparison = EpsilonComparison


class LSHADEEpsLong(_LSHADE):
    """Same L-SHADE, relaxed and longer epsilon schedule (eps reaches 0 at 60% of the budget)."""
    comparison = EpsilonComparisonLong