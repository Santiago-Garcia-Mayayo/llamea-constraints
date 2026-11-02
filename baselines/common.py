"""Comparisons shared by the baselines.

Each baseline is written once and paired with one of two comparisons, so that the pair
differs ONLY in how the constraint information is used:

  PenaltyComparison  (penalty arm)   rank by the penalised scalar y = func(x)
  DebComparison      (free arm)      Deb's feasibility rules on (f, g) = func(x):
                                     feasible < infeasible; feasible by f; infeasible by sum(g)
  EpsilonComparison  (free arm)      epsilon-constrained: like Deb, but a violation up to a
                                     level eps (shrinking to 0 during the run) counts as feasible

Keys are (n, 2) arrays [class, value], class 0 = feasible (or "no information" in the
penalty arm), compared lexicographically.
"""
from __future__ import annotations

import numpy as np

BIG = 1e300


class PenaltyComparison:
    arm = "penalty"

    def evaluate(self, func, X):
        y = np.atleast_1d(np.asarray(func(X), float))
        y = np.where(np.isfinite(y), y, BIG)
        return np.column_stack([np.zeros_like(y), y])


class DebComparison:
    arm = "free"

    def evaluate(self, func, X):
        f, g = func(X)
        f = np.atleast_1d(np.asarray(f, float))
        g = np.atleast_2d(np.asarray(g, float))
        t = g.sum(axis=1)
        feasible = (t == 0) & np.isfinite(f)
        # undefined objective without any recorded violation counts as one unit of violation
        t = np.where(~feasible & (t == 0), 1.0, t)
        val = np.where(feasible, f, np.where(np.isfinite(t), t, BIG))
        return np.column_stack([(~feasible).astype(float), val])


class EpsilonComparison:
    """epsilon-constrained comparison (Takahama & Sakai, 2006; level control as in eps-DEag, 2010).

    phi = sum(g). While phi <= eps, a point counts as feasible and is compared by f; otherwise by phi.
    eps(t) = eps0 * (1 - t / Tc) ** cp for t < Tc, 0 afterwards (t, Tc as fractions of the budget):
    eps0 = phi of the theta-th best initial point (theta = 0.2 * NP); cp chosen so that
    eps(0.95 * Tc) = 1e-5 (eps-DEag), at least CP_MIN. With eps = 0 it is Deb's rules.

    Keys have four columns [class, value, f, phi]: the comparisons use the first two, `rekey`
    recomputes them from (f, phi) when eps changes, so stored points are re-judged every generation.
    """
    arm = "free"
    THETA, TC, CP_MIN, LAMBDA = 0.2, 0.2, 3.0, 0.95

    def __init__(self):
        self.eps0, self.cp, self.eps = None, self.CP_MIN, 0.0

    def _raw(self, func, X):
        f, g = func(X)
        f = np.atleast_1d(np.asarray(f, float))
        phi = np.atleast_2d(np.asarray(g, float)).sum(axis=1)
        phi = np.where(np.isfinite(f) | (phi > 0), phi, 1.0)   # undefined f, no violation: one unit
        phi = np.where(np.isfinite(phi), phi, BIG)
        f = np.where(np.isfinite(f), f, BIG)
        return f, phi

    def _keys(self, f, phi):
        inside = phi <= self.eps
        return np.column_stack([(~inside).astype(float), np.where(inside, f, phi), f, phi])

    def evaluate(self, func, X):
        f, phi = self._raw(func, X)
        if self.eps0 is None:                                  # first call: the initial population
            k = max(0, min(len(phi) - 1, int(self.THETA * len(phi))))
            self.eps0 = float(np.sort(phi)[k])
            if self.eps0 > 0:
                self.cp = max(self.CP_MIN, (-5 - np.log10(self.eps0)) / np.log10(1 - self.LAMBDA))
            self.eps = self.eps0
        return self._keys(f, phi)

    def rekey(self, K, progress):
        t = min(1.0, progress / self.TC)
        self.eps = self.eps0 * (1 - t) ** self.cp if t < 1 else 0.0
        return self._keys(K[:, 2], K[:, 3])


class EpsilonComparisonLong(EpsilonComparison):
    """A more relaxed and longer epsilon schedule, like the epsilon-constrained algorithms LLaMEA
    found for the lens: eps0 = median of the positive violations of the initial population,
    eps(t) = eps0 * (1 - t / 0.6) ** 3, i.e. eps reaches 0 at 60% of the budget."""
    TC, CP = 0.6, 3.0

    def evaluate(self, func, X):
        f, phi = self._raw(func, X)
        if self.eps0 is None:
            pos = phi[phi > 0]
            self.eps0 = float(np.median(pos)) if pos.size else 0.0
            self.cp = self.CP
            self.eps = self.eps0
        return self._keys(f, phi)


def leq(a, b):
    """a <= b elementwise for key arrays (n, 2)."""
    return (a[:, 0] < b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] <= b[:, 1]))


def lt(a, b):
    return (a[:, 0] < b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] < b[:, 1]))


def order(K):
    """Indices sorting keys from best to worst."""
    return np.lexsort((K[:, 1], K[:, 0]))


def improvement(old, new):
    """Positive improvement magnitude old -> new (for success-history weights)."""
    same = old[:, 0] == new[:, 0]
    d = np.where(same, np.abs(old[:, 1] - new[:, 1]), np.minimum(old[:, 1], BIG))
    d = np.where(np.isfinite(d), d, 1.0)
    return np.maximum(d, 1e-300)


def clip_to_bounds(X, lb, ub):
    return np.minimum(np.maximum(X, lb), ub)