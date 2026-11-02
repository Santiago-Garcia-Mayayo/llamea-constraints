"""Baselines: comparisons, correctness on the toy problem, budget use, isolation."""
import numpy as np
import pytest

from baselines import BASELINES
from baselines.cmaes import CMAESDeb, CMAESPenalty
from baselines.random_search import RandomSearch
from baselines.common import DebComparison, PenaltyComparison, leq, lt, order
from baselines.lshade import LSHADEDeb, LSHADEPenalty
from harness.runner import run_job, run_one
from problems import get_problem
from problems.toy import ToyProblem


def _fake_free(F, G):
    return lambda X: (np.asarray(F, float), np.asarray(G, float))


def test_deb_ordering():
    F = [5.0, 1.0, np.inf, 3.0, 0.5]
    G = [[0, 0], [0.2, 0], [0, 0], [0, 0], [3.0, 1.0]]
    K = DebComparison().evaluate(_fake_free(F, G), None)
    # feasible by f (3.0 then 5.0), then infeasible by total violation (0.2, 1.0 for undefined f, 4.0)
    assert order(K).tolist() == [3, 0, 1, 2, 4]


def test_penalty_ordering_and_inf():
    K = PenaltyComparison().evaluate(lambda X: np.array([2.0, np.inf, -1.0]), None)
    assert order(K).tolist() == [2, 0, 1]
    assert np.all(np.isfinite(K))


def test_key_comparisons():
    a = np.array([[0, 1.0], [1, 0.1], [0, 2.0]])
    b = np.array([[1, 0.0], [1, 0.1], [0, 1.0]])
    assert lt(a, b).tolist() == [True, False, False]
    assert leq(a, b).tolist() == [True, True, False]


@pytest.mark.parametrize("cls,arm", [(LSHADEPenalty, "penalty"), (LSHADEDeb, "free")])
def test_lshade_solves_toy(cls, arm):
    r, _ = run_one(cls, ToyProblem(), arm, 0, 3000, 60)
    assert r.status == "budget_exhausted" and r.n_evals == 3000
    assert r.found_feasible and r.best_f - 1.0 < 1e-3


@pytest.mark.parametrize("cls,arm", [(CMAESPenalty, "penalty"), (CMAESDeb, "free")])
def test_cmaes_runs_to_budget_and_is_feasible(cls, arm):
    r, _ = run_one(cls, ToyProblem(), arm, 0, 3000, 60)
    assert r.status == "budget_exhausted" and r.n_evals == 3000
    assert r.found_feasible and r.best_f < 1.5


def test_epsilon_comparison_relaxes_then_equals_deb():
    from baselines.common import EpsilonComparison
    F = [5.0, 1.0, 3.0, 0.5, 2.0]
    G = [[0, 0], [0.2, 0], [0, 0], [3.0, 1.0], [0.05, 0]]
    c = EpsilonComparison()
    K = c.evaluate(_fake_free(F, G), None)
    assert c.eps0 == 0.0          # theta-th best (index 1 of 5) violation is 0
    c.eps0, c.eps = 0.1, 0.1     # with eps = 0.1, the point violating by 0.05 counts as feasible
    K = c.rekey(K, 0.0)
    assert order(K).tolist() == [4, 2, 0, 1, 3]
    K = c.rekey(K, 1.0)           # after Tc: eps = 0, exactly Deb's order
    deb = order(DebComparison().evaluate(_fake_free(F, G), None)).tolist()
    assert c.eps == 0.0 and order(K).tolist() == deb


@pytest.mark.parametrize("name", ["LSHADEEps", "LSHADEEpsLong"])
def test_lshade_eps_solves_toy(name):
    import baselines.lshade as L
    r, _ = run_one(getattr(L, name), ToyProblem(), "free", 0, 3000, 60)
    assert r.status == "budget_exhausted" and r.n_evals == 3000
    assert r.found_feasible and r.best_f - 1.0 < 1e-3


def test_long_epsilon_schedule():
    from baselines.common import EpsilonComparisonLong
    c = EpsilonComparisonLong()
    K = c.evaluate(_fake_free([1.0, 2.0, 3.0, 4.0], [[0.0], [2.0], [4.0], [6.0]]), None)
    assert c.eps0 == 4.0 and c.eps == 4.0                 # median of the positive violations
    c.rekey(K, 0.3)
    assert abs(c.eps - 4.0 * 0.5 ** 3) < 1e-12
    c.rekey(K, 0.6)
    assert c.eps == 0.0


def test_colshade_solves_toy_and_uses_the_whole_budget():
    from baselines.colshade import COLSHADE
    r, _ = run_one(COLSHADE, ToyProblem(), "free", 0, 3000, 60)
    assert r.status in ("ok", "budget_exhausted") and r.n_evals == 3000   # stops exactly at the budget
    assert r.found_feasible and r.best_f - 1.0 < 1e-2


def test_random_search_identical_in_both_arms():
    a, _ = run_one(RandomSearch, ToyProblem(), "free", 3, 500, 60)
    b, _ = run_one(RandomSearch, ToyProblem(), "penalty", 3, 500, 60)
    assert a.best_f == b.best_f and a.aocc == b.aocc


def test_pairs_differ_only_in_comparison():
    p = ToyProblem()
    for pen, deb in [(LSHADEPenalty, LSHADEDeb), (CMAESPenalty, CMAESDeb)]:
        assert pen.__mro__[1] is deb.__mro__[1]                    # same implementation class
        assert {k for k in vars(pen) if not k.startswith("__")} == {"comparison"}
        _, h1 = run_one(pen, p, "penalty", 0, 400, 60, keep_trace=True)
        _, h2 = run_one(deb, p, "free", 0, 400, 60, keep_trace=True)
        x1, x2 = h1.trace_arrays()["x"], h2.trace_arrays()["x"]
        n0 = len(x1) and np.argmax(np.any(x1 != x2, axis=1))
        assert n0 > 0                       # identical start (same seed), then they diverge


def test_registry_paths_and_arms():
    for name, (path, arm) in BASELINES.items():
        rs = run_job(path, "cec/RC15", arm, [0], budget=300, time_limit=60)
        assert rs[0].status in ("budget_exhausted", "ok") and rs[0].n_evals == 300, (name, rs[0].error)


def test_baselines_on_mixed_integer_problems():
    for pid in ("cec/RC12", "cec/RC26"):
        p = get_problem(pid)
        for cls, arm in [(LSHADEDeb, "free"), (CMAESPenalty, "penalty")]:
            r, h = run_one(cls, p, arm, 1, 1500, 120, keep_trace=True)
            x = h.trace_arrays()["x"]
            ii = p.integer_indices
            assert r.n_evals == 1500 and np.all(x[:, ii] == np.round(x[:, ii]))