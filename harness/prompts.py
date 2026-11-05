"""Task prompts for the three arms. penalty and free differ only in the paragraph that
describes what func returns; informed has exactly the free prompt (it differs from free only
in the feedback after each candidate, see evaluate.py). Keep it that way: any other
difference between arms is a confound."""
from __future__ import annotations

import textwrap

ROLE = "You are a highly skilled computer scientist and Python expert in optimisation."

_COMMON_HEAD = """\
Your task is to design a novel, high-performing optimisation algorithm in Python for a
black-box, constrained, mixed-integer optimisation problem from engineering design.
The algorithm is evaluated on this problem in several independent runs (different random
seeds); the problem itself is a black box: only what `func` exposes is known.

The code must define a class with `__init__(self, budget, dim)` and `__call__(self, func)`.
`__call__` must optimise `func` using at most `self.budget` calls to `func`; calls beyond the
budget raise an exception that ends the run (do not catch it). `func` also accepts a batch:
an array of shape (n, dim), counting n evaluations.

Search space (available as attributes of `func`):
- `func.bounds.lb`, `func.bounds.ub`: lower and upper bounds, arrays of length `dim`.
- `func.variable_types[i]` is "continuous", "ordinal", "binary" or "categorical".
  Continuous variables lie in [-1, 1]. A variable of the other types takes the integer values
  0, 1, ..., `func.n_values[i]` - 1; its bounds are [-0.5, n_values[i] - 0.5] and any real input
  is rounded to the nearest valid integer. Ordinal values are ordered; categorical values are
  labels (do not assume that neighbouring indices are similar); binary variables are 0/1 decisions.
- `func.integer_indices`, `func.continuous_indices`: index arrays for the two groups.
"""

_INTERFACE = {
    "penalty": """\
What `func` returns:
- `y = func(x)`: a single float to be minimised. It is the design objective plus penalty
  terms for violated constraints (zero penalty when all constraints are satisfied).
""",
    "free": """\
What `func` returns:
- `f, g = func(x)`: `f` is the design objective (a float, to be minimised); `g` is a numpy
  array with one entry per constraint (`func.n_constraints` entries), the normalised amount
  by which each constraint is violated: `g[j] == 0` means constraint j is satisfied and
  `g[j] > 0` measures the violation. A point is feasible if all entries of `g` are 0.
  `f` may be `inf` for points where the objective is undefined (always infeasible).
""",
}
_INTERFACE["informed"] = _INTERFACE["informed_v2"] = _INTERFACE["free"]

_COMMON_TAIL = """\
Evaluation criterion: only feasible points count. In each run the score is based on the
best objective value among the feasible points evaluated during the run (any feasible point
is better than any infeasible one), and the anytime performance over the whole budget is
measured, so finding good feasible points early matters.

Give an excellent and novel algorithm to solve this task.
"""

_EXAMPLE = {
    "penalty": '''\
An example of such code (a simple random search), is as follows:
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        best_y, best_x = np.inf, None
        for _ in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            y = func(x)
            if y < best_y:
                best_y, best_x = y, x
        return best_y, best_x
```
''',
    "free": '''\
An example of such code (a simple random search), is as follows:
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        best_f, best_x = np.inf, None
        for _ in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            f, g = func(x)
            if np.all(g == 0) and f < best_f:
                best_f, best_x = f, x
        return best_f, best_x
```
''',
}
_EXAMPLE["informed"] = _EXAMPLE["informed_v2"] = _EXAMPLE["free"]


def task_prompt(arm: str) -> str:
    parts = [_COMMON_HEAD, _INTERFACE[arm], _COMMON_TAIL]
    return "\n".join(textwrap.dedent(p) for p in parts)


def example_prompt(arm: str) -> str:
    return _EXAMPLE[arm]


MUTATIONS = {
    "refine": "Refine the strategy of the selected solution to improve it.", 
    "refine_or_redesign": "Either refine or redesign to improve the algorithm.",
    
}