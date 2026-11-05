"""Glue between LLaMEA (pip install llamea) and the harness.

    from harness.llamea_adapter import make_llamea
    es = make_llamea(arm="free", llm=my_llm, problem_id="cec/RC15", experiment_name="free-RC15")
    best = es.run()

One LLaMEA run evolves algorithms for ONE problem (DESIGN.md §8). LLaMEA calls
`f(solution, logger)`; we run solution.code through the harness on that problem with the
training seeds, and set fitness (mean AOCC, maximised), feedback and error.
Everything arm-specific lives in harness.prompts (task prompt) and harness.evaluate (feedback).
"""
from __future__ import annotations

import json
import os

from . import config, prompts
from .evaluate import evaluate_candidate


class Evaluator:
    """LLaMEA's fitness function for one arm and problem(s). A top-level class (not a closure)
    so that LLaMEA can pickle it for its worker processes and checkpoints."""

    def __init__(self, arm: str, problem_ids, seeds=config.TRAIN_SEEDS,
                 n_workers: int | None = None, log_dir: str | None = None):
        self.arm = arm
        self.problem_ids = [problem_ids] if isinstance(problem_ids, str) else list(problem_ids)
        self.seeds, self.n_workers, self.log_dir = tuple(seeds), n_workers, log_dir

    def _parent_summary(self, solution):
        """informed_v2: the parent's per-problem summaries, read from its candidate JSON."""
        if self.arm != "informed_v2" or not self.log_dir or not getattr(solution, "parent_ids", None):
            return None
        path = os.path.join(self.log_dir, f"{solution.parent_ids[0]}.json")
        try:
            with open(path) as fh:
                return json.load(fh)["per_problem"]
        except (OSError, ValueError, KeyError):
            return None

    def __call__(self, solution, logger=None):
        try:
            res = evaluate_candidate(solution.code, self.arm, self.problem_ids, self.seeds,
                                     class_name=solution.name, n_workers=self.n_workers,
                                     algorithm_name=solution.name,
                                     parent_per_problem=self._parent_summary(solution))
        except Exception as e:  # harness-side failure: never let it look like a good algorithm
            solution.set_scores(0.0, f"Evaluation failed: {e}", e)
            return solution
        err = RuntimeError(res.error.splitlines()[0]) if res.error else None
        solution.set_scores(res.fitness, res.feedback, err)
        solution.add_metadata("per_problem", res.per_problem)
        if self.arm == "informed":
            solution.add_metadata("ast_features", sage_features(solution.code, res.per_problem))
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, f"{solution.id}.json"), "w") as fh:
                json.dump({"name": solution.name, "arm": self.arm, "fitness": res.fitness,
                           "per_problem": res.per_problem, "runs": res.runs, "code": solution.code,
                           "feedback": res.feedback}, fh, default=float)
        return solution


def behaviour_features(per_problem) -> dict:
    """How the candidate behaved with respect to the constraints (one problem per run).
    Times are fractions of the budget; "never" counts as the whole budget. The gap to the
    reference value is left out on purpose: it is almost the fitness itself."""
    s = next(iter(per_problem.values()))
    nan = float("nan")
    pct = lambda v, never: never if v is None else v / 100.0   # noqa: E731
    f = {
        "fraction_of_feasible_evaluations": float(s.get("feasible_fraction", 0.0)),
        "time_to_first_feasible_point": pct(s.get("first_feasible_pct_median"), 1.0),
        "time_to_90_percent_of_final_improvement": pct(s.get("eval90_pct_median"), nan),
        "time_of_last_improvement": pct(s.get("last_improvement_pct_median"), nan),
    }
    for j, v in enumerate(s.get("viol_freq_mean", [])):
        f[f"violation_rate_of_constraint_{j}"] = float(v)
    return f


def sage_features(code: str, per_problem) -> dict:
    """Features for LLaMEA-SAGE's archive model in the informed arm: SAGE's own code-structure
    features plus the behaviour features. LLaMEA reads them from the `ast_features` metadata
    and turns the most important one into an instruction such as "try to decrease the time to
    first feasible point of the solution"."""
    from llamea.ast_features import extract_ast_features
    try:
        feats = dict(extract_ast_features(code))
    except Exception:
        feats = {}
    feats.update(behaviour_features(per_problem))
    return feats


def make_evaluator(arm: str, problem_ids, seeds=config.TRAIN_SEEDS,
                   n_workers: int | None = None, log_dir: str | None = None) -> Evaluator:
    return Evaluator(arm, problem_ids, seeds, n_workers, log_dir)


def candidate_timeout(problem_ids, seeds=config.TRAIN_SEEDS) -> int:
    """Upper bound on the wall time of one candidate evaluation (all problem x seed jobs run
    in parallel), used as LLaMEA's eval_timeout."""
    from problems import get_problem
    from .runner import _LensStub
    problem_ids = [problem_ids] if isinstance(problem_ids, str) else list(problem_ids)
    limits = []
    for pid in problem_ids:
        p = _LensStub(pid) if pid.startswith("lens/") else get_problem(pid)
        limits.append(config.time_limit_for(p, config.budget_for(p, "train")))
    return int(config.STARTUP_GRACE_S + max(limits) + 15.0 + 60)


def make_llamea(arm: str, llm, problem_id, budget: int = config.LLAMEA_BUDGET,
                n_parents: int = config.LLAMEA_N_PARENTS, n_offspring: int = config.LLAMEA_N_OFFSPRING,
                elitism: bool = config.LLAMEA_ELITISM, experiment_name: str = "",
                max_workers: int | None = None, n_workers_per_candidate: int | None = None,
                log_dir: str | None = None, seeds=config.TRAIN_SEEDS, llm_timeout: int = 0, **kwargs):
    """LLaMEA instance with the arm's prompts for one problem. Fitness is maximised.

    Parallelism: LLaMEA evaluates up to `max_workers` candidates at once (default: all
    offspring of a generation) and each candidate runs its seeds in parallel, so one run uses
    about n_offspring x len(seeds) processes.

    LLaMEA waits for each parallel job (LLM query + evaluation) at most eval_timeout + 15 s, and
    eval_timeout is only used for that wait (evaluations have their own limits in the runner).
    llm_timeout adds seconds for the LLM itself: 0 for API models, which answer in seconds;
    more for slow local models whose requests also queue on one server."""
    from llamea import LLaMEA
    # informed arm = free prompt + behaviour statistics in the feedback + LLaMEA-SAGE guidance
    # over code-structure and behaviour features (see sage_features).
    # informed_v2 = free prompt + diagnoses and parent comparison in the feedback; no SAGE.
    kwargs.setdefault("feature_guided_mutation", arm == "informed")
    # mutation prompt: LLaMEA's default unless LLAMEA_MUTATION names one of prompts.MUTATIONS
    if os.environ.get("LLAMEA_MUTATION"):
        kwargs.setdefault("operators", [prompts.MUTATIONS[os.environ["LLAMEA_MUTATION"]]])
    evaluator = make_evaluator(arm, problem_id, seeds, n_workers=n_workers_per_candidate, log_dir=log_dir)
    return LLaMEA(evaluator, llm=llm,
                  n_parents=n_parents, n_offspring=n_offspring, elitism=elitism,
                  role_prompt=prompts.ROLE, task_prompt=prompts.task_prompt(arm),
                  example_prompt=prompts.example_prompt(arm),
                  experiment_name=experiment_name or f"{arm}-{problem_id.replace('/', '_')}",
                  budget=budget, eval_timeout=candidate_timeout(problem_id, seeds) + int(llm_timeout),
                  max_workers=max_workers or n_offspring, minimization=False, **kwargs)