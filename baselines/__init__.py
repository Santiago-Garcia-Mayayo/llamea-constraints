"""Baselines of the study.

BASELINES maps a name to (import path, arm). Pairs differ only in the comparison:
penalty scalar (penalty arm) vs Deb's feasibility rules on (f, g) (free arm).
"""
from problems._threads import limit_threads as _limit_threads

_limit_threads()  # before numpy loads (run_baselines imports numpy first)

BASELINES = {
    "random_search":  ("baselines.random_search:RandomSearch", "free"),
    "lshade_penalty": ("baselines.lshade:LSHADEPenalty", "penalty"),
    "lshade_deb":     ("baselines.lshade:LSHADEDeb", "free"),
    "cmaes_penalty":  ("baselines.cmaes:CMAESPenalty", "penalty"),
    "cmaes_deb":      ("baselines.cmaes:CMAESDeb", "free"),
    "colshade":       ("baselines.colshade:COLSHADE", "free"),   # CEC 2020 runner-up (Deb + Levy)
    "lshade_eps":     ("baselines.lshade:LSHADEEps", "free"),    # same L-SHADE, epsilon (eps-DEag schedule)
    "lshade_eps_long": ("baselines.lshade:LSHADEEpsLong", "free"),  # epsilon, relaxed and longer (LLaMEA-like)
}