"""Convergence of LLaMEA: best training fitness so far vs number of evaluated candidates.

    python -m experiments.plot_llamea_convergence                    # every haiku5* repetition
    python -m experiments.plot_llamea_convergence --arms penalty free --out results/figures/conv.png

One panel per problem; per arm, the median over the LLaMEA runs (repetitions) of the
best-so-far training fitness (mean AOCC over the 5 training seeds), with a band from the
worst to the best run. Order of candidates: LLaMEA's own log (<run>/*/log.jsonl, in the order
they were evaluated); runs without it fall back to the candidates' file times.
Note: training fitness (5 seeds), not the test AOCC of the final evaluation.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np

# Categorical slots 1-3 of the reference palette (validated for all pairs), fixed per arm.
COLORS = {"penalty": "#2a78d6", "free": "#eb6834", "informed_v2": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {"penalty": "penalty", "free": "free", "informed_v2": "informed v2", "informed": "informed v1"}
PROBLEMS = ("lens/original", "lens/realistic_draft", "cec/RC12", "cec/RC13", "cec/RC15",
            "cec/RC16", "cec/RC18", "cec/RC21", "cec/RC25", "cec/RC26")


def _fit(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def fitness_sequence(run_dir: str) -> list[float]:
    logs = sorted(glob.glob(os.path.join(run_dir, "*", "log.jsonl")))
    if logs:
        seq = []
        # the most recent log: an interrupted earlier attempt can leave a shorter one
        for line in open(logs[-1]):
            try:
                seq.append(_fit(json.loads(line).get("fitness")))
            except json.JSONDecodeError:
                continue
        if seq:
            return seq
    files = sorted(glob.glob(os.path.join(run_dir, "candidates", "*.json")), key=os.path.getmtime)
    return [_fit(json.load(open(f)).get("fitness")) for f in files]


def best_so_far(seq, n):
    b = np.maximum.accumulate(np.asarray(seq, float)) if seq else np.zeros(0)
    if len(b) < n:   # a run that ended early keeps its last value
        b = np.concatenate([b, np.full(n - len(b), b[-1] if len(b) else np.nan)])
    return b[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/llamea")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=["penalty", "free", "informed_v2"])
    ap.add_argument("--candidates", type=int, default=100)
    ap.add_argument("--out", default="results/figures/llamea_convergence.png")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = a.tags or sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.runs, "haiku5*"))
                            if re.fullmatch(r"haiku5(-r\d+)?", os.path.basename(d)))
    n = a.candidates
    x = np.arange(1, n + 1)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2), sharex=True)
    fig.patch.set_facecolor("white")
    summary = []
    for ax, pid in zip(axes.flat, PROBLEMS):
        for arm in a.arms:
            curves = [best_so_far(fitness_sequence(d), n) for t in tags
                      for d in [os.path.join(a.runs, t, arm, pid.replace("/", "_"))] if os.path.isdir(d)]
            curves = [c for c in curves if np.isfinite(c).any()]
            if not curves:
                continue
            C = np.vstack(curves)
            med = np.nanmedian(C, axis=0)
            ax.fill_between(x, np.nanmin(C, axis=0), np.nanmax(C, axis=0), color=COLORS.get(arm, MUTED),
                            alpha=0.15, linewidth=0)
            ax.plot(x, med, color=COLORS.get(arm, MUTED), linewidth=2, label=f"{LABELS.get(arm, arm)} (n={len(C)})")
            summary.append((pid, arm, len(C), float(med[-1])))
        ax.set_title(pid, fontsize=11, color=INK, loc="left")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
    for ax in axes[1]:
        ax.set_xlabel("candidates evaluated", color=MUTED, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("best training fitness (AOCC)", color=MUTED, fontsize=10)
    handles, labels = axes.flat[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200)
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    print(f"{a.out} (+ .pdf); line = median over runs, band = worst..best run; repetitions: {', '.join(tags)}")
    for pid, arm, k, last in summary:
        print(f"  {pid:22s} {arm:12s} n={k}  final median {last:.3f}")


if __name__ == "__main__":
    main()