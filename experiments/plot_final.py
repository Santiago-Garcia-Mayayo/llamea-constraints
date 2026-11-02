"""Figures of the final evaluation (test seeds, test budget) of LLaMEA's best algorithms.

    python -m experiments.plot_final                 # both figures, every haiku5* repetition
    python -m experiments.plot_final --logx          # anytime curves on a log budget axis

1. results/figures/final_anytime.png - anytime performance of the best algorithm of each
   LLaMEA run: unit score of the best point so far (u in [0, 1]; >= 0.5 = feasible) vs
   fraction of the budget. Per arm: mean over the LLaMEA runs (each run = mean of its 15 test
   seeds), band = worst..best run. The area under each curve is the AOCC. Grey: the baselines
   (by default COLSHADE and the two L-SHADE variants; --all-baselines for all six).
2. results/figures/final_aocc.png - final test AOCC: one dot per LLaMEA run (mean over the
   15 test seeds), a bar at the mean of the runs; then the six baselines in grey, each as the
   mean over its 15 test seeds with a +-1 std whisker (a baseline is one fixed algorithm, so
   its spread is over seeds, not over LLaMEA runs).
Also writes .pdf versions.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from experiments.plot_llamea_convergence import COLORS, GRID, INK, LABELS, MUTED, PROBLEMS

FAILED = ("error", "crashed", "timeout")
BASE = {"colshade": ("COLSHADE (CEC 2020)", "#0b0b0b", (0, (6, 2, 1.5, 2, 1.5, 2))),
        "lshade_deb": ("L-SHADE + Deb", "#52514e", (0, (5, 3))),
        "lshade_penalty": ("L-SHADE + penalty", "#9a9893", (0, (1.5, 2.5))),
        "cmaes_deb": ("CMA-ES + Deb", "#52514e", (0, (8, 2, 1.5, 2))),
        "cmaes_penalty": ("CMA-ES + penalty", "#9a9893", (0, (3, 1.5))),
        "random_search": ("random search", "#bdbbb5", "solid"),
        "lshade_eps": ("L-SHADE + ε (εDEag)", "#7a7975", (0, (2, 1))),
        "lshade_eps_long": ("L-SHADE + ε (long)", "#2b2b29", (0, (1, 1)))}
DEFAULT_CURVES = ("colshade", "lshade_deb", "lshade_penalty", "lshade_eps_long")
SHORT = {"colshade": "COLSHADE", "lshade_deb": "L-SHADE Deb", "lshade_penalty": "L-SHADE pen.", "cmaes_deb": "CMA-ES Deb",
         "cmaes_penalty": "CMA-ES pen.", "random_search": "random",
         "lshade_eps": "L-SHADE ε", "lshade_eps_long": "L-SHADE ε-long"}


def dense(curve, budget, grid):
    """Unit score of the best-so-far point after evaluation t, at t = grid * budget."""
    t = np.maximum(1, np.ceil(grid * budget)).astype(int)
    if not curve:
        return np.zeros_like(grid)
    e = np.array([c[0] for c in curve]); u = np.array([c[1] for c in curve])
    idx = np.searchsorted(e, t, side="right") - 1
    return np.where(idx >= 0, u[np.clip(idx, 0, None)], 0.0)


def load_runs(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def _style(ax, title):
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=["penalty", "free", "informed_v2"])
    ap.add_argument("--baselines", default="results/baselines_1000d/runs.jsonl")
    ap.add_argument("--outdir", default="results/figures")
    ap.add_argument("--logx", action="store_true")
    ap.add_argument("--all-baselines", action="store_true", help="all five baselines in the anytime figure")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = a.tags or sorted(os.path.basename(d) for d in glob.glob("results/llamea_final/haiku5*")
                            if re.fullmatch(r"haiku5(-r\d+)?", os.path.basename(d)))
    grid = np.geomspace(1e-3, 1, 400) if a.logx else np.linspace(1 / 400, 1, 400)
    # per (arm, problem): list over LLaMEA runs of (mean curve over seeds, mean AOCC over seeds)
    llamea = {}
    for tag in tags:
        acc = {}
        for r in load_runs(os.path.join("results", "llamea_final", tag, "runs.jsonl")):
            if not r.get("baseline", "").startswith("llamea_"):
                continue
            arm = r["baseline"][len("llamea_"):]
            bad = r["status"] in FAILED
            c = np.zeros_like(grid) if bad else dense(r["curve"], r["budget"], grid)
            acc.setdefault((arm, r["problem"]), []).append((c, 0.0 if bad else r["aocc"]))
        for k, v in acc.items():
            llamea.setdefault(k, []).append((np.mean([c for c, _ in v], axis=0), float(np.mean([x for _, x in v]))))
    base = {}
    for r in load_runs(a.baselines):
        if r["baseline"] in BASE:
            bad = r["status"] in FAILED
            base.setdefault((r["baseline"], r["problem"]), []).append(
                (np.zeros_like(grid) if bad else dense(r["curve"], r["budget"], grid), 0.0 if bad else r["aocc"]))

    os.makedirs(a.outdir, exist_ok=True)
    # ---- figure 1: anytime curves
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2), sharex=True)
    fig.patch.set_facecolor("white")
    curves_shown = tuple(BASE) if a.all_baselines else DEFAULT_CURVES
    for ax, pid in zip(axes.flat, PROBLEMS):
        for b in curves_shown:
            lab, col, dash = BASE[b]
            v = base.get((b, pid))
            if v:
                ax.plot(grid, np.mean([c for c, _ in v], axis=0), color=col, linewidth=1.5, linestyle=dash, label=lab)
        for arm in a.arms:
            v = llamea.get((arm, pid))
            if not v:
                continue
            C = np.vstack([c for c, _ in v])
            ax.fill_between(grid, C.min(0), C.max(0), color=COLORS.get(arm, MUTED), alpha=0.15, linewidth=0)
            ax.plot(grid, C.mean(0), color=COLORS.get(arm, MUTED), linewidth=2,
                    label=f"LLaMEA {LABELS.get(arm, arm)} (n={len(C)})")
        ax.axhline(0.5, color=MUTED, linewidth=0.8, alpha=0.6)
        if a.logx:
            ax.set_xscale("log")
        _style(ax, pid)
    for ax in axes[1]:
        ax.set_xlabel("fraction of the evaluation budget", color=MUTED, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("score of best point so far", color=MUTED, fontsize=10)
    h, l = axes.flat[2].get_legend_handles_labels()
    ncol = min(len(l), 4)
    fig.legend(h, l, loc="upper center", ncol=ncol, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93 if len(l) <= 4 else 0.89))
    out1 = os.path.join(a.outdir, "final_anytime" + ("_logx" if a.logx else "") + ("_all" if a.all_baselines else ""))
    fig.savefig(out1 + ".png", dpi=200); fig.savefig(out1 + ".pdf")
    plt.close(fig)

    # ---- figure 2: final AOCC per LLaMEA run, then the baselines
    bnames = [b for b in BASE if any((b, p) in base for p in PROBLEMS)]
    cols = list(a.arms) + bnames
    fig, axes = plt.subplots(2, 5, figsize=(17, 8))
    fig.patch.set_facecolor("white")
    rng = np.random.default_rng(0)
    lines = []
    for ax, pid in zip(axes.flat, PROBLEMS):
        for i, arm in enumerate(a.arms):
            v = llamea.get((arm, pid))
            if not v:
                continue
            y = np.array([x for _, x in v])
            ax.hlines(y.mean(), i - 0.3, i + 0.3, color=COLORS.get(arm, MUTED), linewidth=2.5)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=36, color=COLORS.get(arm, MUTED),
                       edgecolor="white", linewidth=1.0, zorder=3, label=f"LLaMEA {LABELS.get(arm, arm)} (one dot per run)")
            lines.append(f"  {pid:22s} {arm:15s} n={len(y)}  mean {y.mean():.3f}  runs {np.round(np.sort(y), 3)}")
        for j, b in enumerate(bnames, start=len(a.arms)):
            v = base.get((b, pid))
            if not v:
                continue
            y = np.array([x for _, x in v])
            ax.errorbar(j, y.mean(), yerr=y.std(), fmt="D", ms=6, color=BASE[b][1], ecolor=BASE[b][1],
                        elinewidth=1.2, capsize=3, zorder=3, label="baseline (mean ± std over seeds)" if j == len(a.arms) else None)
            lines.append(f"  {pid:22s} {b:15s} seeds={len(y)}  mean {y.mean():.3f} ± {y.std():.3f}")
        ax.axvline(len(a.arms) - 0.5, color=GRID, linewidth=1.2)
        ax.set_xticks(range(len(cols)), [LABELS.get(x, SHORT.get(x, x)) for x in cols],
                      fontsize=8.5, rotation=55, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(cols) - 0.4)
        _style(ax, pid)
    for ax in axes[:, 0]:
        ax.set_ylabel("test AOCC (mean of 15 seeds)", color=MUTED, fontsize=10)
    h, l = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out2 = os.path.join(a.outdir, "final_aocc")
    fig.savefig(out2 + ".png", dpi=200); fig.savefig(out2 + ".pdf")
    plt.close(fig)
    print(f"{out1}.png, {out2}.png (+ .pdf); repetitions: {', '.join(tags)}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()