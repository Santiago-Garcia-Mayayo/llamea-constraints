"""How the primary constraint-handling technique of LLaMEA's best algorithm changes from
generation to generation. One panel per problem, one line per LLaMEA run (repetition).

    python -m experiments.plot_technique_trajectory                  # free and informed_v2
    python -m experiments.plot_technique_trajectory --arm free
    python -m experiments.plot_technique_trajectory --mode gen-best  # best of each generation
    python -m experiments.plot_technique_trajectory --detailed       # all techniques, not 4 groups

Reads results/techniques/labels.jsonl (from classify_techniques run), so it only covers the
arms that were classified (free, informed_v2; penalty is a penalty by construction).

    --mode best-so-far (default)  at each generation, the technique of the best algorithm found
                                  so far (what LLaMEA keeps as its elite); the line only moves
                                  when a new best algorithm uses another technique.
    --mode gen-best               the technique of the best algorithm generated in that generation.

y axis: Deb (feasibility rules), epsilon, penalty (static or adaptive) and, below the dotted
line, all other techniques. Lines of different runs are shifted slightly so they do not hide
each other. Also prints, per arm and problem, how many runs change technique and how often.
Writes results/figures/technique_trajectory_<arm>.png / .pdf.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

from experiments.plot_table import PROBLEMS

GROUPS = [("Deb", {"feasibility_rules"}), ("ε", {"epsilon_constrained"}),
          ("Penalty", {"penalty_static", "penalty_adaptive"})]
OTHERS = "Others"
DETAILED = ["feasibility_rules", "epsilon_constrained", "penalty_static", "penalty_adaptive",
            "stochastic_ranking", "multiobjective", "repair", "feasible_archive", "per_constraint_guidance",
            "boundary_search", "constraint_surrogate", "ignores_constraints", "other"]
ARM_NAMES = {"free": "Free", "informed_v2": "Informed"}
COLORS = ["#1f3b73", "#d4a017", "#2e7d32", "#b23a48", "#6a4c93", "#4a4a4a", "#e07b39", "#2a9d8f"]


def group_of(t, detailed):
    if detailed:
        return t
    return next((name for name, s in GROUPS if t in s), OTHERS)


def trajectories(rows, mode):
    """{(arm, problem): {tag: [(generation, technique, fitness), ...]}}, one point per generation."""
    by_run = defaultdict(list)
    for r in rows:
        if r.get("generation") is None:
            continue
        by_run[(r["arm"], r["problem"], r["tag"])].append(r)
    out = defaultdict(dict)
    for (arm, pid, tag), rs in by_run.items():
        rs.sort(key=lambda r: (r["generation"], r.get("order", 0)))
        gens = sorted({r["generation"] for r in rs})
        best, pts = None, []
        for g in gens:
            cand = [r for r in rs if r["generation"] == g and r.get("fitness") is not None
                    and np.isfinite(r["fitness"])]
            gbest = max(cand, key=lambda r: r["fitness"]) if cand else None   # first one wins ties
            if mode == "gen-best":
                if gbest is not None:
                    pts.append((g, gbest["primary"], gbest["fitness"]))
                continue
            if gbest is not None and (best is None or gbest["fitness"] > best["fitness"]):
                best = gbest
            if best is not None:
                pts.append((g, best["primary"], best["fitness"]))
        out[(arm, pid)][tag] = pts
    return out


def draw(arm, traj, problems, levels, detailed, out, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = next((f for f in ("Times New Roman", "Liberation Serif", "Nimbus Roman",
                                                    "STIXGeneral", "DejaVu Serif") if f in names), "serif")
    ncol = 2 if len(problems) > 1 else 1
    nrow = int(np.ceil(len(problems) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, (0.42 * len(levels) + 1.1) * nrow),
                             squeeze=False, sharey=True)
    ypos = {lv: len(levels) - 1 - i for i, lv in enumerate(levels)}
    tags_all = sorted({t for pid in problems for t in traj.get((arm, pid), {})})
    for k, pid in enumerate(problems):
        ax = axes[k // ncol][k % ncol]
        runs = traj.get((arm, pid), {})
        n = max(1, len(tags_all))
        for j, tag in enumerate(tags_all):
            pts = runs.get(tag)
            if not pts:
                continue
            off = (j - (n - 1) / 2) * 0.07
            x = [p[0] for p in pts]
            y = [ypos[group_of(p[1], detailed)] + off for p in pts]
            ax.plot(x, y, "-o", color=COLORS[j % len(COLORS)], lw=1.3, ms=3.5, mfc="white",
                    mec=COLORS[j % len(COLORS)], mew=1.0, label=_short(tag))
        if not detailed:
            ax.axhline(ypos[OTHERS] + 0.5, color="#c0392b", lw=0.9, ls=":")
        ax.set_title(pid, fontsize=11)
        ax.set_yticks(list(ypos.values()), [_lab(lv) for lv in ypos])
        ax.set_ylim(-0.6, len(levels) - 0.4)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="x", color="#dddddd", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if k // ncol == nrow - 1:
            ax.set_xlabel("Generation")
    for k in range(len(problems), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    what = "best algorithm so far" if mode == "best-so-far" else "best algorithm of each generation"
    fig.suptitle(f"{ARM_NAMES.get(arm, arm)}: primary technique of the {what}", fontsize=12)
    if h:
        fig.legend(h, l, loc="lower center", ncol=len(h), frameon=False, fontsize=9, title="LLaMEA run")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".png", dpi=300, facecolor="white")
    fig.savefig(out + ".pdf", facecolor="white")
    plt.close(fig)


def _short(tag):
    m = re.search(r"-r(\d+)$", tag)
    return f"run {int(m.group(1))}" if m else "run 1"


def _lab(lv):
    return lv.replace("_", " ") if lv not in ("Deb", "ε", "Penalty", OTHERS) else lv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="results/techniques/labels.jsonl")
    ap.add_argument("--arm", nargs="*", default=["free", "informed_v2"])
    ap.add_argument("--problems", nargs="*", default=None)
    ap.add_argument("--tags-prefix", default="haiku5")
    ap.add_argument("--mode", choices=("best-so-far", "gen-best"), default="best-so-far")
    ap.add_argument("--detailed", action="store_true")
    ap.add_argument("--outdir", default="results/figures")
    a = ap.parse_args()
    pat = re.compile(re.escape(a.tags_prefix) + r"(-r\d+)?")
    rows = [json.loads(l) for l in open(a.labels)]
    rows = [r for r in rows if pat.fullmatch(r["tag"]) and r["arm"] in a.arm]
    traj = trajectories(rows, a.mode)
    problems = a.problems or [p for p in PROBLEMS if any((arm, p) in traj for arm in a.arm)]
    levels = DETAILED if a.detailed else [g for g, _ in GROUPS] + [OTHERS]
    if a.detailed:   # keep only techniques that appear
        seen = {p[1] for runs in traj.values() for pts in runs.values() for p in pts}
        levels = [t for t in levels if t in seen]
    for arm in a.arm:
        out = os.path.join(a.outdir, f"technique_trajectory_{arm}" + ("_detailed" if a.detailed else "")
                           + ("_genbest" if a.mode == "gen-best" else ""))
        draw(arm, traj, problems, levels, a.detailed, out, a.mode)
        print(f"{out}.png (+ .pdf)")
        for pid in problems:
            runs = traj.get((arm, pid), {})
            ch = [sum(group_of(p[1], a.detailed) != group_of(q[1], a.detailed) for p, q in zip(pts, pts[1:]))
                  for pts in runs.values()]
            final = [group_of(pts[-1][1], a.detailed) for pts in runs.values() if pts]
            print(f"  {pid:22s} runs that change: {sum(c > 0 for c in ch)}/{len(ch)}  "
                  f"changes per run: {ch}  final: {', '.join(final)}")


if __name__ == "__main__":
    main()