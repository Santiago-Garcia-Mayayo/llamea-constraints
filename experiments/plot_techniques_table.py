"""Constraint-handling techniques of the generated algorithms, as a plain table image (slides).

    python -m experiments.plot_techniques_table

Reads results/techniques/labels.jsonl (experiments.classify_techniques). Rows: techniques; two
blocks, the primary technique (what decides selection) and any use of each technique (an
algorithm can use several). Columns: share of all candidates and of the best algorithm of each
run, per arm. Rows that are below 1% everywhere are left out. Same style as plot_table_simple.
Writes results/figures/techniques_table.png (300 dpi) and .pdf.
"""
from __future__ import annotations

import argparse
import json
import os

NAMES = {"penalty_static": "Static penalty", "penalty_adaptive": "Adaptive penalty",
         "feasibility_rules": "Feasibility rules (Deb)", "epsilon_constrained": "ε-constrained",
         "stochastic_ranking": "Stochastic ranking", "multiobjective": "Multi-objective",
         "repair": "Repair", "feasible_archive": "Feasible archive",
         "per_constraint_guidance": "Per-constraint guidance", "boundary_search": "Boundary search",
         "constraint_surrogate": "Constraint surrogate", "ignores_constraints": "Ignores constraints",
         "other": "Other"}
ARMS = (("free", "Free"), ("informed_v2", "Informed"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="results/techniques/labels.jsonl")
    ap.add_argument("--out", default="results/figures/techniques_table")
    ap.add_argument("--min-pct", type=float, default=1.0)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    names = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = next((f for f in ("Times New Roman", "Liberation Serif", "Nimbus Roman",
                                                    "STIXGeneral", "DejaVu Serif") if f in names), "serif")
    rows = [json.loads(l) for l in open(a.labels)]

    groups = []   # (header, filter)
    for arm, lab in ARMS:
        groups.append((f"{lab}\nall", lambda r, arm=arm: r["arm"] == arm))
    for arm, lab in ARMS:
        groups.append((f"{lab}\nbest", lambda r, arm=arm: r["arm"] == arm and r.get("is_best")))
    subsets = [[r for r in rows if f(r)] for _, f in groups]

    def pct(sub, t, multi):
        return 100 * sum((t in r["techniques"]) if multi else (r["primary"] == t) for r in sub) / max(1, len(sub))

    blocks = []
    for title, multi in (("Primary technique", False), ("Any use (several per algorithm)", True)):
        lines = []
        for t in NAMES:
            v = [pct(s, t, multi) for s in subsets]
            if max(v) >= a.min_pct:
                lines.append((NAMES[t], v))
        lines.sort(key=lambda x: -x[1][0])
        blocks.append((title, lines))
    ind = [100 * sum(r["uses_individual_constraints"] for r in s) / max(1, len(s)) for s in subsets]
    blocks.append(("", [("Uses each constraint separately", ind)]))

    header = ["Technique"] + [g for g, _ in groups]
    widths = [3.4] + [1.25] * len(groups)
    W, rh, fs = sum(widths), 0.34, 11
    xs = [sum(widths[:j]) for j in range(len(widths))]
    n_lines = sum(len(b[1]) + (1 if b[0] else 0) for b in blocks)
    H = 0.3 + rh * 1.9 + rh * n_lines + 0.25 * len(blocks) + 0.9
    fig = plt.figure(figsize=(W * 0.8, H * 0.8), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(-0.1, W + 0.1); ax.set_ylim(H, 0); ax.axis("off")

    y = 0.3
    ax.plot([0, W], [y, y], color="black", lw=1.4)
    for j, h in enumerate(header):
        ax.text(xs[j] + (0.1 if j == 0 else widths[j] / 2), y + rh * 0.95, h, ha="left" if j == 0 else "center",
                va="center", fontsize=fs, linespacing=1.1)
    y += rh * 1.9
    ax.plot([0, W], [y, y], color="black", lw=0.8)
    for bi, (title, lines) in enumerate(blocks):
        if bi:
            y += 0.12
            ax.plot([0, W], [y, y], color="black", lw=0.5)
            y += 0.05
        if title:
            ax.text(0.1, y + rh * 0.6, title, ha="left", va="center", fontsize=fs, style="italic")
            y += rh
        for name, v in lines:
            ax.text(0.3 if title else 0.1, y + rh * 0.6, name, ha="left", va="center", fontsize=fs)
            for j, x in enumerate(v, start=1):
                ax.text(xs[j] + widths[j] / 2, y + rh * 0.6, f"{x:.0f}%", ha="center", va="center", fontsize=fs)
            y += rh
    y += 0.1
    ax.plot([0, W], [y, y], color="black", lw=1.4)
    ns = ", ".join(f"{g.replace(chr(10), ' ')} n={len(s)}" for (g, _), s in zip(groups, subsets))
    notes = ["Share of algorithms. 'Best' = the best algorithm of each LLaMEA run.", f"{ns}.",
             f"Classified by {rows[0].get('model', 'an LLM')} from the code."]
    for k, t in enumerate(notes):
        ax.text(0.1, y + 0.3 + 0.28 * k, t, ha="left", va="center", fontsize=9, style="italic")
    y_end = y + 0.3 + 0.28 * len(notes)
    ax.set_ylim(y_end, 0)
    fig.set_size_inches(W * 0.8, y_end * 0.8)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".png", dpi=300, facecolor="white")
    fig.savefig(a.out + ".pdf", facecolor="white")
    print(f"{a.out}.png (+ .pdf)")


if __name__ == "__main__":
    main()