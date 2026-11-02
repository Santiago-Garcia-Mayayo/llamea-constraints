"""Results table as an image (booktabs look: three rules, no colours).

    python -m experiments.plot_table
    python -m experiments.plot_table --baselines colshade lshade_deb lshade_penalty cmaes_deb

Same numbers as experiments.aggregate. Writes
results/figures/results_table.png (300 dpi) and .pdf.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from experiments.aggregate import load

PROBLEMS = ("lens/original", "lens/realistic_draft", "cec/RC12", "cec/RC13", "cec/RC15",
            "cec/RC16", "cec/RC18", "cec/RC21", "cec/RC25", "cec/RC26")
ARMS = (("penalty", "Penalty"), ("free", "Free"), ("informed_v2", "Informed"))
DIFFS_ALL = (("free", "penalty", "Free − Penalty"), ("informed_v2", "free", "Informed − Free"))
BASE = {"colshade": "COLSHADE", "lshade_deb": "L-SHADE Deb", "lshade_penalty": "L-SHADE pen.",
        "cmaes_deb": "CMA-ES Deb", "cmaes_penalty": "CMA-ES pen.", "random_search": "Random",
        "lshade_eps": "L-SHADE ε", "lshade_eps_long": "L-SHADE ε-long"}
FAILED = ("error", "crashed", "timeout")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", nargs="*", default=["colshade", "lshade_deb", "lshade_eps_long"])
    ap.add_argument("--no-diffs", action="store_true", help="hide the difference columns")
    ap.add_argument("--all-baselines", action="store_true", help="the six baselines")
    ap.add_argument("--baselines-file", default="results/baselines_1000d/runs.jsonl")
    ap.add_argument("--out", default="results/figures/results_table")
    a = ap.parse_args()
    if a.all_baselines:
        a.baselines = ["colshade", "lshade_deb", "lshade_penalty", "lshade_eps", "lshade_eps_long", "cmaes_deb",
                       "cmaes_penalty", "random_search"]
    DIFFS = () if a.no_diffs else DIFFS_ALL
    from scipy.stats import mannwhitneyu
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    names = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = next((f for f in ("Times New Roman", "Liberation Serif", "Nimbus Roman",
                                                    "STIXGeneral", "DejaVu Serif") if f in names), "serif")
    plt.rcParams["mathtext.fontset"] = "stix"

    tags = sorted(os.path.basename(d) for d in glob.glob("results/llamea_final/haiku5*")
                  if re.fullmatch(r"haiku5(-r\d+)?", os.path.basename(d)) and os.path.exists(os.path.join(d, "runs.jsonl")))
    per, _ = load(tags)
    runs = {(x, p): list(per.get((x, p), {}).values()) for x, _ in ARMS for p in PROBLEMS}
    for x, _ in ARMS:
        runs[(x, "mean")] = [float(np.mean([per[(x, p)][t] for p in PROBLEMS])) for t in tags
                             if all(t in per.get((x, p), {}) for p in PROBLEMS)]
    bv = {}
    acc = {}
    if os.path.exists(a.baselines_file):
        for line in open(a.baselines_file):
            r = json.loads(line)
            if r["baseline"] in a.baselines:
                acc.setdefault((r["baseline"], r["problem"]), []).append(0.0 if r["status"] in FAILED else r["aocc"])
    bv = {k: float(np.mean(v)) for k, v in acc.items()}
    for b in a.baselines:
        vals = [bv[(b, p)] for p in PROBLEMS if (b, p) in bv]
        if len(vals) == len(PROBLEMS):          # mean only when the baseline ran on every problem
            bv[(b, "mean")] = float(np.mean(vals))

    header = (["Problem"] + [n for _, n in ARMS] + [d[2] for d in DIFFS] + [BASE[b] for b in a.baselines])
    rows = []
    for pid in list(PROBLEMS) + ["mean"]:
        row = ["Mean" if pid == "mean" else pid]
        means = [np.mean(runs[(x, pid)]) for x, _ in ARMS]
        best = max(means + [bv.get((b, pid), -1) for b in a.baselines])
        for (x, _), m in zip(ARMS, means):
            sd = np.std(runs[(x, pid)])
            row.append((f"{m:.3f} ± {sd:.3f}", abs(m - best) < 5e-4))
        for x, y, _ in DIFFS:
            A, B = runs[(x, pid)], runs[(y, pid)]
            d = np.mean(A) - np.mean(B)
            star = "*" if pid != "mean" and mannwhitneyu(A, B, alternative="two-sided").pvalue < 0.05 else ""
            if pid == "mean":
                star = "*" if mannwhitneyu(A, B, alternative="two-sided").pvalue < 0.05 else ""
            row.append((f"{d:+.3f}{star}".replace("-", "−"), False))
        for b in a.baselines:
            v = bv.get((b, pid))
            row.append(("–" if v is None else f"{v:.3f}", v is not None and abs(v - best) < 5e-4))
        rows.append(row)

    widths = [2.2] + [1.7] * len(ARMS) + [2.0] * len(DIFFS) + [1.6] * len(a.baselines)
    W, rh = sum(widths), 0.34
    H = rh * (len(rows) + 2) + 0.9
    fig = plt.figure(figsize=(W * 0.8, H * 0.8 + 0.4), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(-0.1, W + 0.1); ax.set_ylim(H, 0); ax.axis("off")
    xs = np.cumsum([0] + widths)
    fs = 11
    y = 0.3
    ax.plot([0, W], [y, y], color="black", lw=1.4)
    for j, h in enumerate(header):
        ax.text(xs[j] + (0.1 if j == 0 else widths[j] / 2), y + rh * 0.55, h, ha="left" if j == 0 else "center",
                va="center", fontsize=fs)
    y += rh * 1.1
    ax.plot([0, W], [y, y], color="black", lw=0.8)
    for i, row in enumerate(rows):
        if i == len(rows) - 1:
            y += 0.06
            ax.plot([0, W], [y, y], color="black", lw=0.5)
        for j, cell in enumerate(row):
            txt, bold = (cell, False) if isinstance(cell, str) else cell
            ax.text(xs[j] + (0.1 if j == 0 else widths[j] / 2), y + rh * 0.6, txt, ha="left" if j == 0 else "center",
                    va="center", fontsize=fs, fontweight="bold" if bold else "normal")
        y += rh
    y += 0.08
    ax.plot([0, W], [y, y], color="black", lw=1.4)
    ax.text(0.1, y + 0.3, "Test AOCC (higher is better). LLaMEA: mean ± std over 5 runs; baselines: mean over 15 seeds.",
            ha="left", va="center", fontsize=9.5, style="italic")
    ax.text(0.1, y + 0.62, "Bold: best in row." + ("" if a.no_diffs else " * p < 0.05 (Mann–Whitney U, uncorrected)."),
            ha="left", va="center", fontsize=9.5, style="italic")
    ax.set_ylim(y + 0.9, 0)
    fig.set_size_inches(W * 0.8, (y + 0.9) * 0.8)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".png", dpi=300, facecolor="white")
    fig.savefig(a.out + ".pdf", facecolor="white")
    print(f"{a.out}.png (+ .pdf); font: {plt.rcParams['font.family']}")


if __name__ == "__main__":
    main()
    