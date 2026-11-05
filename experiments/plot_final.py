"""Results table as an image (booktabs look: three rules, no colours).

    python -m experiments.plot_table
    python -m experiments.plot_table --no-diffs --all-baselines
    python -m experiments.plot_table --tags-prefix qwennext            # another LLM's repetitions

Every cell is the test AOCC, mean +- std over the individual test runs (one value per seed):
  * LLaMEA arm: the best algorithm of each of the 5 LLaMEA runs, 15 test seeds each, pooled
    (5 x 15 = 75 values);
  * best LLaMEA algorithm of the arm: of the 5 runs' best algorithms, the one with the highest
    training fitness (selected on the training seeds, so the test numbers stay unbiased;
    --select test picks the best on the test seeds instead), over its 15 test seeds;
  * baseline: its 15 test seeds.
Mean row: per test run, the mean over the 10 problems, then mean +- std of those (75 or 15).
Differences between arms (optional): difference of the means; * = two-sided Mann-Whitney
p < 0.05 over the 5 LLaMEA runs (each run = mean of its 15 seeds), the unit the study compares.
Bold: best of the arms and baselines in the row; in the best-algorithm block, best of the three.
Writes results/figures/results_table.png (300 dpi) and .pdf.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

PROBLEMS = ("lens/original", "lens/realistic_draft", "cec/RC12", "cec/RC13", "cec/RC15",
            "cec/RC16", "cec/RC18", "cec/RC21", "cec/RC25", "cec/RC26")
ARMS = (("penalty", "Penalty"), ("free", "Free"), ("informed_v2", "Informed"))
DIFFS_ALL = (("free", "penalty", "Free − Penalty"), ("informed_v2", "free", "Informed − Free"))
BASE = {"colshade": "COLSHADE", "lshade_deb": "L-SHADE Deb", "lshade_penalty": "L-SHADE pen.",
        "cmaes_deb": "CMA-ES Deb", "cmaes_penalty": "CMA-ES pen.", "random_search": "Random",
        "lshade_eps": "L-SHADE ε", "lshade_eps_long": "L-SHADE ε-long"}
FAILED = ("error", "crashed", "timeout")


def _aocc(r):
    return 0.0 if r["status"] in FAILED else r["aocc"]


def load_llamea(tags, arms):
    """{(arm, problem, tag): {seed: test AOCC}} from results/llamea_final/<tag>/runs.jsonl."""
    out = {}
    for tag in tags:
        for line in open(os.path.join("results", "llamea_final", tag, "runs.jsonl")):
            r = json.loads(line)
            b = r.get("baseline", "")
            if b.startswith("llamea_") and b[len("llamea_"):] in arms:
                out.setdefault((b[len("llamea_"):], r["problem"], tag), {})[r["seed"]] = _aocc(r)
    return out


def train_fitness(tag, arm, pid):
    f = os.path.join("results", "llamea", tag, arm, pid.replace("/", "_"), "best.json")
    try:
        return float(json.load(open(f))["fitness"])
    except Exception:
        return -np.inf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", nargs="*", default=["colshade", "lshade_deb", "lshade_eps_long"])
    ap.add_argument("--no-diffs", action="store_true", help="hide the difference columns")
    ap.add_argument("--no-best", action="store_true", help="hide the best-algorithm columns")
    ap.add_argument("--all-baselines", action="store_true", help="all eight baselines")
    ap.add_argument("--select", choices=("train", "test"), default="train",
                    help="how the best algorithm of an arm is chosen (default: training fitness)")
    ap.add_argument("--tags-prefix", default="haiku5", help="repetitions <prefix>, <prefix>-r2, ...")
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

    pat = re.compile(re.escape(a.tags_prefix) + r"(-r\d+)?")
    tags = sorted(os.path.basename(d) for d in glob.glob(os.path.join("results", "llamea_final", a.tags_prefix + "*"))
                  if pat.fullmatch(os.path.basename(d)) and os.path.exists(os.path.join(d, "runs.jsonl")))
    arms = [x for x, _ in ARMS]
    S = load_llamea(tags, arms)
    arms = [x for x in arms if any(k[0] == x for k in S)]
    ARMS_ = [(x, n) for x, n in ARMS if x in arms]
    DIFFS_ = [d for d in DIFFS if d[0] in arms and d[1] in arms]
    problems = [p for p in PROBLEMS if any(k[1] == p for k in S)]

    def runs_of(arm, pid):
        return [t for t in tags if (arm, pid, t) in S]

    # pooled per-seed values of an arm; run means (for the tests); the selected best run
    pooled, run_means, best_tag = {}, {}, {}
    for arm in arms:
        for pid in problems:
            ts = runs_of(arm, pid)
            pooled[(arm, pid)] = [v for t in ts for v in S[(arm, pid, t)].values()]
            run_means[(arm, pid)] = [float(np.mean(list(S[(arm, pid, t)].values()))) for t in ts]
            if ts:
                key = (lambda t: train_fitness(t, arm, pid)) if a.select == "train" \
                    else (lambda t: np.mean(list(S[(arm, pid, t)].values())))
                best_tag[(arm, pid)] = max(ts, key=key)
        # mean over problems, per test run (tag, seed), and per run for the tests
        full = [t for t in tags if all((arm, p, t) in S for p in problems)]
        seeds = sorted(set.intersection(*[set(S[(arm, p, t)]) for p in problems for t in full])) if full else []
        pooled[(arm, "mean")] = [float(np.mean([S[(arm, p, t)][s] for p in problems])) for t in full for s in seeds]
        run_means[(arm, "mean")] = [float(np.mean([run_means[(arm, p)][runs_of(arm, p).index(t)] for p in problems]))
                                    for t in full]
    best_vals = {}
    for arm in arms:
        for pid in problems:
            if (arm, pid) in best_tag:
                best_vals[(arm, pid)] = list(S[(arm, pid, best_tag[(arm, pid)])].values())
        if all((arm, p) in best_tag for p in problems):
            dicts = [S[(arm, p, best_tag[(arm, p)])] for p in problems]
            seeds = sorted(set.intersection(*[set(d) for d in dicts]))
            best_vals[(arm, "mean")] = [float(np.mean([d[s] for d in dicts])) for s in seeds]

    acc = {}
    if os.path.exists(a.baselines_file):
        for line in open(a.baselines_file):
            r = json.loads(line)
            if r["baseline"] in a.baselines:
                acc.setdefault((r["baseline"], r["problem"]), {})[r["seed"]] = _aocc(r)
    bv = {k: list(v.values()) for k, v in acc.items()}
    for b in a.baselines:
        if all((b, p) in acc for p in problems):
            seeds = sorted(set.intersection(*[set(acc[(b, p)]) for p in problems]))
            bv[(b, "mean")] = [float(np.mean([acc[(b, p)][s] for p in problems])) for s in seeds]
    bases = [b for b in a.baselines if any((b, p) in bv for p in problems)]

    def cell(v):
        return ("–", None) if not v else (f"{np.mean(v):.3f} ± {np.std(v):.3f}", float(np.mean(v)))

    show_best = not a.no_best
    header = (["Problem"] + [n for _, n in ARMS_] + [d[2] for d in DIFFS_]
              + ([f"Best {n.lower()}" for _, n in ARMS_] if show_best else []) + [BASE[b] for b in bases])
    rows = []
    for pid in problems + ["mean"]:
        row = ["Mean" if pid == "mean" else pid]
        arm_cells = [cell(pooled.get((x, pid))) for x, _ in ARMS_]
        base_cells = [cell(bv.get((b, pid))) for b in bases]
        best = max([m for _, m in arm_cells + base_cells if m is not None], default=None)
        row += [(t, m is not None and abs(m - best) < 5e-4) for t, m in arm_cells]
        for x, y, _ in DIFFS_:
            A, B = run_means.get((x, pid), []), run_means.get((y, pid), [])
            if not A or not B:
                row.append(("–", False)); continue
            d = np.mean(A) - np.mean(B)
            star = "*" if len(A) > 1 and len(B) > 1 and mannwhitneyu(A, B, alternative="two-sided").pvalue < 0.05 else ""
            row.append((f"{d:+.3f}{star}".replace("-", "−"), False))
        if show_best:
            bc = [cell(best_vals.get((x, pid))) for x, _ in ARMS_]
            bb = max([m for _, m in bc if m is not None], default=None)
            row += [(t, m is not None and abs(m - bb) < 5e-4) for t, m in bc]
        row += [(t, m is not None and abs(m - best) < 5e-4) for t, m in base_cells]
        rows.append(row)

    nb = len(ARMS_) if show_best else 0
    widths = [2.2] + [1.75] * len(ARMS_) + [1.9] * len(DIFFS_) + [1.75] * nb + [1.75] * len(bases)
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
        for j, c in enumerate(row):
            txt, bold = (c, False) if isinstance(c, str) else c
            ax.text(xs[j] + (0.1 if j == 0 else widths[j] / 2), y + rh * 0.6, txt, ha="left" if j == 0 else "center",
                    va="center", fontsize=fs, fontweight="bold" if bold else "normal")
        y += rh
    y += 0.08
    ax.plot([0, W], [y, y], color="black", lw=1.4)
    n_runs = len(tags)
    notes = [f"Test AOCC (higher is better), mean ± std over test runs: LLaMEA arm = best algorithm of each of the "
             f"{n_runs} LLaMEA runs × 15 seeds ({n_runs * 15} values); baselines = 15 seeds."]
    if show_best:
        notes.append("Best: the arm's best algorithm, chosen by "
                     + ("training fitness" if a.select == "train" else "test AOCC") + ", over its 15 test seeds.")
    notes.append("Mean row: per test run, the mean over the problems. Bold: best of arms and baselines"
                 + ("; in the Best block, best of the three." if show_best else ".")
                 + ("" if not DIFFS_ else " * p < 0.05, Mann–Whitney U over the LLaMEA runs (uncorrected)."))
    for k, t in enumerate(notes):
        ax.text(0.1, y + 0.3 + 0.3 * k, t, ha="left", va="center", fontsize=9.5, style="italic")
    y_end = y + 0.3 + 0.3 * len(notes)
    ax.set_ylim(y_end, 0)
    fig.set_size_inches(W * 0.8, y_end * 0.8)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".png", dpi=300, facecolor="white")
    fig.savefig(a.out + ".pdf", facecolor="white")
    print(f"{a.out}.png (+ .pdf); repetitions: {', '.join(tags)}; best selected by {a.select}")
    for x, _ in ARMS_:
        print(f"  best {x:12s}: " + ", ".join(f"{p.split('/')[-1]}={best_tag.get((x, p), '-')}" for p in problems))


if __name__ == "__main__":
    main()