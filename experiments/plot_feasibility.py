"""How fast and how often each method is feasible on the test runs: two table images.

    python -m experiments.plot_feasibility
    python -m experiments.plot_feasibility --all-baselines
    python -m experiments.plot_feasibility --tags-prefix qwennext

1. results/figures/feasibility_first.png - first feasible point, as % of the evaluation budget
   (lower = faster). Runs that never find a feasible point count as 100 %; their share is shown
   in brackets when there are any (e.g. "[3/75 never]").
2. results/figures/feasibility_fraction.png - share of all evaluated points that are feasible.
Every cell is mean +- std over the individual test runs: LLaMEA arm = the best algorithm of
each LLaMEA run x 15 seeds (75 values); baseline = 15 seeds. Mean row: per test run, the mean
over the problems. Bold: best of the row. Same style as plot_table.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from experiments.plot_table import ARMS, BASE, PROBLEMS

ALL_BASELINES = ["colshade", "lshade_deb", "lshade_penalty", "lshade_eps", "lshade_eps_long", "cmaes_deb",
                 "cmaes_penalty", "random_search"]
FAILED = ("error", "crashed", "timeout")


def first_pct(r):
    if r["status"] in FAILED or not r.get("found_feasible") or r.get("first_feasible_eval") is None:
        return 100.0, True
    return 100.0 * r["first_feasible_eval"] / max(1, r["budget"]), False


def frac_pct(r):
    if r["status"] in FAILED:
        return 0.0
    return 100.0 * float(r.get("feasible_fraction") or 0.0)


def collect(tags, baselines, baselines_file):
    """{method: {problem: {(run, seed): record}}}; LLaMEA methods are 'llamea_<arm>'."""
    data = {}
    for tag in tags:
        for line in open(os.path.join("results", "llamea_final", tag, "runs.jsonl")):
            r = json.loads(line)
            if r.get("baseline", "").startswith("llamea_"):
                data.setdefault(r["baseline"], {}).setdefault(r["problem"], {})[(tag, r["seed"])] = r
    if os.path.exists(baselines_file):
        for line in open(baselines_file):
            r = json.loads(line)
            if r["baseline"] in baselines:
                data.setdefault(r["baseline"], {}).setdefault(r["problem"], {})[("", r["seed"])] = r
    return data


def draw(header, rows, notes, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = next((f for f in ("Times New Roman", "Liberation Serif", "Nimbus Roman",
                                                    "STIXGeneral", "DejaVu Serif") if f in names), "serif")
    widths = [2.2] + [max(1.75, 0.095 * max(len(str(r[j][0] if isinstance(r[j], tuple) else r[j])) for r in rows))
                      for j in range(1, len(header))]
    W, rh, fs = sum(widths), 0.34, 11
    fig = plt.figure(figsize=(W * 0.8, 6), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(-0.1, W + 0.1); ax.axis("off")
    xs = np.cumsum([0] + widths)
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
    import textwrap
    lines = [l for t in notes for l in textwrap.wrap(t, width=max(40, int(W * 10.5)))]   # fit the table width
    for k, t in enumerate(lines):
        ax.text(0.1, y + 0.3 + 0.3 * k, t, ha="left", va="center", fontsize=9.5, style="italic")
    y_end = y + 0.3 + 0.3 * len(lines)
    ax.set_ylim(y_end, 0)
    fig.set_size_inches(W * 0.8, y_end * 0.8)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".png", dpi=300, facecolor="white")
    fig.savefig(out + ".pdf", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", nargs="*", default=["colshade", "lshade_deb", "lshade_eps_long"])
    ap.add_argument("--all-baselines", action="store_true")
    ap.add_argument("--tags-prefix", default="haiku5")
    ap.add_argument("--baselines-file", default="results/baselines_1000d/runs.jsonl")
    ap.add_argument("--outdir", default="results/figures")
    a = ap.parse_args()
    if a.all_baselines:
        a.baselines = ALL_BASELINES
    pat = re.compile(re.escape(a.tags_prefix) + r"(-r\d+)?")
    tags = sorted(os.path.basename(d) for d in glob.glob(os.path.join("results", "llamea_final", a.tags_prefix + "*"))
                  if pat.fullmatch(os.path.basename(d)) and os.path.exists(os.path.join(d, "runs.jsonl")))
    data = collect(tags, a.baselines, a.baselines_file)
    methods = [(f"llamea_{x}", n) for x, n in ARMS if f"llamea_{x}" in data] + \
              [(b, BASE[b]) for b in a.baselines if b in data]
    problems = [p for p in PROBLEMS if any(p in data[m] for m, _ in methods)]

    for kind, out_name, lower_better in (("first", "feasibility_first", True), ("fraction", "feasibility_fraction", False)):
        rows = []
        for pid in problems + ["mean"]:
            cells = []
            for m, _ in methods:
                runs = data[m]
                if pid != "mean":
                    recs = runs.get(pid, {})
                    if kind == "first":
                        vals = [first_pct(r) for r in recs.values()]
                        never = sum(n for _, n in vals)
                        v = [x for x, _ in vals]
                    else:
                        v, never = [frac_pct(r) for r in recs.values()], 0
                else:   # per test run (run, seed): mean over problems
                    keys = set.intersection(*[set(runs.get(p, {})) for p in problems]) if all(p in runs for p in problems) else set()
                    f = (lambda r: first_pct(r)[0]) if kind == "first" else frac_pct
                    v = [float(np.mean([f(runs[p][k]) for p in problems])) for k in sorted(keys)]
                    never = 0
                if not v:
                    cells.append(("–", None)); continue
                txt = f"{np.mean(v):.1f} ± {np.std(v):.1f}"
                if never:
                    txt += f" [{never}/{len(v)} never]"
                cells.append((txt, float(np.mean(v))))
            ms = [m for _, m in cells if m is not None]
            best = (min(ms) if lower_better else max(ms)) if ms else None
            rows.append(["Mean" if pid == "mean" else pid] + [(t, m is not None and abs(m - best) < 0.05) for t, m in cells])
        header = ["Problem"] + [n for _, n in methods]
        what = ("First feasible point, % of the evaluation budget (lower = faster); runs that never become feasible "
                "count as 100 %.") if kind == "first" else "Share of evaluated points that are feasible, % (higher = more time in the feasible region)."
        notes = [what, f"Mean ± std over test runs: LLaMEA arm = best algorithm of each of the {len(tags)} LLaMEA runs × "
                       f"15 seeds; baselines = 15 seeds. Mean row: per test run, mean over the problems. Bold: best of the row."]
        out = os.path.join(a.outdir, out_name)
        draw(header, rows, notes, out)
        print(f"{out}.png (+ .pdf)")
        for r in rows:
            print("  " + " | ".join(c if isinstance(c, str) else c[0] for c in r))


if __name__ == "__main__":
    main()