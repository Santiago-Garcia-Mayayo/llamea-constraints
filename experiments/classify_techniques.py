"""Classify the constraint-handling technique of every algorithm LLaMEA generated (RQ3).

    python -m experiments.classify_techniques run --dry-run         # count candidates, estimate cost
    python -m experiments.classify_techniques run --workers 8       # classify (resumable)
    python -m experiments.classify_techniques summary               # tables -> results/techniques/summary.md
    python -m experiments.classify_techniques sample --n 30         # sheet for a manual check
    python -m experiments.classify_techniques agreement             # LLM vs manual labels (Cohen's kappa)

Which algorithms: every candidate (about 100 per run) of the arms that see the constraints
(free, informed_v2) in every repetition haiku5*, read from LLaMEA's own log
(results/llamea/<tag>/<arm>/<problem>/exp-*/log.jsonl), with its generation, parents and
training fitness. The penalty arm is left out: it only sees f + rho * violation, so its
technique is a penalty by construction.

How: one call per algorithm to an Anthropic model (default Claude Haiku 5.5, the model that
wrote the algorithms), forced to answer through a tool with a closed taxonomy, so the output
is always valid JSON with labels from a fixed list, the primary technique, the optimiser family,
and the code lines that justify the labels. The key comes from ANTHROPIC_API_KEY.
Writes results/techniques/labels.jsonl (one line per algorithm; resumable: algorithms already
labelled are skipped).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT = "results/techniques"
ARMS = ("free", "informed_v2")

TECHNIQUES = {
    "penalty_static": "adds the violation to the objective with a fixed weight",
    "penalty_adaptive": "adds the violation to the objective with a weight that changes during the run",
    "feasibility_rules": "Deb's rules: feasible beats infeasible, lower violation beats higher, then objective",
    "epsilon_constrained": "violations below a tolerance epsilon count as feasible; epsilon shrinks over time",
    "stochastic_ranking": "ranks by objective or violation chosen at random (Runarsson and Yao)",
    "multiobjective": "treats violation (or each constraint) as extra objectives, Pareto ranking",
    "repair": "moves infeasible points towards feasibility (projection, bisection or line search towards a feasible point)",
    "feasible_archive": "keeps a separate archive or pool of feasible solutions that guides the search",
    "per_constraint_guidance": "uses the individual constraint values (e.g. the most violated one) to steer moves or weights",
    "boundary_search": "deliberately searches along the boundary between feasible and infeasible regions",
    "constraint_surrogate": "builds a model of the constraints to predict feasibility",
    "ignores_constraints": "does not use g at all (treats the problem as unconstrained)",
    "other": "a constraint-handling idea not covered above (explain in the rationale)",
}
FAMILIES = ("differential_evolution", "evolution_strategy_cmaes", "genetic_algorithm", "particle_swarm",
            "local_search_pattern", "bayesian_surrogate", "hybrid", "random_search", "other")

SYSTEM = (
    "You are an expert in constrained evolutionary optimisation. You read Python optimisation "
    "algorithms and classify how they handle constraints. The algorithm calls func(x) and gets "
    "(f, g): f is the objective to minimise, g is an array with one violation per constraint "
    "(0 means that constraint is satisfied). Classify only what the code actually does, not what "
    "its name, docstring or comments claim.")

TOOL = {
    "name": "classify",
    "description": "Report the constraint-handling techniques used by the algorithm.",
    "input_schema": {
        "type": "object",
        "properties": {
            "techniques": {"type": "array", "items": {"type": "string", "enum": list(TECHNIQUES)},
                           "description": "every constraint-handling technique the code uses"},
            "primary": {"type": "string", "enum": list(TECHNIQUES),
                        "description": "the technique that decides which solutions are selected or kept"},
            "uses_individual_constraints": {"type": "boolean",
                                            "description": "true if it uses the per-constraint values of g, not only their sum or max"},
            "family": {"type": "string", "enum": list(FAMILIES), "description": "the optimiser family"},
            "evidence": {"type": "string", "description": "the few code lines (verbatim) that show the primary technique"},
            "rationale": {"type": "string", "description": "one or two sentences"},
        },
        "required": ["techniques", "primary", "uses_individual_constraints", "family", "evidence", "rationale"],
    },
}


def _prompt(code: str) -> str:
    tax = "\n".join(f"- {k}: {v}" for k, v in TECHNIQUES.items())
    return (f"Techniques to choose from:\n{tax}\n\nOptimiser families: {', '.join(FAMILIES)}.\n\n"
            f"Algorithm:\n```python\n{code[:40000]}\n```\n\nCall the classify tool.")


def candidates(runs="results/llamea", arms=ARMS):
    """Every algorithm in LLaMEA's logs of the haiku5* repetitions, in evaluation order."""
    tags = sorted(os.path.basename(d) for d in glob.glob(os.path.join(runs, "haiku5*"))
                  if re.fullmatch(r"haiku5(-r\d+)?", os.path.basename(d)))
    out = []
    for tag in tags:
        for arm in arms:
            for rd in sorted(glob.glob(os.path.join(runs, tag, arm, "*"))):
                logs = sorted(glob.glob(os.path.join(rd, "exp-*", "log.jsonl")))
                if not logs:
                    continue
                best = None
                if os.path.exists(os.path.join(rd, "best.json")):
                    best = json.load(open(os.path.join(rd, "best.json"))).get("code")
                for k, line in enumerate(open(logs[0])):
                    try:
                        c = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not c.get("code"):
                        continue
                    fit = c.get("fitness")
                    try:
                        fit = float(fit)
                        fit = fit if math.isfinite(fit) else None
                    except (TypeError, ValueError):
                        fit = None
                    out.append({"tag": tag, "arm": arm, "problem": os.path.basename(rd).replace("_", "/", 1),
                                "id": c.get("id"), "order": k, "generation": c.get("generation"),
                                "parent_ids": c.get("parent_ids") or [], "name": c.get("name"),
                                "fitness": fit, "is_best": c.get("code") == best, "code": c["code"]})
    return out


def _classify(client, model, code, retries=6):
    for attempt in range(retries):
        try:
            r = client.messages.create(model=model, max_tokens=800, system=SYSTEM, tools=[TOOL],
                                       tool_choice={"type": "tool", "name": "classify"},
                                       messages=[{"role": "user", "content": _prompt(code)}])
            block = next(b for b in r.content if b.type == "tool_use")
            return dict(block.input), r.usage.input_tokens, r.usage.output_tokens
        except Exception as e:  # rate limits, overload, network: back off and retry
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** attempt * 2) + 0.5 * attempt)


def run(a):
    path = os.path.join(a.out, "labels.jsonl")
    os.makedirs(a.out, exist_ok=True)
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done.add((r["tag"], r["arm"], r["problem"], r["id"]))
    todo = [c for c in candidates(a.runs, a.arms) if (c["tag"], c["arm"], c["problem"], c["id"]) not in done]
    chars = sum(min(len(c["code"]), 40000) for c in todo)
    est_in = chars / 3.5 + len(todo) * 900          # code tokens + prompt, taxonomy and tool
    est_out = len(todo) * 350
    price = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0)}.get(a.model, (None, None))
    cost = f"~${est_in / 1e6 * price[0] + est_out / 1e6 * price[1]:.2f}" if price[0] else "unknown price"
    print(f"{len(todo)} algorithms to classify ({len(done)} already done); ~{est_in / 1e6:.1f}M input and "
          f"~{est_out / 1e6:.1f}M output tokens with {a.model}: {cost}", flush=True)
    if a.dry_run or not todo:
        return
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=4)
    lock, tin, tout, t0 = threading.Lock(), 0, 0, time.time()
    with open(path, "a") as fh, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_classify, client, a.model, c["code"]): c for c in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            rec = {kk: v for kk, v in c.items() if kk != "code"}
            try:
                lab, i, o = fut.result()
                rec.update(lab, model=a.model)
                tin += i; tout += o
            except Exception as e:
                print(f"failed {c['tag']} {c['arm']} {c['problem']} {c['id']}: {e}", flush=True)
                continue
            with lock:
                fh.write(json.dumps(rec) + "\n"); fh.flush()
            if k % 100 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}] {time.time() - t0:.0f} s, {tin / 1e6:.2f}M in / {tout / 1e6:.2f}M out tokens",
                      flush=True)


def _load(out):
    path = os.path.join(out, "labels.jsonl")
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def _pct_table(rows, key, groups, title, multi=False):
    L = [f"### {title}", ""]
    cats = [t for t in TECHNIQUES] if key in ("techniques", "primary") else list(FAMILIES)
    used = [t for t in cats if any((t in r[key]) if multi else (r[key] == t) for r in rows)]
    L += [f"| {'family' if key == 'family' else 'technique'} | " + " | ".join(f"{g} (n={sum(1 for r in rows if gf(r))})" for g, gf in groups) + " |",
          "|---|" + "---|" * len(groups)]
    for t in used:
        cells = []
        for _, gf in groups:
            sub = [r for r in rows if gf(r)]
            n = sum(1 for r in sub if ((t in r[key]) if multi else (r[key] == t)))
            cells.append(f"{100 * n / len(sub):.0f}%" if sub else "—")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    return L + [""]


def summary(a):
    rows = _load(a.out)
    if not rows:
        print("no labels yet"); return
    by_arm = [(arm, (lambda r, arm=arm: r["arm"] == arm)) for arm in a.arms]
    best_by_arm = [(f"{arm}, best", (lambda r, arm=arm: r["arm"] == arm and r.get("is_best"))) for arm in a.arms]
    L = ["# Constraint-handling techniques of the generated algorithms", "",
         f"{len(rows)} algorithms classified ({', '.join(sorted({r['model'] for r in rows}))}). "
         "Share of algorithms; 'best' = the best algorithm of each run.", ""]
    L += _pct_table(rows, "primary", by_arm + best_by_arm, "Primary technique (what decides selection)")
    L += _pct_table(rows, "techniques", by_arm + best_by_arm, "Any use of each technique (multi-label)", multi=True)
    L += _pct_table(rows, "family", by_arm, "Optimiser family")
    ind = {arm: [r["uses_individual_constraints"] for r in rows if r["arm"] == arm] for arm in a.arms}
    L += ["Uses the individual constraint values (not only their sum): " +
          ", ".join(f"{arm} {100 * sum(v) / len(v):.0f}%" for arm, v in ind.items() if v), ""]

    # per problem: primary technique of all candidates
    probs = sorted({r["problem"] for r in rows}, key=lambda p: (not p.startswith("lens"), p))
    L += ["### Primary technique per problem (all candidates of both arms)", "",
          "| problem | n | most common | 2nd | 3rd |", "|---|---|---|---|---|"]
    for p in probs:
        c = Counter(r["primary"] for r in rows if r["problem"] == p)
        n = sum(c.values())
        top = [f"{t} {100 * k / n:.0f}%" for t, k in c.most_common(3)] + ["—"] * 3
        L.append(f"| {p} | {n} | " + " | ".join(top[:3]) + " |")
    L.append("")

    # technique vs training fitness, per problem (does a technique go with better algorithms?)
    L += ["### Mean training fitness by primary technique (techniques with >= 10 algorithms in the problem)", "",
          "| problem | technique | n | mean fitness | best fitness |", "|---|---|---|---|---|"]
    for p in probs:
        g = defaultdict(list)
        for r in rows:
            if r["problem"] == p and r["fitness"] is not None:
                g[r["primary"]].append(r["fitness"])
        for t, v in sorted(g.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            if len(v) >= 10:
                L.append(f"| {p} | {t} | {len(v)} | {sum(v) / len(v):.3f} | {max(v):.3f} |")
    L.append("")

    # jumps: child vs parent primary technique
    key = {(r["tag"], r["arm"], r["problem"], r["id"]): r for r in rows}
    trans, gain = Counter(), defaultdict(list)
    for r in rows:
        for pid in r["parent_ids"][:1]:
            par = key.get((r["tag"], r["arm"], r["problem"], pid))
            if par is None:
                continue
            t = (par["primary"], r["primary"])
            trans[t] += 1
            if r["fitness"] is not None and par["fitness"] is not None:
                gain[t].append(r["fitness"] - par["fitness"])
    n_tr = sum(trans.values())
    switched = sum(k for (x, y), k in trans.items() if x != y)
    L += ["### Jumps between techniques (child vs its parent)", "",
          f"{n_tr} parent-child pairs; the child changes primary technique in {switched} "
          f"({100 * switched / max(1, n_tr):.0f}%).", "",
          "| parent -> child | n | mean fitness change | share improving |", "|---|---|---|---|"]
    for (x, y), k in trans.most_common(20):
        v = gain[(x, y)]
        imp = f"{100 * sum(d > 0 for d in v) / len(v):.0f}%" if v else "—"
        mean = f"{sum(v) / len(v):+.3f}" if v else "—"
        L.append(f"| {x} -> {y} | {k} | {mean} | {imp} |")
    L.append("")

    # how the mix changes during the search: first vs second half of each run
    L += ["### Primary technique early vs late in the runs (first vs last 50 algorithms of each run)", ""]
    early = [r for r in rows if r["order"] < 50]
    late = [r for r in rows if r["order"] >= 50]
    groups = [(f"{arm} early", (lambda r, arm=arm: r["arm"] == arm and r["order"] < 50)) for arm in a.arms] + \
             [(f"{arm} late", (lambda r, arm=arm: r["arm"] == arm and r["order"] >= 50)) for arm in a.arms]
    L += _pct_table(early + late, "primary", groups, "Share of algorithms")

    md = "\n".join(L) + "\n"
    open(os.path.join(a.out, "summary.md"), "w").write(md)
    print(md)


def sample(a):
    """A sheet for a manual check: a stratified random sample of algorithms, with their code in
    separate files and empty columns for your own labels (the LLM's labels are not shown)."""
    import random
    rows = _load(a.out)
    code = {(c["tag"], c["arm"], c["problem"], c["id"]): c["code"] for c in candidates(a.runs, a.arms)}
    rnd = random.Random(a.seed)
    by = defaultdict(list)
    for r in rows:
        by[r["primary"]].append(r)
    pick = []
    while len(pick) < min(a.n, len(rows)):     # round-robin over techniques so rare ones appear too
        for t in sorted(by):
            if by[t] and len(pick) < a.n:
                pick.append(by[t].pop(rnd.randrange(len(by[t]))))
    d = os.path.join(a.out, "manual")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(a.out, "manual_sample.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "file", "manual_primary", "manual_techniques"])
        for i, r in enumerate(pick):
            k = "|".join([r["tag"], r["arm"], r["problem"], r["id"]])
            f = os.path.join(d, f"{i:02d}_{r['name']}.py")
            open(f, "w").write(code.get((r["tag"], r["arm"], r["problem"], r["id"]), "# code not found\n"))
            w.writerow([k, f, "", ""])
    print(f"{len(pick)} algorithms -> {a.out}/manual_sample.csv (code in {d}/). Fill manual_primary with one of: "
          + ", ".join(TECHNIQUES) + "; manual_techniques with several, separated by ';'.")


def agreement(a):
    rows = {"|".join([r["tag"], r["arm"], r["problem"], r["id"]]): r for r in _load(a.out)}
    man = [m for m in csv.DictReader(open(os.path.join(a.out, "manual_sample.csv"))) if m["manual_primary"].strip()]
    pairs = [(m["manual_primary"].strip(), rows[m["key"]]["primary"]) for m in man if m["key"] in rows]
    if not pairs:
        print("no manual labels yet"); return
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    cx, cy = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(cx[t] * cy[t] for t in set(cx) | set(cy)) / n ** 2
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"{n} algorithms: agreement {100 * po:.0f}%, Cohen's kappa {kappa:.2f}")
    jac = []
    for m in man:
        if m["key"] in rows and m["manual_techniques"].strip():
            A = {t.strip() for t in m["manual_techniques"].split(";") if t.strip()}
            B = set(rows[m["key"]]["techniques"])
            jac.append(len(A & B) / len(A | B) if A | B else 1.0)
    if jac:
        print(f"multi-label overlap (Jaccard), mean over {len(jac)}: {sum(jac) / len(jac):.2f}")
    for x, y in pairs:
        if x != y:
            print(f"  manual {x:26s} LLM {y}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("run", "summary", "sample", "agreement"))
    ap.add_argument("--runs", default="results/llamea")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    {"run": run, "summary": summary, "sample": sample, "agreement": agreement}[a.action](a)


if __name__ == "__main__":
    main()