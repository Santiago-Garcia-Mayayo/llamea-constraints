"""LLaMEA runs: one run per (problem, arm), each evolving on its own problem only.

    python -m experiments.run_llamea --problems cec/RC15 --arms free            # one run
    python -m experiments.run_llamea --parallel 4                                # all 30 runs
    python -m experiments.run_llamea --model claude-haiku-5-5 --budget 10 --problems cec/RC15 --arms free --tag pilot
    python -m experiments.run_llamea --provider ollama --model qwen2.5-coder:32b --ollama-host http://localhost:11434 \
        --budget 10 --problems cec/RC15 --arms free --tag pilot-ollama

Every run lives in <out>/<tag>/<arm>/<problem>/ :
    best.json        best algorithm (code, name, training fitness) -- written at the end
    candidates/      one JSON per evaluated candidate (code, fitness, per-seed results, feedback)
    LLaMEA-*/        LLaMEA's own log (conversation, population per generation)
    run.log          stdout/stderr of the run
Runs that already have best.json are skipped (resumable). Each run is its own subprocess.
Needs ANTHROPIC_API_KEY in the environment (provider anthropic), or a reachable Ollama server
(provider ollama; see harness/llms.py).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from harness import config
from harness.core import ARMS
from harness.llms import PROVIDERS


def run_dir(out, tag, arm, pid):
    return os.path.join(out, tag, arm, pid.replace("/", "_"))


def single(pid, arm, out, tag, model, budget, n_parents, n_offspring, elitism, temperature,
           max_workers=None, provider="anthropic", ollama_host=None, num_ctx=32768, llm_timeout=0):
    """Run one LLaMEA in this process (called in a subprocess by `main`)."""
    from harness.llms import make_llm
    from harness.llamea_adapter import make_llamea
    d = os.path.abspath(run_dir(out, tag, arm, pid))
    os.makedirs(d, exist_ok=True)
    os.chdir(d)                                   # LLaMEA writes its log to the working directory
    llm = make_llm(provider, model, temperature=temperature, ollama_host=ollama_host, num_ctx=num_ctx)
    es = make_llamea(arm, llm, pid, budget=budget, n_parents=n_parents, n_offspring=n_offspring,
                     elitism=elitism, experiment_name=f"{arm}-{pid.replace('/', '_')}",
                     log_dir=os.path.join(d, "candidates"), max_workers=max_workers, llm_timeout=llm_timeout)
    t0 = time.time()
    best = es.run()
    rec = {"problem": pid, "arm": arm, "provider": provider, "model": model, "llm_timeout": llm_timeout, "budget": budget, "n_parents": n_parents,
           "n_offspring": n_offspring, "elitism": elitism, "temperature": temperature,
           "name": best.name, "fitness": best.fitness, "code": best.code, "description": best.description,
                      "wall_time_s": time.time() - t0}
    with open(os.path.join(d, "best.json"), "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print(f"done {arm} {pid}: fitness {best.fitness:.4f}", flush=True)
    

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", nargs="*", default=list(config.PROBLEMS))
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=ARMS)
    ap.add_argument("--provider", default="anthropic", choices=PROVIDERS)
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--ollama-host", default=None, help="Ollama server (default: OLLAMA_HOST or localhost:11434)")
    ap.add_argument("--num-ctx", type=int, default=32768, help="Ollama context window (tokens)")
    ap.add_argument("--llm-timeout", type=int, default=None,
                    help="extra seconds LLaMEA waits for each LLM answer (default 0 for anthropic, 1800 for ollama)")
    ap.add_argument("--temperature", type=float, default=config.LLM_TEMPERATURE)
    ap.add_argument("--budget", type=int, default=config.LLAMEA_BUDGET, help="candidates per run")
    ap.add_argument("--n-parents", type=int, default=config.LLAMEA_N_PARENTS)
    ap.add_argument("--n-offspring", type=int, default=config.LLAMEA_N_OFFSPRING)
    ap.add_argument("--no-elitism", action="store_true")
    ap.add_argument("--out", default="results/llamea")
    ap.add_argument("--tag", default="main", help="sub-folder, e.g. main / pilot / ablation-mu10")
    ap.add_argument("--parallel", type=int, default=2,
                    help="LLaMEA runs at the same time; each uses ~n_offspring x 3 processes")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.llm_timeout is None:
        a.llm_timeout = 1800 if a.provider == "ollama" else 0
    if a.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        ap.error("ANTHROPIC_API_KEY is not set")
    if a.single:
        single(a.problems[0], a.arms[0], a.out, a.tag, a.model, a.budget, a.n_parents, a.n_offspring,
               not a.no_elitism, a.temperature, provider=a.provider, ollama_host=a.ollama_host,
               num_ctx=a.num_ctx, llm_timeout=a.llm_timeout)
        return
    todo = [(pid, arm) for pid in a.problems for arm in a.arms
            if not os.path.exists(os.path.join(run_dir(a.out, a.tag, arm, pid), "best.json"))]
    print(f"{len(todo)} LLaMEA runs to do, {a.parallel} at a time -> {os.path.join(a.out, a.tag)}", flush=True)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def launch(pid, arm):
        d = run_dir(a.out, a.tag, arm, pid)
        os.makedirs(d, exist_ok=True)
        cmd = [sys.executable, "-m", "experiments.run_llamea", "--single", "--problems", pid, "--arms", arm,
               "--model", a.model, "--budget", str(a.budget), "--n-parents", str(a.n_parents),
               "--n-offspring", str(a.n_offspring), "--out", os.path.abspath(a.out), "--tag", a.tag,
               "--provider", a.provider, "--num-ctx", str(a.num_ctx), "--llm-timeout", str(a.llm_timeout)]
        if a.ollama_host:
            cmd += ["--ollama-host", a.ollama_host]
        if a.temperature is not None:
            cmd += ["--temperature", str(a.temperature)]
        if a.no_elitism:
            cmd.append("--no-elitism")
        env = dict(os.environ, PYTHONPATH=here + os.pathsep + os.environ.get("PYTHONPATH", ""))
        with open(os.path.join(d, "run.log"), "a") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=here)
        return pid, arm, rc

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        futs = [ex.submit(launch, pid, arm) for pid, arm in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            pid, arm, rc = fut.result()
            status = "ok" if rc == 0 else f"FAILED (exit {rc}, see run.log)"
            print(f"[{k}/{len(todo)}] {arm} {pid}: {status} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()