import numpy as np


class Algorithm:
    """
    (mu+1) steady-state EA for constrained mixed-integer problems.
    - Deb feasibility rules: feasible ranked by f, infeasible by total violation.
    - Operator credit with probability floor:
        0: self-adaptive Gaussian mutation with type-aware integer moves
        1: DE current-to-best with binomial crossover
        2: integer moves plus small continuous jitter
        3: archive/population uniform crossover followed by light mutation
        4: feasibility bisection between an infeasible parent and an archive design
    - Elite feasible archive seeds restarts, recombination and bisection.
    - On each new elite: Hooke-Jeeves pattern search with momentum extrapolation
      (doubling the net displacement after successive successful pattern moves),
      then an integer neighbourhood sweep (full if small, sampled otherwise).
    - Duplicate cache: identical repaired points are never re-evaluated.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        budget, d = self.budget, self.dim
        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        span = ub - lb
        vt = np.array([str(t) for t in func.variable_types])
        is_cont = vt == "continuous"
        is_ord = vt == "ordinal"
        is_cat = ~is_cont & ~is_ord
        ii = np.where(~is_cont)[0]
        ci = np.where(is_cont)[0]
        nv = np.ones(d)
        for i in ii:
            nv[i] = float(func.n_values[i])

        state = {"n": 0, "best_f": np.inf, "best_x": None, "best_key": (2, np.inf)}
        cache = {}
        archive = []
        ARCH_CAP = 12
        step_reset = 0.1 * span.copy()
        step = step_reset.copy()
        swept = set()

        def repair(x):
            x = np.clip(x, lb, ub)
            if len(ii):
                x[ii] = np.clip(np.rint(x[ii]), 0, nv[ii] - 1)
            return x

        def random_x():
            x = lb + span * np.random.rand(d)
            if len(ii):
                x[ii] = np.floor(np.random.rand(len(ii)) * nv[ii])
            return x

        def update_archive(f, x):
            kx = x.tobytes()
            for (_, xa) in archive:
                if xa.tobytes() == kx:
                    return
            archive.append((f, x.copy()))
            archive.sort(key=lambda t: t[0])
            del archive[ARCH_CAP:]

        def evaluate(x):
            x = repair(np.asarray(x, dtype=float))
            k = x.tobytes()
            if k in cache:
                return cache[k]
            if state["n"] >= budget:
                return None
            f, g = func(x)
            state["n"] += 1
            g = np.nan_to_num(np.asarray(g, dtype=float).ravel(),
                              nan=1e12, posinf=1e12, neginf=0.0)
            viol = float(np.sum(np.maximum(g, 0.0)))
            f = float(f)
            if not np.isfinite(f):
                f = np.inf
                viol = max(viol, 1e12)
            if viol > 0:
                key = (1, viol)
            else:
                key = (0, f)
                if key < state["best_key"]:
                    state["best_key"] = key
                    state["best_f"] = f
                    state["best_x"] = x.copy()
                if np.isfinite(f):
                    update_archive(f, x)
            ind = {"x": x, "key": key, "s": 0.2}
            cache[k] = ind
            return ind

        def int_move(x, i):
            if is_ord[i]:
                if np.random.rand() < 0.7:
                    x[i] = x[i] + (1.0 if np.random.rand() < 0.5 else -1.0)
                else:
                    x[i] = np.floor(np.random.rand() * nv[i])
            elif is_cat[i]:
                v = np.floor(np.random.rand() * (nv[i] - 1))
                x[i] = v + float(v >= x[i])
            return x

        def gauss_mut(x, s):
            mask = np.random.rand(d) < 1.0 / d
            if not mask.any():
                mask[np.random.randint(d)] = True
            cm = mask & is_cont
            if cm.any():
                x[cm] += s * span[cm] * np.random.randn(cm.sum())
            for i in np.where(mask & ~is_cont)[0]:
                int_move(x, i)
            return x

        def op_gauss(p, best, pool):
            s = p["s"] * np.exp(np.random.randn() / np.sqrt(d))
            s = float(np.clip(s, 1e-3, 0.5))
            return gauss_mut(p["x"].copy(), s), s

        def op_de(p, best, pool):
            a = pool[np.random.randint(len(pool))]
            b = pool[np.random.randint(len(pool))]
            F = np.random.uniform(0.4, 0.9)
            mutant = p["x"].copy()
            if len(ci):
                mutant[ci] = (p["x"][ci] + F * (best["x"][ci] - p["x"][ci])
                              + F * (a["x"][ci] - b["x"][ci]))
            if len(ii):
                use_best = np.random.rand(len(ii)) < 0.5
                mutant[ii] = np.where(use_best, best["x"][ii], a["x"][ii])
            cross = np.random.rand(d) < 0.9
            cross[np.random.randint(d)] = True
            x = np.where(cross, mutant, p["x"])
            return x, p["s"]

        def op_int(p, best, pool):
            x = p["x"].copy()
            if len(ii):
                k = 1 if np.random.rand() < 0.7 else 2
                for i in np.random.choice(ii, size=min(k, len(ii)), replace=False):
                    int_move(x, i)
            if len(ci) and np.random.rand() < 0.3:
                i = ci[np.random.randint(len(ci))]
                x[i] += 0.02 * span[i] * np.random.randn()
            return x, p["s"]

        def op_cross(p, best, pool):
            if archive and np.random.rand() < 0.3:
                q = {"x": archive[np.random.randint(len(archive))][1]}
            else:
                q = pool[np.random.randint(len(pool))]
            m = np.random.rand(d) < 0.5
            x = np.where(m, p["x"], q["x"])
            s = float(np.clip(p["s"], 1e-3, 0.5))
            return gauss_mut(x, 0.5 * s), s

        def op_bisect(p, best, pool):
            # Bisection between a (possibly infeasible) parent and a feasible archive design;
            # the Beta(0.6,0.6) draw concentrates the step near the two endpoints,
            # so the child tends to land close to the feasibility boundary.
            if not archive:
                return op_gauss(p, best, pool)
            _, xf = archive[np.random.randint(len(archive))]
            t = np.random.beta(0.6, 0.6)
            x = p["x"] + t * (xf - p["x"])
            if len(ci):
                x[ci] += 0.01 * span[ci] * np.random.randn(len(ci))
            return x, p["s"]

        ops = [op_gauss, op_de, op_int, op_cross, op_bisect]
        n_ops = len(ops)
        score = np.ones(n_ops) * 0.5
        floor = 0.08

        def pick_op():
            p = score + floor
            p = p / p.sum()
            return np.random.choice(n_ops, p=p)

        def best_of(pop):
            return min(pop, key=lambda p: p["key"])

        def insert(pop, ind):
            if ind is None:
                return
            worst = max(range(len(pop)), key=lambda i: pop[i]["key"])
            if ind["key"] <= pop[worst]["key"] and all(
                    ind["x"].tobytes() != q["x"].tobytes() for q in pop):
                pop[worst] = ind

        def pattern_search(elite, cap):
            # Hooke-Jeeves with momentum: after consecutive successful pattern
            # moves the net displacement is extrapolated by a growing factor.
            if len(ci) == 0:
                return elite
            cur = elite
            used = 0
            mom = 1.0
            while used < cap and state["n"] < budget:
                start = cur
                improved_any = False
                for i in np.random.permutation(ci):
                    if used >= cap or state["n"] >= budget:
                        break
                    for sgn in (1.0, -1.0):
                        if used >= cap:
                            break
                        y = cur["x"].copy()
                        y[i] += sgn * step[i]
                        ind = evaluate(y)
                        used += 1
                        if ind is None:
                            return cur
                        if ind["key"] < cur["key"]:
                            cur = ind
                            improved_any = True
                            step[i] = min(step[i] * 1.5, span[i])
                            break
                if improved_any:
                    if used < cap and state["n"] < budget:
                        disp = np.zeros(d)
                        disp[ci] = cur["x"][ci] - start["x"][ci]
                        if np.any(disp != 0):
                            z = repair(cur["x"] + mom * disp)
                            ez = evaluate(z)
                            used += 1
                            if ez is None:
                                return cur
                            if ez["key"] < cur["key"]:
                                cur = ez
                                mom = min(mom * 2.0, 8.0)
                            else:
                                mom = 1.0
                else:
                    mom = 1.0
                    for i in ci:
                        step[i] = max(step[i] * 0.5, 1e-6 * span[i])
                    if np.max(step[ci] / span[ci]) < 1e-4:
                        break
            return cur

        def int_sweep(elite, cap):
            if len(ii) == 0:
                return None
            x0 = elite["x"]
            best_ind = elite
            used = 0
            for i in np.random.permutation(ii):
                cands = [v for v in range(int(nv[i])) if v != int(x0[i])]
                if len(cands) > 6 and int(nv[i]) > 12:
                    cands = list(np.random.choice(cands, size=6, replace=False))
                for v in cands:
                    if used >= cap or state["n"] >= budget:
                        return best_ind
                    y = best_ind["x"].copy()
                    y[i] = v
                    ind = evaluate(y)
                    used += 1
                    if ind is None:
                        return best_ind
                    if ind["key"] < best_ind["key"]:
                        best_ind = ind
            return best_ind

        def local_search(pop, elite):
            cand = pattern_search(elite, cap=3 * len(ci) + 6)
            if cand is not None:
                insert(pop, cand)
                elite = best_of(pop)
            if elite["key"][0] == 0:
                sig = elite["x"].tobytes()
                if sig not in swept and state["n"] < budget:
                    swept.add(sig)
                    cand = int_sweep(elite, cap=4 * len(ii) + 6)
                    if cand is not None:
                        insert(pop, cand)
            return best_of(pop)

        mu = max(8, min(40, budget // 12))
        n_init = min(max(mu, 2 * d), max(1, budget // 4))
        stall_limit = 12 * d + 25

        pop = []
        while len(pop) < n_init and state["n"] < budget:
            ind = evaluate(random_x())
            if ind is None:
                break
            pop.append(ind)
        if not pop:
            return state["best_f"], state["best_x"]

        best_pop = best_of(pop)["key"]
        last_ls = None
        stall = 0

        def restart(elite):
            base = elite
            if archive and np.random.rand() < 0.5:
                _, xa = archive[np.random.randint(len(archive))]
                base_ind = evaluate(xa)
                if base_ind is not None:
                    base = base_ind
            newpop = [elite]
            k = 0
            while len(newpop) < n_init and state["n"] < budget:
                if k % 3 != 2:
                    s = float(np.random.choice([0.02, 0.1, 0.3]))
                    ind = evaluate(gauss_mut(base["x"].copy(), s))
                else:
                    ind = evaluate(random_x())
                if ind is None:
                    break
                newpop.append(ind)
                k += 1
            return newpop

        while state["n"] < budget:
            if np.random.rand() < 0.5:
                a, b = np.random.randint(len(pop), size=2)
                p1 = pop[a] if pop[a]["key"] <= pop[b]["key"] else pop[b]
            else:
                p1 = pop[np.random.randint(len(pop))]
            best = best_of(pop)
            o = pick_op()
            x, s = ops[o](p1, best, pop)
            child = evaluate(x)
            if child is None:
                break
            child["s"] = s
            stall += 1
            reward = 1.0 if child["key"] < p1["key"] else (
                0.5 if child["key"] < best["key"] else 0.0)
            score[o] = 0.9 * score[o] + 0.1 * reward
            if child["key"] < best_pop:
                best_pop = child["key"]
                stall = 0
            insert(pop, child)

            if best_pop != last_ls and state["n"] < budget:
                elite = local_search(pop, best_of(pop))
                last_ls = elite["key"]
                best_pop = min(best_pop, elite["key"])

            if stall > stall_limit and state["n"] < budget:
                elite = best_of(pop)
                pop = restart(elite)
                best_pop = best_of(pop)["key"]
                step[:] = step_reset
                last_ls = None
                stall = 0

        return state["best_f"], state["best_x"]