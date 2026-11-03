import numpy as np


class Algorithm:
    """
    Feasibility-first mixed-integer (mu+lambda) ES with:
    - Cached evaluator (repeated points cost no budget) plus a guard against
      generations that consume no budget.
    - Latin-hypercube initial design.
    - UCB-biased credit selection over: local 1/5-rule moves on the incumbent,
      violation-descent moves on the least-infeasible parent, multi-anchor
      feasibility bisection, DE/best/1, crossover+mutation, archive crossover.
    - Deb's-rule survivor selection.
    - Staged stagnation handling: pattern search, full discrete sweep,
      then elite-core restarts from the feasible archive.
    """

    OPS = ["ls", "vdesc", "bis", "de", "ga", "arch"]

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lb = np.asarray(func.bounds.lb, dtype=float).copy()
        ub = np.asarray(func.bounds.ub, dtype=float).copy()
        vtypes = list(func.variable_types)

        nvals = np.zeros(dim, dtype=int)
        for i in range(dim):
            if vtypes[i] != "continuous":
                nvals[i] = int(func.n_values[i])
        cont = np.array([i for i in range(dim) if vtypes[i] == "continuous"], dtype=int)
        disc = np.array([i for i in range(dim) if vtypes[i] != "continuous"], dtype=int)
        is_ord = np.array([vtypes[i] == "ordinal" for i in range(dim)], dtype=bool)
        span = ub - lb

        used = [0]
        cache = {}
        best = {"f": np.inf, "x": None, "rec": None}
        archive = []
        ARCH_MAX = 50

        def repair(x):
            x = np.clip(x, lb, ub)
            if disc.size:
                x[disc] = np.clip(np.rint(x[disc]), 0, nvals[disc] - 1)
            return x

        def add_archive(rec):
            if rec["v"] != 0.0 or not np.isfinite(rec["f"]):
                return
            for a in archive:
                if np.array_equal(a["x"], rec["x"]):
                    return
            archive.append(dict(rec))
            archive.sort(key=lambda r: r["f"])
            del archive[ARCH_MAX:]

        def evaluate(x, s):
            x = repair(np.array(x, dtype=float))
            k = x.tobytes()
            if k in cache:
                c = cache[k]
                return {"x": c["x"], "f": c["f"], "v": c["v"], "s": s}
            if used[0] >= budget:
                return None
            f, g = func(x)
            used[0] += 1
            g = np.atleast_1d(np.asarray(g, dtype=float))
            g = np.nan_to_num(g, nan=1e6, posinf=1e6, neginf=0.0)
            v = float(np.sum(np.maximum(g, 0.0)))
            f = float(f)
            if not np.isfinite(f):
                f = np.inf
                if v == 0.0:
                    v = 1e6
            rec = {"x": x.copy(), "f": f, "v": v, "s": s}
            cache[k] = rec
            if v == 0.0 and f < best["f"]:
                best["f"] = f
                best["x"] = x.copy()
                best["rec"] = dict(rec)
            add_archive(rec)
            return {"x": rec["x"], "f": f, "v": v, "s": s}

        def key(ind):
            if ind["v"] == 0.0:
                return (0, ind["f"])
            return (1, ind["v"])

        def select(pool, mu):
            seen = set()
            uniq = []
            for p in sorted(pool, key=key):
                k = p["x"].tobytes()
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(p)
            return uniq[:mu]

        def tournament(pop):
            a = pop[np.random.randint(len(pop))]
            b = pop[np.random.randint(len(pop))]
            return a if key(a) <= key(b) else b

        def lhs_points(n):
            pts = np.empty((n, dim), dtype=float)
            for i in cont:
                strata = (np.random.permutation(n) + np.random.rand(n)) / n
                pts[:, i] = lb[i] + strata * span[i]
            for i in disc:
                base = np.arange(n) % nvals[i]
                pts[:, i] = np.random.permutation(base)
            return pts

        def random_point():
            x = np.empty(dim, dtype=float)
            if cont.size:
                x[cont] = np.random.uniform(lb[cont], ub[cont])
            if disc.size:
                x[disc] = np.random.randint(0, nvals[disc])
            return x

        def mutate_discrete_gene(y, i):
            n = nvals[i]
            if n <= 1:
                return
            if is_ord[i] and np.random.rand() < 0.7:
                step = 1 if np.random.rand() < 0.5 else -1
                y[i] = np.clip(y[i] + step, 0, n - 1)
            else:
                choices = [v for v in range(n) if v != int(y[i])]
                y[i] = choices[np.random.randint(len(choices))]

        def mutate(x, s):
            y = x.copy()
            s_new = float(np.clip(s * np.exp(0.3 * np.random.randn()), 1e-4, 0.5))
            if cont.size:
                y[cont] = y[cont] + s_new * span[cont] * np.random.randn(cont.size)
            if disc.size:
                pd = min(1.0, 1.5 / dim)
                for i in disc:
                    if np.random.rand() < pd:
                        mutate_discrete_gene(y, i)
            return y, s_new

        def mutate_multi(x, sig):
            y = x.copy()
            m = 1 if np.random.rand() < 0.6 else (2 if np.random.rand() < 0.7 else 3)
            idx = np.random.choice(dim, size=min(m, dim), replace=False)
            for j in idx:
                if vtypes[j] == "continuous":
                    y[j] = y[j] + sig * span[j] * np.random.randn()
                else:
                    mutate_discrete_gene(y, j)
            return y

        def de_move(base_x, a_x, b_x, F):
            y = base_x.copy()
            if cont.size:
                y[cont] = base_x[cont] + F * (a_x[cont] - b_x[cont])
            if disc.size:
                take = np.random.rand(disc.size) < 0.5
                y[disc] = np.where(take, a_x[disc], base_x[disc])
                for i in disc:
                    if np.random.rand() < 0.5 / max(1, disc.size):
                        mutate_discrete_gene(y, i)
            return y

        def violation_descent(rec, n_moves):
            # Greedy single-coordinate moves on an infeasible parent that
            # reduce total violation; returns best candidate found.
            cur = {"x": rec["x"].copy(), "f": rec["f"], "v": rec["v"], "s": rec["s"]}
            spent = 0
            order = list(range(dim))
            np.random.shuffle(order)
            for i in order:
                if spent >= n_moves or used[0] >= budget:
                    break
                if vtypes[i] == "continuous":
                    y = cur["x"].copy()
                    y[i] = y[i] + 0.05 * span[i] * np.random.randn()
                    cands = [y]
                else:
                    cands = []
                    levels = [v for v in range(nvals[i]) if v != int(cur["x"][i])]
                    np.random.shuffle(levels)
                    for v in levels[:3]:
                        y = cur["x"].copy()
                        y[i] = v
                        cands.append(y)
                for y in cands:
                    if spent >= n_moves or used[0] >= budget:
                        break
                    cand = evaluate(y, cur["s"])
                    spent += 1
                    if cand is None:
                        return cur
                    if key(cand) < key(cur):
                        cur = cand
                        break
            return cur

        def pattern_search(start, cap, full_discrete=False):
            cur = {"x": start["x"].copy(), "f": start["f"], "v": start["v"], "s": start["s"]}
            steps = {int(i): 0.1 * span[i] for i in cont}
            spent = 0
            while spent < cap and used[0] < budget:
                improved = False
                order = list(range(dim))
                np.random.shuffle(order)
                for i in order:
                    if spent >= cap or used[0] >= budget:
                        break
                    if vtypes[i] == "continuous":
                        for d in (1.0, -1.0):
                            y = cur["x"].copy()
                            y[i] = y[i] + d * steps[int(i)]
                            cand = evaluate(y, cur["s"])
                            spent += 1
                            if cand is None:
                                return cur
                            if key(cand) < key(cur):
                                cur = cand
                                improved = True
                                break
                    else:
                        n = nvals[i]
                        levels = [v for v in range(n) if v != int(cur["x"][i])]
                        if not full_discrete and len(levels) > 6:
                            levels = list(np.random.choice(levels, 6, replace=False))
                        else:
                            np.random.shuffle(levels)
                        for v in levels:
                            if spent >= cap or used[0] >= budget:
                                break
                            y = cur["x"].copy()
                            y[i] = v
                            cand = evaluate(y, cur["s"])
                            spent += 1
                            if cand is None:
                                return cur
                            if key(cand) < key(cur):
                                cur = cand
                                improved = True
                                break
                if not improved:
                    for i in steps:
                        steps[i] *= 0.5
                    if not cont.size or all(steps[i] < 1e-4 * max(span[i], 1e-12) for i in steps):
                        break
            return cur

        nops = len(self.OPS)
        S = np.ones(nops)
        T = 2.0 * np.ones(nops)

        def choose_op():
            rate = S / T
            bonus = 0.25 * np.sqrt(np.log(max(2.0, T.sum())) / T)
            w = rate + bonus + 0.03
            p = w / w.sum()
            return int(np.random.choice(nops, p=p))

        mu = int(max(8, min(30, budget // 50)))
        n_init = min(3 * mu, budget)
        pop = []
        for x0 in lhs_points(n_init):
            ind = evaluate(x0, 0.1)
            if ind is None:
                break
            pop.append(ind)
        if not pop:
            return best["f"], best["x"]
        pop = select(pop, mu)

        sigma_ls = 0.05
        ls_trials = 0
        ls_succ = 0
        t_lo, t_hi = 0.2, 0.8
        stag = 0
        dry_gens = 0
        ls_done_this_stag = False
        sweep_done_this_stag = False

        while used[0] < budget:
            prev_best_f = best["f"]
            prev_has = best["x"] is not None
            used_before = used[0]
            offspring = []
            n_off = 2 * mu
            thr = key(pop[-1]) if len(pop) >= mu else (2, np.inf)
            S *= 0.95
            T *= 0.95
            for _ in range(n_off):
                if used[0] >= budget:
                    break
                k = choose_op()
                op = self.OPS[k]
                feas_exists = best["rec"] is not None
                infeas_in_pop = [p for p in pop if p["v"] > 0.0]
                ind = None

                if op == "ls" and feas_exists:
                    base = best["rec"]
                    child = mutate_multi(base["x"], sigma_ls)
                    ind = evaluate(child, sigma_ls)
                    if ind is not None:
                        ls_trials += 1
                        if ind["v"] == 0.0 and ind["f"] < base["f"]:
                            ls_succ += 1
                    if ls_trials >= 10:
                        rate = ls_succ / ls_trials
                        sigma_ls *= 1.22 if rate > 0.2 else 0.82
                        sigma_ls = float(np.clip(sigma_ls, 1e-3, 0.3))
                        ls_trials = 0
                        ls_succ = 0

                elif op == "vdesc" and infeas_in_pop:
                    # descend violation on the least-infeasible parent
                    cand_pool = sorted(infeas_in_pop, key=lambda p: p["v"])[:max(1, len(infeas_in_pop) // 3)]
                    par = cand_pool[np.random.randint(len(cand_pool))]
                    ind = violation_descent(par, n_moves=4)

                elif op == "bis" and feas_exists and infeas_in_pop:
                    if archive and np.random.rand() < 0.5:
                        feas = archive[np.random.randint(min(len(archive), 5))]["x"]
                    else:
                        feas = best["rec"]["x"]
                    inf_ind = infeas_in_pop[np.random.randint(len(infeas_in_pop))]
                    t = np.random.uniform(t_lo, t_hi)
                    y = feas + t * (inf_ind["x"] - feas)
                    if disc.size:
                        take = np.random.rand(disc.size) < 0.5
                        y[disc] = np.where(take, inf_ind["x"][disc], feas[disc])
                    ind = evaluate(y, 0.1)
                    if ind is not None and ind["v"] == 0.0:
                        t_hi = min(0.95, t_hi + 0.01)
                    elif ind is not None:
                        t_lo = max(0.02, t_lo - 0.01)
                        t_hi = max(t_lo + 0.1, t_hi - 0.01)

                elif op == "de" and feas_exists and len(archive) >= 3:
                    i1, i2 = np.random.choice(len(archive), 2, replace=False)
                    F = float(np.random.uniform(0.4, 0.9))
                    child = de_move(best["rec"]["x"], archive[i1]["x"], archive[i2]["x"], F)
                    ind = evaluate(child, 0.1)

                elif op == "arch" and archive:
                    p1 = tournament(pop)
                    a = archive[np.random.randint(min(len(archive), 8))]
                    mask = np.random.rand(dim) < 0.5
                    xc = np.where(mask, p1["x"], a["x"])
                    child, s_new = mutate(xc, p1["s"])
                    ind = evaluate(child, s_new)

                else:  # "ga" and fallbacks
                    p1 = tournament(pop)
                    p2 = tournament(pop)
                    mask = np.random.rand(dim) < 0.5
                    xc = np.where(mask, p1["x"], p2["x"])
                    sc = p1["s"] if np.random.rand() < 0.5 else p2["s"]
                    child, s_new = mutate(xc, sc)
                    ind = evaluate(child, s_new)

                if ind is None:
                    break
                ind["op"] = k
                offspring.append(ind)

            for ind in offspring:
                k = ind["op"]
                T[k] += 1.0
                if key(ind) < thr:
                    S[k] += 1.0

            pop = select(pop + offspring, mu)

            # guard: generations that consumed no budget (all cache hits)
            if used[0] == used_before:
                dry_gens += 1
                if dry_gens >= 3 and used[0] < budget:
                    for _ in range(mu // 2):
                        if used[0] >= budget:
                            break
                        ind = evaluate(random_point(), 0.2)
                        if ind is None:
                            break
                        pop.append(ind)
                    pop = select(pop, mu)
                    dry_gens = 0
            else:
                dry_gens = 0

            improved = best["x"] is not None and ((not prev_has) or best["f"] < prev_best_f)
            stag = 0 if improved else stag + 1
            if improved:
                ls_done_this_stag = False
                sweep_done_this_stag = False

            if stag >= 5 and not ls_done_this_stag and best["rec"] is not None and used[0] < budget:
                res = pattern_search(best["rec"], cap=3 * dim + 6)
                ls_done_this_stag = True
                pop = select(pop + [res], mu)

            if stag >= 7 and not sweep_done_this_stag and disc.size and best["rec"] is not None and used[0] < budget:
                res = pattern_search(best["rec"], cap=int(sum(int(nvals[i]) for i in disc)) + 2 * dim, full_discrete=True)
                sweep_done_this_stag = True
                pop = select(pop + [res], mu)

            if stag >= 10 and used[0] < budget:
                keep = max(1, mu // 3)
                new_pop = pop[:keep]
                n_new = mu - keep
                for _ in range(n_new):
                    if used[0] >= budget:
                        break
                    u = np.random.rand()
                    if archive and u < 0.4:
                        a = archive[np.random.randint(min(len(archive), 10))]
                        sc = float(np.exp(np.random.uniform(np.log(0.005), np.log(0.1))))
                        child, _ = mutate(a["x"], sc)
                        ind = evaluate(child, sc)
                    elif archive and u < 0.7 and len(archive) >= 2:
                        i1, i2 = np.random.choice(len(archive), 2, replace=False)
                        a, b = archive[i1], archive[i2]
                        mask = np.random.rand(dim) < 0.5
                        xc = np.where(mask, a["x"], b["x"])
                        child, _ = mutate(xc, 0.05)
                        ind = evaluate(child, 0.05)
                    else:
                        ind = evaluate(random_point(), 0.2)
                    if ind is None:
                        break
                    new_pop.append(ind)
                pop = select(new_pop, mu)
                stag = 0
                ls_done_this_stag = False
                sweep_done_this_stag = False

        return best["f"], best["x"]