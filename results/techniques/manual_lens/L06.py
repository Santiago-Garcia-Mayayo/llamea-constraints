import numpy as np


class Algorithm:
    """Epsilon-constrained current-to-pbest JADE for mixed variables with an elite feasible archive.
    - Epsilon-level constraint handling with annealing and early drop once most of the population is feasible.
    - Elite archive of distinct feasible points: seeds restarts and centres local clouds.
    - Integer-descent repair of near-feasible trials while no feasible point is known.
    - Incumbent refinement: integer flips, coordinate pattern search, random-direction moves
      and mixed integer+continuous moves with adaptive step sizes."""

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng()
        d = self.dim
        budget = self.budget

        lb = np.asarray(func.bounds.lb, dtype=float).copy()
        ub = np.asarray(func.bounds.ub, dtype=float).copy()
        vtypes = [str(v) for v in func.variable_types]
        is_int = np.array([v != "continuous" for v in vtypes], dtype=bool)
        is_cat = np.array([v == "categorical" for v in vtypes], dtype=bool)
        ci = np.where(~is_int)[0]
        ii = np.where(is_int)[0]
        nv = np.ones(d, dtype=int)
        for i in ii:
            nv[i] = max(2, int(func.n_values[i]))
        span = ub - lb

        st = {"evals": 0, "best_f": np.inf, "best_x": None,
              "feas_frac": 0.0, "steps": 0.05 * span.copy()}
        cache = {}
        elite = []
        tend = max(1.0, 0.7 * budget)
        eps_cfg = {"eps0": 0.0}
        rb = int(max(20, min(150, budget // 40)))

        def eps_now():
            if st["evals"] >= tend:
                return 0.0
            if st["feas_frac"] >= 0.6:
                return 0.0
            return eps_cfg["eps0"] * (1.0 - st["evals"] / tend) ** 5

        def key(v, f, eps):
            return (0, f) if v <= eps else (1, v)

        def repair(x):
            x = np.clip(np.asarray(x, dtype=float), lb, ub)
            if len(ii):
                x[ii] = np.clip(np.round(x[ii]), 0, nv[ii] - 1)
            return np.ascontiguousarray(x, dtype=np.float64)

        def add_elite(f, x):
            for _, ex in elite:
                if np.array_equal(ex, x):
                    return
            elite.append((f, x.copy()))
            elite.sort(key=lambda t: t[0])
            del elite[8:]

        def ev(x):
            x = repair(x)
            k = x.tobytes()
            if k in cache:
                f, v = cache[k]
                return x, f, v
            if st["evals"] >= budget:
                return None
            fr, g = func(x)
            st["evals"] += 1
            g = np.atleast_1d(np.asarray(g, dtype=float))
            if not np.all(np.isfinite(g)):
                v = 1e6
            else:
                v = float(np.sum(np.maximum(g, 0.0)))
            f = float(fr)
            if not np.isfinite(f):
                f = 1e30
                v = v + 1.0
            cache[k] = (f, v)
            if v == 0:
                add_elite(f, x)
                if f < st["best_f"]:
                    st["best_f"] = f
                    st["best_x"] = x.copy()
            return x, f, v

        def lhs_points(m):
            if m <= 0:
                return np.empty((0, d))
            X = np.empty((m, d))
            for j in ci:
                perm = rng.permutation(m)
                X[:, j] = lb[j] + (perm + rng.random(m)) / m * span[j]
            if len(ii):
                X[:, ii] = rng.integers(0, nv[ii], size=(m, len(ii)))
            return X

        def mutate_gene(x, j, scale):
            if is_int[j]:
                if nv[j] == 2:
                    x[j] = 1 - x[j]
                elif is_cat[j]:
                    other = int(rng.integers(0, nv[j] - 1))
                    if other >= x[j]:
                        other += 1
                    x[j] = other
                else:
                    step = 1 if rng.random() < 0.5 else -1
                    nxt = x[j] + step
                    if nxt < 0 or nxt > nv[j] - 1:
                        nxt = x[j] - step
                    x[j] = np.clip(nxt, 0, nv[j] - 1)
            else:
                x[j] = np.clip(x[j] + scale * span[j] * rng.standard_normal(), lb[j], ub[j])

        def local_child(base, sigma):
            child = base.copy()
            pm_loc = max(1.0 / d, 0.1)
            mask = rng.random(d) < pm_loc
            if not mask.any():
                mask[rng.integers(d)] = True
            for j in np.where(mask)[0]:
                mutate_gene(child, j, sigma)
            return child

        def int_descent(x, fx, vx, eps, max_ev):
            """First-improvement descent over single integer changes, key-based."""
            cur = [x, fx, vx]
            used = 0
            if len(ii) == 0:
                return cur
            improved = True
            while improved and used < max_ev and st["evals"] < budget:
                improved = False
                for j in rng.permutation(ii):
                    vals = [v for v in range(nv[j]) if v != int(cur[0][j])]
                    rng.shuffle(vals)
                    for v in vals:
                        if used >= max_ev or st["evals"] >= budget:
                            return cur
                        c = cur[0].copy()
                        c[j] = v
                        used += 1
                        r = ev(c)
                        if r is None:
                            return cur
                        if key(r[2], r[1], eps) < key(cur[2], cur[1], eps):
                            cur = [r[0], r[1], r[2]]
                            improved = True
                            break
                    if improved:
                        break
            return cur

        def refine(max_ev):
            """Budgeted incumbent refinement: integer flips, coordinate pattern search,
            random-direction moves and mixed integer+continuous moves."""
            used = 0
            if st["best_x"] is None:
                return
            lim_int = max_ev // 2
            cands = []
            b = st["best_x"]
            for j in ii:
                for v in range(nv[j]):
                    if v != int(b[j]):
                        c = b.copy()
                        c[j] = v
                        cands.append(c)
            if cands:
                for idx in rng.permutation(len(cands))[:lim_int]:
                    if st["evals"] >= budget or used >= lim_int:
                        return
                    used += 1
                    if ev(cands[idx]) is None:
                        return
            if len(ci) == 0:
                return
            steps = st["steps"]
            while used < max_ev and st["evals"] < budget:
                if st["best_x"] is None:
                    return
                x = st["best_x"].copy()
                fx = st["best_f"]
                improved = False
                for j in rng.permutation(ci):
                    for sgn in (1.0, -1.0):
                        if used >= max_ev or st["evals"] >= budget:
                            return
                        c = x.copy()
                        c[j] = np.clip(x[j] + sgn * steps[j], lb[j], ub[j])
                        if c[j] == x[j]:
                            continue
                        used += 1
                        r = ev(c)
                        if r is None:
                            return
                        xc, fc, vc = r
                        if vc == 0 and fc < fx:
                            x = xc
                            fx = fc
                            steps[j] *= 1.5
                            improved = True
                            break
                if not improved and used < max_ev and st["evals"] < budget:
                    c = x.copy()
                    dvec = rng.standard_normal(len(ci)) * steps[ci]
                    c[ci] = np.clip(x[ci] + dvec, lb[ci], ub[ci])
                    if len(ii) and rng.random() < 0.3:
                        j = int(rng.choice(ii))
                        vals = [v for v in range(nv[j]) if v != int(x[j])]
                        c[j] = vals[int(rng.integers(len(vals)))]
                    used += 1
                    r = ev(c)
                    if r is None:
                        return
                    if r[2] == 0 and r[1] < fx:
                        improved = True
                if not improved:
                    steps[ci] *= 0.5
                    if np.all(steps[ci] < 1e-4 * span[ci]):
                        steps[ci] = 0.02 * span[ci]
                        return

        def base_x(pop):
            if st["best_x"] is not None:
                return st["best_x"]
            return min(pop, key=lambda p: p[2])[0]

        def progress_metric(pop):
            if st["best_x"] is not None:
                return st["best_f"]
            return 1e9 + min(p[2] for p in pop)

        P0 = int(min(max(12, 5 * d), 80, max(4, budget // 5)))
        Ninit = P0
        pop = []
        for x in lhs_points(P0):
            r = ev(x)
            if r is None:
                break
            pop.append(list(r))
        if len(pop) < 4:
            return st["best_f"], st["best_x"]

        pos_v = [p[2] for p in pop if p[2] > 0]
        eps_cfg["eps0"] = float(np.median(pos_v)) if pos_v else 0.0
        st["feas_frac"] = float(np.mean([p[2] <= 0 for p in pop]))

        muF, muCR = 0.5, 0.5
        c_adapt = 0.1
        archive = []
        p_ls = 0.15
        sigma = 0.05
        ls_succ, ls_tot = 0, 0
        last_prog = None
        no_imp = 0
        G_STALL = 8
        last_scan_f = None
        restarts = 0
        phase_start = st["evals"]
        phase_len = max(4 * P0, (budget - st["evals"]) // 2)
        attempts = 0
        max_attempts = 60 * budget + 1000

        while st["evals"] < budget and len(pop) >= 4 and attempts < max_attempts:
            eps = eps_now()
            order = sorted(range(len(pop)), key=lambda k: key(pop[k][2], pop[k][1], eps))
            pbest_n = max(2, int(round(0.15 * len(pop))))
            top = order[:pbest_n]
            SF, SCR = [], []
            n_now = len(pop)

            for i in range(n_now):
                if st["evals"] >= budget or attempts >= max_attempts:
                    break
                attempts += 1
                xi, fi, vi = pop[i]
                ki = key(vi, fi, eps)

                if rng.random() < p_ls:
                    if elite and rng.random() < 0.5:
                        base = elite[int(rng.integers(min(3, len(elite))))][1]
                    else:
                        base = base_x(pop)
                    child = local_child(base, sigma)
                    r = ev(child)
                    if r is None:
                        break
                    xc, fc, vc = r
                    ls_tot += 1
                    if key(vc, fc, eps) < ki:
                        ls_succ += 1
                    if ls_tot >= 10:
                        rate = ls_succ / ls_tot
                        sigma = sigma * 1.3 if rate > 0.2 else sigma * 0.8
                        sigma = float(np.clip(sigma, 1e-4, 0.5))
                        ls_succ, ls_tot = 0, 0
                else:
                    F = muF + 0.1 * rng.standard_cauchy()
                    tries = 0
                    while F <= 0 and tries < 50:
                        F = muF + 0.1 * rng.standard_cauchy()
                        tries += 1
                    F = 0.5 if F <= 0 else min(float(F), 1.0)
                    CR = float(np.clip(rng.normal(muCR, 0.1), 0.0, 1.0))

                    others = [k for k in range(len(pop)) if k != i]
                    r1 = others[int(rng.integers(len(others)))]
                    pool_size = len(pop) + len(archive)
                    r2idx = int(rng.integers(pool_size))
                    xr2 = pop[r2idx][0] if r2idx < len(pop) else archive[r2idx - len(pop)]
                    xb = pop[top[int(rng.integers(len(top)))]][0]
                    xr1 = pop[r1][0]

                    mutant = xi + F * (xb - xi) + F * (xr1 - xr2)
                    trial = xi.copy()
                    mask = rng.random(d) < CR
                    mask[rng.integers(d)] = True
                    for j in ci:
                        if mask[j]:
                            trial[j] = mutant[j]
                    pm = max(1.0 / max(1, len(ii)), 0.05)
                    for j in ii:
                        if mask[j]:
                            pick = rng.random()
                            if pick < 0.4:
                                trial[j] = xb[j]
                            elif pick < 0.7:
                                trial[j] = xr1[j]
                            else:
                                trial[j] = xr2[j] if is_cat[j] else np.round(mutant[j])
                        if rng.random() < pm:
                            mutate_gene(trial, j, 0.0)
                    r = ev(trial)
                    if r is None:
                        break
                    xc, fc, vc = r

                    # Integer-descent repair of infeasible trials while no feasible point is known
                    if vc > 0 and st["best_x"] is None and len(ii) and rng.random() < 0.05:
                        rr = int_descent(xc, fc, vc, eps, max_ev=2 * len(ii))
                        xc, fc, vc = rr

                    if key(vc, fc, eps) < ki:
                        SF.append(F)
                        SCR.append(CR)
                        archive.append(xi.copy())

                if key(vc, fc, eps) <= ki:
                    pop[i] = [xc, fc, vc]

            if len(pop) < 4:
                break

            while len(archive) > len(pop):
                archive.pop(int(rng.integers(len(archive))))

            if SF:
                SF_arr = np.asarray(SF)
                lehmer = float(np.sum(SF_arr ** 2) / max(np.sum(SF_arr), 1e-12))
                muF = (1 - c_adapt) * muF + c_adapt * lehmer
                muCR = (1 - c_adapt) * muCR + c_adapt * float(np.mean(SCR))

            frac = min(1.0, (st["evals"] - phase_start) / max(1, phase_len))
            Ntarget = max(4, int(round(Ninit + (4 - Ninit) * frac)))
            if len(pop) > Ntarget:
                eps = eps_now()
                pop.sort(key=lambda p: key(p[2], p[1], eps))
                pop = pop[:Ntarget]

            st["feas_frac"] = float(np.mean([p[2] <= 0 for p in pop])) if pop else 0.0

            prog = progress_metric(pop)
            if last_prog is None or prog < last_prog - 1e-12:
                last_prog = prog
                no_imp = 0
            else:
                no_imp += 1

            if st["evals"] >= budget:
                break

            if no_imp >= G_STALL or (frac >= 1.0 and no_imp >= 3):
                no_imp = 0
                if st["best_x"] is not None and st["best_f"] != last_scan_f:
                    last_scan_f = st["best_f"]
                    refine(rb)
                    if st["evals"] >= budget:
                        break
                restarts += 1
                if restarts % 2 == 0:
                    Ninit = P0
                else:
                    Ninit = int(min(80, max(P0, round(1.5 * P0))))
                newpop = []
                if elite:
                    k_el = min(len(elite), max(1, Ninit // 4))
                    for idx in range(k_el):
                        if st["evals"] >= budget:
                            break
                        r = ev(elite[idx][1])
                        if r is not None:
                            newpop.append(list(r))
                    base = elite[int(rng.integers(len(elite)))][1]
                else:
                    base = base_x(pop)
                if restarts % 2 == 1:
                    n_local = Ninit // 3
                else:
                    n_local = Ninit // 5
                for _ in range(n_local):
                    if st["evals"] >= budget:
                        break
                    s = float(np.exp(rng.uniform(np.log(0.003), np.log(0.3))))
                    r = ev(local_child(base, s))
                    if r is None:
                        break
                    newpop.append(list(r))
                for x in lhs_points(max(0, Ninit - len(newpop))):
                    if st["evals"] >= budget:
                        break
                    r = ev(x)
                    if r is None:
                        break
                    newpop.append(list(r))
                if len(newpop) >= 4:
                    pop = newpop
                    st["feas_frac"] = float(np.mean([p[2] <= 0 for p in pop]))
                archive = []
                last_prog = None
                phase_start = st["evals"]
                phase_len = max(4 * Ninit, (budget - st["evals"]) // 2)

        return st["best_f"], st["best_x"]