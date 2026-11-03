import numpy as np


class Algorithm:
    """Steady-state mixed-integer evolution with epsilon-constrained ranking.
    Builds on RefinedAdaptiveEvolutionV4 with:
      * boundary-line probes: infeasible elites are interpolated towards the
        best feasible incumbent to locate the feasible/infeasible boundary;
      * success-adaptive restart radius (grows when restarts yield improvements);
      * a discrete sweep capped by remaining budget fraction;
      * more frequent repair while no feasible point is known.
    Cached evaluations cost no budget."""

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng(np.random.randint(2**31 - 1))
        d = self.dim
        budget = self.budget
        if budget <= 0:
            return np.inf, None

        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        vtypes = [str(t) for t in func.variable_types]
        cont = np.array([t == "continuous" for t in vtypes], dtype=bool)
        disc = ~cont
        cat = np.array([t == "categorical" for t in vtypes], dtype=bool)
        span = ub - lb
        nv = np.ones(d, dtype=int)
        nv[disc] = np.maximum(np.round(span[disc]).astype(int), 1)
        cspan = np.where(cont, span, 0.0)

        def repair(x):
            x = np.clip(np.asarray(x, dtype=float), lb, ub)
            if disc.any():
                x[disc] = np.clip(np.round(x[disc]), 0, nv[disc] - 1)
            return x

        def lhs_points(n):
            pts = np.zeros((n, d))
            for i in range(d):
                if cont[i]:
                    u = (rng.permutation(n) + rng.random(n)) / n
                    pts[:, i] = lb[i] + u * span[i]
                else:
                    pts[:, i] = rng.permutation(n) % nv[i]
            return pts

        def random_value(i):
            if cont[i]:
                return rng.uniform(lb[i], ub[i])
            return float(rng.integers(0, nv[i]))

        # ---------------- evaluation with cache and budget accounting -------
        cache = {}
        used = [0]
        best = {"f": np.inf, "x": None, "ind": None}
        archive = []
        eps_state = {"eps0": None}

        def eps_now():
            e0 = eps_state["eps0"]
            if e0 is None or e0 <= 0.0:
                return 0.0
            frac = min(1.0, used[0] / (0.8 * budget))
            return e0 * (1.0 - frac) ** 3

        def evaluate(x):
            x = repair(x)
            k = tuple(x.tolist())
            if k in cache:
                v, f = cache[k]
            else:
                if used[0] >= budget:
                    return None
                f, g = func(x)
                used[0] += 1
                g = np.nan_to_num(np.asarray(g, dtype=float).ravel(),
                                  nan=1e6, posinf=1e6)
                v = float(np.sum(np.maximum(g, 0.0)))
                f = float(f)
                if not np.isfinite(f):
                    f = np.inf
                    v += 1.0
                cache[k] = (v, f)
                if eps_state["eps0"] is None and len(cache) >= 20:
                    vs = np.array([c[0] for c in cache.values()])
                    pos = vs[vs > 0]
                    eps_state["eps0"] = float(np.median(pos)) if pos.size else 0.0
            ind = {"x": x, "v": v, "f": f}
            if v <= 0.0:
                if f < best["f"]:
                    best["f"] = f
                    best["x"] = x.copy()
                    best["ind"] = {"x": x.copy(), "v": 0.0, "f": f}
                if np.isfinite(f):
                    archive.append((f, x.copy()))
                    if len(archive) > 60:
                        archive.sort(key=lambda t: t[0])
                        del archive[30:]
            return ind

        def rank(ind):
            if ind["v"] <= eps_now() and np.isfinite(ind["f"]):
                return (0, ind["f"])
            if ind["v"] <= 0.0:
                return (0, ind["f"])
            return (1, ind["v"])

        def key(x):
            return tuple(np.asarray(x).tolist())

        # ---------------- variation operators -------------------------------
        def mutate_mask(p=None):
            p = 1.0 / d if p is None else p
            m = rng.random(d) < p
            if not m.any():
                m[rng.integers(d)] = True
            return m

        def discrete_step(x, i):
            n = nv[i]
            if n <= 1:
                return
            if cat[i]:
                r = int(rng.integers(0, n - 1))
                x[i] = r + (1 if r >= x[i] else 0)
            else:
                jump = 1 if rng.random() < 0.75 else 2
                s = 1 if rng.random() < 0.5 else -1
                nxt = x[i] + s * jump
                if nxt < 0 or nxt > n - 1:
                    nxt = x[i] - s
                x[i] = min(max(nxt, 0.0), n - 1.0)

        def pick_mate(pop):
            if archive and rng.random() < 0.4:
                return archive[rng.integers(len(archive))][1]
            return pop[rng.integers(len(pop))]["x"]

        def make_child(op, p1, pop, best_x, sigma):
            x = p1["x"].copy()
            if op == 0:
                m = mutate_mask() & cont
                if m.any():
                    x[m] += sigma * cspan[m] * rng.standard_normal(m.sum())
                elif disc.any():
                    for i in np.where(mutate_mask() & disc)[0]:
                        discrete_step(x, i)
            elif op == 1:
                idx = rng.choice(len(pop), 3, replace=len(pop) < 3)
                a, b, c = (pop[j]["x"] for j in idx)
                F = rng.uniform(0.4, 0.9)
                donor = a.copy()
                donor[cont] = a[cont] + F * (b[cont] - c[cont])
                cr = rng.random(d) < 0.7
                cr[rng.integers(d)] = True
                x = np.where(cr, donor, x)
            elif op == 2:
                m = mutate_mask() & disc
                if not m.any() and disc.any():
                    m = np.zeros(d, dtype=bool)
                    m[rng.choice(np.where(disc)[0])] = True
                for i in np.where(m)[0]:
                    discrete_step(x, i)
            elif op == 3:
                i = rng.integers(d)
                x[i] = random_value(i)
            elif op == 4:
                if best_x is not None and rng.random() < 0.5:
                    mate = best_x
                else:
                    mate = pick_mate(pop)
                cr = rng.random(d) < 0.5
                x = np.where(cr, mate, x)
                m = mutate_mask()
                for i in np.where(m & disc)[0]:
                    discrete_step(x, i)
                mc = m & cont
                if mc.any():
                    x[mc] += 0.05 * cspan[mc] * rng.standard_normal(mc.sum())
            elif op == 5:
                ref = best_x if best_x is not None else p1["x"]
                x = ref.copy()
                m = mutate_mask(p=min(1.0, 2.0 / d))
                mc = m & cont
                if mc.any():
                    x[mc] += 0.3 * sigma * cspan[mc] * rng.standard_normal(mc.sum())
                for i in np.where(m & disc)[0]:
                    if rng.random() < 0.5:
                        discrete_step(x, i)
            else:
                ref = best_x if best_x is not None else p1["x"]
                x = ref.copy()
                if cont.any():
                    dirn = rng.standard_normal(d) * cont
                    nrm = np.linalg.norm(dirn)
                    if nrm > 0:
                        dirn /= nrm
                    x = x + sigma * np.sqrt(max(cont.sum(), 1)) * cspan * dirn * 0.5
                if rng.random() < 0.3:
                    for i in np.where(disc & (rng.random(d) < 0.2))[0]:
                        discrete_step(x, i)
            return repair(x)

        # ---------------- UCB-style bandit over operators -------------------
        n_ops = 7
        S = np.ones(n_ops)
        N = 2.0 * np.ones(n_ops)
        T = [float(n_ops)]

        def pick_op():
            mean = S / N
            bonus = 0.5 * np.sqrt(np.log(T[0] + 1.0) / N)
            score = mean + bonus
            p = np.exp(3.0 * (score - score.max()))
            p = p / p.sum()
            p = 0.9 * p + 0.1 / n_ops
            return int(rng.choice(n_ops, p=p))

        hs = np.where(cont, 0.05, 1.0)

        # ---------------- discrete neighbourhood sweep ----------------------
        def discrete_sweep(start, max_evals):
            cur = start
            spent = 0
            moves = []
            for i in np.where(disc & (nv > 1))[0]:
                for val in range(nv[i]):
                    if val != int(round(cur["x"][i])):
                        moves.append((i, float(val)))
            if not moves:
                return cur
            order = rng.permutation(len(moves))
            for idx in order:
                if spent >= max_evals or used[0] >= budget:
                    break
                i, val = moves[idx]
                t = cur["x"].copy()
                t[i] = val
                before = used[0]
                ind = evaluate(t)
                if used[0] > before:
                    spent += 1
                if ind is None:
                    return cur
                if rank(ind) < rank(cur):
                    cur = ind
            return cur

        # ---------------- adaptive coordinate pattern search ----------------
        def pattern_search(start, max_evals):
            cur = start
            spent = 0
            while spent < max_evals and used[0] < budget:
                improved_any = False
                for i in rng.permutation(d):
                    if spent >= max_evals or used[0] >= budget:
                        break
                    if not cont[i] and nv[i] <= 1:
                        continue
                    moved = False
                    if cont[i]:
                        trials = (1.0, -1.0)
                    elif cat[i]:
                        trials = (0.0, 0.0)
                    else:
                        trials = (1.0, -1.0)
                    for s in trials:
                        if spent >= max_evals or used[0] >= budget:
                            break
                        t = cur["x"].copy()
                        if cont[i]:
                            t[i] += s * hs[i] * span[i]
                        elif cat[i]:
                            t[i] = random_value(i)
                        else:
                            t[i] += s
                        before = used[0]
                        ind = evaluate(t)
                        if used[0] > before:
                            spent += 1
                        if ind is None:
                            return cur
                        if rank(ind) < rank(cur):
                            cur = ind
                            moved = True
                            improved_any = True
                            break
                    if cont[i]:
                        if moved:
                            hs[i] = min(hs[i] * 1.5, 0.5)
                        else:
                            hs[i] = max(hs[i] * 0.5, 1e-4)
                if not improved_any:
                    cont_steps = hs[cont]
                    if cont_steps.size == 0 or cont_steps.max() < 1e-3:
                        break
            return cur

        # ---------------- boundary-line probes ------------------------------
        def boundary_probe(infeas_ind, n_probe=2):
            """Interpolate from best feasible towards an infeasible elite."""
            if best["x"] is None or infeas_ind is None:
                return None
            bx = best["x"]
            ix = infeas_ind["x"]
            out = None
            ts = (0.5, 0.8, 0.95)
            picks = rng.choice(len(ts), size=min(n_probe, len(ts)), replace=False)
            for j in picks:
                if used[0] >= budget:
                    break
                t = ts[j]
                z = bx + t * (ix - bx)
                # small continuous jitter to escape exact-boundary ties
                if cont.any():
                    z[cont] += 0.01 * cspan[cont] * rng.standard_normal(cont.sum())
                ind = evaluate(z)
                if ind is None:
                    break
                if out is None or rank(ind) < rank(out):
                    out = ind
            return out

        # ---------------- population helpers --------------------------------
        def init_pop(n, keep=(), seeds=()):
            new = list(keep)
            for s in seeds:
                if len(new) >= n or used[0] >= budget:
                    break
                ind = evaluate(s)
                if ind is not None:
                    new.append(ind)
            if len(new) < n and used[0] < budget:
                pts = lhs_points(n - len(new) + 5)
                for p in pts:
                    if len(new) >= n or used[0] >= budget:
                        break
                    ind = evaluate(p)
                    if ind is not None:
                        new.append(ind)
            tries = 0
            while len(new) < n and used[0] < budget and tries < 100 * n + 100:
                tries += 1
                ind = evaluate(np.array([random_value(i) for i in range(d)]))
                if ind is not None:
                    new.append(ind)
            return new

        mu = int(min(max(10, 3 * d), max(4, budget // 8)))
        mu = max(2, min(mu, budget))
        pop = init_pop(mu)
        if not pop:
            return best["f"], best["x"]

        sigma = 0.1
        succ_hist = []
        last_imp = used[0]
        prev_best = best["f"]
        stall = max(150, 25 * d)
        ps_budget = max(2 * d, 12)
        sweep_cap = max(2 * d, 16)
        radii = [0.01, 0.05, 0.2]
        rad_idx = 0
        restart_gain = {}
        n_restart = 0
        iters = 0
        max_iters = 50 * budget + 1000
        last_restart_best = None

        def sweep_allow():
            # shrink discrete sweep as the budget is consumed
            frac = used[0] / budget
            return int(sweep_cap * max(0.25, 1.0 - frac))

        def refine_incumbent():
            nonlocal prev_best
            if used[0] >= budget or best["ind"] is None:
                return
            refined = pattern_search(best["ind"], ps_budget)
            if disc.any() and used[0] < budget:
                refined = discrete_sweep(refined, sweep_allow())
            worst = max(range(len(pop)), key=lambda j: rank(pop[j]))
            if rank(refined) <= rank(pop[worst]):
                pop[worst] = refined
            prev_best = min(prev_best, best["f"])

        if best["ind"] is not None and used[0] < budget:
            refine_incumbent()

        # ---------------- main loop -----------------------------------------
        while used[0] < budget and iters < max_iters:
            iters += 1
            op = pick_op()
            T[0] += 1.0
            i1, i2 = rng.integers(len(pop), size=2)
            p1 = pop[i1] if rank(pop[i1]) <= rank(pop[i2]) else pop[i2]

            cx = make_child(op, p1, pop, best["x"], sigma)
            child = evaluate(cx)
            if child is None:
                break

            r = 1.0 if rank(child) < rank(p1) else 0.0
            S *= 0.98
            N *= 0.98
            S[op] += r
            N[op] += 1.0

            succ_hist.append(r)
            if len(succ_hist) >= 20:
                rate = float(np.mean(succ_hist[-20:]))
                if rate > 0.2:
                    sigma *= 1.22
                elif rate < 0.2:
                    sigma *= 0.82
                sigma = float(np.clip(sigma, 1e-4, 0.5))

            worst = max(range(len(pop)), key=lambda j: rank(pop[j]))
            ck = key(child["x"])
            if rank(child) <= rank(pop[worst]) and all(key(q["x"]) != ck for q in pop):
                pop[worst] = child

            # feasible improvement -> refine incumbent
            if best["f"] < prev_best - 1e-12:
                prev_best = best["f"]
                last_imp = used[0]
                refine_incumbent()

            # boundary-line probes between best feasible and infeasible elites
            if best["x"] is not None and iters % 8 == 0 and used[0] < budget:
                infeas = [q for q in pop if q["v"] > 0.0]
                if infeas:
                    cand = min(infeas, key=lambda q: q["v"])
                    bp = boundary_probe(cand)
                    if bp is not None:
                        worst = max(range(len(pop)), key=lambda j: rank(pop[j]))
                        if rank(bp) <= rank(pop[worst]):
                            pop[worst] = bp

            # no feasible point yet: repair the least-violated individual more often
            if best["x"] is None and used[0] < budget and iters % 8 == 0:
                cand = min(pop, key=lambda q: q["v"])
                if cand["v"] > 0.0:
                    rep = pattern_search(cand, max(d, 8))
                    worst = max(range(len(pop)), key=lambda j: rank(pop[j]))
                    if rank(rep) <= rank(pop[worst]):
                        pop[worst] = rep

            if used[0] - last_imp > stall and used[0] < budget:
                # credit the previous restart radius if it produced an improvement
                if last_restart_best is not None and best["f"] < last_restart_best - 1e-12:
                    restart_gain[radii[rad_idx]] = restart_gain.get(radii[rad_idx], 0) + 1
                last_restart_best = best["f"]

                pop.sort(key=rank)
                keep = pop[:max(1, mu // 4)]
                seeds = []
                if best["x"] is not None:
                    # choose radius: prefer those that have yielded gains
                    if restart_gain and rng.random() < 0.6:
                        rad = max(radii, key=lambda rr: restart_gain.get(rr, 0))
                    else:
                        rad = radii[rng.integers(len(radii))]
                    rad_idx = radii.index(rad)
                    n_restart += 1
                    for _ in range(mu // 4):
                        if archive and rng.random() < 0.5:
                            base = archive[rng.integers(len(archive))][1]
                        else:
                            base = best["x"]
                        s = base.copy()
                        m = mutate_mask(p=0.3)
                        mc = m & cont
                        s[mc] += rad * cspan[mc] * rng.standard_normal(mc.sum())
                        for i in np.where(m & disc)[0]:
                            discrete_step(s, i)
                        seeds.append(repair(s))
                infeas = [q for q in pop if q["v"] > 0.0]
                infeas.sort(key=lambda q: q["v"])
                for q in infeas[:max(1, mu // 8)]:
                    s = q["x"].copy()
                    m = mutate_mask(p=0.2)
                    mc = m & cont
                    s[mc] += 0.05 * cspan[mc] * rng.standard_normal(mc.sum())
                    seeds.append(repair(s))
                pop = init_pop(mu, keep=keep, seeds=seeds)
                if not pop:
                    break
                hs[:] = np.where(cont, 0.05, 1.0)
                sigma = 0.1
                succ_hist = []
                last_imp = used[0]

        if best["x"] is None:
            return np.inf, None
        return best["f"], best["x"]