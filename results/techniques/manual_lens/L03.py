import numpy as np


class Algorithm:
    """
    (mu+1) mixed-integer ES with Deb's feasibility rules and a discounted-UCB operator bandit.
    Operators:
      0: Gaussian step on continuous part + type-aware integer mutation
      1: DE current-to-best/1 (base = incumbent) with binomial crossover on integers
      2: uniform crossover + sparse mutation (1/5-rule sigma)
      3: global random immigrant
      4: systematic incumbent integer-neighbour queue with optional continuous jitter
      5: boundary line search / bisection from incumbent toward near-feasible archive
      6: compass pattern search on incumbent: per-coordinate step and direction, flipping
         direction on failure, expanding on success, contracting after two failures;
         optionally coupled with one integer move.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        vtypes = [str(t) for t in func.variable_types]
        cont = np.array([t == "continuous" for t in vtypes])
        ints = ~cont
        binm = np.array([t == "binary" for t in vtypes])
        catm = np.array([t == "categorical" for t in vtypes])
        ordm = np.array([t == "ordinal" for t in vtypes])
        nvals = np.ones(dim, dtype=int)
        for i in range(dim):
            if not cont[i]:
                nvals[i] = max(1, int(func.n_values[i]))
        n_cont = int(cont.sum())
        cont_idx = np.where(cont)[0]
        int_idx = np.where(ints)[0]

        def repair(x):
            x = np.asarray(x, dtype=float).copy()
            x[cont] = np.clip(x[cont], -1.0, 1.0)
            x[ints] = np.clip(np.round(x[ints]), 0, nvals[ints] - 1)
            return x

        def random_point():
            x = np.empty(dim)
            x[cont] = np.random.uniform(-1.0, 1.0, size=n_cont)
            x[ints] = np.random.randint(0, nvals[ints])
            return x

        def stratified_points(n):
            pts = np.empty((n, dim))
            for j in range(dim):
                if cont[j]:
                    u = (np.random.permutation(n) + np.random.rand(n)) / n
                    pts[:, j] = 2.0 * u - 1.0
                else:
                    pts[:, j] = np.random.randint(0, nvals[j], size=n)
            return pts

        evals = 0
        seen = set()
        best_f = np.inf
        best_x = None
        elites = []
        near = []

        def add_elite(f, x):
            nonlocal elites
            for _, ex in elites:
                if np.array_equal(ex, x):
                    return
            elites.append((f, x.copy()))
            elites.sort(key=lambda t: t[0])
            elites = elites[:15]

        def add_near(v, x):
            nonlocal near
            for _, ex in near:
                if np.array_equal(ex, x):
                    return
            near.append((v, x.copy()))
            near.sort(key=lambda t: t[0])
            near = near[:12]

        def evaluate(x):
            nonlocal evals, best_f, best_x
            x = repair(x)
            seen.add(tuple(np.round(x, 9)))
            f, g = func(x)
            evals += 1
            g = np.nan_to_num(np.asarray(g, dtype=float).ravel(),
                              nan=1e6, posinf=1e6, neginf=0.0)
            viol = float(np.sum(np.maximum(g, 0.0)))
            f = float(f)
            if not np.isfinite(f):
                f = np.inf
                viol += 1.0
            feas = (viol == 0.0) and np.isfinite(f)
            if feas:
                add_elite(f, x)
                if f < best_f:
                    best_f, best_x = f, x.copy()
            elif np.isfinite(viol) and viol < 1e5:
                add_near(viol, x)
            return {"x": x, "f": f, "viol": viol, "feas": feas}

        def score(ind):
            return (0, ind["f"]) if ind["feas"] else (1, ind["viol"])

        # ---- stratified initial design ----
        n_init = min(self.budget, max(2 * dim + 5, min(100, self.budget // 8)))
        pts = stratified_points(n_init)
        pop = []
        for k in range(n_init):
            if evals >= self.budget:
                break
            pop.append(evaluate(pts[k]))
        if not pop:
            return np.inf, None
        mu = min(40, max(4, len(pop) // 2))
        pop.sort(key=score)
        pop = pop[:mu]

        def tournament():
            i, j = np.random.randint(len(pop), size=2)
            return pop[i] if score(pop[i]) <= score(pop[j]) else pop[j]

        def base_point():
            if best_x is not None:
                return best_x
            return min(pop, key=score)["x"]

        def mutate_int(x, i):
            n = nvals[i]
            if n <= 1:
                return
            old = int(round(x[i]))
            if binm[i]:
                x[i] = 1 - old
            elif catm[i]:
                v = np.random.randint(n - 1)
                x[i] = v + (1 if v >= old else 0)
            else:
                if np.random.rand() < 0.7:
                    nv = old + (1 if np.random.rand() < 0.5 else -1)
                    if nv < 0 or nv >= n:
                        nv = old - (nv - old)
                    x[i] = nv
                else:
                    x[i] = np.random.randint(n)

        # ---- systematic incumbent integer-neighbour queue ----
        qstate = {"key": None, "q": []}

        def build_queue():
            q_near, q_far = [], []
            if best_x is None:
                return []
            for i in int_idx:
                n = nvals[i]
                if n <= 1:
                    continue
                cur = int(round(best_x[i]))
                for v in range(n):
                    if v == cur:
                        continue
                    if ordm[i] and abs(v - cur) == 1:
                        q_near.append((i, v))
                    elif (not ordm[i]) and (not catm[i]) and (not binm[i]) and abs(v - cur) == 1:
                        q_near.append((i, v))
                    elif binm[i]:
                        q_near.append((i, v))
                    else:
                        q_far.append((i, v))
            np.random.shuffle(q_near)
            np.random.shuffle(q_far)
            return q_near + q_far

        def next_queue_move():
            if qstate["key"] != best_f:
                qstate["key"] = best_f
                qstate["q"] = build_queue()
            q = qstate["q"]
            while q:
                i, v = q.pop(0)
                x = best_x.copy()
                x[i] = v
                if np.random.rand() < 0.5:
                    x[cont] += 0.01 * np.random.randn(n_cont)
                return x
            return None

        def sweep_int(x, i):
            n = nvals[i]
            if n <= 1:
                return
            old = int(round(x[i]))
            if binm[i]:
                x[i] = 1 - old
            elif catm[i]:
                v = np.random.randint(n - 1)
                x[i] = v + (1 if v >= old else 0)
            else:
                step = 1 if np.random.rand() < 0.5 else -1
                nv = old + step
                if nv < 0 or nv >= n:
                    nv = old - step
                x[i] = nv

        # ---- compass pattern-search state (per coordinate) ----
        ps = {
            "step": np.full(max(1, n_cont), 0.05),
            "dir": np.ones(max(1, n_cont)),
            "fail": np.zeros(max(1, n_cont), dtype=int),
            "ptr": 0,
            "last": None,
            "last_dir": 1.0,
        }

        def make_child(op, sig, sig_loc):
            if op == 0:
                p1 = tournament()
                x = p1["x"].copy()
                mut = np.random.rand(dim) < 1.0 / max(1.0, np.sqrt(dim))
                if not mut.any():
                    mut[np.random.randint(dim)] = True
                m = mut & cont
                if m.any():
                    x[m] += sig * 2.0 * np.random.randn(int(m.sum()))
                for i in np.where(mut & ints)[0]:
                    mutate_int(x, i)
            elif op == 1:
                p1 = tournament()
                p2 = tournament()
                base = base_point()
                x = p1["x"].copy()
                F = np.random.uniform(0.4, 0.9)
                cr = np.random.uniform(0.5, 0.95)
                donor = base + F * (p1["x"] - p2["x"])
                cmask = (np.random.rand(dim) < cr) | (np.arange(dim) == np.random.randint(dim))
                m = cmask & cont
                x[m] = donor[m]
                take_best = np.random.rand(dim) < 0.5
                im = take_best & ints
                x[im] = base[im]
                for i in np.where(ints & (np.random.rand(dim) < 0.05))[0]:
                    mutate_int(x, i)
            elif op == 2:
                p1 = tournament()
                p2 = tournament()
                mask = np.random.rand(dim) < 0.5
                x = np.where(mask, p1["x"], p2["x"]).astype(float)
                for i in np.where(ints & (np.random.rand(dim) < 1.0 / dim))[0]:
                    mutate_int(x, i)
                m = cont & (np.random.rand(dim) < 1.0 / dim)
                if m.any():
                    x[m] += 0.5 * sig * np.random.randn(int(m.sum()))
            elif op == 3:
                x = random_point()
            elif op == 4:
                if best_x is None or len(int_idx) == 0:
                    x = random_point()
                else:
                    x = next_queue_move()
                    if x is None:
                        x = best_x.copy()
                        k = 1 if (np.random.rand() < 0.7 or len(int_idx) < 2) else 2
                        for i in np.random.choice(int_idx, size=k, replace=False):
                            sweep_int(x, i)
                        x[cont] += sig_loc * np.random.randn(n_cont)
            elif op == 5:
                if best_x is None or not near:
                    x = random_point()
                else:
                    z = near[np.random.randint(len(near))][1]
                    if np.random.rand() < 0.4:
                        t = 0.5
                    else:
                        t = np.random.uniform(0.2, 1.0)
                    x = best_x + t * (z - best_x)
                    take_z = np.random.rand(dim) < 0.5
                    x[ints] = np.where(take_z[ints], z[ints], best_x[ints])
                    for i in np.where(ints & (np.random.rand(dim) < 0.05))[0]:
                        mutate_int(x, i)
                    x[cont] += 0.3 * sig * np.random.randn(n_cont)
            else:  # op 6: compass pattern search on incumbent
                if best_x is None or n_cont == 0:
                    x = random_point()
                    ps["last"] = None
                else:
                    x = best_x.copy()
                    k = int(ps["ptr"] % n_cont)
                    ps["ptr"] += 1
                    ps["last"] = k
                    ci = cont_idx[k]
                    d = ps["dir"][k]
                    ps["last_dir"] = d
                    x[ci] += d * ps["step"][k] * np.random.uniform(0.8, 1.2)
                    # occasionally couple with a single integer move
                    if len(int_idx) > 0 and np.random.rand() < 0.25:
                        i = int(np.random.choice(int_idx))
                        sweep_int(x, i)
                    if np.random.rand() < 0.15 and n_cont > 1:
                        k2 = np.random.randint(n_cont)
                        x[cont_idx[k2]] += ps["step"][k2] * np.random.randn()
            return repair(x)

        # discounted UCB operator bandit
        n_ops = 7
        gam = 0.97
        cnt = np.zeros(n_ops)
        rew = np.zeros(n_ops)

        def pick_op():
            untried = np.where(cnt == 0)[0]
            if len(untried) > 0:
                return int(np.random.choice(untried))
            total = cnt.sum()
            ucb = rew / cnt + 0.4 * np.sqrt(2.0 * np.log(total) / cnt)
            return int(np.argmax(ucb))

        def credit(op, r):
            cnt[:] *= gam
            rew[:] *= gam
            cnt[op] += 1.0
            rew[op] += r

        sigma = 0.1
        sigma_loc = 0.02
        succ_hist = []
        best_before = best_f
        last_improve = evals
        stagn_limit = max(40, 30 * dim)

        def restart():
            nonlocal pop, sigma, succ_hist, best_before, last_improve
            pop.sort(key=score)
            pop = pop[:max(1, len(pop) // 2)]
            n_seed = max(1, mu // 4)
            for _ in range(n_seed):
                if evals >= self.budget or not elites:
                    break
                if np.random.rand() < 0.5 or len(elites) == 1:
                    seed = best_x if best_x is not None else elites[0][1]
                else:
                    seed = elites[np.random.randint(len(elites))][1]
                x = seed.copy()
                scale = np.exp(np.random.uniform(np.log(1e-3), np.log(0.3)))
                x[cont] += scale * np.random.randn(n_cont)
                for i in np.where(ints & (np.random.rand(dim) < 0.1))[0]:
                    mutate_int(x, i)
                pop.append(evaluate(x))
            while len(pop) < mu and evals < self.budget:
                pop.append(evaluate(random_point()))
            sigma = 0.15
            succ_hist = []
            last_improve = evals
            best_before = best_f
            ps["step"] = np.full(max(1, n_cont), 0.05)
            ps["dir"] = np.ones(max(1, n_cont))
            ps["fail"] = np.zeros(max(1, n_cont), dtype=int)
            qstate["key"] = None

        # ---- main loop ----
        while evals < self.budget:
            if evals - last_improve > stagn_limit:
                restart()
                if evals >= self.budget:
                    break

            op = pick_op()
            child = None
            for _ in range(6):
                cand = make_child(op, sigma, sigma_loc)
                if tuple(np.round(cand, 9)) not in seen:
                    child = cand
                    break
            if child is None:
                op = 3
                child = random_point()

            ind = evaluate(child)

            new_best = ind["feas"] and ind["f"] < best_before - 1e-12
            if new_best:
                best_before = best_f
                last_improve = evals
            elif best_f < best_before:
                best_before = best_f
                last_improve = evals

            # compass pattern-search adaptation
            if op == 6 and ps["last"] is not None:
                k = ps["last"]
                if new_best or (ind["feas"] and ind["f"] <= best_f):
                    ps["step"][k] = min(0.5, ps["step"][k] * 1.8)
                    ps["fail"][k] = 0
                else:
                    ps["dir"][k] = -ps["last_dir"]   # probe the opposite direction next
                    ps["fail"][k] += 1
                    if ps["fail"][k] >= 2:
                        ps["step"][k] = max(1e-4, ps["step"][k] * 0.5)
                        ps["fail"][k] = 0

            if len(pop) < mu:
                pop.append(ind)
                success = True
            else:
                w = max(range(len(pop)), key=lambda k: score(pop[k]))
                success = score(ind) < score(pop[w])
                if success:
                    pop[w] = ind

            reward = 1.0 if new_best else (0.4 if success else 0.0)
            credit(op, reward)

            if op in (0, 2):
                succ_hist.append(1.0 if success else 0.0)
                if len(succ_hist) >= 10:
                    rate = float(np.mean(succ_hist))
                    sigma *= 1.25 if rate > 0.2 else 0.82
                    sigma = float(np.clip(sigma, 1e-5, 0.5))
                    succ_hist = []
            if op == 4:
                sigma_loc *= 1.1 if new_best else 0.95
                sigma_loc = float(np.clip(sigma_loc, 1e-4, 0.1))

        if best_x is None:
            pop.sort(key=score)
            return np.inf, pop[0]["x"]
        return best_f, best_x