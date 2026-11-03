import numpy as np


class Algorithm:
    """
    (mu + lambda) mixed-integer ES with Deb feasibility rules and surrogate screening.
      * Ridge models per constraint predict violation magnitude and violation probability.
      * Children are screened by predicted weighted violation plus joint-infeasibility risk.
      * Feasibility pursuit targets the two most frequently violated constraints with a
        model-based line search (dominant constraint gets double weight).
      * Feasible-preserving local search around the incumbent with adaptive step size,
        so that once a feasible point exists, budget is spent on improving it.
      * Feasible/infeasible interpolation biased toward the feasibility boundary.
      * Exact-evaluation cache, GA-style children, stagnation restarts.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        lb = np.asarray(func.bounds.lb, dtype=float).ravel()
        ub = np.asarray(func.bounds.ub, dtype=float).ravel()
        vt = list(func.variable_types)
        n_values = np.asarray(getattr(func, "n_values", np.zeros(dim)), dtype=float).ravel()
        n_values = np.where(np.isfinite(n_values), n_values, 2).astype(int)
        span = np.maximum(ub - lb, 1e-12)

        cont = np.array([i for i in range(dim) if vt[i] == "continuous"], dtype=int)
        disc = np.array([i for i in range(dim) if vt[i] != "continuous"], dtype=int)
        is_binary = {i: vt[i] == "binary" for i in disc}
        is_ordinal = {i: vt[i] == "ordinal" for i in disc}
        n_c = len(cont)

        pm_c = max(1.0 / max(dim, 1), 0.3)
        pm_d = max(1.0 / max(dim, 1), 0.15)
        tau = 1.0 / np.sqrt(max(dim, 1))

        self.evals = 0
        best = {"f": np.inf, "x": None}
        best_viol = {"v": np.inf, "x": None, "f": np.inf}
        state = {"m": None, "hist": None}
        archive = {"X": [], "G": [], "F": [], "feas": []}
        cache = {}
        model = {"Wg": None, "Wi": None, "Wf": None}
        sig = {"loc": 0.02, "feas": 0.01}

        def repair(x):
            y = np.array(x, dtype=float).copy()
            if n_c:
                y[cont] = np.clip(y[cont], lb[cont], ub[cont])
            for i in disc:
                y[i] = float(np.clip(np.rint(y[i]), 0, n_values[i] - 1))
            return y

        def features(X):
            U = (np.atleast_2d(X) - lb) / span
            return np.hstack([U, np.ones((U.shape[0], 1))])

        def weights():
            if state["hist"] is None:
                return np.ones(0)
            h = state["hist"]
            w = 1.0 + 6.0 * h ** 2
            w = w + 4.0 * (h > 0.8)
            return w

        def feas_rate():
            recent = archive["feas"][-100:]
            return float(np.mean(recent)) if recent else 0.0

        def evaluate(x, s=0.1):
            x = repair(x)
            ck = tuple(np.round(x, 9))
            if ck in cache:
                f, gv = cache[ck]
            else:
                if self.evals >= self.budget:
                    raise RuntimeError("budget exhausted")
                f, g = func(x)
                self.evals += 1
                f = float(f) if np.isfinite(f) else np.inf
                g = np.nan_to_num(np.atleast_1d(np.asarray(g, dtype=float)).ravel(),
                                  nan=1e6, posinf=1e6, neginf=0.0)
                gv = np.maximum(g, 0.0)
                cache[ck] = (f, gv)
                if state["m"] is None:
                    state["m"] = gv.size
                    state["hist"] = np.full(gv.size, 0.5)
                if gv.size == state["m"] and gv.size > 0:
                    state["hist"] = 0.9 * state["hist"] + 0.1 * (gv > 0).astype(float)
                viol = float(gv.sum())
                feas = (viol <= 0.0) and np.isfinite(f)
                archive["X"].append(x.copy())
                archive["G"].append(gv.copy())
                archive["F"].append(f if np.isfinite(f) else np.nan)
                archive["feas"].append(bool(feas))
                if feas and f < best["f"]:
                    best["f"], best["x"] = f, x.copy()
                if (not feas) and viol < best_viol["v"]:
                    best_viol["v"], best_viol["x"], best_viol["f"] = viol, x.copy(), f
            viol = float(gv.sum())
            feas = (viol <= 0.0) and np.isfinite(f)
            return {"x": x, "f": f, "g": gv, "viol": viol, "feas": feas, "s": s}

        def ridge(Z, Y, lam=1e-3):
            A = Z.T @ Z + lam * np.eye(Z.shape[1])
            return np.linalg.solve(A, Z.T @ Y)

        def fit_model():
            m = state["m"]
            n = len(archive["X"])
            if m is None or m == 0 or n < max(2 * dim + 5, 20):
                model["Wg"] = model["Wi"] = None
                model["Wf"] = None
                return
            lo = max(0, n - 400)
            Z = features(np.array(archive["X"][lo:n]))
            G = np.array(archive["G"][lo:n])
            I = (G > 0).astype(float)
            try:
                model["Wg"] = ridge(Z, G)
                model["Wi"] = ridge(Z, I)
            except np.linalg.LinAlgError:
                model["Wg"] = model["Wi"] = None
            feas_idx = [k for k in range(lo, n) if archive["feas"][k]]
            if len(feas_idx) >= max(8, dim):
                try:
                    Zf = features(np.array([archive["X"][k] for k in feas_idx]))
                    Ff = np.array([archive["F"][k] for k in feas_idx])
                    model["Wf"] = ridge(Zf, Ff)
                except np.linalg.LinAlgError:
                    model["Wf"] = None
            else:
                model["Wf"] = None

        def predicted_viol(X, w):
            Z = features(np.array(X))
            P = np.maximum(Z @ model["Wg"], 0.0)
            Q = np.clip(Z @ model["Wi"], 0.0, 1.0)
            pv = (w * (P + 0.5 * Q)).sum(axis=1)
            risk = -np.log(np.clip(1.0 - Q, 1e-3, 1.0)).sum(axis=1)
            return pv, risk

        def zs(a):
            return (a - a.mean()) / (a.std() + 1e-12)

        def predicted_score(X, w):
            if model["Wg"] is None or w.size != model["Wg"].shape[1]:
                return np.random.rand(len(X))
            pv, risk = predicted_viol(X, w)
            score = zs(pv) + 0.5 * zs(risk)
            if model["Wf"] is not None:
                Z = features(np.array(X))
                pf = Z @ model["Wf"]
                rank = np.argsort(np.argsort(pf)) / max(len(pf) - 1, 1)
                score = score + 0.3 * rank
            return score

        def key(ind, w):
            if ind["feas"]:
                return (0, ind["f"])
            if w.size == ind["g"].size and w.size > 0:
                return (1, float(np.sum(w * ind["g"])))
            return (1, ind["viol"])

        def sort_pop(lst):
            w = weights()
            lst.sort(key=lambda ind: key(ind, w))
            return w

        def random_x():
            x = np.zeros(dim)
            if n_c:
                x[cont] = np.random.uniform(lb[cont], ub[cont])
            for i in disc:
                x[i] = np.random.randint(0, n_values[i])
            return x

        def mutate_discrete(v, i):
            nv = n_values[i]
            if nv <= 1:
                return v
            if is_binary.get(i, False) or nv == 2:
                return 1.0 - v
            if is_ordinal.get(i, False) and np.random.rand() < 0.5:
                step = 1 if np.random.rand() < 0.5 else -1
                nxt = v + step
                if 0 <= nxt <= nv - 1:
                    return float(nxt)
                return float(v - step) if 0 <= v - step <= nv - 1 else float(v)
            choices = [c for c in range(nv) if c != int(v)]
            return float(np.random.choice(choices))

        def make_child(p1, p2):
            mask = np.random.rand(dim) < 0.5
            x = np.where(mask, p1["x"], p2["x"]).astype(float)
            s = 0.5 * (p1["s"] + p2["s"])
            s = float(np.clip(s * np.exp(tau * np.random.randn()), 1e-3, 0.5))
            if n_c:
                m = np.random.rand(n_c) < pm_c
                if not m.any():
                    m[np.random.randint(n_c)] = True
                x[cont] = x[cont] + m * s * span[cont] * np.random.randn(n_c)
            for i in disc:
                if np.random.rand() < pm_d:
                    x[i] = mutate_discrete(x[i], i)
            return x, s

        def local_move(base, sg):
            x = np.array(base, dtype=float).copy()
            if n_c:
                m = np.random.rand(n_c) < 0.3
                if not m.any():
                    m[np.random.randint(n_c)] = True
                x[cont] = x[cont] + m * sg * span[cont] * np.random.randn(n_c)
            for i in disc:
                if np.random.rand() < 0.05:
                    x[i] = mutate_discrete(x[i], i)
            return x

        def grad_step(base, w):
            # Pursue the two most frequently violated constraints (dominant one doubled).
            if model["Wg"] is None or n_c == 0 or w.size != model["Wg"].shape[1]:
                return local_move(base, 0.02)
            Wc = model["Wg"][:-1]
            d = Wc @ w
            order_h = np.argsort(-state["hist"])
            d = d + 2.0 * Wc[:, order_h[0]]
            if len(order_h) > 1:
                d = d + 1.0 * Wc[:, order_h[1]]
            dc = d[cont]
            nrm = np.linalg.norm(dc)
            if nrm < 1e-12:
                return local_move(base, 0.02)
            u0 = (np.asarray(base, dtype=float)[cont] - lb[cont]) / span[cont]
            alphas = [0.03, 0.1, 0.25, 0.5]
            cands = []
            for a in alphas:
                u = np.clip(u0 - a * dc / nrm + 0.01 * np.random.randn(n_c), 0.0, 1.0)
                x = np.array(base, dtype=float).copy()
                x[cont] = lb[cont] + u * span[cont]
                for i in disc:
                    if np.random.rand() < 0.05:
                        x[i] = mutate_discrete(x[i], i)
                cands.append(repair(x))
            pv, risk = predicted_viol(cands, w)
            k = int(np.argmin(pv + 0.5 * risk))
            return cands[k]

        def interpolate(a, b):
            t = np.random.rand() ** 0.7
            x = np.array(a, dtype=float).copy()
            if n_c:
                x[cont] = a[cont] + t * (b[cont] - a[cont])
                x[cont] += 0.01 * span[cont] * np.random.randn(n_c)
            mask = np.random.rand(dim) < 0.5
            for i in disc:
                x[i] = a[i] if mask[i] else b[i]
            return x

        def tournament(pop, w):
            a, b = np.random.randint(len(pop)), np.random.randint(len(pop))
            ka, kb = key(pop[a], w), key(pop[b], w)
            return pop[a] if ka <= kb else pop[b]

        mu = max(8, min(20, dim + 2))
        lam = 2 * mu
        n_pool = 6 * lam
        init_n = min(self.budget, max(30, 5 * dim))
        stall_limit = 12

        pop = []
        for _ in range(init_n):
            if self.evals >= self.budget:
                break
            pop.append(evaluate(random_x(), 0.3))
        if not pop:
            return np.inf, None
        w = sort_pop(pop)
        pop = pop[:mu]
        prev_best_key = key(pop[0], w)
        stall = 0

        while self.evals < self.budget:
            before_evals = self.evals
            w = weights()
            fit_model()
            inc = best["x"]
            elite = best_viol["x"]
            g_frac = 0.12 + 0.2 * (1.0 - feas_rate())
            f_frac = 0.12 if inc is not None else 0.0

            pool = []
            for _ in range(n_pool):
                u = np.random.rand()
                if inc is not None and u < 0.12:
                    pool.append((local_move(inc, sig["loc"]), 0.02))
                elif inc is not None and u < 0.12 + f_frac:
                    # feasible-preserving local search around incumbent
                    pool.append((local_move(inc, sig["feas"]), 0.01))
                elif elite is not None and u < 0.12 + f_frac + g_frac:
                    pool.append((grad_step(elite, w), 0.05))
                elif inc is not None and elite is not None and u < 0.12 + f_frac + g_frac + 0.15:
                    pool.append((interpolate(inc, elite), 0.05))
                elif elite is not None and u < 0.12 + f_frac + g_frac + 0.25:
                    pool.append((local_move(elite, 0.05), 0.05))
                else:
                    p1 = tournament(pop, w)
                    p2 = tournament(pop, w)
                    x, s = make_child(p1, p2)
                    pool.append((x, s))
            X = [repair(c[0]) for c in pool]
            sc = predicted_score(X, w) + 1e-9 * np.random.rand(len(X))
            order = np.argsort(sc)

            chosen = list(order[: max(1, lam - 1)])
            rest = order[max(1, lam - 1):]
            if len(rest) > 0:
                chosen.append(int(np.random.choice(rest)))

            best_before = best["f"]
            best_before_viol = best_viol["v"]
            offspring = []
            n_feas_local = 0
            n_local = 0
            for idx in chosen:
                if self.evals >= self.budget:
                    break
                x, s = pool[idx]
                ev = evaluate(x, s)
                offspring.append(ev)
                if s == 0.01:
                    n_local += 1
                    n_feas_local += int(ev["feas"])

            improved = (best["f"] < best_before) or (best_viol["v"] < best_before_viol)
            if inc is not None:
                sig["loc"] = float(np.clip(sig["loc"] * (1.2 if improved else 0.93), 1e-4, 0.1))
            if n_local > 0:
                # grow feasible-local step when many children remain feasible, shrink otherwise
                rate = n_feas_local / n_local
                factor = 1.15 if rate >= 0.5 else (0.9 if rate < 0.2 else 1.0)
                sig["feas"] = float(np.clip(sig["feas"] * factor, 1e-4, 0.05))

            if offspring:
                combined = pop + offspring
                w = sort_pop(combined)
                pop = combined[:mu]

                cur_key = key(pop[0], w)
                if cur_key < prev_best_key:
                    prev_best_key = cur_key
                    stall = 0
                else:
                    stall += 1

            if stall >= stall_limit and self.evals < self.budget:
                new_pop = []
                if best["x"] is not None:
                    new_pop.append(evaluate(best["x"], 0.1))
                if best_viol["x"] is not None and self.evals < self.budget:
                    new_pop.append(evaluate(best_viol["x"], 0.2))
                n_fresh = min(mu, self.budget - self.evals)
                for _ in range(n_fresh):
                    if self.evals >= self.budget:
                        break
                    new_pop.append(evaluate(random_x(), 0.3))
                if new_pop:
                    w = sort_pop(new_pop)
                    pop = new_pop[:mu]
                    prev_best_key = key(pop[0], w)
                stall = 0

            if self.evals == before_evals and self.evals < self.budget:
                pop.append(evaluate(random_x(), 0.3))
                w = sort_pop(pop)
                pop = pop[:mu]

        if best["x"] is not None:
            return best["f"], best["x"]
        return best_viol["f"], best_viol["x"]