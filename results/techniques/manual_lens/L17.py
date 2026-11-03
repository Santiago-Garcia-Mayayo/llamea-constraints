import numpy as np


class Algorithm:
    """
    Feasibility-rule mixed-integer EA with:
    - a free evaluation cache (repeated exact designs cost no budget),
    - an idle-generation guard that injects fresh designs when proposals are all cached,
    - self-adaptive Gaussian steps and five success-credited operators,
    - a feasible elite archive (diverse, deduplicated) that seeds restarts,
    - coordinate search with pattern extrapolation around the incumbent,
    - violation-guided local moves on every generation before the first feasible point.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng()
        d = self.dim
        budget = self.budget

        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        span = ub - lb
        vt = [str(t) for t in func.variable_types]

        is_int = np.array([t != "continuous" for t in vt], dtype=bool)
        cont_idx = np.where(~is_int)[0]
        int_idx = np.where(is_int)[0]
        nc, ni = len(cont_idx), len(int_idx)
        nv = np.ones(d, dtype=int)
        for i in int_idx:
            nv[i] = max(int(func.n_values[i]), 1)
        is_bin = np.array([vt[i] == "binary" for i in range(d)], dtype=bool)
        is_ord = np.array([vt[i] == "ordinal" for i in range(d)], dtype=bool)

        pm_int = min(0.5, max(1.0 / max(ni, 1), 0.1))
        pc = min(1.0, max(2.0 / max(nc, 1), 0.3))

        st = {"evals": 0, "best_f": np.inf, "best_x": None}
        cache = {}  # exact-design cache: key -> (f, v)
        archive = []  # list of (f, x) feasible elites
        ARCH_MAX = 30

        def add_archive(f, x):
            for fa, xa in archive:
                if fa == f and np.array_equal(xa, x):
                    return
            archive.append((f, np.array(x, copy=True)))
            archive.sort(key=lambda t: t[0])
            del archive[ARCH_MAX:]

        def repair(X):
            X = np.clip(X, lb, ub)
            if ni:
                X[:, int_idx] = np.clip(np.rint(X[:, int_idx]), 0, nv[int_idx] - 1)
            return X

        def key_of(x):
            k = np.array(x, dtype=float, copy=True)
            if ni:
                k[int_idx] = np.rint(k[int_idx])
            return k.tobytes()

        def rand_pop(n):
            X = np.empty((n, d))
            if nc:
                X[:, cont_idx] = rng.uniform(lb[cont_idx], ub[cont_idx], size=(n, nc))
            if ni:
                X[:, int_idx] = rng.integers(0, nv[int_idx], size=(n, ni))
            return X

        def lhs_pop(n):
            X = np.empty((n, d))
            if nc:
                perm = np.argsort(rng.random((n, nc)), axis=0)
                u = (perm + rng.random((n, nc))) / n
                X[:, cont_idx] = lb[cont_idx] + u * span[cont_idx]
            if ni:
                X[:, int_idx] = rng.integers(0, nv[int_idx], size=(n, ni))
            return X

        def mutate_int(i, v):
            if nv[i] <= 1:
                return v
            if is_bin[i]:
                return 1.0 - v
            if is_ord[i] and rng.random() < 0.7:
                nvv = v + (1.0 if rng.random() < 0.5 else -1.0)
                return float(np.clip(nvv, 0, nv[i] - 1))
            if is_ord[i]:
                return float(rng.integers(0, nv[i]))
            r = int(rng.integers(0, nv[i] - 1))
            if r >= int(v):
                r += 1
            return float(r)

        def evaluate(X):
            F, V = [], []
            for x in X:
                k = key_of(x)
                if k in cache:
                    f, v = cache[k]
                else:
                    if st["evals"] >= budget:
                        break
                    f, g = func(np.array(x, copy=True))
                    st["evals"] += 1
                    f = float(f)
                    if not np.isfinite(f):
                        f = np.inf
                    g = np.nan_to_num(np.asarray(g, dtype=float).ravel(), nan=1e9, posinf=1e9)
                    v = float(np.sum(np.maximum(g, 0.0)))
                    if v <= 0.0 and np.isfinite(f):
                        v = 0.0
                    else:
                        v = max(v, 1e-9)
                    cache[k] = (f, v)
                if v == 0.0:
                    add_archive(f, x)
                    if f < st["best_f"]:
                        st["best_f"] = f
                        st["best_x"] = np.array(x, copy=True)
                F.append(f)
                V.append(v)
            return np.array(F, dtype=float), np.array(V, dtype=float), len(F)

        def better(f1, v1, f0, v0):
            if v1 == 0.0 and v0 > 0.0:
                return True
            if v1 > 0.0 and v0 > 0.0:
                return v1 < v0
            if v1 > 0.0 and v0 == 0.0:
                return False
            return f1 < f0

        def rank(F, V):
            infeas = V > 0
            sec = np.where(infeas, V, F)
            return np.lexsort((sec, infeas))

        def eval1(y):
            F, V, m = evaluate(np.asarray(y, dtype=float)[None, :])
            if m == 0:
                return None
            return float(F[0]), float(V[0])

        def local_search(x0, f0, ls_budget):
            """Coordinate search with pattern extrapolation from a feasible x0."""
            x, fx = x0.copy(), f0
            step = 0.1 * span.copy()
            used = 0
            start_f = f0
            while used < ls_budget and st["evals"] < budget:
                improved = False
                sweep_start = x.copy()
                for i in rng.permutation(d):
                    if used >= ls_budget or st["evals"] >= budget:
                        break
                    if is_int[i]:
                        if nv[i] <= 1:
                            continue
                        if nv[i] <= 6:
                            cands = [float(v) for v in range(nv[i]) if v != int(x[i])]
                        else:
                            cands = [mutate_int(i, x[i]), mutate_int(i, x[i])]
                        rng.shuffle(cands)
                        for c in cands:
                            if used >= ls_budget or st["evals"] >= budget:
                                break
                            y = x.copy()
                            y[i] = c
                            res = eval1(y)
                            used += 1
                            if res is None:
                                return x, fx, fx < start_f
                            if better(res[0], 0.0, fx, 0.0):
                                x, fx = y, res[0]
                                improved = True
                                break
                    else:
                        for sgn in (1.0, -1.0):
                            if used >= ls_budget or st["evals"] >= budget:
                                break
                            y = x.copy()
                            y[i] = np.clip(x[i] + sgn * step[i], lb[i], ub[i])
                            if y[i] == x[i]:
                                continue
                            res = eval1(y)
                            used += 1
                            if res is None:
                                return x, fx, fx < start_f
                            if better(res[0], 0.0, fx, 0.0):
                                x, fx = y, res[0]
                                improved = True
                                step[i] = min(step[i] * 1.5, 0.3 * span[i])
                                break
                            else:
                                step[i] *= 0.8
                # pattern extrapolation along the successful sweep displacement
                if improved and nc and used < ls_budget and st["evals"] < budget:
                    disp = x - sweep_start
                    if np.any(disp[cont_idx] != 0):
                        y = x.copy()
                        y[cont_idx] = np.clip(x[cont_idx] + disp[cont_idx], lb[cont_idx], ub[cont_idx])
                        res = eval1(y)
                        used += 1
                        if res is None:
                            return x, fx, fx < start_f
                        if better(res[0], 0.0, fx, 0.0):
                            x, fx = y, res[0]
                if not improved:
                    if nc:
                        step[cont_idx] *= 0.5
                        if np.all(step[cont_idx] < 1e-4 * span[cont_idx]):
                            break
                    else:
                        break
            return x, fx, fx < start_f

        def violation_moves(x0, v0, n_try):
            """Before a feasible point exists: greedy violation-reducing moves."""
            x, vx = x0.copy(), v0
            used = 0
            while used < n_try and st["evals"] < budget:
                improved = False
                for i in rng.permutation(d):
                    if used >= n_try or st["evals"] >= budget:
                        break
                    y = x.copy()
                    if is_int[i]:
                        if nv[i] <= 1:
                            continue
                        y[i] = mutate_int(i, x[i])
                    else:
                        y[i] = np.clip(x[i] + rng.normal(0, 0.05) * span[i], lb[i], ub[i])
                    res = eval1(y)
                    used += 1
                    if res is None:
                        return x, vx, False
                    if res[1] < vx:
                        x, vx = y, res[1]
                        improved = True
                if vx == 0.0:
                    break
                if not improved and used >= d:
                    break
            return x, vx, vx == 0.0

        mu = int(np.clip(2 * d, 12, 40))

        X0 = repair(lhs_pop(min(budget, max(5 * mu, 60))))
        F, V, m = evaluate(X0)
        if m == 0:
            return st["best_f"], st["best_x"]
        X0 = X0[:m]
        o = rank(F, V)[:mu]
        X, F, V = X0[o], F[o], V[o]
        S = rng.uniform(0.05, 0.3, size=len(X))

        ops_sr = np.full(5, 0.2)
        last_best = np.inf
        stall = 0
        idle = 0
        last_ls_eval = -10**9
        ls_every = max(4 * mu, 80)
        ls_base = max(20, 3 * d)
        ls_budget = ls_base
        ls_pending = False

        while st["evals"] < budget and len(X) > 0:
            evals_before = st["evals"]
            lam = int(max(1, min(2 * mu, budget - st["evals"])))
            n_par = len(X)

            w = np.maximum(ops_sr, 0.05) + 0.02
            p = w / w.sum()
            ops_c = rng.choice(5, size=lam, p=p)
            if idle >= 3:
                # all recent proposals were cached: force fresh exploration
                ops_c = np.full(lam, 4)

            def tour():
                return rng.integers(0, n_par, size=(lam, 2)).min(axis=1)

            p1, p2 = tour(), tour()
            p3 = rng.integers(0, n_par, size=lam)
            bestX = st["best_x"] if st["best_x"] is not None else X[0]

            C = np.empty((lam, d))
            Sc = np.empty(lam)
            for k in range(lam):
                op = ops_c[k]
                a, b, c = p1[k], p2[k], p3[k]
                if op == 0:
                    x = np.where(rng.random(d) < 0.5, X[a], X[b]).astype(float)
                    s = 0.5 * (S[a] + S[b]) * np.exp(0.3 * rng.standard_normal())
                    s = float(np.clip(s, 1e-3, 0.5))
                    if nc:
                        M = rng.random(nc) < pc
                        x[cont_idx] += M * rng.standard_normal(nc) * s * span[cont_idx]
                    for i in int_idx:
                        if rng.random() < pm_int:
                            x[i] = mutate_int(i, x[i])
                elif op == 1:
                    Fm = rng.uniform(0.4, 0.9)
                    base = X[a].astype(float)
                    x = base.copy()
                    if nc:
                        x[cont_idx] = (base[cont_idx]
                                       + Fm * (bestX[cont_idx] - base[cont_idx])
                                       + Fm * (X[b, cont_idx] - X[c, cont_idx]))
                    cm = rng.random(d) < 0.9
                    cm[rng.integers(d)] = True
                    x = np.where(cm, x, base)
                    for i in int_idx:
                        if cm[i]:
                            x[i] = bestX[i] if rng.random() < 0.5 else X[b, i]
                        if rng.random() < 0.5 * pm_int:
                            x[i] = mutate_int(i, x[i])
                    s = float(S[a])
                elif op == 2:
                    x = (bestX if rng.random() < 0.5 else X[a]).astype(float).copy()
                    s = float(np.clip(0.3 * S[a], 1e-3, 0.5))
                    if nc:
                        M = rng.random(nc) < pc
                        x[cont_idx] += M * rng.standard_normal(nc) * s * span[cont_idx]
                    if ni:
                        nm = int(rng.integers(1, min(2, ni) + 1))
                        for i in rng.choice(int_idx, size=nm, replace=False):
                            x[i] = mutate_int(i, x[i])
                    s = float(S[a])
                elif op == 3:
                    x = X[a].astype(float).copy()
                    if nc:
                        g = rng.uniform(0.5, 1.5)
                        x[cont_idx] += g * (X[b, cont_idx] - X[c, cont_idx])
                        x[cont_idx] += 0.02 * rng.standard_normal(nc) * span[cont_idx]
                    for i in int_idx:
                        if rng.random() < pm_int:
                            x[i] = X[b, i] if rng.random() < 0.5 else mutate_int(i, x[i])
                    s = float(S[a])
                else:
                    x = rand_pop(1)[0]
                    s = 0.2
                C[k] = x
                Sc[k] = s

            C = repair(C)
            Fc, Vc, m = evaluate(C)
            if m == 0:
                break
            C, Sc, ops_c = C[:m], Sc[:m], ops_c[:m]

            # idle detection: no new real evaluations consumed budget this generation
            if st["evals"] == evals_before:
                idle += 1
            else:
                idle = 0

            XX = np.vstack([X, C])
            FF = np.concatenate([F, Fc])
            VV = np.concatenate([V, Vc])
            SS = np.concatenate([S, Sc])
            o = rank(FF, VV)[:mu]

            kept = o[o >= n_par] - n_par
            for op in range(5):
                t = int(np.sum(ops_c == op))
                if t > 0:
                    sc = int(np.sum(ops_c[kept] == op))
                    ops_sr[op] = 0.7 * ops_sr[op] + 0.3 * (sc / t)

            X, F, V, S = XX[o], FF[o], VV[o], SS[o]

            # Violation-guided moves while no feasible point is known (every generation)
            if st["best_x"] is None and st["evals"] < budget and len(X) > 0:
                xv, vv, ok = violation_moves(X[0], V[0], max(d, 8))
                if vv > 0 or ok:
                    XX = np.vstack([X, xv[None, :]])
                    FF = np.concatenate([F, [np.inf]])
                    VV = np.concatenate([V, [vv]])
                    SS = np.concatenate([S, [0.1]])
                    o = rank(FF, VV)[:mu]
                    X, F, V, S = XX[o], FF[o], VV[o], SS[o]

            # Detect new incumbent -> schedule local search
            if st["best_f"] < last_best - 1e-12:
                last_best = st["best_f"]
                stall = 0
                ls_pending = True
            else:
                stall += 1

            # Local refinement of the incumbent
            if (st["best_x"] is not None and st["evals"] < budget
                    and (ls_pending or st["evals"] - last_ls_eval >= ls_every)):
                ls_pending = False
                last_ls_eval = st["evals"]
                xb = st["best_x"].copy()
                fb = st["best_f"]
                xr, fr, succ = local_search(xb, fb, ls_budget)
                if succ:
                    ls_budget = min(ls_base * 4, int(ls_budget * 1.25) + 1)
                    ls_pending = True
                else:
                    ls_budget = max(ls_base // 2, int(ls_budget * 0.8))
                if not np.array_equal(xr, xb) and st["evals"] <= budget:
                    add_archive(fr, xr)
                    XX = np.vstack([X, xr[None, :]])
                    FF = np.concatenate([F, [fr]])
                    VV = np.concatenate([V, [0.0]])
                    SS = np.concatenate([S, [float(np.clip(0.3 * np.mean(S), 1e-3, 0.5))]])
                    o = rank(FF, VV)[:mu]
                    X, F, V, S = XX[o], FF[o], VV[o], SS[o]

            # Restarts: keep the best, reseed from archive elites and the incumbent
            if (stall >= 15 or idle >= 6) and st["evals"] < budget and mu > 3:
                nkeep = max(3, mu // 4)
                nre = mu - nkeep
                n_elite = nre // 3
                n_near = nre // 3
                parts = []
                if st["best_x"] is not None and n_near > 0:
                    base = st["best_x"].astype(float)
                    Xn_arr = np.repeat(base[None, :], n_near, axis=0)
                    if nc:
                        Xn_arr[:, cont_idx] += (rng.standard_normal((n_near, nc))
                                                * 0.1 * span[cont_idx])
                    for r in range(n_near):
                        for i in int_idx:
                            if rng.random() < 0.15:
                                Xn_arr[r, i] = mutate_int(i, Xn_arr[r, i])
                    parts.append(repair(Xn_arr))
                if archive and n_elite > 0:
                    picks = rng.integers(0, len(archive), size=n_elite)
                    Xe = np.array([archive[j][1] for j in picks], dtype=float)
                    if nc:
                        Xe[:, cont_idx] += (rng.standard_normal((n_elite, nc))
                                            * 0.05 * span[cont_idx])
                    for r in range(n_elite):
                        for i in int_idx:
                            if rng.random() < 0.1:
                                Xe[r, i] = mutate_int(i, Xe[r, i])
                    parts.append(repair(Xe))
                n_rand = nre - sum(len(p) for p in parts)
                if n_rand > 0:
                    parts.append(repair(lhs_pop(n_rand)))
                Xnew = np.vstack(parts) if parts else np.empty((0, d))
                if len(Xnew) > 0:
                    Fr, Vr, mr = evaluate(Xnew)
                    if mr > 0:
                        Xnew = Xnew[:mr]
                        X = np.vstack([X[:nkeep], Xnew])
                        F = np.concatenate([F[:nkeep], Fr])
                        V = np.concatenate([V[:nkeep], Vr])
                        S = np.concatenate([S[:nkeep], np.full(mr, 0.2)])
                        o = rank(F, V)
                        X, F, V, S = X[o], F[o], V[o], S[o]
                stall = 0
                idle = 0

        return st["best_f"], st["best_x"]