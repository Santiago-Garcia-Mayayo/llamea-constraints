import numpy as np


class Algorithm:
    """
    (mu + lambda) mixed-integer ES with six adaptively selected operators:
      0: GA-style type-aware mutation/crossover
      1: DE-style difference moves (continuous) with integer crossover
      2: dominant-constraint pull toward feasible archive points or the incumbent
      3: local refinement of the incumbent
      4: archive-based integer sensitivity move for the dominant violated constraint
      5: bottleneck line search: interpolate an infeasible parent toward a feasible
         archive point that satisfies the dominant constraint, at several step sizes
    Constraint weights use EMA violation frequencies (squared) and scale-normalised violations.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng()
        d = self.dim
        budget = self.budget

        lb = np.asarray(func.bounds.lb, dtype=float).reshape(-1)[:d]
        ub = np.asarray(func.bounds.ub, dtype=float).reshape(-1)[:d]
        vtypes = [str(v).lower() for v in func.variable_types][:d]
        n_values = getattr(func, "n_values", None)
        if n_values is None:
            n_values = np.zeros(d)
        n_values = np.asarray(n_values, dtype=float).reshape(-1)[:d]

        is_int = np.array([v != "continuous" for v in vtypes])
        is_bin = np.array([v == "binary" for v in vtypes])
        is_ord = np.array([v == "ordinal" for v in vtypes])
        cont = ~is_int
        nv = np.maximum(np.where(is_int, n_values, 1).astype(int), 1)

        lo = np.where(is_int, 0.0, lb)
        hi = np.where(is_int, nv - 1.0, ub)
        span = np.where(is_int, np.maximum(nv - 1.0, 1.0), np.maximum(ub - lb, 1e-12))
        int_idx = np.where(is_int)[0]

        NOPS = 6

        def repair(X):
            X = np.clip(X, lo, hi)
            X[:, is_int] = np.round(X[:, is_int])
            return X

        def sample(n):
            X = np.zeros((n, d))
            for j in range(d):
                if is_int[j]:
                    X[:, j] = rng.integers(0, nv[j], size=n)
                else:
                    strata = (rng.permutation(n) + rng.random(n)) / n
                    X[:, j] = lb[j] + strata * (ub[j] - lb[j])
            return repair(X)

        st = {"count": 0, "best_f": np.inf, "best_x": None}
        arch_X, arch_G = [], []
        cw = {"m": None, "ema": None, "sum_pos": None, "npos": None}
        succ = np.ones(NOPS)
        tries = np.full(NOPS, 2.0)

        def clean_g(g):
            g = np.asarray(g, dtype=float).ravel()
            g = np.where(np.isfinite(g), g, 1e6)
            return np.maximum(g, 0.0)

        def update_stats(g):
            pos = g > 0
            if cw["m"] is None:
                m = g.shape[0]
                cw["m"] = m
                cw["ema"] = pos.astype(float)
                cw["sum_pos"] = np.zeros(m)
                cw["npos"] = np.zeros(m)
            else:
                cw["ema"] = 0.9 * cw["ema"] + 0.1 * pos
            cw["sum_pos"] += np.where(pos, g, 0.0)
            cw["npos"] += pos

        def weights():
            if cw["m"] is None:
                return np.ones(1), np.ones(1)
            w = 1.0 + 5.0 * cw["ema"] ** 2
            scale = np.where(cw["npos"] > 0, cw["sum_pos"] / np.maximum(cw["npos"], 1), 1.0)
            scale = np.maximum(scale, 1e-9)
            return w, scale

        def viol(G):
            if cw["m"] is None or G.shape[1] == 0:
                return G.sum(1)
            w, s = weights()
            return ((G / s) * w).sum(1)

        def evaluate(X):
            F, G = [], []
            for x in X:
                if st["count"] >= budget:
                    break
                f, g = func(x)
                st["count"] += 1
                f = float(np.asarray(f, dtype=float).reshape(-1)[0])
                g = clean_g(g)
                if cw["m"] is not None and g.shape[0] != cw["m"]:
                    gg = np.zeros(cw["m"])
                    k = min(cw["m"], g.shape[0])
                    gg[:k] = g[:k]
                    g = gg
                if not np.isfinite(f):
                    f = 1e300
                    g = np.maximum(g, 1.0)
                if np.all(g <= 0.0) and f < st["best_f"]:
                    st["best_f"] = f
                    st["best_x"] = np.array(x, dtype=float).copy()
                update_stats(g)
                arch_X.append(np.array(x, dtype=float).copy())
                arch_G.append(g)
                F.append(f)
                G.append(g)
            if len(G) == 0:
                return np.array([]), np.zeros((0, 0))
            m = max(len(g) for g in G)
            Gm = np.zeros((len(G), m))
            for i, g in enumerate(G):
                Gm[i, :len(g)] = g
            return np.array(F, dtype=float), Gm

        def rank(X, F, G):
            V = viol(G)
            order = np.lexsort((F, V))
            return X[order], F[order], G[order]

        def pull_child(p, pg, A, AG, w, s, target):
            if target is not None:
                nb = target
            else:
                score = pg / s * w
                j = int(np.argmax(score))
                if pg[j] <= 0 or A.shape[0] == 0:
                    return None
                mask = AG[:, j] <= 0
                if not mask.any():
                    return None
                Cand = A[mask]
                zc = (Cand - lo) / span
                zp = (p - lo) / span
                dist = ((zc - zp) ** 2).sum(1)
                k = min(3, Cand.shape[0])
                idx = np.argsort(dist)[:k]
                nb = Cand[rng.choice(idx)]
            t = rng.uniform(0.3, 1.0)
            c = p.copy()
            c[cont] = p[cont] + t * (nb[cont] - p[cont])
            c[cont] += 0.02 * span[cont] * rng.standard_normal(cont.sum())
            take = rng.random(d) < 0.5
            c[is_int & take] = nb[is_int & take]
            return c

        def bottleneck_child(p, pg, A, AG, s, w):
            """Interpolate p toward a feasible-on-bottleneck archive point at several step sizes."""
            if A.shape[0] == 0:
                return None
            j = int(np.argmax(pg / s * w))
            if pg[j] <= 0:
                return None
            mask = AG[:, j] <= 0
            if not mask.any():
                return None
            Cand = A[mask]
            zc = (Cand - lo) / span
            zp = (p - lo) / span
            dist = ((zc - zp) ** 2).sum(1)
            k = min(4, Cand.shape[0])
            idx = np.argsort(dist)[:k]
            q = Cand[rng.choice(idx)]
            t = rng.choice([0.5, 0.75, 0.9])
            c = p.copy()
            c[cont] = p[cont] + t * (q[cont] - p[cont])
            take = rng.random(d) < 0.5
            c[is_int & take] = q[is_int & take]
            if not cont.any():
                c[is_int & ~take] = p[is_int & ~take]
            return c

        def sens_child(p, A, AG, j):
            """Set integer variables to the values most associated with low violation of j."""
            if A.shape[0] < 8 or int_idx.size == 0:
                return None
            A2 = A[-3000:]
            gj = AG[-3000:, j]
            gains = []
            for k in int_idx:
                vals = np.round(A2[:, k])
                cur = int(np.round(p[k]))
                cur_mask = vals == cur
                if cur_mask.sum() < 1:
                    continue
                cur_mean = gj[cur_mask].mean()
                best_v, best_m = None, cur_mean
                for v in np.unique(vals):
                    if int(v) == cur:
                        continue
                    msk = vals == v
                    if msk.sum() < 2:
                        continue
                    m = gj[msk].mean()
                    if m < best_m - 1e-12:
                        best_m, best_v = m, v
                if best_v is not None:
                    gains.append((cur_mean - best_m, k, best_v))
            if not gains:
                return None
            gains.sort(key=lambda t: -t[0])
            top = gains[:3]
            n_flip = rng.integers(1, min(2, len(top)) + 1)
            chosen = rng.choice(len(top), size=n_flip, replace=False)
            c = p.copy()
            for ci in chosen:
                _, k, v = top[ci]
                c[k] = v
            return c

        def ga_child(X):
            mu_ = X.shape[0]
            t1 = rng.integers(mu_, size=2).min()
            t2 = rng.integers(mu_, size=2).min()
            p1, p2 = X[t1], X[t2]
            c = p1.copy()
            if rng.random() < 0.5:
                mask = rng.random(d) < 0.5
                c[mask] = p2[mask]
            pm = rng.choice([1.0 / d, 3.0 / d, 0.5])
            mut = rng.random(d) < pm
            if not mut.any():
                mut[rng.integers(d)] = True
            mc = mut & cont
            if mc.any():
                s = 10 ** rng.uniform(-3.0, -0.3)
                if rng.random() < 0.1:
                    noise = rng.standard_cauchy(mc.sum())
                else:
                    noise = rng.standard_normal(mc.sum())
                c[mc] += s * span[mc] * noise
            for j in np.where(mut & is_int)[0]:
                if is_bin[j]:
                    c[j] = 1.0 - c[j]
                elif is_ord[j]:
                    if rng.random() < 0.7:
                        c[j] += rng.choice([-1.0, 1.0])
                    else:
                        c[j] = rng.integers(nv[j])
                else:
                    if nv[j] > 1:
                        cur = int(np.round(c[j]))
                        r = rng.integers(nv[j] - 1)
                        if r >= cur:
                            r += 1
                        c[j] = r
            return c

        def de_child(X):
            mu_ = X.shape[0]
            ia, ib, ic = rng.choice(mu_, size=3, replace=False)
            a, b, c_ = X[ia], X[ib], X[ic]
            c = a.copy()
            F_de = rng.uniform(0.4, 0.9)
            if np.any(cont):
                c[cont] = a[cont] + F_de * (b[cont] - c_[cont])
            cross = rng.random(d) < 0.5
            c[cross & is_int] = b[cross & is_int]
            return c

        def local_child(best):
            c = best.copy()
            ints = int_idx
            if rng.random() < 0.5 and ints.size > 0:
                k = rng.integers(1, min(3, ints.size) + 1)
                for j in rng.choice(ints, size=k, replace=False):
                    if nv[j] > 1:
                        if is_ord[j] and rng.random() < 0.7:
                            c[j] = np.clip(c[j] + rng.choice([-1.0, 1.0]), 0, nv[j] - 1)
                        else:
                            cur = int(np.round(c[j]))
                            r = rng.integers(nv[j] - 1)
                            if r >= cur:
                                r += 1
                            c[j] = r
            elif cont.any():
                sig = 10 ** rng.uniform(-3.0, -1.3)
                mask = cont & (rng.random(d) < 0.3)
                if not mask.any():
                    mask = cont.copy()
                c[mask] += sig * span[mask] * rng.standard_normal(mask.sum())
            return c

        def make_children(X, G, n):
            mu_ = X.shape[0]
            pfeas = np.all(G <= 0, axis=1)
            w, s = weights() if cw["m"] is not None else (np.ones(1), np.ones(1))
            if cw["m"] is not None and len(arch_X) > 0:
                A = np.array(arch_X)
                AG = np.array(arch_G)
            else:
                A = np.zeros((0, d))
                AG = np.zeros((0, 0))
            best = st["best_x"]

            score = (succ + 1.0) / (tries + 4.0)
            p = score / score.sum()
            p = 0.85 * p + 0.15 / NOPS
            if pfeas.all() or cw["m"] is None:
                p[2] = 0.0
                p[4] = 0.0
                p[5] = 0.0
            if best is None:
                p[3] = 0.0
            if int_idx.size == 0 or AG.shape[0] < 8:
                p[4] = 0.0
            if AG.shape[0] == 0:
                p[5] = 0.0
            if p.sum() <= 0:
                p = np.ones(NOPS) / NOPS
            p = p / p.sum()

            kids, ops = [], []
            for _ in range(n):
                k = int(rng.choice(NOPS, p=p))
                c = None
                if k == 2:
                    infeas = np.where(~pfeas)[0]
                    i = rng.choice(infeas)
                    target = best if (best is not None and rng.random() < 0.5) else None
                    c = pull_child(X[i], G[i], A, AG, w, s, target)
                elif k == 4:
                    infeas = np.where(~pfeas)[0]
                    i = rng.choice(infeas)
                    pg = G[i]
                    j = int(np.argmax(pg / s * w))
                    if pg[j] > 0:
                        c = sens_child(X[i], A, AG, j)
                elif k == 5:
                    infeas = np.where(~pfeas)[0]
                    i = rng.choice(infeas)
                    c = bottleneck_child(X[i], G[i], A, AG, s, w)
                elif k == 1 and mu_ >= 3:
                    c = de_child(X)
                elif k == 3 and best is not None:
                    c = local_child(best)
                if c is None:
                    k = 0
                    c = ga_child(X)
                kids.append(c)
                ops.append(k)
            return repair(np.array(kids)), np.array(ops, dtype=int)

        mu = int(np.clip(2 * d, 8, 40))
        lam = 2 * mu
        stall_limit = 15

        X = sample(mu)
        F, G = evaluate(X)
        X = X[:len(F)]
        if len(F) == 0:
            return np.inf, None
        X, F, G = rank(X, F, G)
        last_best = st["best_f"]
        stall = 0

        while st["count"] < budget:
            kids, ops = make_children(X, G, lam)
            Fc, Gc = evaluate(kids)
            if len(Fc) > 0:
                m_c = len(Fc)
                Xc = kids[:m_c]
                opsc = ops[:m_c]
                mP = max(G.shape[1], Gc.shape[1])
                Gp = np.zeros((G.shape[0], mP)); Gp[:, :G.shape[1]] = G
                Gq = np.zeros((Gc.shape[0], mP)); Gq[:, :Gc.shape[1]] = Gc
                allX = np.vstack([X, Xc])
                allF = np.concatenate([F, Fc])
                allG = np.vstack([Gp, Gq])
                V = viol(allG)
                order = np.lexsort((allF, V))
                sel = order[:mu]
                X, F, G = allX[sel], allF[sel], allG[sel]

                succ *= 0.98
                tries *= 0.98
                np.add.at(tries, opsc, 1.0)
                chosen = sel[sel >= mu] - mu
                if chosen.size > 0:
                    np.add.at(succ, opsc[chosen], 1.0)

            if st["best_f"] < last_best - 1e-12:
                last_best = st["best_f"]
                stall = 0
            else:
                stall += 1

            if stall >= stall_limit and st["count"] < budget:
                stall = 0
                keep = max(1, mu // 4)
                keepX, keepF, keepG = X[:keep], F[:keep], G[:keep]
                n_new = mu - keep
                newX = sample(n_new)
                newF, newG = evaluate(newX)
                newX = newX[:len(newF)]
                if len(newF) > 0:
                    mP = max(keepG.shape[1], newG.shape[1])
                    Kp = np.zeros((keepG.shape[0], mP)); Kp[:, :keepG.shape[1]] = keepG
                    Np = np.zeros((newG.shape[0], mP)); Np[:, :newG.shape[1]] = newG
                    aX = np.vstack([keepX, newX])
                    aF = np.concatenate([keepF, newF])
                    aG = np.vstack([Kp, Np])
                    X, F, G = rank(aX, aF, aG)
                    last_best = st["best_f"]

        if st["best_x"] is None:
            return np.inf, None
        return st["best_f"], st["best_x"]