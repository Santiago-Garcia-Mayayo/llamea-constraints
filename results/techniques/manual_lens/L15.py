import numpy as np


class Algorithm:
    """
    Feasibility-first mixed-integer DE with:
    - Per-constraint kNN surrogate on log(1+violation), weighted by chronic-violation importance,
      plus a kNN feasibility-probability term in the screening score.
    - Adaptive screening (3x over-generation); exploitation share grows with recent feasible rate.
    - Chronic-aware repair: violated children that break chronic constraints are repaired first,
      by line search toward either the incumbent or the nearest feasible archive point.
    - Deb's rules with weighted violation; evaluation cache (duplicates cost no budget).
    - Continuous DE/current-to-best and rand/1 with binomial crossover; discrete uniform crossover
      with type-aware mutation; local Gaussian refinement with 1/5 step rule.
    - Stagnation restarts that keep feasible elites.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget

        lb = np.asarray(func.bounds.lb, dtype=float).reshape(d)
        ub = np.asarray(func.bounds.ub, dtype=float).reshape(d)
        vtypes = list(func.variable_types)
        is_cont = np.array([t == "continuous" for t in vtypes], dtype=bool)
        is_disc = ~is_cont
        is_bin = np.array([t == "binary" for t in vtypes], dtype=bool)
        is_cat = np.array([t == "categorical" for t in vtypes], dtype=bool)
        is_ord = is_disc & ~is_bin & ~is_cat

        nv = np.ones(d)
        nv[is_disc] = np.maximum(np.round(ub[is_disc] - lb[is_disc]), 1)
        n_disc = int(is_disc.sum())
        disc_idx = np.where(is_disc)[0]
        pm = min(0.5, max(1.0 / max(n_disc, 1), 0.05))
        cont_scale = (ub - lb) / 2.0
        rng_span = np.maximum(ub - lb, 1e-12)

        st = {"used": 0, "best_f": np.inf, "best_x": None}
        cache = {}
        arc_X, arc_G = [], []

        # ---------- helpers ----------
        def repair(X):
            X = np.clip(X, lb, ub)
            if n_disc:
                X[:, is_disc] = np.clip(np.rint(X[:, is_disc]), 0, nv[is_disc] - 1)
            return X

        def sample(n):
            X = np.random.uniform(lb, ub, size=(n, d))
            if n_disc:
                U = np.random.rand(n, n_disc)
                X[:, is_disc] = np.floor(U * nv[is_disc])
            return X

        def evaluate(X):
            X = repair(np.array(X, dtype=float, copy=True))
            Fs, Gs, Xs = [], [], []
            for i in range(len(X)):
                x = X[i].copy()
                key = x.tobytes()
                if key in cache:
                    f, g = cache[key]
                else:
                    if st["used"] >= budget:
                        break
                    raw_f, raw_g = func(x)
                    st["used"] += 1
                    f = float(raw_f)
                    if not np.isfinite(f):
                        f = np.inf
                    g = np.nan_to_num(np.asarray(raw_g, dtype=float).ravel(),
                                      nan=1e6, posinf=1e6, neginf=0.0)
                    g = np.maximum(g, 0.0)
                    if not np.isfinite(f):
                        g = np.maximum(g, 1e-6)
                    cache[key] = (f, g)
                    arc_X.append(x.copy())
                    arc_G.append(g)
                    if g.sum() <= 0.0 and f < st["best_f"]:
                        st["best_f"] = f
                        st["best_x"] = x.copy()
                Xs.append(x)
                Fs.append(f)
                Gs.append(g)
            if not Xs:
                return None
            return np.array(Xs), np.array(Fs, dtype=float), np.array(Gs, dtype=float)

        def scale(X):
            return (X - lb) / rng_span

        def chronic_mask(G):
            if len(G) == 0:
                return np.zeros(0, dtype=bool)
            return np.mean(G > 0, axis=0) > 0.3

        def cons_weights(G):
            nc = G.shape[1] if G.ndim == 2 else 0
            if len(G) == 0 or nc == 0:
                return np.ones(nc)
            freq = np.mean(G > 0, axis=0)
            mag = np.mean(G, axis=0)
            mag = mag / (mag.max() + 1e-12)
            chronic = (freq > 0.3).astype(float)
            return 1.0 + 4.0 * freq + 2.0 * freq ** 2 + mag + 2.0 * chronic

        def predict_score(Xc, w):
            """Weighted per-constraint kNN log-violation plus kNN infeasibility-probability term."""
            A = np.array(arc_X)
            Gar = np.array(arc_G)
            Ga = np.log1p(Gar)
            feasA = Gar.sum(axis=1) <= 0.0
            Za = scale(A)
            Zc = scale(Xc)
            D = (Zc ** 2).sum(1)[:, None] + (Za ** 2).sum(1)[None, :] - 2.0 * Zc @ Za.T
            k = min(7, len(A))
            nb = np.argpartition(D, k - 1, axis=1)[:, :k]
            pred_w = Ga[nb].mean(axis=1) @ w
            pfeas = feasA[nb].mean(axis=1)
            return pred_w + 0.5 * (1.0 - pfeas)

        def order(F, G, w):
            V = G.sum(axis=1)
            feas = V <= 0.0
            k1 = (~feas).astype(int)
            k2 = np.where(feas, F, G @ w)
            return np.lexsort((k2, k1))

        def mutate_disc(child, prob):
            for j in disc_idx:
                if np.random.rand() < prob:
                    if is_bin[j]:
                        child[j] = 1.0 - child[j]
                    elif is_ord[j]:
                        if np.random.rand() < 0.7:
                            child[j] = child[j] + (1.0 if np.random.rand() < 0.5 else -1.0)
                        else:
                            child[j] = np.random.randint(int(nv[j]))
                    else:
                        if nv[j] > 1:
                            cur = int(round(child[j]))
                            r = np.random.randint(int(nv[j]) - 1)
                            child[j] = r if r < cur else r + 1
            return child

        def breed(Xp, lam, sigma):
            n = len(Xp)
            children = np.empty((lam, d))
            have_best = st["best_x"] is not None
            for k in range(lam):
                base = Xp[k % n]
                a = min(n - 1, int(n * np.random.rand() ** 2))
                b = np.random.randint(n)
                c = np.random.randint(n)
                u = np.random.rand()
                if have_best and u < 0.2:
                    child = st["best_x"].copy()
                    noise = np.random.randn(d) * sigma * cont_scale
                    child[is_cont] += noise[is_cont]
                    child = mutate_disc(child, 1.0 / max(n_disc, 1))
                elif u < 0.3:
                    child = base.copy()
                    noise = np.random.randn(d) * sigma * cont_scale
                    child[is_cont] += noise[is_cont]
                    child = mutate_disc(child, 1.0 / max(n_disc, 1))
                else:
                    Fd = np.random.uniform(0.4, 0.9)
                    CR = np.random.uniform(0.5, 1.0)
                    if np.random.rand() < 0.5:
                        mut = base + Fd * (Xp[0] - base) + Fd * (Xp[a] - Xp[b])
                    else:
                        mut = Xp[a] + Fd * (Xp[b] - Xp[c])
                    cmask = np.random.rand(d) < CR
                    cmask[np.random.randint(d)] = True
                    child = base.copy()
                    cm = is_cont & cmask
                    child[cm] = mut[cm]
                    if n_disc:
                        dm = is_disc & (np.random.rand(d) < 0.5)
                        child[dm] = Xp[a][dm]
                        child = mutate_disc(child, pm)
                children[k] = child
            return children

        def screen(pool, lam, w, exploit_frac):
            if len(pool) <= lam:
                return pool
            if len(arc_X) == 0:
                return pool[np.random.permutation(len(pool))[:lam]]
            pred = predict_score(pool, w)
            n_exploit = int(round(exploit_frac * lam))
            ordr = np.argsort(pred)
            chosen = list(ordr[:n_exploit])
            rest = ordr[n_exploit:]
            n_rand = lam - n_exploit
            if n_rand > 0 and len(rest) > 0:
                chosen += list(np.random.choice(rest, size=min(n_rand, len(rest)), replace=False))
            return pool[np.array(chosen, dtype=int)]

        def nearest_feasible(Xq):
            """Nearest feasible archive point (scaled distance) for each query row; None if none."""
            Gar = np.array(arc_G)
            feasA = Gar.sum(axis=1) <= 0.0
            if not feasA.any():
                return None
            Af = np.array(arc_X)[feasA]
            Zf = scale(Af)
            Zq = scale(Xq)
            D = (Zq ** 2).sum(1)[:, None] + (Zf ** 2).sum(1)[None, :] - 2.0 * Zq @ Zf.T
            return Af[np.argmin(D, axis=1)]

        # ---------- initialisation ----------
        N = int(np.clip(5 * d, 20, 80))
        if budget <= N:
            evaluate(sample(budget))
            return st["best_f"], st["best_x"]

        Xp, Fp, Gp = evaluate(sample(N))
        sigma = 0.1
        last_best = np.inf
        stagn = 0
        feas_rate = 0.5

        # ---------- main loop ----------
        while st["used"] < budget:
            n_old = len(Xp)
            lam = min(n_old, budget - st["used"])
            if lam <= 0:
                break

            Gar_all = np.array(arc_G)
            w_pred = cons_weights(Gar_all)
            exploit_frac = float(np.clip(0.6 + 0.3 * feas_rate, 0.6, 0.9))

            pool = breed(Xp, 3 * lam, sigma)
            Xc = screen(pool, lam, w_pred, exploit_frac)
            out = evaluate(Xc)
            if out is None:
                break
            Xc, Fc, Gc = out
            if len(Gc):
                feas_rate = 0.7 * feas_rate + 0.3 * float(np.mean(Gc.sum(axis=1) <= 0.0))

            # Chronic-aware repair of near-feasible children
            if st["best_x"] is not None and st["used"] < budget and len(Xc) > 0:
                Vc = Gc.sum(axis=1)
                viol = np.where(Vc > 0.0)[0]
                if len(viol) > 0:
                    ch = chronic_mask(Gar_all)
                    if ch.any():
                        chronic_v = (Gc[:, ch] > 0).any(axis=1)
                    else:
                        chronic_v = np.zeros(len(Xc), dtype=bool)
                    # chronic violators first, then smallest total violation
                    ordv = np.lexsort((Vc[viol], ~chronic_v[viol]))
                    nsel = max(2, n_old // 5)
                    cand = viol[ordv][:nsel]
                    Xcand = Xc[cand]
                    near = nearest_feasible(Xcand)
                    use_best = np.random.rand(len(cand)) < 0.5
                    if near is None:
                        use_best[:] = True
                    T = np.where(use_best[:, None], st["best_x"][None, :], near if near is not None else st["best_x"][None, :])
                    t = np.random.choice(np.array([0.25, 0.5, 0.75]), size=(len(cand), 1))
                    Xm = Xcand + t * (T - Xcand)
                    out2 = evaluate(Xm)
                    if out2 is not None:
                        Xm2, Fm, Gm = out2
                        Xc = np.vstack([Xc, Xm2])
                        Fc = np.concatenate([Fc, Fm])
                        Gc = np.vstack([Gc, Gm])

            XA = np.vstack([Xp, Xc])
            FA = np.concatenate([Fp, Fc])
            GA = np.vstack([Gp, Gc])
            w = cons_weights(GA)
            idx = order(FA, GA, w)[:n_old]
            succ = float(np.mean(idx >= n_old))
            Xp, Fp, Gp = XA[idx], FA[idx], GA[idx]

            sigma = float(np.clip(sigma * (1.25 if succ > 0.2 else 0.85), 1e-3, 0.5))

            if st["best_f"] < last_best:
                last_best = st["best_f"]
                stagn = 0
            else:
                stagn += 1

            # partial restart: keep feasible elites and incumbent
            if stagn >= 30 and st["used"] + n_old < budget:
                k = max(2, n_old // 4)
                fresh = evaluate(sample(n_old - k))
                if fresh is None:
                    break
                Xr, Fr, Gr = fresh
                XA = np.vstack([Xp[:k], Xr])
                FA = np.concatenate([Fp[:k], Fr])
                GA = np.vstack([Gp[:k], Gr])
                w = cons_weights(GA)
                idx = order(FA, GA, w)
                Xp, Fp, Gp = XA[idx], FA[idx], GA[idx]
                stagn = 0
                sigma = 0.1

        return st["best_f"], st["best_x"]