import numpy as np


class Algorithm:
    """
    Feasibility-first mixed-integer DE with:
    - EMA of violation frequency AND magnitude per constraint; the chronic violator receives a boost;
    - chronic-aware bisection: infeasible members are pulled toward the normalised-nearest feasible
      archive elite (cheapest boundary crossing), with an adaptive interpolation fraction;
    - diverse elite archive of feasible points, each with its own 1/5-rule step size;
    - current-to-pbest/1 for continuous variables, label-based donors for categorical/binary;
    - jDE-style self-adaptation of F and CR; restarts seeded from the archive on stagnation.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    # ------------------------------------------------------------ utilities
    def _repair(self, x):
        x = np.clip(x, self.lb, self.ub)
        if self.any_int:
            x[self.is_int] = np.clip(np.rint(x[self.is_int]), 0,
                                     self.nvals[self.is_int] - 1)
        return x

    def _random_point(self):
        return self._repair(np.random.uniform(self.lb, self.ub))

    def _perturb(self, xb, scale):
        x = xb.copy()
        c = self.is_cont
        if c.any():
            x[c] += np.random.normal(0.0, scale, c.sum()) * (self.ub[c] - self.lb[c]) / 2.0
        if self.any_int:
            m = self.is_int & (np.random.rand(self.dim) < 0.15)
            if m.any():
                x[m] = np.floor(np.random.rand(m.sum()) * self.nvals[m])
        return self._repair(x)

    def _local_step(self, xb, sig):
        """Small exploitation move: Gaussian on a few continuous vars or +/-1 / resample on ints."""
        x = xb.copy()
        if self.cont_idx.size and (np.random.rand() < 0.6 or not self.any_int):
            n = self.cont_idx.size
            k = self.cont_idx[np.random.rand(n) < max(1.0 / n, 0.2)]
            if k.size == 0:
                k = self.cont_idx[[np.random.randint(n)]]
            x[k] += np.random.normal(0.0, 1.0, k.size) * sig * (self.ub[k] - self.lb[k]) / 2.0
        elif self.any_int:
            j = self.int_idx[np.random.randint(self.int_idx.size)]
            if self.is_cat[j]:
                x[j] = np.floor(np.random.rand() * self.nvals[j])
            else:
                x[j] += np.random.choice([-1.0, 1.0])
        return self._repair(x)

    def _bisect_move(self, xf, xi):
        """Interpolate between a feasible elite xf and an infeasible member xi, then jitter."""
        # adaptive fraction: biased toward the feasible side when bisection has been successful
        lo, hi = self.bis_range
        t = np.random.uniform(lo, hi)
        x = xf + t * (xi - xf)
        c = self.is_cont
        if c.any():
            x[c] += np.random.normal(0.0, 0.02, c.sum()) * (self.ub[c] - self.lb[c]) / 2.0
        if self.any_int:
            pick = np.random.rand(self.dim) < 0.5
            x[self.is_int] = np.where(pick, xf, xi)[self.is_int]
        return self._repair(x)

    # ------------------------------------------------------- archive / evaluation
    def _archive_add(self, f, x):
        for el in self.archive:
            if el["f"] == f and np.array_equal(el["x"], x):
                return
        self.archive.append({"f": f, "x": x.copy(), "sig": 0.1, "g": None})
        self.archive.sort(key=lambda t: t["f"])
        del self.archive[self.A:]
        self.best_f = self.archive[0]["f"]
        self.best_x = self.archive[0]["x"].copy()

    def _eval(self, x):
        f, g = self.func(x)
        self.used += 1
        self.n_eval += 1
        g = np.atleast_1d(np.asarray(g, dtype=float)).ravel()
        g = np.nan_to_num(g, nan=1e6, posinf=1e6, neginf=0.0)
        g = np.maximum(g, 0.0)
        f = float(f)
        if not np.isfinite(f):
            f = np.inf
        if self.ema is None:
            self.m = len(g)
            self.ema = np.zeros(self.m)
            self.vema = np.zeros(self.m)
            self.gscale = np.full(self.m, 1e-12)
        alpha = max(0.02, 1.0 / self.n_eval)
        self.ema = (1.0 - alpha) * self.ema + alpha * (g > 0)
        self.gscale = np.maximum(self.gscale, g)
        scaled_now = g / np.maximum(self.gscale, 1e-12)
        self.vema = (1.0 - alpha) * self.vema + alpha * scaled_now
        if np.isfinite(f) and g.sum() <= 0.0:
            self._archive_add(f, x)
        return g, f

    # ------------------------------------------------------- ranking
    def _chronic(self):
        """Index of the chronic violator: frequency x normalised violation magnitude."""
        mag = self.vema / np.maximum(self.vema.max(), 1e-12)
        return int(np.argmax(self.ema * (1.0 + mag)))

    def _weights(self):
        if self.ema is None:
            return np.ones(1)
        mag = self.vema / np.maximum(self.vema.max(), 1e-12)
        w = 1.0 + 4.0 * self.ema + 1.5 * mag
        j = self._chronic()
        if self.ema[j] > 0.3:
            w[j] *= 1.0 + 2.0 * self.ema[j] + 0.5 * mag[j]
        return w

    def _scores(self, G):
        G = np.atleast_2d(G)
        feas = G.sum(axis=1) <= 0.0
        scaled = G / np.maximum(self.gscale, 1e-12)
        S = scaled @ self._weights()
        return np.where(feas, 0.0, np.maximum(S, 1e-9))

    @staticmethod
    def _better(s1, f1, s2, f2):
        feas1, feas2 = s1 <= 0.0, s2 <= 0.0
        if feas1 and feas2:
            return f1 < f2
        if feas1 != feas2:
            return feas1
        return s1 < s2

    @staticmethod
    def _pop_best(S, F):
        feas = S <= 0.0
        if np.any(feas):
            idx = np.where(feas)[0]
            return int(idx[np.argmin(F[idx])])
        return int(np.argmin(S))

    def _chronic_pick_elite(self, xi):
        """Pick the feasible elite nearest (normalised) to the infeasible member xi, so the
        bisection crosses the chronic boundary over the shortest feasible-infeasible segment.
        Occasionally pick a random top elite to keep diversity."""
        k = len(self.archive)
        if np.random.rand() < 0.3 or k == 1:
            return self.archive[np.random.randint(min(3, k))]["x"]
        E = np.array([el["x"] for el in self.archive])
        span = np.maximum(self.ub - self.lb, 1e-12)
        d = np.sum(((E - xi) / span) ** 2, axis=1)
        d[d < 1e-12] = np.inf  # avoid picking xi itself
        cand = np.where(np.isfinite(d))[0]
        if cand.size == 0:
            return self.archive[0]["x"]
        return E[cand[np.argmin(d[cand])]]

    # ------------------------------------------------------- main
    def __call__(self, func):
        self.func = func
        self.used = 0
        self.n_eval = 0
        self.best_f = np.inf
        self.best_x = None
        self.archive = []
        self.A = 10
        self.m = 0
        self.ema = None
        self.vema = None
        self.gscale = None
        self.bis_range = (0.25, 0.75)
        bis_ok, bis_try = 1.0, 2.0

        self.lb = np.asarray(func.bounds.lb, dtype=float).copy()
        self.ub = np.asarray(func.bounds.ub, dtype=float).copy()
        vt = list(func.variable_types)
        self.is_int = np.array([t != "continuous" for t in vt], dtype=bool)
        self.is_cat = np.array([t in ("categorical", "binary") for t in vt], dtype=bool)
        self.is_cont = ~self.is_int
        self.cont_idx = np.where(self.is_cont)[0]
        self.int_idx = np.where(self.is_int)[0]
        self.any_int = self.int_idx.size > 0
        self.nvals = np.ones(self.dim, dtype=int)
        for i in self.int_idx:
            self.nvals[i] = int(func.n_values[i])

        NP = int(min(max(6 * self.dim, 20), 80))
        if self.budget < 10 * NP:
            NP = max(5, self.budget // 10)
        patience = max(10, 3 * self.dim)

        while self.used < self.budget:
            X, G, Fo = [], [], []
            n_seed = NP // 3 if self.archive else 0
            for k in range(NP):
                if self.used >= self.budget:
                    return self.best_f, self.best_x
                if k < n_seed:
                    el = self.archive[k % len(self.archive)]
                    x = self._perturb(el["x"], 0.05 + 0.2 * np.random.rand())
                else:
                    x = self._random_point()
                g, f = self._eval(x)
                X.append(x)
                G.append(g)
                Fo.append(f)
            X = np.array(X)
            G = np.array(G)
            Fo = np.array(Fo)
            Fp = np.full(NP, 0.5)
            CRp = np.full(NP, 0.9)

            S = self._scores(G)
            bi = self._pop_best(S, Fo)
            bestS, bestF = S[bi], Fo[bi]
            stall = 0

            while self.used < self.budget and stall < patience:
                for i in range(NP):
                    if self.used >= self.budget:
                        return self.best_f, self.best_x

                    S = self._scores(G)
                    order = np.lexsort((Fo, S))
                    p = max(2, int(0.15 * NP))
                    pb = order[np.random.randint(p)]
                    bi = self._pop_best(S, Fo)

                    r = np.random.rand()
                    mode = "de"
                    if self.archive and r < 0.12 and S[i] > 0.0:
                        mode = "bisect"
                    elif self.archive and r < 0.21:
                        mode = "ls"

                    Fi, CRi = None, None
                    if mode == "ls":
                        el = self.archive[np.random.randint(min(3, len(self.archive)))]
                        ef = el["f"]
                        trial = self._local_step(el["x"], el["sig"])
                    elif mode == "bisect":
                        xf = self._chronic_pick_elite(X[i])
                        trial = self._bisect_move(xf, X[i])
                    else:
                        Fi = np.random.uniform(0.1, 1.0) if np.random.rand() < 0.1 else Fp[i]
                        CRi = np.random.rand() if np.random.rand() < 0.1 else CRp[i]
                        others = np.delete(np.arange(NP), i)
                        r1, r2, r3 = np.random.choice(others, 3, replace=False)
                        xi = X[i]
                        base = X[pb] if (self.best_x is None or np.random.rand() < 0.5) else X[bi]
                        mut = xi + Fi * (base - xi) + Fi * (X[r1] - X[r2])
                        pick = np.random.rand(self.dim) < 0.5
                        cat_donor = np.where(pick, X[pb], X[r3])
                        donor = np.where(self.is_cat, cat_donor, mut)
                        mask = np.random.rand(self.dim) < CRi
                        mask[np.random.randint(self.dim)] = True
                        trial = np.where(mask, donor, xi)
                        mm = self.is_cat & (np.random.rand(self.dim) < 1.0 / self.dim)
                        if mm.any():
                            trial[mm] = np.floor(np.random.rand(mm.sum()) * self.nvals[mm])
                        trial = self._repair(trial)

                    gt, ft = self._eval(trial)
                    st = self._scores(gt)[0]

                    if mode == "ls":
                        success = (st <= 0.0) and (ft < ef - 1e-12)
                        if success:
                            el["sig"] = min(el["sig"] * 1.5, 0.5)
                        else:
                            el["sig"] = max(el["sig"] * 1.5 ** (-0.25), 1e-4)

                    if mode == "bisect":
                        # track whether bisection produced a feasible point (adapt fraction range)
                        if st <= 0.0:
                            bis_ok += 1.0
                        else:
                            bis_try += 1.0
                        rate = bis_ok / (bis_ok + bis_try)
                        if rate > 0.4:
                            self.bis_range = (0.15, 0.75)
                        elif rate < 0.15:
                            self.bis_range = (0.35, 0.85)
                        else:
                            self.bis_range = (0.25, 0.75)

                    if self._better(st, ft, S[i], Fo[i]):
                        X[i] = trial
                        G[i] = gt
                        Fo[i] = ft
                        if mode == "de":
                            Fp[i], CRp[i] = Fi, CRi

                S = self._scores(G)
                bi = self._pop_best(S, Fo)
                sb, fb = S[bi], Fo[bi]
                if (sb <= 0.0) != (bestS <= 0.0):
                    improved = sb <= 0.0
                elif sb <= 0.0:
                    improved = fb < bestF - 1e-12
                else:
                    improved = sb < bestS - 1e-12
                if self._better(sb, fb, bestS, bestF):
                    bestS, bestF = sb, fb
                stall = 0 if improved else stall + 1

        return self.best_f, self.best_x