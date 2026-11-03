import numpy as np


class Algorithm:
    """
    (mu+1) steady-state ES for mixed-integer constrained black-box problems.
    - Constraint weights: violation-frequency exponential, top-3 boost and magnitude correction.
    - kNN surrogate predicts weighted violation, the hardest-constraint violation and a joint
      feasibility log-probability over ALL constraints that are violated in >=2% of evaluations
      (so frequently violated constraints such as the hardest one are never drowned out).
    - Multi-level bisection of predicted-infeasible children toward the nearest feasible archived anchor.
    - Feasibility-adaptive pool: when feasible points are rare, screen more children per evaluation.
    - Adaptive local search around the incumbent, 1/5-rule step sizes, interpolation repair.
    - Stagnation restarts that always keep the global best feasible solution.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    # ------------------------------------------------------------------ setup
    def _setup(self, func):
        d = self.dim
        types = [str(t).lower() for t in func.variable_types]
        self.is_cont = np.array([t == "continuous" for t in types], dtype=bool)
        self.is_bin = np.array([t == "binary" for t in types], dtype=bool)
        self.is_ord = np.array([t == "ordinal" for t in types], dtype=bool)
        self.is_cat = np.array([t == "categorical" for t in types], dtype=bool)
        self.cont_idx = np.where(self.is_cont)[0]
        self.int_idx = np.where(~self.is_cont)[0]
        self.nc = len(self.cont_idx)
        self.ni = len(self.int_idx)

        nv = np.full(d, 2.0)
        for i in self.int_idx:
            nv[i] = float(func.n_values[i])
        self.nv = nv
        self.lo = np.where(self.is_cont, -1.0, 0.0)
        self.hi = np.where(self.is_cont, 1.0, nv - 1.0)
        self.scale = np.where(self.is_cont, 1.0, 1.0 / np.maximum(nv - 1.0, 1.0))
        self.tau = 1.0 / np.sqrt(2.0 * np.sqrt(max(self.nc, 1)))

    def _repair(self, x):
        x = np.clip(x, self.lo, self.hi)
        if self.ni > 0:
            x[self.int_idx] = np.rint(x[self.int_idx])
        return x

    # ------------------------------------------------------------ surrogate
    def _pred(self, X):
        """kNN prediction: weighted violation, joint feasibility log-probability
        over active constraints, and violation of the hardest constraint."""
        na = self.na
        n = X.shape[0]
        if na < 5:
            return np.zeros(n), np.zeros(n), np.zeros(n)
        lo_i = max(0, na - self.cap)
        A = self.aZ[lo_i:na]
        G = self.aG[lo_i:na]
        Z = X * self.scale
        A = A * self.scale
        dist = ((Z[:, None, :] - A[None, :, :]) ** 2).sum(axis=-1)
        k = min(5, A.shape[0])
        idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
        Gn = G[idx]                                       # (n, k, m)
        pred = Gn.mean(axis=1) @ self.w
        # joint feasibility over active constraints (log-probability)
        act = self.act
        if act.size > 0:
            sat = (Gn[:, :, act] <= 0.0).mean(axis=1)     # (n, a)
            sat = np.clip(sat, 0.05, 1.0)
            logp = np.log(sat).sum(axis=1)
        else:
            logp = np.zeros(n)
        hard = Gn[:, :, self.hard_h].mean(axis=1)
        return pred, logp, hard

    # ------------------------------------------------------------ evaluation
    def _evaluate(self, func, x, s):
        x = self._repair(np.asarray(x, dtype=float))
        f, g = func(x)
        self.used += 1
        f = float(f)
        if not np.isfinite(f):
            f = np.inf
        g = np.atleast_1d(np.asarray(g, dtype=float))
        g = np.where(np.isfinite(g), g, 1e6)
        gp = np.maximum(g, 0.0)
        v = float(gp.sum())
        if v <= 0.0 and not np.isfinite(f):
            v = 1e-9

        if self.na < self.aZ.shape[0]:
            self.aZ[self.na] = x
            self.aG[self.na] = gp
            self.aV[self.na] = v
            self.na += 1

        viol = gp > 0
        self.viol_cnt = self.viol_cnt + viol.astype(float)
        self.vmag_sum = self.vmag_sum + gp * viol
        self.vmag_n = self.vmag_n + viol
        vmag = np.where(self.vmag_n > 0, self.vmag_sum / np.maximum(self.vmag_n, 1), 1.0)

        rate = self.viol_cnt / max(self.used, 1)
        w = np.exp(4.0 * rate)
        mag_ratio = vmag.mean() / np.maximum(vmag, 1e-9)
        w *= np.clip(mag_ratio ** 0.25, 0.6, 1.6)
        order = np.argsort(-rate)
        boosts = [1.6, 1.3, 1.1]
        for r_, b in enumerate(boosts):
            if r_ < len(order):
                w[order[r_]] *= b
        self.w = w
        self.hard_h = int(order[0])
        # active constraints: violated in at least 2% of evaluations
        self.act = np.where(rate >= 0.02)[0]

        feas = (v <= 0.0) and np.isfinite(f)
        if feas:
            self.n_feas_eval += 1
            if self.nf < self.fX.shape[0]:
                self.fX[self.nf] = x
                self.nf += 1
        if feas and f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
            self.best_gp = gp.copy()
        return [x, s, f, gp, v]

    def _key(self, ind):
        if ind[4] <= 0.0 and np.isfinite(ind[2]):
            return (0, ind[2])
        return (1, float(self.w @ ind[3]) + ind[4])

    # ------------------------------------------------------------ operators
    def _lhs(self, n):
        X = np.empty((n, self.dim))
        for j in range(self.dim):
            if self.is_cont[j]:
                perm = np.random.permutation(n)
                u = (perm + np.random.rand(n)) / n
                X[:, j] = -1.0 + 2.0 * u
            else:
                X[:, j] = np.random.randint(0, int(self.nv[j]), size=n)
        return X

    def _tournament(self, pop, k=2):
        idx = np.random.randint(0, len(pop), size=k)
        best = idx[0]
        for i in idx[1:]:
            if self._key(pop[i]) < self._key(pop[best]):
                best = i
        return best

    def _mutate_int(self, x, j):
        cur = int(round(x[j]))
        n = int(self.nv[j])
        if self.is_bin[j]:
            x[j] = 1 - cur
        elif self.is_ord[j]:
            if np.random.rand() < 0.8:
                step = 1 if np.random.rand() < 0.5 else -1
                x[j] = np.clip(cur + step, 0, n - 1)
            else:
                x[j] = np.random.randint(0, n)
        else:
            if n > 1:
                r = np.random.randint(0, n - 1)
                x[j] = r + 1 if r >= cur else r
            else:
                x[j] = cur

    def _make_child(self, pop):
        d = self.dim
        p1 = pop[self._tournament(pop)]
        p2 = pop[self._tournament(pop)]
        if np.random.rand() < 0.5:
            mask = np.random.rand(d) < 0.5
            x = np.where(mask, p1[0], p2[0]).astype(float)
        else:
            x = p1[0].copy()
        s = np.sqrt(p1[1] * p2[1])
        s = float(np.clip(s * np.exp(self.tau * np.random.randn()), 1e-3, 0.5))

        pc = min(1.0, 3.0 / max(self.nc, 1))
        pm = min(1.0, 1.5 / max(self.ni, 1))
        mut_c = (np.random.rand(d) < pc) & self.is_cont
        mut_i = (np.random.rand(d) < pm) & (~self.is_cont)
        if not (mut_c.any() or mut_i.any()):
            j = np.random.randint(d)
            if self.is_cont[j]:
                mut_c[j] = True
            else:
                mut_i[j] = True
        if mut_c.any():
            x[mut_c] = x[mut_c] + s * np.random.randn(int(mut_c.sum()))
        for j in np.where(mut_i)[0]:
            self._mutate_int(x, j)
        return self._repair(x), s

    def _nearest_feasible(self, c):
        """Nearest feasible archived point to c (random among 3 nearest for diversity)."""
        if self.nf == 0:
            return self.best_x
        F = self.fX[:self.nf]
        dd = (((F - c) * self.scale) ** 2).sum(axis=1)
        k = min(3, self.nf)
        idx = np.argpartition(dd, k - 1)[:k]
        return F[idx[np.random.randint(len(idx))]]

    def _select_child(self, pop):
        """Children plus bisection toward nearest feasible anchors; screened by
        weighted predicted violation, joint infeasibility and hardest-constraint violation."""
        cands = [self._make_child(pop) for _ in range(self.pool)]
        if self.best_x is not None:
            base = list(cands)
            for c, s in base:
                a = self._nearest_feasible(c)
                if a is None:
                    continue
                t2 = np.random.uniform(0.3, 0.7)
                cands.append((self._repair(c + 0.5 * (a - c)), s))
                cands.append((self._repair(c + t2 * (a - c)), s))
        if self.na < 20 or np.random.rand() < 0.1:
            return cands[0]
        X = np.array([c[0] for c in cands])
        pred, logp, hard = self._pred(X)
        score = pred + 0.3 * (-logp) + 0.5 * hard
        return cands[int(np.argmin(score))]

    def _local_move(self, sig):
        x = self.best_x.copy()
        pc = min(1.0, 2.0 / max(self.nc, 1))
        mask = (np.random.rand(self.dim) < pc) & self.is_cont
        if not mask.any() and self.nc > 0:
            mask[np.random.choice(self.cont_idx)] = True
        x[mask] = x[mask] + sig * np.random.randn(int(mask.sum()))
        if self.ni > 0 and np.random.rand() < 0.2:
            j = np.random.choice(self.int_idx)
            self._mutate_int(x, j)
        return self._repair(x)

    # ------------------------------------------------------------ main loop
    def __call__(self, func):
        self._setup(func)
        m = int(func.n_constraints)
        self.used = 0
        self.n_feas_eval = 0
        self.best_f = np.inf
        self.best_x = None
        self.best_gp = np.zeros(m)
        self.viol_cnt = np.zeros(m)
        self.vmag_sum = np.zeros(m)
        self.vmag_n = np.zeros(m)
        self.w = np.ones(m)
        self.hard_h = 0
        self.act = np.arange(m)
        self.cap = 1500
        self.pool = 4
        self.aZ = np.empty((self.budget, self.dim))
        self.aG = np.empty((self.budget, m))
        self.aV = np.empty(self.budget)
        self.na = 0
        self.fX = np.empty((self.budget, self.dim))
        self.nf = 0

        mu = int(np.clip(2 * self.dim, 10, 40))
        patience = 15 * mu
        s0 = 0.3
        sig_loc = 0.05
        loc_trials = 0
        loc_succ = 0
        p_loc, p_interp = 0.2, 0.15

        while self.used < self.budget:
            if self.used > 0:
                frate = self.n_feas_eval / self.used
                # more screening when feasibility is rare (screening is cheap)
                if frate < 0.2:
                    self.pool = 6
                elif frate < 0.4:
                    self.pool = 4
                else:
                    self.pool = 3
                p_loc = 0.3 if frate < 0.2 else 0.2

            n0 = min(mu, self.budget - self.used)
            pop = []
            for x in self._lhs(n0):
                if self.used >= self.budget:
                    break
                pop.append(self._evaluate(func, x, s0))
            if self.best_x is not None and len(pop) > 0:
                worst = max(range(len(pop)), key=lambda i: self._key(pop[i]))
                cand = [self.best_x.copy(), s0, self.best_f, self.best_gp.copy(), 0.0]
                if self._key(cand) < self._key(pop[worst]):
                    pop[worst] = cand
            if len(pop) < 2:
                break

            best_key = min(self._key(p) for p in pop)
            stall = 0

            while self.used < self.budget and stall < patience:
                r = np.random.rand()
                op = "std"
                if self.best_x is not None and r < p_loc:
                    xc = self._local_move(sig_loc)
                    sc = s0
                    op = "loc"
                elif self.best_x is not None and r < p_loc + p_interp:
                    p = pop[self._tournament(pop)]
                    if p[4] > 0.0:
                        t = np.random.uniform(0.2, 0.8)
                        xc = self._repair(p[0] + t * (self.best_x - p[0]))
                        sc = p[1]
                        op = "interp"
                    else:
                        xc, sc = self._select_child(pop)
                else:
                    xc, sc = self._select_child(pop)

                ind = self._evaluate(func, xc, sc)
                ck = self._key(ind)

                if op == "loc":
                    loc_trials += 1
                    if ck < best_key:
                        loc_succ += 1
                    if loc_trials == 10:
                        if loc_succ / 10.0 > 0.2:
                            sig_loc *= 1.22
                        else:
                            sig_loc *= 0.82
                        sig_loc = float(np.clip(sig_loc, 1e-3, 0.3))
                        loc_trials = 0
                        loc_succ = 0

                worst = max(range(len(pop)), key=lambda i: self._key(pop[i]))
                if ck <= self._key(pop[worst]):
                    pop[worst] = ind
                cur_best = min(self._key(p) for p in pop)
                if cur_best < best_key:
                    best_key = cur_best
                    stall = 0
                else:
                    stall += 1

        return self.best_f, self.best_x