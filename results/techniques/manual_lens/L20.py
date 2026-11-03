import numpy as np


class Algorithm:
    """
    Feasibility-rule (mu+lambda) search with epsilon-constrained ranking (tolerance decays to zero
    by 30% of the budget) and a bandit over five mixed-integer operators:
      0: ES crossover + self-adaptive Gaussian + type-aware discrete mutation
      1: current-to-best DE on continuous variables, donor-based discrete recombination
      2: discrete sweep queue of the incumbent (single moves, then sampled pairwise moves), then random moves
      3: sparse Gaussian steps around the incumbent, with a Hooke-Jeeves style extrapolation along the
         last successful incumbent direction (gain grows on success, shrinks on failure)
      4: archive-guided current-to-pbest DE (pbest drawn from the elite top-20% of the population)
    Restarts cycle through Latin-hypercube populations, archive perturbations, incumbent perturbations,
    archive recombinations, incumbent-elite crossovers and tight incumbent-centred populations.
    Patience grows across restarts; evaluations are cached.
    """

    N_OPS = 5
    ELITE_K = 20
    N_PAIRS = 120
    P_FLOOR = 0.12
    EPS_END = 0.3

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    # ------------------------------------------------------------ setup
    def _setup(self, func):
        self.lb = np.asarray(func.bounds.lb, dtype=float).copy()
        self.ub = np.asarray(func.bounds.ub, dtype=float).copy()
        self.vt = list(func.variable_types)
        cont = np.array([t == "continuous" for t in self.vt], dtype=bool)
        self.cidx = np.where(cont)[0]
        self.didx = np.where(~cont)[0]
        self.nv = np.ones(self.dim)
        for j in self.didx:
            self.nv[j] = float(func.n_values[j])
        self.width = self.ub - self.lb
        self.cache = {}
        self.archive = []

    def _repair(self, X):
        X = np.clip(X, self.lb, self.ub)
        if len(self.didx):
            X[:, self.didx] = np.clip(np.round(X[:, self.didx]), 0, self.nv[self.didx] - 1)
        return X

    def _lhs_pop(self, n):
        X = np.zeros((n, self.dim))
        for j in range(self.dim):
            u = (np.random.permutation(n) + np.random.rand(n)) / n
            if j in self.didx:
                X[:, j] = np.floor(u * self.nv[j])
            else:
                X[:, j] = self.lb[j] + u * self.width[j]
        return self._repair(X)

    def _rank_order(self, F, V):
        infeas = V > self.eps
        key2 = np.where(infeas, V, F)
        return np.lexsort((key2, infeas.astype(int)))

    def _archive_add(self, f, xr):
        if not np.isfinite(f):
            return
        if len(self.archive) >= self.ELITE_K and f >= self.archive[-1][0]:
            return
        for fa, xa in self.archive:
            if np.array_equal(xa, xr):
                return
        self.archive.append((f, xr.copy()))
        self.archive.sort(key=lambda t: t[0])
        del self.archive[self.ELITE_K:]

    def _evaluate(self, func, X):
        Fs, Vs, Xs = [], [], []
        for x in X:
            xr = np.array(x, dtype=float)
            key = tuple(np.round(xr, 9))
            if key in self.cache:
                f, v = self.cache[key]
            else:
                if self.used >= self.budget:
                    break
                f, g = func(xr)
                self.used += 1
                f = float(f)
                g = np.nan_to_num(np.atleast_1d(np.asarray(g, dtype=float)),
                                  nan=1e10, posinf=1e10, neginf=0.0)
                v = float(np.sum(np.maximum(g, 0.0)))
                if not np.isfinite(f):
                    f = np.inf
                    v = max(v, 1e-12)
                self.cache[key] = (f, v)
                if v <= 0.0 and f < self.best_f:
                    self.best_f = f
                    self.best_x = xr.copy()
                if v <= 0.0:
                    self._archive_add(f, xr)
                if v < self.fb_v:
                    self.fb_v = v
                    self.fb_x = xr.copy()
            Fs.append(f)
            Vs.append(v)
            Xs.append(xr)
        if not Xs:
            return np.array([]), np.array([]), np.empty((0, self.dim))
        return np.array(Fs), np.array(Vs), np.array(Xs)

    def _base(self):
        return self.best_x if self.best_x is not None else self.fb_x

    # ------------------------------------------------------------ discrete helpers
    def _disc_move(self, child, j):
        n = int(self.nv[j])
        if n <= 1:
            return
        cur = int(round(child[j]))
        if self.vt[j] == "binary":
            child[j] = 1 - cur
        elif self.vt[j] == "ordinal":
            step = 1 if np.random.rand() < 0.5 else -1
            new = cur + step
            if new < 0 or new > n - 1:
                new = cur - step
            child[j] = min(n - 1, max(0, new))
        else:
            new = np.random.randint(0, n - 1)
            if new >= cur:
                new += 1
            child[j] = new

    def _mutate_discrete(self, child, rate):
        if len(self.didx) == 0:
            return child
        for t in np.where(np.random.rand(len(self.didx)) < rate)[0]:
            self._disc_move(child, self.didx[t])
        return child

    def _full_discrete_neighbourhood(self, x):
        singles = []
        for j in self.didx:
            cur = int(round(x[j]))
            for v in range(int(self.nv[j])):
                if v != cur:
                    c = np.array(x, dtype=float)
                    c[j] = v
                    singles.append(c)
        if len(singles) > 1:
            perm = np.random.permutation(len(singles))
            singles = [singles[i] for i in perm]
        pairs = []
        if len(self.didx) >= 2:
            for _ in range(self.N_PAIRS):
                a, b = np.random.choice(len(self.didx), 2, replace=False)
                c = np.array(x, dtype=float)
                self._disc_move(c, self.didx[a])
                self._disc_move(c, self.didx[b])
                pairs.append(c)
        return pairs + singles

    # ------------------------------------------------------------ operators
    def _op_es(self, P, rank, PS, n):
        dim = self.dim
        tau = 1.0 / np.sqrt(max(1, dim))
        C, S = [], []
        for _ in range(n):
            a, b = np.random.randint(0, len(P), 2)
            p1 = a if rank[a] < rank[b] else b
            a, b = np.random.randint(0, len(P), 2)
            p2 = a if rank[a] < rank[b] else b
            if np.random.rand() < 0.8:
                mask = np.random.rand(dim) < 0.5
                child = np.where(mask, P[p1], P[p2]).astype(float)
            else:
                child = P[p1].astype(float).copy()
            sig = float(np.clip(PS[p1] * np.exp(tau * np.random.randn()), 1e-4, 0.3))
            if len(self.cidx):
                child[self.cidx] += sig * self.width[self.cidx] * np.random.randn(len(self.cidx))
            child = self._mutate_discrete(child, 1.0 / dim)
            C.append(child)
            S.append(sig)
        return C, S

    def _op_de(self, P, base, n):
        N = len(P)
        C = []
        for i in range(n):
            r1, r2 = np.random.choice(N, 2, replace=False)
            Fm = np.random.uniform(0.4, 0.9)
            donor = P[i].astype(float).copy()
            if len(self.cidx):
                ci = self.cidx
                donor[ci] = P[i][ci] + Fm * (base[ci] - P[i][ci]) + Fm * (P[r1][ci] - P[r2][ci])
            if len(self.didx):
                di = self.didx
                pick_r1 = np.random.rand(len(di)) < 0.5
                donor[di] = np.where(pick_r1, P[r1][di], base[di])
            cross = np.random.rand(self.dim) < 0.8
            cross[np.random.randint(self.dim)] = True
            child = np.where(cross, donor, P[i]).astype(float)
            child = self._mutate_discrete(child, 0.05)
            C.append(child)
        return C

    def _op_arch_de(self, P, order, n):
        N = len(P)
        n_top = max(2, N // 5)
        C = []
        for i in range(n):
            pb = P[order[np.random.randint(n_top)]]
            r1, r2 = np.random.choice(N, 2, replace=False)
            Fm = np.random.uniform(0.5, 1.0)
            cur = P[np.random.randint(N)]
            donor = cur.astype(float).copy()
            if len(self.cidx):
                ci = self.cidx
                donor[ci] = cur[ci] + Fm * (pb[ci] - cur[ci]) + Fm * (P[r1][ci] - P[r2][ci])
            if len(self.didx):
                di = self.didx
                pick = np.random.rand(len(di)) < 0.5
                donor[di] = np.where(pick, pb[di], cur[di])
            cross = np.random.rand(self.dim) < 0.7
            cross[np.random.randint(self.dim)] = True
            child = np.where(cross, donor, cur).astype(float)
            child = self._mutate_discrete(child, 0.03)
            C.append(child)
        return C

    def _op_disc(self, base, n):
        C = []
        while self.pending and len(C) < n:
            C.append(self.pending.pop())
        for _ in range(n - len(C)):
            child = np.array(base, dtype=float)
            if len(self.didx):
                n_moves = 1 if np.random.rand() < 0.7 else 2
                for _m in range(n_moves):
                    self._disc_move(child, self.didx[np.random.randint(len(self.didx))])
            elif len(self.cidx):
                j = self.cidx[np.random.randint(len(self.cidx))]
                child[j] += 0.05 * self.width[j] * np.random.randn()
            C.append(child)
        return C

    def _op_gauss(self, base, scale, n):
        C = []
        for _ in range(n):
            child = np.array(base, dtype=float)
            if len(self.cidx):
                r = np.random.rand()
                if self.mom is not None and r < 0.3:
                    # Hooke-Jeeves style extrapolation along last successful incumbent direction
                    g = self.mom_gain * np.random.uniform(0.5, 1.5)
                    child[self.cidx] += g * self.mom[self.cidx]
                    child[self.cidx] += 0.1 * scale * self.width[self.cidx] * np.random.randn(len(self.cidx))
                elif self.mom is not None and r < 0.4:
                    # half-step extrapolation with coordinate-restricted noise
                    g = 0.5 * self.mom_gain
                    k = np.random.randint(len(self.cidx))
                    j = self.cidx[k]
                    child[j] += g * self.mom[j]
                else:
                    m = min(len(self.cidx), max(1, np.random.binomial(len(self.cidx), 0.3)))
                    idx = np.random.choice(self.cidx, size=m, replace=False)
                    s = scale * np.exp(0.3 * np.random.randn())
                    child[idx] += s * self.width[idx] * np.random.randn(m)
            elif len(self.didx):
                self._disc_move(child, self.didx[np.random.randint(len(self.didx))])
            C.append(child)
        return C

    # ------------------------------------------------------------ restart seeding
    def _perturbed(self, centre, strengths, mu):
        X = []
        for i in range(mu):
            s = strengths[i % len(strengths)]
            child = np.array(centre, dtype=float)
            if len(self.cidx):
                child[self.cidx] += s * self.width[self.cidx] * np.random.randn(len(self.cidx))
            child = self._mutate_discrete(child, min(0.5, max(1.0 / self.dim, s)))
            X.append(child)
        return self._repair(np.array(X))

    def _recombine_archive(self, mu, anchor=None):
        X = []
        na = len(self.archive)
        for i in range(mu):
            _, e1 = self.archive[np.random.randint(na)]
            if anchor is not None and i % 2 == 0:
                e1 = anchor
            _, e2 = self.archive[np.random.randint(na)]
            mask = np.random.rand(self.dim) < 0.5
            child = np.where(mask, e1, e2).astype(float)
            if len(self.cidx):
                a = np.random.rand(len(self.cidx))
                child[self.cidx] = a * e1[self.cidx] + (1 - a) * e2[self.cidx]
                child[self.cidx] += 0.02 * self.width[self.cidx] * np.random.randn(len(self.cidx))
            child = self._mutate_discrete(child, 1.0 / self.dim)
            X.append(child)
        return self._repair(np.array(X))

    # ------------------------------------------------------------ main loop
    def __call__(self, func):
        self._setup(func)
        dim = self.dim
        self.used = 0
        self.best_f = np.inf
        self.best_x = None
        self.fb_v = np.inf
        self.fb_x = None
        self.eps = 0.0
        self.eps0 = None
        self.pending = []
        self._pend_key = None
        self.mom = None
        self.mom_gain = 1.0
        self._prev_best = None

        mu = max(12, min(40, 3 * dim))
        lam = mu
        base_patience = 12 + dim // 2
        score = np.ones(self.N_OPS)
        ls_scale = 0.05
        restarts = 0

        while self.used < self.budget:
            base0 = self._base()
            mode = restarts % 6
            if base0 is None or mode == 0:
                X = self._lhs_pop(mu)
            elif mode == 1 and len(self.archive) > 0:
                X = []
                strengths = [0.03, 0.1, 0.25]
                for i in range(mu):
                    _, ex = self.archive[np.random.randint(len(self.archive))]
                    X.append(self._perturbed(ex, [strengths[i % 3]], 1)[0])
                X = np.array(X)
            elif mode == 2 and len(self.archive) >= 2:
                X = self._recombine_archive(mu)
            elif mode == 3 and len(self.archive) >= 2:
                X = self._recombine_archive(mu, anchor=np.array(base0, dtype=float))
            elif mode == 4:
                # tight incumbent-centred population for local refinement
                X = self._perturbed(base0, [0.01, 0.03, 0.08], mu)
            else:
                X = self._perturbed(base0, [0.05, 0.15, 0.3], mu)
            if self._base() is not None:
                X[0] = self._base()
            restarts += 1
            patience = base_patience + min(12, restarts)
            self.mom = None
            self.mom_gain = 1.0
            self._prev_best = None

            F, V, X = self._evaluate(func, X)
            if len(F) == 0:
                break

            if self.eps0 is None:
                pos = V[V > 0]
                self.eps0 = float(np.median(pos)) if len(pos) else 0.0
                self.eps0 = min(self.eps0, 1.0)

            self._update_eps()
            P, PF, PV = X, F, V
            PS = np.full(len(F), 0.2)

            b0 = self._rank_order(PF, PV)[0]
            last_key = (PV[b0] > 0, PV[b0] if PV[b0] > 0 else PF[b0])
            stagn = 0

            while self.used < self.budget:
                self._update_eps()
                order = self._rank_order(PF, PV)
                rank = np.empty(len(PF), dtype=int)
                rank[order] = np.arange(len(PF))

                if self.best_x is not None:
                    pk = tuple(np.round(self.best_x, 9))
                    if pk != self._pend_key:
                        self._pend_key = pk
                        self.pending = self._full_discrete_neighbourhood(self.best_x)
                        if self._prev_best is not None:
                            self.mom = np.array(self.best_x, dtype=float) - self._prev_best
                        self._prev_best = np.array(self.best_x, dtype=float)

                p = score / score.sum()
                p = self.P_FLOOR + (1 - self.P_FLOOR * self.N_OPS) * p
                p = p / p.sum()
                counts = np.random.multinomial(lam, p)

                base = self._base()
                if base is None:
                    base = P[order[0]]

                C_es, S_es = self._op_es(P, rank, PS, counts[0])
                C_de = self._op_de(P, base, counts[1])
                C_dm = self._op_disc(base, counts[2])
                C_gs = self._op_gauss(base, ls_scale, counts[3])
                C_ad = self._op_arch_de(P, order, counts[4])

                children = C_es + C_de + C_dm + C_gs + C_ad
                ops_lab = np.repeat(np.arange(self.N_OPS), counts)
                csig = np.concatenate([np.array(S_es, dtype=float),
                                       np.full(lam - counts[0], 0.2)])

                Xc = self._repair(np.array(children))
                best_before = self.best_f
                Fc, Vc, Xc = self._evaluate(func, Xc)
                n_c = len(Fc)
                if n_c == 0:
                    break
                ops_lab = ops_lab[:n_c]
                csig = csig[:n_c]

                new_best = (Vc <= 0) & (Fc < best_before)
                gs_mask = ops_lab == 3
                if gs_mask.any():
                    if np.any(new_best & gs_mask):
                        ls_scale = min(0.2, ls_scale * 1.2)
                    else:
                        ls_scale = max(1e-3, ls_scale * 0.92)

                # adapt extrapolation gain: grow when the incumbent improves, shrink otherwise
                if self.best_f < best_before:
                    if self.mom is not None:
                        self.mom_gain = min(3.0, self.mom_gain * 1.3)
                else:
                    self.mom_gain = max(0.2, self.mom_gain * 0.9)

                nP = len(P)
                allX = np.vstack([P, Xc])
                allF = np.concatenate([PF, Fc])
                allV = np.concatenate([PV, Vc])
                allS = np.concatenate([PS, csig])
                order = self._rank_order(allF, allV)[:mu]

                surv_child = order[order >= nP] - nP
                n_per = np.bincount(ops_lab, minlength=self.N_OPS).astype(float)
                surv_per = np.bincount(ops_lab[surv_child], minlength=self.N_OPS).astype(float)
                nb_per = np.bincount(ops_lab[new_best], minlength=self.N_OPS).astype(float)
                reward = (surv_per + 2.0 * nb_per) / np.maximum(1.0, n_per)
                used_ops = n_per > 0
                score[used_ops] = 0.6 * score[used_ops] + 0.4 * (reward[used_ops] + 0.03)

                P, PF, PV, PS = allX[order], allF[order], allV[order], allS[order]

                strict = PV <= 0
                if strict.any():
                    idxs = np.where(strict)[0]
                    b = idxs[np.argmin(PF[idxs])]
                    key = (False, PF[b])
                else:
                    b = np.argmin(PV)
                    key = (True, PV[b])
                improved = (key[0] < last_key[0]) or (
                    key[0] == last_key[0] and key[1] < last_key[1] - 1e-12)
                if improved:
                    stagn = 0
                    last_key = key
                else:
                    stagn += 1
                if stagn >= patience:
                    break

        if self.best_x is None:
            return self.best_f, self.fb_x
        return self.best_f, self.best_x

    def _update_eps(self):
        if not self.eps0:
            self.eps = 0.0
            return
        frac = self.used / max(1, self.budget)
        self.eps = self.eps0 * max(0.0, 1.0 - frac / self.EPS_END) ** 2