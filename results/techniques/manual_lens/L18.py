import numpy as np


class Algorithm:
    """
    Mixed-integer constrained DE with:
      * Deb-style feasibility rules, where infeasible points are ranked by a weighted violation,
      * per-constraint weights from exponentially smoothed violation frequencies, so that the
        constraints that block feasibility (e.g. constraint 23) receive more weight,
      * a frontier operator that interpolates between the best feasible point and infeasible
        population members to find feasible points near the constraint boundary,
      * jDE self-adaptation of F and CR, current-to-best/rand DE moves with discrete-aware genes,
      * an incumbent exploitation operator, a cache (duplicates cost no budget) and restarts.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng()
        dim = self.dim
        budget = self.budget

        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        vtypes = list(func.variable_types)
        nv = np.asarray(getattr(func, "n_values", np.zeros(dim)), dtype=float)
        nv = np.where(np.isfinite(nv) & (nv > 0), nv, 1.0)

        is_int = np.array([t != "continuous" for t in vtypes], dtype=bool)
        is_cat = np.array([t == "categorical" for t in vtypes], dtype=bool)
        is_bin = np.array([t == "binary" for t in vtypes], dtype=bool)
        is_ord = np.array([t == "ordinal" for t in vtypes], dtype=bool)
        is_cont = ~is_int
        int_idx = np.where(is_int)[0]
        cont_idx = np.where(is_cont)[0]

        lo = np.where(is_int, 0.0, lb)
        hi = np.where(is_int, nv - 1.0, ub)
        span = hi - lo

        # ---------- state ----------
        cache = {}
        used = [0]
        best = {"f": np.inf, "x": None}
        m = [None]          # number of constraints (known after first evaluation)
        w = [None]          # per-constraint weights
        freq = [None]       # smoothed violation frequency per constraint
        pending = []        # raw constraint rows of new evaluations in current generation

        # ---------- helpers ----------
        def random_points(n):
            X = lo + rng.random((n, dim)) * (hi - lo)
            if len(int_idx):
                X[:, int_idx] = np.floor(rng.random((n, len(int_idx))) * nv[int_idx])
            return np.clip(X, lo, hi)

        def repair(x):
            x = np.clip(x, lo, hi)
            if len(int_idx):
                x[int_idx] = np.rint(x[int_idx])
            return np.clip(x, lo, hi)

        def viol(F, G):
            """Weighted violation for arrays F (n,), G (n,m)."""
            G = np.atleast_2d(G)
            V = np.maximum(G, 0.0) @ w[0]
            F = np.atleast_1d(F)
            V = np.where(np.isinf(F), np.maximum(V, 1e9), V)
            return V

        def rank(F, V):
            feas = V <= 0
            pri = np.where(feas, 0, 1)
            sec = np.where(feas, F, V)
            return np.lexsort((sec, pri))

        def big_g():
            return np.full(m[0] if m[0] is not None else 1, 1e10)

        def evaluate(x):
            key = x.tobytes()
            if key in cache:
                return cache[key]
            if used[0] >= budget:
                return np.inf, big_g()
            f, g = func(x.copy())
            used[0] += 1
            f = float(f)
            if not np.isfinite(f):
                f = np.inf
            g = np.nan_to_num(np.asarray(g, dtype=float).ravel(),
                              nan=1e10, posinf=1e10, neginf=0.0)
            if m[0] is None:
                m[0] = g.size
                w[0] = np.ones(g.size)
                freq[0] = np.zeros(g.size)
            if g.size != m[0]:
                g = np.resize(g, m[0])
            cache[key] = (f, g)
            pending.append(np.maximum(g, 0.0))
            feasible = (f < np.inf) and bool(np.all(g <= 0))
            if feasible and f < best["f"]:
                best["f"] = f
                best["x"] = x.copy()
            return f, g

        def update_weights():
            if not pending:
                return
            Gn = np.array(pending)
            pending.clear()
            fr = (Gn > 0).mean(axis=0)
            freq[0] = 0.9 * freq[0] + 0.1 * fr
            ww = 0.2 + freq[0]
            w[0] = ww / ww.mean()

        # ---------- initialisation ----------
        mu = int(np.clip(5 * dim, 20, 100))
        mu = max(4, min(mu, max(4, budget // 5)))
        PX = random_points(mu)
        PF = np.empty(mu)
        PG = None
        rows = []
        for i in range(mu):
            if used[0] >= budget:
                break
            f, g = evaluate(PX[i])
            rows.append((f, g))
        if len(rows) == 0:
            return np.inf, PX[0].copy()
        n0 = len(rows)
        PX = PX[:n0]
        PF = np.array([r[0] for r in rows])
        PG = np.array([r[1] for r in rows])
        update_weights()
        PV = viol(PF, PG)
        mu = n0
        Fp = np.full(mu, 0.5)
        CRp = np.full(mu, 0.9)

        sigma_scale = 0.1
        stall = 0
        stag_limit = 10 + dim // 2
        prev_best = best["f"]

        # ---------- main loop ----------
        while used[0] < budget:
            npop = len(PX)
            order = rank(PF, PV)
            ptop = max(1, int(0.15 * npop))
            have_best = best["x"] is not None
            used_before = used[0]

            infeas_idx = np.where(PV > 0)[0]
            off = np.empty((npop, dim))
            offF = np.empty(npop)
            offCR = np.empty(npop)

            for k in range(npop):
                r = rng.random()
                if have_best and r < 0.12:
                    # Incumbent-centred exploitation with discrete-aware moves
                    x = best["x"].copy()
                    mm = int(rng.integers(1, 3))
                    for j in rng.choice(dim, size=min(mm, dim), replace=False):
                        if is_cont[j]:
                            x[j] += rng.normal() * sigma_scale * span[j]
                        elif is_bin[j]:
                            x[j] = 1.0 - x[j]
                        elif is_ord[j]:
                            x[j] += rng.choice([-1.0, 1.0])
                        else:
                            x[j] = float(rng.integers(0, int(nv[j])))
                    off[k] = repair(x)
                    offF[k], offCR[k] = 0.5, 0.9
                    continue

                if have_best and len(infeas_idx) > 0 and r < 0.22:
                    # Frontier operator: move from the feasible incumbent toward an infeasible
                    # member, mixing discrete genes, to cross blocking constraints at the boundary
                    i = int(rng.choice(infeas_idx))
                    xb = best["x"]
                    t = rng.uniform(0.3, 1.0)
                    x = xb + t * (PX[i] - xb)
                    take = rng.random(dim) < 0.5
                    if len(int_idx):
                        d = take & is_int
                        x[d] = PX[i][d]
                    if len(cont_idx):
                        x[cont_idx] += rng.normal(0, 0.02, len(cont_idx)) * span[cont_idx]
                    off[k] = repair(x)
                    offF[k], offCR[k] = 0.5, 0.9
                    continue

                # Differential evolution step with jDE self-adaptation
                i = k % npop
                Fi = Fp[i]
                CRi = CRp[i]
                if rng.random() < 0.1:
                    Fi = 0.1 + 0.9 * rng.random()
                if rng.random() < 0.1:
                    CRi = rng.random()

                others = rng.choice(npop - 1, 3, replace=False)
                others[others >= i] += 1
                r1, r2, r3 = others
                top = order[rng.integers(ptop)]

                if rng.random() < 0.5:
                    mutant = PX[i] + Fi * (PX[top] - PX[i]) + Fi * (PX[r1] - PX[r2])
                else:
                    mutant = PX[r1] + Fi * (PX[r2] - PX[r3])

                mask = rng.random(dim) < CRi
                mask[rng.integers(dim)] = True
                x = np.where(mask, mutant, PX[i])

                reset = is_cat & (rng.random(dim) < 1.0 / dim)
                if np.any(reset):
                    x[reset] = np.floor(rng.random(int(reset.sum())) * nv[reset])

                off[k] = repair(x)
                offF[k] = Fi
                offCR[k] = CRi

            # Evaluate offspring
            offPF = np.empty(npop)
            offG = []
            for k in range(npop):
                if used[0] >= budget:
                    offPF[k] = np.inf
                    offG.append(big_g())
                else:
                    f, g = evaluate(off[k])
                    offPF[k] = f
                    offG.append(g)
            offG = np.array(offG)

            # Update constraint weights from this generation's violation frequencies
            update_weights()

            # Elitist (mu + lambda) survivor selection with weighted feasibility rules
            CX = np.vstack([PX, off])
            CF = np.concatenate([PF, offPF])
            CG = np.vstack([PG, offG])
            CV = viol(CF, CG)
            CFp = np.concatenate([Fp, offF])
            CCR = np.concatenate([CRp, offCR])
            sel = rank(CF, CV)[:npop]
            PX, PF, PG, PV = CX[sel], CF[sel], CG[sel], CV[sel]
            Fp, CRp = CFp[sel], CCR[sel]

            # Adaptation of exploitation step and stagnation tracking
            if best["f"] < prev_best - 1e-12:
                stall = 0
                sigma_scale = min(0.5, sigma_scale * 1.2)
            else:
                stall += 1
                sigma_scale = max(1e-3, sigma_scale * 0.97)
            prev_best = best["f"]

            # No new evaluations possible (all duplicates): force diversification
            if used[0] == used_before:
                stall += stag_limit + 1

            # Restart: keep incumbent, resample the rest
            if stall > stag_limit and used[0] < budget:
                n_new = len(PX)
                NX = random_points(n_new)
                if best["x"] is not None:
                    NX[0] = best["x"].copy()
                NF = np.empty(n_new)
                NG = []
                for i in range(n_new):
                    if used[0] >= budget:
                        NF[i] = np.inf
                        NG.append(big_g())
                    else:
                        f, g = evaluate(NX[i])
                        NF[i] = f
                        NG.append(g)
                NG = np.array(NG)
                update_weights()
                PX, PF, PG = NX, NF, NG
                PV = viol(PF, PG)
                Fp = np.full(n_new, 0.5)
                CRp = np.full(n_new, 0.9)
                stall = 0
                sigma_scale = 0.1

        # ---------- result ----------
        if best["x"] is None:
            order = rank(PF, PV)
            return np.inf, PX[order[0]].copy()
        return best["f"], best["x"]