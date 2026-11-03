import numpy as np


class Algorithm:
    """
    Epsilon-constrained SHADE with discounted bandit restarts and adaptive local search.
      * Epsilon tolerance starts at the median violation of the initial population and decays
        cubically; forced to zero if no feasible point is found by 50% of budget.
      * Duplicate cache: repeated points cost no budget.
      * Feasible / least-violating elite archives (diversity filtered) seed restarts.
      * Restart type (feasible elite, infeasible elite, fresh LHS) is chosen by a discounted
        Thompson bandit, so recent successes dominate and non-stationarity is tolerated.
      * Incumbent local search uses per-variable 1/5-rule step sizes (continuous) and
        ordinal +-1/+-2 moves (integer); stalls trigger boundary bisection, integer sweeps
        and an elite-centred restart whose radius adapts to elite spread.
    """

    BIG = 1e30

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        d = self.dim
        budget = self.budget

        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        is_int = np.array([t != "continuous" for t in func.variable_types], dtype=bool)
        idx_int = np.where(is_int)[0]
        idx_cont = np.where(~is_int)[0]
        nv = np.ones(d)
        for i in idx_int:
            nv[i] = float(int(func.n_values[i]))

        BIG = self.BIG
        cache = {}
        st = {"evals": 0, "best_u": None, "best_f": np.inf,
              "lv_u": None, "lv_v": np.inf, "eps0": 0.0, "feas_ratio": 0.0}
        feas_elite = []
        inf_elite = []
        ELITE_K = 10

        def decode(u):
            x = lb + u * (ub - lb)
            if idx_int.size:
                x[idx_int] = np.clip(np.floor(u[idx_int] * nv[idx_int]), 0, nv[idx_int] - 1)
            return x

        def add_elite(u, viol, f):
            if viol <= 0:
                for _, ue in feas_elite:
                    if np.array_equal(ue, u):
                        return
                feas_elite.append((f, u.copy()))
                feas_elite.sort(key=lambda t: t[0])
                del feas_elite[ELITE_K:]
            else:
                for _, ue in inf_elite:
                    if np.array_equal(ue, u):
                        return
                inf_elite.append((viol, u.copy()))
                inf_elite.sort(key=lambda t: t[0])
                del inf_elite[ELITE_K:]

        def ev(u):
            u = np.clip(u, 0.0, 1.0)
            x = decode(u)
            key = tuple(np.round(x, 9))
            if key in cache:
                return cache[key]
            if st["evals"] >= budget:
                return None
            f, g = func(x)
            st["evals"] += 1
            g = np.atleast_1d(np.asarray(g, dtype=float))
            g = np.nan_to_num(g, nan=1e6, posinf=1e6, neginf=0.0)
            viol = float(np.sum(np.maximum(g, 0.0)))
            f = float(f)
            if np.isnan(f) or not np.isfinite(f):
                f = BIG
                if viol <= 0:
                    viol = 1e6
            res = (viol, f)
            if viol <= 0:
                if f < st["best_f"]:
                    st["best_f"] = f
                    st["best_u"] = u.copy()
            else:
                if viol < st["lv_v"]:
                    st["lv_v"] = viol
                    st["lv_u"] = u.copy()
            add_elite(u, viol, f)
            cache[key] = res
            return res

        def lt(a, b, eps):
            va, fa = a
            vb, fb = b
            if va <= eps and vb <= eps:
                return fa < fb
            if va != vb:
                return va < vb
            return fa < fb

        def skey(t, eps):
            if t[0] <= eps:
                return (0, t[1], 0.0)
            return (1, t[0], t[1])

        def eps_now():
            frac = st["evals"] / budget
            prog = min(1.0, frac + 0.15 * st["feas_ratio"])
            if st["best_u"] is None and frac > 0.5:
                return 0.0
            return st["eps0"] * max(0.0, 1.0 - prog / 0.6) ** 3

        def rand_level(k):
            return (rng.integers(int(nv[k])) + 0.5) / nv[k]

        def lhs(n):
            u = np.empty((n, d))
            for j in range(d):
                u[:, j] = (rng.permutation(n) + rng.random(n)) / n
            for k in idx_int:
                u[:, k] = (np.floor(u[:, k] * nv[k]) + 0.5) / nv[k]
            return u

        def elite_spread():
            if len(feas_elite) < 2:
                return 0.2
            P = np.array([e[1] for e in feas_elite])
            s = float(np.mean(np.std(P, axis=0)))
            return float(np.clip(s, 0.01, 0.3))

        def make_pop(n, center=None, scale=0.0):
            if center is None:
                pop = lhs(n)
            else:
                pop = center[None, :] + scale * rng.standard_normal((n, d))
                for i in range(n):
                    for k in idx_int:
                        if rng.random() < 0.2:
                            pop[i, k] = rand_level(k)
                pop = np.clip(pop, 0.0, 1.0)
                pop[0] = center.copy()
            fit = []
            for i in range(n):
                fi = ev(pop[i])
                if fi is None:
                    return None, None
                fit.append(fi)
            return pop, fit

        def int_sweep(base, bfit, eps):
            improved_any = True
            while improved_any:
                improved_any = False
                for k in idx_int:
                    lvl = int(np.floor(base[k] * nv[k]))
                    lvl = min(max(lvl, 0), int(nv[k]) - 1)
                    for dl in (-1, 1, -2, 2):
                        nl = lvl + dl
                        if nl < 0 or nl >= int(nv[k]):
                            continue
                        t = base.copy()
                        t[k] = (nl + 0.5) / nv[k]
                        ft = ev(t)
                        if ft is None:
                            return None, None
                        if lt(ft, bfit, eps):
                            base, bfit = t, ft
                            improved_any = True
                            break
            return base, bfit

        def boundary_search(u_feas, u_inf, steps=4):
            lo, hi = 0.0, 1.0
            for _ in range(steps):
                t = 0.5 * (lo + hi)
                m = np.clip(u_feas + t * (u_inf - u_feas), 0.0, 1.0)
                r = ev(m)
                if r is None:
                    return None, False
                if r[0] <= 0:
                    lo = t
                else:
                    hi = t
            m = np.clip(u_feas + lo * (u_inf - u_feas), 0.0, 1.0)
            if ev(m) is None:
                return None, False
            return m, True

        NP0 = max(10, min(40, 4 * d))
        NP = NP0
        restarts = 0
        H = 6
        MF = np.full(H, 0.5)
        MCR = np.full(H, 0.5)
        mk = 0
        archive = []
        sgv = np.full(d, 0.05)
        succ_rate_cont = np.full(d, 0.2)  # per-variable 1/5-rule tracker

        DECAY = 0.9
        arm_succ = np.ones(3)
        arm_tries = np.ones(3) * 2.0

        pop, fit = make_pop(NP)
        if pop is None:
            return self._result(st, decode)
        viols = np.array([t[0] for t in fit])
        st["eps0"] = float(np.median(viols)) if np.any(viols > 0) else 0.0

        prev_best = st["best_f"]
        prev_lv = st["lv_v"]
        stall = 0
        out = False
        gens = 0
        max_gens = 5 * budget

        while st["evals"] < budget and not out and gens < max_gens:
            gens += 1
            eps = eps_now()
            order = sorted(range(NP), key=lambda k: skey(fit[k], eps))
            st["feas_ratio"] = float(np.mean([t[0] <= eps for t in fit]))

            SF, SCR = [], []
            for i in range(NP):
                r = rng.integers(H)
                Fi = MF[r] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
                while Fi <= 0:
                    Fi = MF[r] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
                Fi = min(Fi, 1.0)
                CRi = float(np.clip(rng.normal(MCR[r], 0.1), 0.0, 1.0))

                p = rng.uniform(2.0 / NP, 0.2)
                ntop = max(1, int(p * NP))
                pb = order[rng.integers(ntop)]

                cand = [k for k in range(NP) if k != i]
                r1 = cand[rng.integers(len(cand))]
                pool_n = NP + len(archive)
                while True:
                    r2 = rng.integers(pool_n)
                    if r2 != i and r2 != r1:
                        break
                x_r2 = pop[r2] if r2 < NP else archive[r2 - NP]

                mutant = pop[i] + Fi * (pop[pb] - pop[i]) + Fi * (pop[r1] - x_r2)
                mutant = np.where(mutant < 0.0, pop[i] / 2.0, mutant)
                mutant = np.where(mutant > 1.0, (pop[i] + 1.0) / 2.0, mutant)

                jr = rng.integers(d)
                cmask = rng.random(d) < CRi
                cmask[jr] = True
                trial = np.where(cmask, mutant, pop[i])
                for k in idx_int:
                    if rng.random() < 1.0 / d:
                        trial[k] = rand_level(k)

                ft = ev(trial)
                if ft is None:
                    out = True
                    break

                old = fit[i]
                if not lt(old, ft, eps):
                    if lt(ft, old, eps):
                        SF.append(Fi)
                        SCR.append(CRi)
                    archive.append(pop[i].copy())
                    pop[i] = trial
                    fit[i] = ft

            if out:
                break

            if SF:
                SFa = np.asarray(SF)
                MF[mk] = np.sum(SFa ** 2) / max(np.sum(SFa), 1e-12)
                MCR[mk] = float(np.mean(SCR))
                mk = (mk + 1) % H
            while len(archive) > NP:
                archive.pop(rng.integers(len(archive)))

            if st["best_u"] is not None:
                base = st["best_u"].copy()
                ls_n = max(4, d // 2) + 2
            elif st["lv_u"] is not None:
                base = st["lv_u"].copy()
                ls_n = max(4, d // 2)
            else:
                base = None
                ls_n = 0

            bfit = None
            if base is not None:
                bfit = ev(base)
                if bfit is None:
                    break

            for _ in range(ls_n):
                if base is None:
                    break
                trial = base.copy()
                move_k = None
                if idx_int.size and (idx_cont.size == 0 or rng.random() < 0.5):
                    k = int(rng.choice(idx_int))
                    lvl = int(np.floor(trial[k] * nv[k]))
                    lvl = min(max(lvl, 0), int(nv[k]) - 1)
                    u_ = rng.random()
                    if u_ < 0.1:
                        newl = int(rng.integers(int(nv[k])))
                    elif u_ < 0.25:
                        newl = lvl + int(rng.choice([-2, 2]))
                    else:
                        newl = lvl + int(rng.choice([-1, 1]))
                    newl = min(max(newl, 0), int(nv[k]) - 1)
                    trial[k] = (newl + 0.5) / nv[k]
                elif idx_cont.size:
                    if rng.random() < 0.3 and idx_cont.size > 1:
                        m = max(2, idx_cont.size // 3)
                        ks = rng.choice(idx_cont, size=m, replace=False)
                        s = float(np.mean(sgv[ks]))
                        trial[ks] += s * rng.normal(size=m)
                    else:
                        k = int(rng.choice(idx_cont))
                        trial[k] += sgv[k] * rng.normal()
                        move_k = k
                else:
                    continue
                trial = np.clip(trial, 0.0, 1.0)
                ft = ev(trial)
                if ft is None:
                    out = True
                    break
                if not lt(bfit, ft, eps):
                    base = trial
                    bfit = ft
                    if move_k is not None:
                        sgv[move_k] = min(sgv[move_k] * 1.3, 0.3)
                        succ_rate_cont[move_k] = 0.9 * succ_rate_cont[move_k] + 0.1
                elif move_k is not None:
                    sgv[move_k] = max(sgv[move_k] * 0.93, 1e-4)
                    succ_rate_cont[move_k] = 0.9 * succ_rate_cont[move_k]

            if out:
                break

            if base is not None:
                worst = max(range(NP), key=lambda k: skey(fit[k], eps))
                if lt(bfit, fit[worst], eps):
                    pop[worst] = base.copy()
                    fit[worst] = bfit

            improved = (st["best_f"] < prev_best) or (st["lv_v"] < prev_lv)
            prev_best = st["best_f"]
            prev_lv = st["lv_v"]
            if improved:
                stall = 0
            else:
                stall += 1

            if stall >= 10:
                best_before_stall = st["best_f"]
                lv_before_stall = st["lv_v"]

                # (i) feasibility-boundary bisection, inject feasible-side point
                if st["best_u"] is not None and inf_elite:
                    ub_feas = st["best_u"].copy()
                    for _, ui in inf_elite[:2]:
                        m, ok = boundary_search(ub_feas, ui.copy())
                        if not ok:
                            out = True
                            break
                        if m is not None:
                            rm = ev(m)
                            if rm is not None and rm[0] <= 0:
                                wk = max(range(NP), key=lambda k: skey(fit[k], eps))
                                if lt(rm, fit[wk], eps):
                                    pop[wk] = m.copy()
                                    fit[wk] = rm
                    if out:
                        break

                # (ii) integer sweep of incumbent
                if st["best_u"] is not None and idx_int.size:
                    sb = st["best_u"].copy()
                    sf = ev(sb)
                    if sf is None:
                        break
                    sb, sf = int_sweep(sb, sf, eps)
                    if sb is None:
                        break

                # (iii) integer sweep of second feasible elite
                if idx_int.size and len(feas_elite) > 1:
                    e_u = feas_elite[1][1].copy()
                    ef = ev(e_u)
                    if ef is None:
                        break
                    e_u, ef = int_sweep(e_u, ef, eps)
                    if e_u is None:
                        break

                # discounted Thompson bandit over restart types
                samples = rng.beta(arm_succ, arm_tries - arm_succ + 1e-9 + 1.0)
                avail = np.array([len(feas_elite) > 0, len(inf_elite) > 0, True])
                samples = np.where(avail, samples, -1.0)
                arm_used = int(np.argmax(samples))

                restarts += 1
                MF[:] = 0.5
                MCR[:] = 0.5
                mk = 0
                archive = []
                sgv = np.full(d, 0.05)
                eps = eps_now()

                if arm_used == 0:
                    j = min(int(rng.exponential(2.0)), len(feas_elite) - 1)
                    center = feas_elite[j][1].copy()
                    scale = float(elite_spread() * rng.choice([0.5, 1.0, 2.0]))
                    NPl = max(8, NP0 // 2)
                    pop, fit = make_pop(NPl, center=center, scale=scale)
                    NP = NPl
                elif arm_used == 1:
                    j = min(int(rng.exponential(2.0)), len(inf_elite) - 1)
                    center = inf_elite[j][1].copy()
                    scale = float(rng.choice([0.05, 0.15]))
                    NPl = max(8, NP0 // 2)
                    pop, fit = make_pop(NPl, center=center, scale=scale)
                    NP = NPl
                else:
                    NP = min(80, int(NP0 * (1.5 ** ((restarts + 1) // 2))))
                    pop, fit = make_pop(NP)

                if pop is None:
                    break

                new_best = st["best_f"] < best_before_stall
                new_lv = st["lv_v"] < lv_before_stall
                arm_succ *= DECAY
                arm_tries *= DECAY
                arm_tries[arm_used] += 1.0
                if new_best or new_lv:
                    arm_succ[arm_used] += 1.0

                stall = 0
                prev_best = st["best_f"]
                prev_lv = st["lv_v"]

        return self._result(st, decode)

    @staticmethod
    def _result(st, decode):
        if st["best_u"] is None:
            return np.inf, None
        return st["best_f"], decode(st["best_u"])