import numpy as np


class Algorithm:
    """Epsilon-constrained mixed-integer DE with SHADE-style F/CR memory, current-to-pbest/1
    mutation with an external archive, boundary crossover between tournament-selected feasible
    and near-boundary infeasible members, faster epsilon decay, and pattern search injected
    into the population (with doubled effort in the final 15% of budget)."""

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng()
        lo = np.asarray(func.bounds.lb, dtype=float).copy()
        hi = np.asarray(func.bounds.ub, dtype=float).copy()
        d = lo.size
        span = hi - lo
        budget = self.budget

        disc = np.zeros(d, dtype=bool)
        cat = np.zeros(d, dtype=bool)
        nv = np.ones(d)
        for i in range(d):
            t = func.variable_types[i]
            if t != "continuous":
                disc[i] = True
                nv[i] = float(func.n_values[i])
                cat[i] = (t == "categorical")
        dis_idx = np.where(disc)[0]
        cont_idx = np.where(~disc)[0]
        any_disc = bool(disc.any())

        used = 0
        best_f = np.inf
        best_x = None
        best_u = None
        inf_v = np.inf
        inf_f = np.inf
        inf_u = None
        elites = []
        ELITE_CAP = 12
        eps0 = 0.0
        T_EPS = 0.6
        N_STRAT = 4

        def canon(U):
            U = np.array(U, dtype=float)
            U = np.clip(U, 0.0, 1.0 - 1e-9)
            if any_disc:
                U[..., disc] = (np.floor(U[..., disc] * nv[disc]) + 0.5) / nv[disc]
            return U

        def decode(u):
            x = lo + u * span
            if any_disc:
                x[disc] = np.clip(np.round(x[disc]), 0, nv[disc] - 1)
            return x

        def reflect(v):
            v = np.abs(v)
            v = np.where(v > 1.0, 2.0 - v, v)
            return np.clip(v, 0.0, 1.0)

        def eps_cur():
            lim = T_EPS * budget
            if used >= lim or eps0 <= 0.0:
                return 0.0
            return eps0 * (1.0 - used / lim) ** 3

        def add_elite(f, u):
            nonlocal elites
            for ef, eu in elites:
                if np.array_equal(eu, u):
                    return
            elites.append((f, u.copy()))
            elites.sort(key=lambda t: t[0])
            elites = elites[:ELITE_CAP]

        def evaluate(u):
            nonlocal used, best_f, best_x, best_u, inf_v, inf_f, inf_u
            x = decode(u)
            f, g = func(x)
            used += 1
            g = np.asarray(g, dtype=float).ravel()
            g = np.where(np.isfinite(g), g, 1e6)
            v = float(np.sum(np.maximum(g, 0.0)))
            f = float(f)
            if not np.isfinite(f):
                f = 1e30
            if v <= 0.0:
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    best_u = u.copy()
                add_elite(f, u)
            elif v < inf_v:
                inf_v = v
                inf_f = f
                inf_u = u.copy()
            return f, v

        def rules_better(fa, va, fb, vb):
            e = eps_cur()
            a_ok = va <= e
            b_ok = vb <= e
            if a_ok and b_ok:
                return fa < fb
            if a_ok != b_ok:
                return a_ok
            return va < vb

        def rank_order(F, V):
            e = eps_cur()
            key = np.where(V <= e, F, 1e29 + V)
            return np.argsort(key, kind="stable")

        def mutate_discrete(u, pd):
            if dis_idx.size == 0:
                return u
            sel = dis_idx[rng.random(dis_idx.size) < pd]
            for j in sel:
                n = int(nv[j])
                if n <= 1:
                    continue
                k = int(np.floor(u[j] * n))
                if cat[j]:
                    r = int(rng.integers(n - 1))
                    k2 = r if r < k else r + 1
                else:
                    if k == 0:
                        k2 = 1
                    elif k == n - 1:
                        k2 = n - 2
                    else:
                        k2 = k + (1 if rng.random() < 0.5 else -1)
                u[j] = (k2 + 0.5) / n
            return u

        def lhs(n):
            U = np.empty((n, d))
            for j in range(d):
                U[:, j] = (rng.permutation(n) + rng.random(n)) / n
            return canon(U)

        NP = int(np.clip(5 * d, 15, 80))
        NP = min(NP, max(5, budget // 8))

        def init_pop(seeds=None, sigma=0.1):
            nonlocal eps0
            U = lhs(NP)
            if seeds:
                for k, s in enumerate(seeds[:NP]):
                    U[k] = canon(reflect(s + sigma * rng.standard_normal(d)))
            F = np.full(NP, np.inf)
            V = np.full(NP, np.inf)
            for i in range(NP):
                if used >= budget:
                    return None
                F[i], V[i] = evaluate(U[i])
            if eps0 == 0.0 and used < budget:
                sv = np.sort(V[np.isfinite(V)])
                if sv.size > 0:
                    eps0 = float(sv[min(int(0.2 * NP), sv.size - 1)])
            return U, F, V

        def local_search(u0, f0, v0, max_eval):
            u = u0.copy()
            f, v = f0, v0
            steps = np.full(d, 0.08)
            evals = 0
            while evals < max_eval and used < budget:
                improved_any = False
                for j in rng.permutation(d):
                    if evals >= max_eval or used >= budget:
                        break
                    cands = []
                    if disc[j]:
                        n = int(nv[j])
                        if n <= 1:
                            continue
                        k = int(np.floor(u[j] * n))
                        if cat[j]:
                            lv = [l for l in range(n) if l != k]
                            pick = rng.choice(len(lv), size=min(2, len(lv)), replace=False)
                            for p_ in pick:
                                c = u.copy()
                                c[j] = (lv[p_] + 0.5) / n
                                cands.append(c)
                        else:
                            for kk in (k - 1, k + 1):
                                if 0 <= kk < n:
                                    c = u.copy()
                                    c[j] = (kk + 0.5) / n
                                    cands.append(c)
                    else:
                        if steps[j] < 1e-3:
                            continue
                        for s in (steps[j], -steps[j]):
                            c = u.copy()
                            c[j] = np.clip(u[j] + s, 0.0, 1.0)
                            cands.append(c)
                    success = False
                    for c in cands:
                        if evals >= max_eval or used >= budget:
                            break
                        c = canon(c)
                        fc, vc = evaluate(c)
                        evals += 1
                        if rules_better(fc, vc, f, v):
                            u, f, v = c, fc, vc
                            success = True
                            improved_any = True
                            if not disc[j]:
                                steps[j] = min(steps[j] * 1.5, 0.3)
                            break
                    if not success and not disc[j]:
                        steps[j] *= 0.5
                if not improved_any:
                    if cont_idx.size == 0:
                        break
                    steps[cont_idx] *= 0.5
                    if steps[cont_idx].max() < 1e-3:
                        break
            return u, f, v

        def sample_cauchy_F(m):
            for _ in range(20):
                f_ = m + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
                if f_ > 0:
                    return min(f_, 1.0)
            return 0.5

        # ---- initialisation
        state = init_pop(None)
        if state is None:
            if best_x is None:
                return np.inf, decode(np.full(d, 0.5))
            return best_f, best_x
        U, F, V = state

        H = 6
        MF = np.full(H, 0.5)
        MCR = np.full(H, 0.9)
        mem_k = 0
        sigma_g = 0.1
        strat_succ = np.ones(N_STRAT)
        strat_try = np.full(N_STRAT, 2.0)
        archive = []
        no_imp = 0
        gen = 0
        last_best = best_f
        restarts = 0
        p_frac = 0.2

        def boundary_point(order):
            """Tournament-selected feasible parent and near-boundary infeasible parent,
            combined with mild extrapolation to walk along the constraint boundary."""
            e = eps_cur()
            feas = np.where(V <= 0.0)[0]
            near = np.where((V > 0.0) & (V <= max(e, 1e-12) * 3.0 + 1e-12))[0]
            if near.size == 0:
                near = np.where(V > 0.0)[0]
            if feas.size > 0 and near.size > 0:
                # binary tournament on objective among feasible members
                c1, c2 = feas[rng.integers(feas.size)], feas[rng.integers(feas.size)]
                a = c1 if F[c1] <= F[c2] else c2
                # binary tournament on violation among near-boundary members
                d1, d2 = near[rng.integers(near.size)], near[rng.integers(near.size)]
                b = d1 if V[d1] <= V[d2] else d2
                t = rng.uniform(-0.2, 1.0)
                return U[a] + t * (U[b] - U[a]) + 0.3 * sigma_g * rng.standard_normal(d)
            if best_u is not None and inf_u is not None:
                t = rng.random()
                return best_u + t * (inf_u - best_u) + 0.5 * sigma_g * rng.standard_normal(d)
            return U[order[0]] + sigma_g * rng.standard_normal(d)

        while used < budget:
            gen += 1
            order = rank_order(F, V)
            pmax = max(2, int(p_frac * NP))
            pool = order[:pmax]

            p = strat_succ / strat_try + 0.05
            p = p / p.sum()

            SF, SCR, dwts = [], [], []
            arr_pool = np.vstack([U] + archive) if archive else U

            for i in range(NP):
                if used >= budget:
                    break
                r = rng.integers(H)
                Ft = sample_cauchy_F(MF[r])
                CRt = float(np.clip(MCR[r] + 0.1 * rng.standard_normal(), 0.0, 1.0))

                others = np.delete(np.arange(NP), i)
                strat = int(rng.choice(N_STRAT, p=p))
                pb = pool[rng.integers(pool.size)]

                if strat == 0:
                    r1 = others[rng.integers(others.size)]
                    r2 = rng.integers(arr_pool.shape[0])
                    v = U[i] + Ft * (U[pb] - U[i]) + Ft * (U[r1] - arr_pool[r2])
                    mask = rng.random(d) < CRt
                    mask[rng.integers(d)] = True
                    v = reflect(v)
                    u = np.where(mask, v, U[i])
                elif strat == 1:
                    r1, r2, r3 = rng.choice(others, 3, replace=False)
                    v = U[r1] + Ft * (U[r2] - U[r3])
                    mask = rng.random(d) < CRt
                    mask[rng.integers(d)] = True
                    v = reflect(v)
                    u = np.where(mask, v, U[i])
                elif strat == 2:
                    v = U[order[0]] + sigma_g * rng.standard_normal(d)
                    mask = rng.random(d) < CRt
                    mask[rng.integers(d)] = True
                    v = reflect(v)
                    u = np.where(mask, v, U[i])
                else:
                    v = reflect(boundary_point(order))
                    u = v
                u = canon(u)
                u = mutate_discrete(u, 1.5 / d)

                ft, vt = evaluate(u)
                strat_try[strat] += 1
                if rules_better(ft, vt, F[i], V[i]):
                    strat_succ[strat] += 1
                    SF.append(Ft)
                    SCR.append(CRt)
                    dwts.append(max(abs(F[i] - ft), 1e-12) if (V[i] <= 0 and vt <= 0)
                                else 1.0)
                    archive.append(U[i].copy())
                    U[i] = u
                    F[i] = ft
                    V[i] = vt
                    arr_pool = np.vstack([U] + archive) if archive else U
                elif not rules_better(F[i], V[i], ft, vt):
                    U[i] = u
                    F[i] = ft
                    V[i] = vt
            if len(archive) > NP:
                keep = rng.choice(len(archive), size=NP, replace=False)
                archive = [archive[k] for k in keep]

            strat_succ *= 0.95
            strat_try *= 0.95
            strat_try = np.maximum(strat_try, 1.0)

            if SF:
                w = np.asarray(dwts, dtype=float)
                w = w / w.sum()
                sf = np.asarray(SF)
                scr = np.asarray(SCR)
                mf = np.sum(w * sf * sf) / max(np.sum(w * sf), 1e-12)
                s_cr = np.sum(w * scr)
                mcr = np.sum(w * scr * scr) / s_cr if s_cr > 0 else 0.0
                MF[mem_k] = mf
                MCR[mem_k] = mcr
                mem_k = (mem_k + 1) % H

            if strat_try[2] > 2.0:
                rate = (strat_succ[2] - 1.0) / max(strat_try[2] - 2.0, 1.0)
                sigma_g *= 1.2 if rate > 0.2 else 0.85
                sigma_g = float(np.clip(sigma_g, 1e-4, 0.3))

            p_frac = max(0.05, 0.2 - 0.15 * (used / budget))

            if best_f < last_best - 1e-12:
                no_imp = 0
                last_best = best_f
            else:
                no_imp += 1

            # pattern search on best feasible (or least violated if none), injected back;
            # effort doubled in the final 15% of the budget
            stagn = no_imp > 15 + d or (no_imp > 5 and np.mean(np.std(U, axis=0)) < 1e-4)
            periodic = (gen % 10 == 0)
            start_u = best_u if best_u is not None else inf_u
            if (periodic or stagn) and start_u is not None and used < budget:
                rem = budget - used
                if best_u is not None:
                    s_f, s_v = best_f, 0.0
                else:
                    s_f, s_v = inf_f, inf_v
                mult = 2 if used >= 0.85 * budget else 1
                ls_u, ls_f, ls_v = local_search(start_u, s_f, s_v,
                                                min(rem, mult * (4 * d + 10)))
                worst = int(rank_order(F, V)[-1])
                if rules_better(ls_f, ls_v, F[worst], V[worst]):
                    U[worst] = ls_u
                    F[worst] = ls_f
                    V[worst] = ls_v
                last_best = min(last_best, best_f)

            if stagn and used < budget:
                restarts += 1
                seeds = None
                sigma = 0.1
                if elites and restarts % 2 == 1:
                    seeds = [eu for _, eu in elites]
                    rng.shuffle(seeds)
                    sigma = max(0.01, 0.2 * (0.6 ** restarts))
                elif best_u is None and inf_u is not None and restarts % 2 == 0:
                    seeds = [inf_u]
                    sigma = 0.1
                state = init_pop(seeds, sigma=sigma)
                if state is None:
                    break
                U, F, V = state
                MF = np.full(H, 0.5)
                MCR = np.full(H, 0.9)
                mem_k = 0
                sigma_g = 0.1
                strat_succ = np.ones(N_STRAT)
                strat_try = np.full(N_STRAT, 2.0)
                archive = []
                no_imp = 0
                last_best = best_f
                p_frac = 0.2

        if best_x is None:
            return np.inf, decode(np.full(d, 0.5))
        return best_f, best_x