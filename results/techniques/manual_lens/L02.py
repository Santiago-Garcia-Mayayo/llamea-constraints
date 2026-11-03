import numpy as np


class Algorithm:
    """
    Constrained mixed-integer optimiser: epsilon-constrained SHADE with a feasibility-gated epsilon
    schedule, anchor-centred differential moves and success-adaptive restarts.
      * epsilon-constrained ranking; epsilon starts at the median positive violation and decays cubically
        to zero at 40% of the budget, but only while enough population members are epsilon-feasible
        (otherwise the tolerance is held so the search keeps a usable feasible-side population),
      * current-to-pbest/1 DE with SHADE success-history memories for F and CR,
      * type-aware variation (categorical inheritance/resampling, sparse ordinal/binary moves),
      * elastic bisection repair toward the anchor (incumbent, or least-violating point before feasibility),
      * feasibility-seeking interpolation from the anchor toward feasible population members,
      * anchor-centred DE moves using two random population/archive members,
      * integer sweeps whose variable selection is weighted by past improvement success,
      * evaluation cache so duplicate points never consume budget,
      * per-coordinate adaptive pattern steps, extrapolation along a smoothed improving direction,
        Gaussian 1/5-rule steps, and elite-preserving multi-scale restarts whose trigger adapts to success.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random
        d = self.dim
        budget = self.budget

        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)

        cont = np.zeros(d, dtype=bool)
        cont[np.asarray(func.continuous_indices, dtype=int)] = True
        ints = ~cont

        vtypes = list(getattr(func, "variable_types", ["continuous"] * d))
        nv = np.ones(d, dtype=float)
        if getattr(func, "n_values", None) is not None:
            nv_given = np.asarray(func.n_values, dtype=float)
            nv[ints] = nv_given[ints]
        if len(vtypes) == d:
            ordinal = np.array([t == "ordinal" for t in vtypes], dtype=bool)
            binary = np.array([t == "binary" for t in vtypes], dtype=bool)
        else:
            ordinal = np.zeros(d, dtype=bool)
            binary = np.zeros(d, dtype=bool)
        cat = ints & ~ordinal & ~binary
        ord_d = ints & ordinal
        bin_d = ints & binary
        int_idx = np.flatnonzero(ints)
        cont_idx = np.flatnonzero(cont)

        lo = np.where(cont, lb, 0.0)
        hi = np.where(cont, ub, nv - 1.0)
        span = np.maximum(hi - lo, 1e-12)

        def repair(x):
            x = np.clip(x, lo, hi)
            x[ints] = np.rint(x[ints])
            return x

        def random_point():
            x = lo + rng.rand(d) * (hi - lo)
            if ints.any():
                x[ints] = np.floor(rng.rand(ints.sum()) * nv[ints])
            return repair(x)

        state = {"evals": 0, "best_f": np.inf, "best_x": None,
                 "lv_v": np.inf, "lv_f": np.inf, "lv_x": None}
        cache = {}

        def evaluate(x):
            k = x.tobytes()
            if k in cache:
                return cache[k]
            f, g = func(x)
            state["evals"] += 1
            f = float(f)
            g = np.asarray(g, dtype=float).ravel()
            g = np.nan_to_num(g, nan=1e10, posinf=1e10, neginf=0.0)
            viol = float(np.sum(np.maximum(g, 0.0)))
            if not np.isfinite(f):
                f = np.inf
                viol = max(viol, 1e10)
            if viol <= 0.0 and np.isfinite(f):
                if f < state["best_f"]:
                    state["best_f"] = f
                    state["best_x"] = np.array(x, dtype=float, copy=True)
            else:
                viol = max(viol, 1e-300)
                if (viol < state["lv_v"]) or (viol == state["lv_v"] and f < state["lv_f"]):
                    state["lv_v"] = viol
                    state["lv_f"] = f
                    state["lv_x"] = np.array(x, dtype=float, copy=True)
            res = (f, viol)
            cache[k] = res
            return res

        def anchor():
            if state["best_x"] is not None:
                return state["best_x"]
            return state["lv_x"]

        def better(f1, v1, f2, v2, eps):
            a = v1 <= eps
            b = v2 <= eps
            if a and b:
                return f1 < f2
            if a != b:
                return a
            return v1 < v2

        def rank_order(Fo, Vi, n, eps):
            return sorted(range(n),
                          key=lambda k: (0, Fo[k]) if Vi[k] <= eps else (1, Vi[k]))

        int_w = np.ones(d)

        def pick_int_vars(k):
            if len(int_idx) == 0:
                return []
            w = int_w[int_idx]
            w = w / w.sum()
            k = min(k, len(int_idx))
            return list(rng.choice(int_idx, size=k, replace=False, p=w))

        def int_candidates(t, j):
            cur = int(round(t[j]))
            if ord_d[j]:
                vals = [cur - 1, cur + 1]
            elif bin_d[j]:
                vals = [1 - cur]
            else:
                vals = [v for v in range(int(nv[j])) if v != cur]
                if len(vals) > 4:
                    vals = [int(v) for v in rng.permutation(vals)[:4]]
            out = []
            for v in vals:
                if 0 <= v < nv[j]:
                    u = t.copy()
                    u[j] = float(v)
                    out.append(u)
            return out

        NP = int(np.clip(6 * d, 12, 60))
        NP = min(NP, max(4, budget // 10))

        X = np.zeros((NP, d))
        Fo = np.full(NP, np.inf)
        Vi = np.full(NP, np.inf)

        n = 0
        while n < NP and state["evals"] < budget:
            X[n] = random_point()
            Fo[n], Vi[n] = evaluate(X[n])
            n += 1
        if n < NP:
            NP = n
            X, Fo, Vi = X[:n], Fo[:n], Vi[:n]
        if NP < 4:
            return state["best_f"], state["best_x"]

        pos_v = Vi[np.isfinite(Vi) & (Vi < 1e9)]
        eps0 = float(np.quantile(pos_v, 0.5)) if pos_v.size > 0 else 0.0
        eps = eps0
        eps_cur = eps0
        decay_end = 0.4

        H = 6
        MF = np.full(H, 0.5)
        MCR = np.full(H, 0.5)
        mem_k = 0

        archive = []
        sigma_ls = 0.1
        pat_step = 0.05
        cstep = np.full(d, 0.05) * span
        ls_succ_ema = 0.2
        stagnation = 0
        restarts = 0
        restart_thr = 8
        dvec = None

        def ls_eval(t):
            nonlocal dvec
            if state["evals"] >= budget:
                return False
            t = repair(t)
            old_anchor = None if anchor() is None else anchor().copy()
            bb_f = state["best_f"]
            ft, vt = evaluate(t)
            imp = (vt <= 0 and ft < bb_f)
            w = rank_order(Fo, Vi, NP, eps)[-1]
            if better(ft, vt, Fo[w], Vi[w], eps):
                X[w] = t
                Fo[w] = ft
                Vi[w] = vt
            moved = imp or (state["best_x"] is None and old_anchor is not None
                            and better(ft, vt, state["lv_f"], state["lv_v"], 0.0)
                            and not np.array_equal(old_anchor, anchor()))
            if imp and old_anchor is not None:
                newd = state["best_x"] - old_anchor
                dvec = newd if dvec is None else 0.6 * newd + 0.4 * dvec
            elif moved and old_anchor is not None and anchor() is not None:
                dvec = anchor() - old_anchor
            return imp

        while state["evals"] < budget:
            progress = state["evals"] / float(budget)
            eps_sched = eps0 * max(0.0, 1.0 - progress / decay_end) ** 3
            # feasibility-gated epsilon: decay only when enough members are epsilon-feasible
            feas_frac = float(np.mean(Vi <= eps_cur))
            if feas_frac >= 0.3 or progress >= 0.8 * decay_end:
                eps_cur = min(eps_cur, eps_sched)
            eps = eps_cur

            order = rank_order(Fo, Vi, NP, eps)
            prev_best = (Fo[order[0]], Vi[order[0]])
            successes = 0
            SF, SCR = [], []
            A = np.array(archive) if len(archive) > 0 else None
            pool = X if A is None else np.vstack([X, A])

            for i in range(NP):
                if state["evals"] >= budget:
                    break

                r = rng.randint(H)
                if MCR[r] < 0:
                    CR = 0.0
                else:
                    CR = float(np.clip(rng.normal(MCR[r], 0.1), 0.0, 1.0))
                F = -1.0
                while F <= 0.0:
                    F = MF[r] + 0.1 * np.tan(np.pi * (rng.rand() - 0.5))
                F = min(F, 1.0)

                p = rng.uniform(2.0 / NP, 0.2)
                n_top = max(2, int(round(p * NP)))
                pb = X[order[rng.randint(n_top)]]

                cands = [k for k in range(pool.shape[0]) if k != i]
                r1 = rng.randint(NP)
                while r1 == i:
                    r1 = rng.randint(NP)
                r2 = cands[rng.randint(len(cands))]
                while r2 == r1:
                    r2 = cands[rng.randint(len(cands))]

                mutant = X[i] + F * (pb - X[i]) + F * (X[r1] - pool[r2])
                jr = rng.randint(d)
                mask = rng.rand(d) < CR
                mask[jr] = True
                trial = np.where(mask, mutant, X[i])

                cm = mask & cat
                if cm.any():
                    pick = rng.rand(d) < 0.5
                    resample = np.floor(rng.rand(d) * nv)
                    trial = np.where(cm & pick, pb, trial)
                    trial = np.where(cm & ~pick, resample, trial)

                if ord_d.any():
                    sm = ord_d & (rng.rand(d) < 1.0 / d)
                    if sm.any():
                        trial[sm] = trial[sm] + rng.choice([-1.0, 1.0], sm.sum())
                if bin_d.any():
                    bm = bin_d & (rng.rand(d) < 1.0 / d)
                    if bm.any():
                        trial[bm] = 1.0 - trial[bm]
                if cat.any():
                    cmut = cat & (rng.rand(d) < 0.5 / d)
                    if cmut.any():
                        trial[cmut] = np.floor(rng.rand(cmut.sum()) * nv[cmut])

                if cont.any() and rng.rand() < 0.1:
                    cm2 = cont & (rng.rand(d) < 0.5)
                    trial[cm2] = pb[cm2] + rng.randn(cm2.sum()) * 0.1 * span[cm2]

                trial = repair(trial)
                ft, vt = evaluate(trial)

                # elastic bisection repair toward the anchor
                if vt > 0 and state["evals"] < budget:
                    has_feas = state["best_x"] is not None
                    p_bis = 0.3 if has_feas else 0.5
                    if rng.rand() < p_bis:
                        target = anchor()
                        if target is None:
                            target = X[order[0]]
                        alpha = 0.2 + 0.6 * rng.rand()
                        mid = repair(alpha * trial + (1.0 - alpha) * target)
                        if not np.array_equal(mid, trial):
                            fm, vm = evaluate(mid)
                            if better(fm, vm, ft, vt, eps):
                                trial, ft, vt = mid, fm, vm

                if better(ft, vt, Fo[i], Vi[i], eps):
                    if len(archive) < NP:
                        archive.append(X[i].copy())
                    else:
                        archive[rng.randint(NP)] = X[i].copy()
                    X[i], Fo[i], Vi[i] = trial, ft, vt
                    SF.append(F)
                    SCR.append(CR)
                    successes += 1
                elif ft == Fo[i] and vt == Vi[i]:
                    X[i] = trial

            if len(SF) > 0:
                sf = np.asarray(SF)
                MF[mem_k] = float(np.sum(sf ** 2) / max(np.sum(sf), 1e-12))
                MCR[mem_k] = float(np.mean(SCR))
                mem_k = (mem_k + 1) % H

            # ---------------- Incumbent / anchor search ----------------
            if anchor() is not None and state["evals"] < budget:
                # 1. extrapolation along the smoothed improving direction
                if dvec is not None and rng.rand() < 0.5 and state["evals"] < budget:
                    ls_eval(anchor() + rng.uniform(0.5, 1.5) * dvec)

                # 1b. feasibility-seeking interpolation toward a feasible population member
                if state["best_x"] is None and state["evals"] < budget:
                    feas_idx = [k for k in range(NP) if Vi[k] <= 0.0 and np.isfinite(Fo[k])]
                    if feas_idx:
                        xf = X[feas_idx[rng.randint(len(feas_idx))]]
                        t = anchor() + rng.uniform(0.3, 0.7) * (xf - anchor())
                        ls_eval(t)
                elif state["best_x"] is not None and state["evals"] < budget and rng.rand() < 0.2:
                    k = rng.randint(NP)
                    t = anchor() + rng.uniform(0.05, 0.3) * (X[k] - anchor())
                    ls_eval(t)

                # 1c. anchor-centred differential move using two random pool members
                if state["evals"] < budget and anchor() is not None and rng.rand() < 0.4:
                    a1 = pool[rng.randint(pool.shape[0])]
                    a2 = pool[rng.randint(pool.shape[0])]
                    Fd = rng.uniform(0.3, 0.8)
                    t = anchor() + Fd * (a1 - a2)
                    if cont.any():
                        t[cont] = t[cont] + rng.randn(cont.sum()) * 0.01 * span[cont]
                    ls_eval(t)

                # 2. Gaussian 1/5-rule steps
                for _ in range(2):
                    if state["evals"] >= budget or anchor() is None:
                        break
                    t = anchor().copy()
                    if cont.any():
                        t[cont] = t[cont] + rng.randn(cont.sum()) * sigma_ls * span[cont]
                    if len(int_idx) > 0 and rng.rand() < 0.3:
                        j = pick_int_vars(1)[0]
                        cands_j = int_candidates(t, j)
                        if len(cands_j) > 0:
                            t = cands_j[0]
                    imp = ls_eval(t)
                    ls_succ_ema = 0.85 * ls_succ_ema + 0.15 * (1.0 if imp else 0.0)
                    if ls_succ_ema > 0.2:
                        sigma_ls *= 1.3
                    else:
                        sigma_ls *= 1.3 ** (-0.25)
                    sigma_ls = float(np.clip(sigma_ls, 1e-5, 0.3))

                # 3. per-coordinate adaptive pattern moves
                if len(cont_idx) > 0:
                    k_c = min(len(cont_idx), 3)
                    pat_imp = False
                    for j in rng.choice(cont_idx, size=k_c, replace=False):
                        if state["evals"] >= budget or anchor() is None:
                            break
                        moved = False
                        for sgn in (1.0, -1.0):
                            t = anchor().copy()
                            t[j] = t[j] + sgn * cstep[j]
                            if ls_eval(t):
                                pat_imp = True
                                moved = True
                                cstep[j] = min(cstep[j] * 1.5, 0.5 * span[j])
                                break
                        if not moved:
                            cstep[j] = max(cstep[j] * 0.7, 1e-4 * span[j])
                    pat_step = float(np.clip(pat_step * (1.5 if pat_imp else 0.8), 1e-4, 0.2))

                # 4. pairwise-direction move
                if dvec is not None and len(cont_idx) > 0 and state["evals"] < budget \
                        and anchor() is not None and rng.rand() < 0.3:
                    t = anchor().copy()
                    t[cont] = t[cont] + 0.5 * rng.uniform(0.5, 1.0) * dvec[cont]
                    j = cont_idx[rng.randint(len(cont_idx))]
                    t[j] = t[j] + rng.randn() * cstep[j]
                    ls_eval(t)

                # 5. success-weighted first-improvement integer sweep
                if len(int_idx) > 0 and state["evals"] < budget and anchor() is not None:
                    k_sel = min(len(int_idx), 3 if stagnation < 2 else 8)
                    for j in pick_int_vars(k_sel):
                        if state["evals"] >= budget or anchor() is None:
                            break
                        for cand in int_candidates(anchor().copy(), j):
                            if state["evals"] >= budget:
                                break
                            if ls_eval(cand):
                                int_w[j] = min(int_w[j] * 1.5, 10.0)
                                break
                        else:
                            int_w[j] = max(int_w[j] * 0.9, 0.2)
                    int_w[int_idx] = np.clip(int_w[int_idx], 0.2, 10.0)

            order = rank_order(Fo, Vi, NP, eps)
            cur = order[0]
            if better(Fo[cur], Vi[cur], *prev_best, eps) or successes > 0:
                stagnation = 0
                restart_thr = min(restart_thr + 0.0, 8)
            else:
                stagnation += 1

            # Elite-preserving multi-scale restart (trigger adapts: earlier restarts when the
            # search repeatedly stalls after a restart without finding a new incumbent)
            if stagnation >= restart_thr and state["evals"] < budget:
                restarts += 1
                keep_n = max(2, NP // 3) if restarts % 3 else max(2, NP // 5)
                archive = []
                for k in order[keep_n:]:
                    if state["evals"] >= budget:
                        break
                    if anchor() is not None and rng.rand() < 0.6:
                        scale = rng.choice([0.02, 0.1, 0.3])
                        p = anchor() + rng.randn(d) * scale * span
                        X[k] = repair(p)
                        if cat.any():
                            X[k][cat] = np.floor(rng.rand(cat.sum()) * nv[cat])
                    else:
                        X[k] = random_point()
                    Fo[k], Vi[k] = evaluate(X[k])
                sigma_ls = max(sigma_ls, 0.05)
                pat_step = max(pat_step, 0.02)
                cstep = np.maximum(cstep, 0.02 * span)
                ls_succ_ema = 0.2
                int_w[:] = 1.0
                stagnation = 0
                restart_thr = 6 if state["best_x"] is not None else 8

        return state["best_f"], state["best_x"]