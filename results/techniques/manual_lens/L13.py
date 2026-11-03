import numpy as np


class Algorithm:
    """
    Mixed-integer DE with epsilon-constrained ranking, JADE-style success-history
    F/CR adaptation, a diversity-aware feasible archive, and dominant-constraint
    repair by bisection towards the nearest feasible archive member.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng()
        dim = self.dim
        budget = self.budget

        lb = np.array(func.bounds.lb, dtype=float).copy()
        ub = np.array(func.bounds.ub, dtype=float).copy()
        vtypes = list(func.variable_types)
        is_int = np.array([vt != "continuous" for vt in vtypes], dtype=bool)
        nvals = np.ones(dim, dtype=int)
        for i in range(dim):
            if is_int[i]:
                nvals[i] = int(func.n_values[i])
                lb[i] = 0.0
                ub[i] = nvals[i] - 1
        ilist = np.where(is_int)[0]
        clist = np.where(~is_int)[0]
        m = max(1, int(getattr(func, "n_constraints", 1)))
        maxv = int(max(2, nvals.max()))

        span = ub - lb
        sspan = np.where(span > 0, span, 1.0)

        st = {"evals": 0, "bx": None, "bf": np.inf}
        archive = []  # list of (f, x) feasible points sorted by f
        ARCH_MAX = 40
        eps_s = {"e": 0.0, "e0": 0.0}
        loc = {"sigma": 0.02}
        credit = np.ones((dim, m))
        imem = np.zeros((dim, maxv, m))
        pstep = np.where(is_int, 0.0, 0.05 * span)
        share = np.full(m, 1.0 / m)
        jade = {"muF": 0.5, "muCR": 0.5}

        def repair(x):
            x = np.array(x, dtype=float)
            if len(ilist):
                x[ilist] = np.clip(np.rint(x[ilist]), 0, nvals[ilist] - 1)
            if len(clist):
                x[clist] = np.clip(x[clist], lb[clist], ub[clist])
            return x

        def random_point():
            x = np.zeros(dim)
            if len(clist):
                x[clist] = rng.uniform(lb[clist], ub[clist])
            if len(ilist):
                x[ilist] = np.floor(rng.random(len(ilist)) * nvals[ilist])
            return x

        def random_ints():
            return np.floor(rng.random(len(ilist)) * nvals[ilist])

        def dist(a, b):
            return float(np.mean(np.abs(a - b) / sspan))

        def add_archive(f, x):
            # diversity-aware insertion: near-duplicates only replace if better
            for k in range(len(archive)):
                fa, xa = archive[k]
                if dist(xa, x) < 1e-3:
                    if f < fa:
                        archive[k] = (f, x.copy())
                        archive.sort(key=lambda t: t[0])
                    return
            archive.append((f, x.copy()))
            archive.sort(key=lambda t: t[0])
            if len(archive) > ARCH_MAX:
                del archive[ARCH_MAX:]

        def nearest_feas(x):
            if not archive:
                return None
            best = None
            bd = np.inf
            for fa, xa in archive:
                d = dist(xa, x)
                if d < bd and d > 1e-9:
                    bd = d
                    best = xa
            return best

        def ev(x):
            if st["evals"] >= budget:
                return None
            x = repair(x)
            f, g = func(x)
            st["evals"] += 1
            try:
                f = float(f)
            except Exception:
                f = np.inf
            g = np.asarray(g, dtype=float).reshape(-1)
            if g.size < m:
                g = np.concatenate([g, np.zeros(m - g.size)])
            g = g[:m]
            g = np.nan_to_num(g, nan=1e6, posinf=1e6, neginf=0.0)
            g = np.maximum(g, 0.0)
            if not np.isfinite(f):
                f = np.inf
                g = g + 1.0
            if not np.any(g > 0):
                if f < st["bf"]:
                    st["bf"] = f
                    st["bx"] = x.copy()
                if np.isfinite(f):
                    add_archive(f, x)
            return x, f, g

        def feas(g):
            return not np.any(g > 0)

        def V(g):
            return float(np.sum(np.maximum(g, 0.0)))

        def less(fa, ga, fb, gb):
            va, vb = V(ga), V(gb)
            e = eps_s["e"]
            if va <= e and vb <= e:
                return fa < fb
            return va < vb

        def rank_order(F, G):
            e = eps_s["e"]
            keys = []
            for k in range(len(F)):
                v = V(G[k])
                if v <= e:
                    keys.append((0, F[k], 0.0))
                else:
                    keys.append((1, v, F[k]))
            return sorted(range(len(F)), key=lambda k: keys[k])

        def learn_int(j, v, parent_g, child_g, dc):
            red = parent_g[dc] - child_g[dc]
            imem[j, int(v), dc] = 0.8 * imem[j, int(v), dc] + 0.2 * red

        def mem_pick(j, cur, dc):
            nv = int(nvals[j])
            vals = [v for v in range(nv) if v != int(cur)]
            if not vals:
                return None
            if rng.random() < 0.3:
                return float(vals[rng.integers(len(vals))])
            sc = np.array([imem[j, v, dc] for v in vals])
            sc = sc - sc.max()
            pr = np.exp(3.0 * sc)
            pr = pr / pr.sum()
            return float(vals[rng.choice(len(vals), p=pr)])

        def boundary_search(x, d, mask_int, steps):
            # bisection from infeasible x towards feasible d; returns best feasible found
            base = x.copy()
            if len(ilist):
                base[ilist] = np.where(mask_int, d[ilist], x[ilist])
            lo, hi = 0.0, 1.0
            best = None
            for _ in range(steps):
                t = 0.5 * (lo + hi)
                y = base.copy()
                if len(clist):
                    y[clist] = base[clist] + t * (d[clist] - base[clist])
                r = ev(y)
                if r is None:
                    break
                if feas(r[2]):
                    hi = t
                    if best is None or less(r[1], r[2], best[1], best[2]):
                        best = r
                else:
                    lo = t
            return best

        def dom_sweep(bx, bf, bg, dc, k_vars):
            p = credit[:, dc] / credit[:, dc].sum()
            cand = rng.choice(dim, size=min(k_vars, dim), replace=False, p=p)
            for j in cand:
                if is_int[j]:
                    if nvals[j] <= 1:
                        continue
                    val = mem_pick(j, bx[j], dc)
                    if val is None:
                        continue
                    opts = [val]
                    pool = [v for v in range(int(nvals[j])) if v != int(bx[j])]
                    if pool:
                        opts.append(float(pool[rng.integers(len(pool))]))
                else:
                    opts = [bx[j] + rng.normal(0.0, 0.1 * span[j]) for _ in range(2)]
                for val in opts:
                    y = bx.copy()
                    y[j] = val
                    r = ev(y)
                    if r is None:
                        return None
                    red = bg[dc] - r[2][dc]
                    if is_int[j]:
                        learn_int(j, val, bg, r[2], dc)
                    if red > 1e-12:
                        credit[j, dc] += min(1.0, red)
                    if less(r[1], r[2], bf, bg):
                        bx, bf, bg = r[0], r[1], r[2]
                        credit[j, dc] += 1.0
                        break
            credit[:, dc] = np.maximum(credit[:, dc] * 0.99, 0.1)
            imem[:, :, dc] *= 0.995
            return bx, bf, bg

        def donor_patch(xi, fi, gi, xf):
            diff = [j for j in ilist if xi[j] != xf[j]]
            if not diff:
                return xi, fi, gi
            rng.shuffle(diff)
            for j in diff[:4]:
                y = xi.copy()
                y[j] = xf[j]
                r = ev(y)
                if r is None:
                    return None
                if less(r[1], r[2], fi, gi):
                    xi, fi, gi = r[0], r[1], r[2]
            return xi, fi, gi

        def pattern_probe(x, f, g, n_try):
            xb, fb, gb = x.copy(), f, g.copy()
            cands = rng.permutation(dim)[:min(dim, n_try)]
            for j in cands:
                if is_int[j]:
                    if nvals[j] <= 1:
                        continue
                    opts = [v for v in range(int(nvals[j])) if v != int(xb[j])]
                    if not opts:
                        continue
                    y = xb.copy()
                    y[j] = float(opts[rng.integers(len(opts))])
                    r = ev(y)
                    if r is None:
                        return None
                    if feas(r[2]) and r[1] < fb:
                        xb, fb, gb = r[0], r[1], r[2]
                else:
                    s = pstep[j]
                    improved = False
                    for sgn in (1.0, -1.0):
                        y = xb.copy()
                        y[j] = xb[j] + sgn * s
                        r = ev(y)
                        if r is None:
                            return None
                        if feas(r[2]) and r[1] < fb:
                            xb, fb, gb = r[0], r[1], r[2]
                            improved = True
                            break
                    if improved:
                        pstep[j] = min(0.5 * span[j], pstep[j] * 1.5)
                    else:
                        pstep[j] = max(1e-5 * span[j], pstep[j] * 0.7)
            return xb, fb, gb

        NP = max(10, min(50, 3 * dim))
        NP = min(NP, max(4, budget // 5))

        X = np.zeros((NP, dim))
        Fo = np.full(NP, np.inf)
        Go = np.full((NP, m), 1.0)
        Fs = np.full(NP, 0.5)
        CRs = np.full(NP, 0.9)

        # ---- initialisation ----
        for i in range(NP):
            r = ev(random_point())
            if r is None:
                return st["bf"], st["bx"]
            X[i], Fo[i], Go[i] = r

        Vs = np.array([V(Go[k]) for k in range(NP)])
        eps_s["e0"] = float(np.median(Vs)) if np.median(Vs) > 0 else 0.0
        eps_s["e"] = eps_s["e0"]

        order = rank_order(Fo, Go)
        last_best = (Fo[order[0]], Go[order[0]].copy())
        stall = 0
        stall_limit = 8 + dim // 2
        succ_ref = 0
        tot_ref = 0
        gen = 0
        hard = np.full(m, 0.5)

        # ---- main loop ----
        while st["evals"] < budget:
            gen += 1
            frac = st["evals"] / budget
            feas_mask = np.array([feas(Go[k]) for k in range(NP)])
            feas_ratio = float(np.mean(feas_mask))
            feas_any = bool(feas_mask.any()) or len(archive) > 0

            # adaptive epsilon relaxation
            if eps_s["e0"] > 0:
                e_time = eps_s["e0"] * max(0.0, 1.0 - frac / 0.6) ** 3
                e_cur = eps_s["e"]
                if feas_ratio >= 0.5:
                    e_cur *= 0.9
                eps_s["e"] = 0.0 if min(e_time, e_cur) < 1e-9 else min(e_time, e_cur)

            viol = np.maximum(Go, 0.0)
            rate = np.mean(viol > 0, axis=0)
            hard = 0.8 * hard + 0.2 * rate
            mv = np.mean(viol, axis=0)
            share = 0.8 * share + 0.2 * (mv / (mv.sum() + 1e-12))
            w = 1.0 + 4.0 * hard / (np.max(hard) + 1e-12)
            dom = int(np.argmax(hard * mv * (0.5 + share)))
            if not feas_any:
                w[dom] *= 2.0

            order = rank_order(Fo, Go)
            bi = order[0]
            pnum = max(2, int(0.15 * NP))
            pset = order[:pnum]
            ib = None
            for k in order:
                if not feas(Go[k]):
                    ib = k
                    break

            repair_scale = 1.0 - 0.6 * feas_ratio
            sweep_p = 0.8 * repair_scale if feas_any else 0.8
            if ib is not None and (not feas_any or gen % 3 == 0) and rng.random() < sweep_p:
                dc = int(np.argmax(w * np.maximum(Go[ib], 0.0) * (0.5 + share)))
                k_vars = 6 if not feas_any else 4
                res = dom_sweep(X[ib], Fo[ib], Go[ib], dc, k_vars)
                if res is None:
                    return st["bf"], st["bx"]
                xb, fb, gb = res
                if less(fb, gb, Fo[ib], Go[ib]):
                    X[ib], Fo[ib], Go[ib] = xb, fb, gb
                order = rank_order(Fo, Go)
                bi = order[0]

            # nearest-feasible boundary repair for the most promising infeasible member
            if ib is not None and feas_any and rng.random() < 0.5 * repair_scale + 0.1:
                fe_near = nearest_feas(X[ib])
                if fe_near is None and st["bx"] is not None:
                    fe_near = st["bx"]
                if fe_near is not None:
                    mask = rng.random(len(ilist)) < 0.5
                    r = boundary_search(X[ib], fe_near, mask, 4)
                    if st["evals"] >= budget:
                        return st["bf"], st["bx"]
                    if r is not None and less(r[1], r[2], Fo[ib], Go[ib]):
                        X[ib], Fo[ib], Go[ib] = r[0], r[1], r[2]

            fe = [k for k in order if feas(Go[k])]
            if fe and ib is not None and rng.random() < 0.5 * repair_scale + 0.1:
                res = donor_patch(X[ib], Fo[ib], Go[ib], X[fe[0]])
                if res is None:
                    return st["bf"], st["bx"]
                xi, fi, gi = res
                if less(fi, gi, Fo[ib], Go[ib]):
                    X[ib], Fo[ib], Go[ib] = xi, fi, gi

            if fe and (gen % 2 == 0 or feas_ratio > 0.5):
                fi0 = fe[0]
                n_try = 6 if feas_ratio < 0.5 else 10
                res = pattern_probe(X[fi0], Fo[fi0], Go[fi0], n_try)
                if res is None:
                    return st["bf"], st["bx"]
                X[fi0], Fo[fi0], Go[fi0] = res

            # ---- DE offspring (JADE-style) ----
            order = rank_order(Fo, Go)
            pset = order[:pnum]
            succF, succCR = [], []
            for i in range(NP):
                if st["evals"] >= budget:
                    return st["bf"], st["bx"]

                # sample F (Cauchy) and CR (Normal) around success-history means
                Fi = -1.0
                for _ in range(10):
                    Fi = jade["muF"] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
                    if Fi > 0:
                        break
                Fi = 0.5 if Fi <= 0 else min(1.0, Fi)
                CRi = float(np.clip(rng.normal(jade["muCR"], 0.1), 0.0, 1.0))

                others = [j for j in range(NP) if j != i]
                r1, r2 = rng.choice(others, 2, replace=False)
                p = pset[rng.integers(len(pset))]
                rr = rng.random()
                is_refine = False
                use_arch = len(archive) >= 3 and st["bx"] is not None

                if (not feas_any) and rr < 0.25:
                    # constraint-driven move towards the best-ranked member
                    tgt = int(np.argmin(Go[:, dom]))
                    if tgt == i:
                        tgt = bi
                    alpha = rng.uniform(0.1, 1.0)
                    u = X[i].copy()
                    if len(clist):
                        u[clist] = X[i][clist] + alpha * (X[tgt][clist] - X[i][clist])
                    if len(ilist):
                        msk = rng.random(len(ilist)) < alpha
                        u[ilist] = np.where(msk, X[tgt][ilist], X[i][ilist])
                    if len(ilist) and rng.random() < 0.3:
                        j = ilist[rng.integers(len(ilist))]
                        val = mem_pick(j, u[j], dom)
                        if val is not None:
                            u[j] = val
                elif use_arch and rr < 0.18:
                    # feasible-archive difference move
                    a, b = rng.choice(len(archive), 2, replace=False)
                    xa = archive[a][1]
                    xc = archive[b][1]
                    base = st["bx"]
                    u = base.copy()
                    if len(clist):
                        Fd = rng.uniform(0.3, 0.9)
                        u[clist] = base[clist] + Fd * (xa[clist] - xc[clist])
                    if len(ilist):
                        msk = rng.random(len(ilist)) < 0.2
                        u[ilist] = np.where(msk, xa[ilist], base[ilist])
                elif feas_any and rr < 0.10 and st["bx"] is not None:
                    is_refine = True
                    u = st["bx"].copy()
                    if len(clist):
                        u[clist] = u[clist] + rng.normal(0, 1, len(clist)) * loc["sigma"] * (ub[clist] - lb[clist])
                    if len(ilist):
                        msk = rng.random(len(ilist)) < 1.0 / max(1, len(ilist))
                        u[ilist] = np.where(msk, random_ints(), u[ilist])
                elif feas_any and rr < 0.22 and st["bx"] is not None and not feas(Go[i]):
                    alpha = rng.uniform(0.2, 0.9)
                    u = X[i] + alpha * (st["bx"] - X[i])
                    if len(ilist):
                        msk = rng.random(len(ilist)) < 0.5
                        u[ilist] = np.where(msk, st["bx"][ilist], X[i][ilist])
                else:
                    # current-to-pbest/1 with binomial crossover
                    v = X[i].copy()
                    if len(clist):
                        v[clist] = (X[i][clist] + Fi * (X[p][clist] - X[i][clist])
                                    + Fi * (X[r1][clist] - X[r2][clist]))
                    if len(ilist):
                        pick = rng.random(len(ilist)) < 0.5
                        v[ilist] = np.where(pick, X[p][ilist], X[r1][ilist])
                        res_m = rng.random(len(ilist)) < 0.1
                        v[ilist] = np.where(res_m, random_ints(), v[ilist])
                    jr = rng.integers(dim)
                    mask = rng.random(dim) < CRi
                    mask[jr] = True
                    u = np.where(mask, v, X[i])

                r = ev(u)
                if r is None:
                    return st["bf"], st["bx"]
                u, fu, gu = r
                if not feas(Go[i]) and len(ilist):
                    diffs = [j for j in ilist if u[j] != X[i][j]]
                    if len(diffs) == 1:
                        learn_int(diffs[0], u[diffs[0]], Go[i], gu, dom)
                if is_refine:
                    tot_ref += 1
                if less(fu, gu, Fo[i], Go[i]) or (fu == Fo[i] and np.array_equal(gu, Go[i])):
                    if is_refine and feas(gu) and fu < Fo[i]:
                        succ_ref += 1
                    if less(fu, gu, Fo[i], Go[i]) and rr >= 0.25:
                        succF.append(Fi)
                        succCR.append(CRi)
                    X[i] = u
                    Fo[i] = fu
                    Go[i] = gu
                    Fs[i] = Fi
                    CRs[i] = CRi
                if i % 5 == 4:
                    order = rank_order(Fo, Go)

            # JADE update of success-history means
            if succF:
                sF = np.array(succF)
                sCR = np.array(succCR)
                lehmer = np.sum(sF ** 2) / max(np.sum(sF), 1e-12)
                jade["muF"] = 0.9 * jade["muF"] + 0.1 * lehmer
                jade["muCR"] = 0.9 * jade["muCR"] + 0.1 * float(np.mean(sCR))

            # integer-neighbour sweep around the feasible incumbent
            order = rank_order(Fo, Go)
            bi = order[0]
            if feas(Go[bi]) and gen % 5 == 0 and len(ilist) > 0:
                xb = X[bi].copy()
                fb, gb = Fo[bi], Go[bi].copy()
                cand_idx = rng.permutation(ilist)[:min(len(ilist), 6)]
                for j in cand_idx:
                    opts = [v for v in range(int(nvals[j])) if v != int(xb[j])]
                    if not opts:
                        continue
                    rng.shuffle(opts)
                    for v in opts[:2]:
                        y = xb.copy()
                        y[j] = v
                        r = ev(y)
                        if r is None:
                            return st["bf"], st["bx"]
                        if feas(r[2]) and r[1] < fb:
                            xb, fb, gb = r[0], r[1], r[2]
                            break
                X[bi], Fo[bi], Go[bi] = xb, fb, gb

            # 1/5-style adaptation of refinement step size
            if st["bx"] is not None and tot_ref >= 5:
                rate_s = succ_ref / max(1, tot_ref)
                if rate_s > 0.2:
                    loc["sigma"] *= 1.2
                else:
                    loc["sigma"] *= 0.9
                loc["sigma"] = float(np.clip(loc["sigma"], 1e-4, 0.1))
                succ_ref = 0
                tot_ref = 0

            # stagnation check
            order = rank_order(Fo, Go)
            bi = order[0]
            cur_best = (Fo[bi], Go[bi].copy())
            if less(cur_best[0], cur_best[1], last_best[0], last_best[1]):
                stall = 0
            else:
                stall += 1
            last_best = cur_best

            # restart: half re-seeded around incumbent, half random
            if stall >= stall_limit and st["evals"] < budget:
                stall = 0
                best_x = X[bi].copy()
                best_F, best_G = Fo[bi], Go[bi].copy()
                sig_scale = 0.1 if feas(best_G) else 0.2
                for k in range(NP):
                    if k == bi:
                        continue
                    if k % 2 == 0:
                        x = best_x.copy()
                        if len(clist):
                            sigma = sig_scale * (ub[clist] - lb[clist])
                            x[clist] = x[clist] + rng.normal(0, 1, len(clist)) * sigma
                        if len(ilist):
                            msk = rng.random(len(ilist)) < 0.2
                            x[ilist] = np.where(msk, random_ints(), x[ilist])
                    else:
                        x = random_point()
                    r = ev(x)
                    if r is None:
                        return st["bf"], st["bx"]
                    X[k], Fo[k], Go[k] = r
                    Fs[k] = 0.5
                    CRs[k] = 0.9
                X[bi] = best_x
                Fo[bi] = best_F
                Go[bi] = best_G
                Fs[bi] = 0.5
                CRs[bi] = 0.9
                last_best = (Fo[bi], Go[bi].copy())

        return st["bf"], st["bx"]