import numpy as np


class Algorithm:
    """
    Mixed-integer (mu + lambda) ES built on archive-diverse bisection.
    Changes over DominantRepairBisectionES:
    - Dominant weight grows faster with the dominant constraint's violation rate.
    - Two-stage dominant repair: randomised interpolation toward the integer-matched
      feasible anchor, followed (when still infeasible) by a short bisection refinement.
    - Near-feasible archive: infeasible points with small dominant-constraint violation
      are kept and used as seeds for repair, so constraint-23-type failures are revisited.
    """

    OPS = ["bisect", "local", "probe", "xover", "de", "es", "domrep"]

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng()
        d = self.dim
        budget = self.budget
        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        vt = list(func.variable_types)
        n_values = np.asarray(func.n_values, dtype=float)

        is_int = np.array([t != "continuous" for t in vt], dtype=bool)
        I = np.flatnonzero(is_int)
        C = np.flatnonzero(~is_int)
        nI, nC = len(I), len(C)

        nv = np.ones(d, dtype=int)
        if nI:
            nv[I] = np.maximum(1, np.rint(n_values[I]).astype(int))
        isbin = np.array([vt[i] == "binary" for i in range(d)], dtype=bool)
        isord = np.array([vt[i] == "ordinal" for i in range(d)], dtype=bool)
        span = np.maximum(ub - lb, 1e-12)

        pm_lo = 0.5 / max(nI, 1)
        pm_hi = 0.5
        pm0 = min(pm_hi, max(pm_lo, 1.0 / max(nI, 1)))
        ls0 = np.log(0.1)
        ls_min = np.log(1e-4)
        ls_max = 0.0

        sig = np.full(d, 0.1)
        sig_lo, sig_hi = 1e-4, 0.5

        mu, lam = 20, 40
        patience = max(200, 20 * lam)
        K_ARCH = 60
        K_NEAR = 30
        OPS = self.OPS
        n_ops = len(OPS)
        ES_IDX = OPS.index("es")
        DOM_IDX = OPS.index("domrep")
        op_score = np.full(n_ops, 0.2)

        S = {"used": 0, "best_f": np.inf, "best_x": None, "last_imp": 0,
             "gsum": None, "gcnt": None, "rate": None, "arch": [], "near": []}

        def repair(x):
            x = np.asarray(x, dtype=float).copy()
            if nI:
                x[I] = np.clip(np.rint(x[I]), 0, nv[I] - 1)
            if nC:
                x[C] = np.clip(x[C], lb[C], ub[C])
            return x

        def random_x():
            x = np.empty(d, dtype=float)
            if nC:
                x[C] = rng.uniform(lb[C], ub[C])
            if nI:
                x[I] = rng.integers(0, nv[I])
            return x

        def archive_add(x, f):
            arch = S["arch"]
            for a in arch:
                if np.array_equal(a["x"], x):
                    return
            arch.append({"x": x.copy(), "f": f})
            if len(arch) > K_ARCH:
                arch.sort(key=lambda a: a["f"])
                del arch[K_ARCH:]

        def near_add(x, g, f):
            near = S["near"]
            dom = int(np.argmax(S["rate"])) if S["rate"] is not None else 0
            near.append({"x": x.copy(), "g": g.copy(), "f": f, "dv": g[dom]})
            if len(near) > K_NEAR:
                near.sort(key=lambda a: a["dv"])
                del near[K_NEAR:]

        def make_ind(x, ls, pm):
            if S["used"] >= budget:
                return None
            x = repair(x)
            S["used"] += 1
            f, g = func(x)
            g = np.nan_to_num(np.asarray(g, dtype=float).ravel(), nan=1e12,
                              posinf=1e12, neginf=0.0)
            g = np.maximum(g, 0.0)
            try:
                f = float(f)
            except Exception:
                f = np.inf
            if not np.isfinite(f):
                f = np.inf
                g = g + 1.0
            if S["gsum"] is None:
                m = len(g)
                S["gsum"] = np.zeros(m)
                S["gcnt"] = np.zeros(m)
                S["rate"] = np.zeros(m)
            pos = g > 0
            S["rate"] = 0.96 * S["rate"] + 0.04 * pos.astype(float)
            feas = bool(np.all(~pos))
            if not feas:
                S["gsum"][pos] += g[pos]
                S["gcnt"][pos] += 1
                if np.isfinite(f):
                    near_add(x, g, f)
            else:
                archive_add(x, f)
                if f < S["best_f"]:
                    S["best_f"] = f
                    S["best_x"] = x.copy()
                    S["last_imp"] = S["used"]
            return {"x": x.copy(), "f": f, "g": g, "feas": feas, "ls": ls, "pm": pm}

        def scale_vec():
            s = np.ones_like(S["gsum"])
            ok = S["gcnt"] > 0
            s[ok] = S["gsum"][ok] / S["gcnt"][ok]
            return np.maximum(s, 1e-9)

        def dominant():
            return int(np.argmax(S["rate"]))

        def weights():
            rate = S["rate"]
            w = 0.5 + 3.0 * rate ** 1.5
            dom = dominant()
            # dominant constraint: boost grows faster with its violation rate
            w[dom] *= 2.0 + 6.0 * rate[dom]
            w[rate > 0.3] *= 1.3
            return w

        def rank(pool, k):
            G = np.array([p["g"] for p in pool], dtype=float)
            F = np.array([p["f"] for p in pool], dtype=float)
            feas = np.all(G <= 0.0, axis=1)
            if S["gsum"] is None:
                V = np.zeros(len(pool))
            else:
                sc = scale_vec()
                Gn = G / sc
                V = (Gn * weights()).sum(axis=1)
                dom = dominant()
                V = V + 8.0 * Gn[:, dom]
            prim = (~feas).astype(int)
            sec = np.where(feas, F, V)
            order = np.lexsort((rng.random(len(pool)), sec, prim))[:k]
            return [pool[i] for i in order]

        def flip_discrete(x, i):
            n = nv[i]
            if n <= 1:
                return
            cur = int(x[i])
            if isbin[i]:
                x[i] = 1 - cur
            elif isord[i] and rng.random() < 0.6:
                new = cur + (1 if rng.random() < 0.5 else -1)
                if new < 0 or new >= n:
                    new = cur - (new - cur)
                x[i] = min(max(new, 0), n - 1)
            else:
                new = int(rng.integers(0, n - 1))
                if new >= cur:
                    new += 1
                x[i] = new

        def mutate_discrete(x, pm):
            if not nI:
                return
            mut = np.flatnonzero(rng.random(nI) < pm)
            for m in mut:
                flip_discrete(x, I[m])

        def update_sig(i, success):
            if success:
                sig[i] = min(sig_hi, sig[i] * 1.3)
            else:
                sig[i] = max(sig_lo, sig[i] * 0.93)

        def base_point():
            arch = S["arch"]
            if len(arch) < 2 or rng.random() < 0.5:
                return S["best_x"].astype(float).copy(), S["best_f"]
            top = sorted(arch, key=lambda a: a["f"])[:min(8, len(arch))]
            a = top[rng.integers(len(top))]
            return a["x"].astype(float).copy(), a["f"]

        def nearest_anchor(x):
            arch = S["arch"]
            r = rng.random()
            if r < 0.5:
                Ax = np.array([a["x"] for a in arch])
                dist = (np.abs(Ax - x) / span).sum(axis=1)
                return arch[int(np.argmin(dist))]
            if r < 0.75:
                return min(arch, key=lambda a: a["f"])
            top = sorted(arch, key=lambda a: a["f"])[:min(10, len(arch))]
            return top[rng.integers(len(top))]

        def integer_matched_anchor(x):
            arch = S["arch"]
            Ax = np.array([a["x"] for a in arch])
            dist = (np.abs(Ax - x) / span).sum(axis=1)
            if nI:
                ham = (Ax[:, I] != x[I]).sum(axis=1).astype(float)
            else:
                ham = np.zeros(len(arch))
            key = ham + 0.1 * dist + 1e-3 * rng.random(len(arch))
            return arch[int(np.argmin(key))]

        def bisect(pa, steps, ls, pm):
            out = []
            if not S["arch"]:
                return out
            anchor = nearest_anchor(pa["x"])
            lo = anchor["x"].copy()
            hi = pa["x"].copy()
            for _ in range(steps):
                if S["used"] >= budget:
                    break
                mid = 0.5 * (lo + hi)
                ind = make_ind(mid, ls, pm)
                if ind is None:
                    break
                out.append(ind)
                if ind["feas"]:
                    lo = ind["x"].copy()
                else:
                    hi = ind["x"].copy()
            return out

        def dom_repair(pa, ls, pm):
            # Stage 1: randomised interpolation toward integer-matched feasible anchor.
            # Stage 2 (if still infeasible): short bisection refinement on the segment.
            if not S["arch"]:
                return []
            anc = integer_matched_anchor(pa["x"])
            t = rng.uniform(0.2, 0.9)
            x = pa["x"].astype(float).copy()
            ax = anc["x"]
            if nC:
                x[C] = ax[C] + t * (pa["x"][C] - ax[C])
                x[C] += 0.1 * sig[C] * span[C] * rng.standard_normal(nC)
            if nI:
                take_anc = rng.random(nI) < t
                x[I] = np.where(take_anc, ax[I], pa["x"][I])
            ind = make_ind(x, ls, pm)
            if ind is None:
                return []
            out = [ind]
            if not ind["feas"] and nC and S["used"] < budget:
                # refinement: bisect between the new point and the anchor
                lo = ax.astype(float).copy()
                hi = ind["x"].copy()
                for _ in range(2):
                    if S["used"] >= budget:
                        break
                    mid = lo.copy()
                    mid[C] = 0.5 * (lo[C] + hi[C])
                    if nI:
                        mid[I] = hi[I]
                    sub = make_ind(mid, ls, pm)
                    if sub is None:
                        break
                    out.append(sub)
                    if sub["feas"]:
                        lo = sub["x"].copy()
                        break
                    hi = sub["x"].copy()
            return out

        def near_seed(pa):
            # use a near-feasible point (small dominant violation) as the repair anchor
            near = S["near"]
            if not near or not S["arch"]:
                return None
            pick = near[rng.integers(min(len(near), 8))]
            return pick

        def local_move(ls, pm):
            x, bf = base_point()
            k = int(rng.integers(1, 4))
            idx = rng.choice(d, size=min(k, d), replace=False)
            cont_touched = []
            for i in idx:
                if is_int[i]:
                    flip_discrete(x, i)
                else:
                    x[i] = x[i] + sig[i] * span[i] * rng.standard_normal()
                    cont_touched.append(i)
            ind = make_ind(x, ls, pm)
            if ind is None:
                return []
            succ = ind["feas"] and ind["f"] < bf
            for i in cont_touched:
                update_sig(i, succ)
            return [ind]

        def probe_move(ls, pm):
            x, bf = base_point()
            i = int(rng.integers(d))
            if is_int[i]:
                flip_discrete(x, i)
                ind = make_ind(x, ls, pm)
                return [ind] if ind is not None else []
            step = sig[i] * span[i] * rng.uniform(0.5, 1.5)
            y = x.copy()
            y[i] += step
            z = x.copy()
            z[i] -= step
            out = []
            for cand in (y, z):
                ind = make_ind(cand, ls, pm)
                if ind is not None:
                    out.append(ind)
            succ = any(o["feas"] and o["f"] < bf for o in out)
            update_sig(i, succ)
            return out

        # Initial population
        pop = []
        for _ in range(min(budget, 3 * mu)):
            if S["used"] >= budget:
                break
            ind = make_ind(random_x(), ls0, pm0)
            if ind is not None:
                pop.append(ind)
        if not pop:
            return S["best_f"], S["best_x"]
        pop = rank(pop, mu)

        while S["used"] < budget:
            # Stagnation restart keeping incumbent and archive anchors
            if S["used"] - S["last_imp"] > patience:
                newpop = []
                if S["best_x"] is not None:
                    newpop.append({"x": S["best_x"].copy(), "f": S["best_f"],
                                   "g": np.zeros(len(S["gsum"])), "feas": True,
                                   "ls": np.log(0.05), "pm": pm0})
                    sig[:] = np.maximum(sig, 0.02)
                for a in S["arch"][:5]:
                    ind = make_ind(a["x"], np.log(0.05), pm0)
                    if ind is not None:
                        newpop.append(ind)
                while len(newpop) < mu and S["used"] < budget:
                    ind = make_ind(random_x(), ls0, pm0)
                    if ind is not None:
                        newpop.append(ind)
                S["last_imp"] = S["used"]
                if not newpop:
                    break
                pop = rank(newpop, mu)

            P = len(pop)
            best_before = S["best_f"]
            offspring, tags = [], []

            probs = op_score + 0.05
            if np.isfinite(S["best_f"]):
                progress = S["used"] / budget
                boost = 1.0 + 2.0 * progress
                probs[1] *= boost
                probs[2] *= boost
                probs[DOM_IDX] *= 0.8
            else:
                probs[0] *= 2.5
                probs[DOM_IDX] *= 1.5
            probs = probs / probs.sum()

            t_seg = rng.uniform(0.0, 1.0)

            for _ in range(lam):
                if S["used"] >= budget:
                    break
                a, b, c = rng.integers(0, P, 3)
                pa, pb = pop[a], pop[b]
                ls = float(np.clip(0.5 * (pa["ls"] + pb["ls"]) + 0.3 * rng.standard_normal(),
                                   ls_min, ls_max))
                pm = float(np.clip(0.5 * (pa["pm"] + pb["pm"]) * np.exp(0.2 * rng.standard_normal()),
                                   pm_lo, pm_hi))
                op = int(rng.choice(n_ops, p=probs))
                name = OPS[op]
                new = None

                if name == "bisect" and S["arch"] and not pa["feas"]:
                    new = bisect(pa, int(rng.integers(3, 7)), ls, pm)
                elif name == "domrep" and S["arch"] and not pa["feas"]:
                    # with some probability, repair from a near-feasible seed instead
                    if rng.random() < 0.3:
                        seed = near_seed(pa)
                        if seed is not None:
                            pa_seed = {"x": seed["x"], "feas": False, "ls": ls, "pm": pm}
                            new = dom_repair(pa_seed, ls, pm)
                        else:
                            new = dom_repair(pa, ls, pm)
                    else:
                        new = dom_repair(pa, ls, pm)
                elif name == "local" and S["best_x"] is not None:
                    new = local_move(ls, pm)
                elif name == "probe" and S["best_x"] is not None:
                    new = probe_move(ls, pm)
                elif name == "xover" and S["best_x"] is not None:
                    base, _ = base_point()
                    mask = rng.random(d) < t_seg
                    x = np.where(mask, pa["x"], base).astype(float)
                    if nC:
                        x[C] = x[C] + np.exp(ls) * 0.5 * span[C] * rng.standard_normal(nC)
                    mutate_discrete(x, pm)
                    ind = make_ind(x, ls, pm)
                    new = [ind] if ind is not None else []
                elif name == "de" and nC >= 2 and P >= 3:
                    F_de = rng.uniform(0.4, 0.9)
                    donor = pa["x"].copy()
                    donor[C] = pa["x"][C] + F_de * (pb["x"][C] - pop[c]["x"][C])
                    cross = rng.random(d) < 0.9
                    cross[rng.integers(d)] = True
                    x = np.where(cross, donor, pa["x"]).astype(float)
                    if nI:
                        keep = rng.random(nI) < 0.5
                        x[I] = np.where(keep, pa["x"][I], pb["x"][I])
                    mutate_discrete(x, pm)
                    ind = make_ind(x, pa["ls"], pm)
                    new = [ind] if ind is not None else []
                else:
                    mask = rng.random(d) < 0.5
                    x = np.where(mask, pa["x"], pb["x"]).astype(float)
                    if nC:
                        x[C] = x[C] + np.exp(ls) * span[C] * rng.standard_normal(nC)
                    mutate_discrete(x, pm)
                    ind = make_ind(x, ls, pm)
                    new = [ind] if ind is not None else []
                    op = ES_IDX

                for o in new:
                    offspring.append(o)
                    tags.append(op)

            if not offspring:
                break

            # Operator reward: feasibility yield plus improvement bonus
            for op in range(n_ops):
                sel = [o for o, t in zip(offspring, tags) if t == op]
                if sel:
                    r = [(1.0 if o["feas"] else 0.0) + (1.0 if o["f"] < best_before else 0.0)
                         for o in sel]
                    yld = min(1.0, float(np.mean(r)))
                    op_score[op] = 0.8 * op_score[op] + 0.2 * yld

            pop = rank(pop + offspring, mu)

        return S["best_f"], S["best_x"]