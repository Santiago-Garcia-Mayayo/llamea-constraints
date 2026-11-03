import numpy as np


class Algorithm:
    """
    (mu+lambda) mixed-integer ES in normalised [0,1]^dim space.
    - Deb-style ranking: feasible (by f) beats infeasible (by weighted violation).
    - Multi-persistent-violator focus: the top-2 constraints by EMA violation rate
      are tracked; repairs, gradient steps and ranking weights target them.
    - Multi-scale ridge gradient repair with predicted-violation screening.
    - Feasibility-boundary bisection (4 steps) toward the feasible elite.
    - Adaptive operator credit with exponential forgetting, DE recombination,
      self-adaptive steps, and stagnation restarts that keep the feasible elite.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        lb = np.asarray(func.bounds.lb, dtype=float).copy()
        ub = np.asarray(func.bounds.ub, dtype=float).copy()
        vt = list(func.variable_types)

        disc = np.array([vt[i] != "continuous" for i in range(dim)])
        cont = ~disc
        binary = np.array([vt[i] == "binary" for i in range(dim)])
        ordinal = np.array([vt[i] == "ordinal" for i in range(dim)])

        nv = np.ones(dim, dtype=int)
        for i in range(dim):
            if disc[i]:
                nv[i] = max(1, int(func.n_values[i]))

        span = ub - lb
        span = np.where(span <= 0, 1.0, span)

        st = {"evals": 0}
        hist_U = []
        hist_G = []
        best = {"f": np.inf, "x": None}
        elite = []
        W = {"w": None, "rate": None, "mag": None}
        CB = {"val": None, "u": None}
        OPS = ["mut", "de", "exploit", "bisect", "grad", "repair", "line"]
        credit = {op: [1.0, 2.0] for op in OPS}

        def decode(u):
            u = np.clip(u, 0.0, 1.0)
            x = lb + u * span
            if np.any(disc):
                x[disc] = np.clip(np.rint(x[disc]), 0, nv[disc] - 1)
            return x

        def encode(x):
            return (x - lb) / span

        def can_eval():
            return st["evals"] < budget

        def key(ind):
            if ind["feas"]:
                return (0, ind["f"])
            if W["w"] is None:
                return (1, float(np.sum(ind["g"])))
            return (1, float(np.dot(W["w"], ind["g"])))

        def persist_set():
            """Indices of up to two most persistently violated constraints."""
            if W["rate"] is None:
                return []
            idx = np.where(W["rate"] >= 0.25)[0]
            if len(idx) == 0:
                return []
            idx = idx[np.argsort(-W["rate"][idx])]
            return [int(j) for j in idx[:2]]

        def persist_idx():
            ps = persist_set()
            return ps[0] if ps else None

        def evaluate(u, s=0.2, op=None, pk=None):
            if not can_eval():
                return None
            u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
            x = decode(u)
            f, g = func(x)
            st["evals"] += 1
            g = np.atleast_1d(np.asarray(g, dtype=float)).ravel()
            g = np.where(np.isfinite(g), g, 1e6)
            g = np.maximum(g, 0.0)
            f = float(f)
            if not np.isfinite(f):
                f = np.inf
            feas = bool(np.all(g <= 0.0) and np.isfinite(f))
            if W["w"] is None:
                m = len(g)
                W["w"] = np.ones(m)
                W["rate"] = np.zeros(m)
                W["mag"] = np.zeros(m)
                CB["val"] = np.full(m, np.inf)
                CB["u"] = [None] * m
            better = g < CB["val"]
            for j in np.where(better)[0]:
                CB["val"][j] = g[j]
                CB["u"][j] = encode(x)
            hist_U.append(encode(x))
            hist_G.append(g.copy())
            if len(hist_U) > 400:
                del hist_U[0]
                del hist_G[0]
            ind = {"u": encode(x), "x": x, "f": f, "g": g, "s": float(s), "feas": feas}
            if feas:
                if f < best["f"]:
                    best["f"] = f
                    best["x"] = x.copy()
                elite.append(ind)
                elite.sort(key=lambda d: d["f"])
                del elite[10:]
            ind["k"] = key(ind)
            if op is not None and pk is not None:
                c = credit[op]
                c[0] *= 0.99  # exponential forgetting of operator credit
                c[1] *= 0.99
                c[1] += 1.0
                if ind["k"] < pk:
                    c[0] += 1.0
            return ind

        def refresh(pop):
            for ind in pop:
                ind["k"] = key(ind)

        def update_weights(pop):
            G = np.array([ind["g"] for ind in pop])
            viol = G > 0.0
            rate = np.mean(viol, axis=0)
            cnt = np.maximum(np.sum(viol, axis=0), 1)
            mag = np.minimum(np.sum(G, axis=0) / cnt, 10.0)
            W["rate"] = 0.7 * W["rate"] + 0.3 * rate
            W["mag"] = 0.7 * W["mag"] + 0.3 * mag
            persist = np.where(W["rate"] > 0.4, 6.0, 0.0)
            target = 1.0 + 9.0 * W["rate"] ** 2 + 2.0 * W["rate"] * np.tanh(W["mag"]) + persist
            W["w"] = np.clip(0.6 * W["w"] + 0.4 * target, 1.0, 100.0)

        def tournament(pop):
            a = pop[np.random.randint(len(pop))]
            b = pop[np.random.randint(len(pop))]
            return a if a["k"] <= b["k"] else b

        def choose_op():
            p = np.array([credit[o][0] / credit[o][1] for o in OPS]) + 0.05
            p = p / p.sum()
            return OPS[np.random.choice(len(OPS), p=p)]

        def worst_constraint(parent):
            if parent["feas"] or W["w"] is None:
                return None
            ps = persist_set()
            cands = [j for j in ps if parent["g"][j] > 0]
            if cands:
                if len(cands) == 1 or np.random.rand() < 0.7:
                    return cands[0]
                return cands[np.random.randint(len(cands))]
            wg = W["w"] * parent["g"]
            if np.max(wg) <= 0:
                return None
            return int(np.argmax(wg))

        def mutate(u, s):
            u = np.array(u, dtype=float)
            pm = min(1.0, 2.0 / dim)
            mask = np.random.rand(dim) < pm
            if not np.any(mask):
                mask[np.random.randint(dim)] = True
            for i in np.where(mask)[0]:
                if cont[i]:
                    step = np.random.standard_cauchy() if np.random.rand() < 0.1 else np.random.randn()
                    u[i] = u[i] + s * np.clip(step, -5, 5)
                else:
                    cur = int(np.clip(np.rint(lb[i] + u[i] * span[i]), 0, nv[i] - 1))
                    if nv[i] <= 1:
                        continue
                    if ordinal[i] and np.random.rand() < 0.7:
                        v = int(np.clip(cur + (1 if np.random.rand() < 0.5 else -1), 0, nv[i] - 1))
                    else:
                        v = np.random.randint(nv[i] - 1)
                        if v >= cur:
                            v += 1
                    if binary[i]:
                        v = 1 - cur
                    u[i] = (v + 0.5) / nv[i]
            return np.clip(u, 0.0, 1.0)

        def random_u():
            u = np.random.rand(dim)
            for i in np.where(disc)[0]:
                v = np.random.randint(nv[i])
                u[i] = (v + 0.5) / nv[i]
            return u

        def targeted_repair(parent):
            j = worst_constraint(parent)
            tgt = None
            p_use = 0.85 if (j is not None and j in persist_set()) else 0.6
            if j is not None and CB["u"][j] is not None and np.random.rand() < p_use:
                tgt = CB["u"][j]
            if tgt is None and elite:
                tgt = elite[np.random.randint(min(len(elite), 3))]["u"]
            if tgt is None:
                return mutate(parent["u"], 0.1)
            t = np.random.uniform(0.3, 1.0)
            u = parent["u"] + t * (tgt - parent["u"])
            u[cont] += 0.03 * np.random.randn(int(np.sum(cont)))
            mask = (np.random.rand(dim) < 0.3) & disc
            u[mask] = tgt[mask]
            return np.clip(u, 0.0, 1.0)

        def grad_step(parent, s):
            j = worst_constraint(parent)
            n = len(hist_U)
            if j is None or n < dim + 5:
                return mutate(parent["u"], s)
            lo = max(0, n - 300)
            U = np.array(hist_U[lo:])
            y = np.array([gg[j] for gg in hist_G[lo:]])
            A = np.hstack([U, np.ones((len(U), 1))])
            reg = 1e-2 * np.eye(A.shape[1])
            reg[-1, -1] = 0.0
            try:
                beta = np.linalg.solve(A.T @ A + reg, A.T @ y)
            except np.linalg.LinAlgError:
                return mutate(parent["u"], s)
            gv = beta[:-1]
            b0 = beta[-1]
            nrm = np.linalg.norm(gv)
            if nrm < 1e-12:
                return mutate(parent["u"], s)
            d = -gv / nrm
            pu = parent["u"]
            pool = []
            # multi-scale steps along the descent direction, plus a jittered direction
            for t in [0.02, 0.05, 0.1, 0.2, 0.35]:
                pool.append(np.clip(pu + t * d + 0.01 * np.random.randn(dim), 0.0, 1.0))
            d2 = d + 0.3 * np.random.randn(dim) / np.sqrt(dim)
            d2 = d2 / (np.linalg.norm(d2) + 1e-12)
            for t in [0.05, 0.15]:
                pool.append(np.clip(pu + t * d2, 0.0, 1.0))
            preds = [b0 + c @ gv for c in pool]
            return pool[int(np.argmin(preds))]

        def bisect_child(parent, tgt):
            a = np.random.choice([0.25, 0.5, 0.75]) if np.random.rand() < 0.6 else np.random.uniform(0.05, 0.95)
            u = a * parent["u"] + (1 - a) * tgt["u"]
            mask = (np.random.rand(dim) < 0.5) & disc
            u[mask] = tgt["u"][mask]
            return np.clip(u, 0.0, 1.0)

        def de_child(pop):
            idx = np.random.choice(len(pop), 3, replace=True)
            a, b, c = pop[idx[0]], pop[idx[1]], pop[idx[2]]
            F = np.random.uniform(0.4, 0.9)
            u = a["u"].copy()
            u[cont] = a["u"][cont] + F * (b["u"][cont] - c["u"][cont])
            mask = (np.random.rand(dim) < 0.3) & disc
            u[mask] = b["u"][mask]
            return np.clip(u, 0.0, 1.0)

        def exploit(inc):
            u = inc["u"].copy()
            s = float(np.clip(inc["s"] * np.exp(0.3 * np.random.randn()), 1e-4, 0.1))
            cidx = np.where(cont)[0]
            if len(cidx) > 0:
                k = np.random.rand(len(cidx)) < max(1.0 / len(cidx), 0.3)
                if not np.any(k):
                    k[np.random.randint(len(cidx))] = True
                idx = cidx[k]
                u[idx] += s * np.random.randn(len(idx))
            didx = np.where(disc)[0]
            if len(didx) > 0 and np.random.rand() < 0.5:
                i = didx[np.random.randint(len(didx))]
                cur = int(np.clip(np.rint(lb[i] + u[i] * span[i]), 0, nv[i] - 1))
                v = int(np.clip(cur + (1 if np.random.rand() < 0.5 else -1), 0, nv[i] - 1))
                u[i] = (v + 0.5) / nv[i]
            return np.clip(u, 0.0, 1.0), s

        def safe_eval(u, s, op=None, pk=None):
            if not can_eval():
                return None
            return evaluate(u, s, op, pk)

        def line_search(p, tgt, n_steps=4):
            """Bisection on segment tgt(feasible) -> p(infeasible); returns evaluated children."""
            out = []
            lo, hi = 0.0, 1.0  # t=0 at tgt (feasible), t=1 at p (infeasible)
            for _ in range(n_steps):
                if not can_eval():
                    break
                t = 0.5 * (lo + hi)
                u = tgt["u"] + t * (p["u"] - tgt["u"])
                child = safe_eval(u, 0.1)
                if child is None:
                    break
                out.append(child)
                if child["feas"]:
                    lo = t
                else:
                    hi = t
            return out

        # ---------- initialisation ----------
        n_init = int(min(max(3 * dim, 30), max(budget // 4, 1)))
        pop = []
        for _ in range(n_init):
            ind = safe_eval(random_u(), 0.2)
            if ind is None:
                break
            pop.append(ind)
        if not pop:
            return best["f"], best["x"]

        update_weights(pop)
        refresh(pop)
        pop.sort(key=lambda d: d["k"])
        mu = int(max(5, min(15, len(pop) // 2)))
        lam = 2 * mu
        pop = pop[:mu]

        best_f_seen = np.inf
        best_k = pop[0]["k"]
        stall = 0
        stall_limit = 12

        # ---------- main loop ----------
        while can_eval():
            update_weights(pop)
            refresh(pop)
            pop.sort(key=lambda d: d["k"])
            infeas = [ind for ind in pop if not ind["feas"]]
            target = elite[0] if elite else None

            offspring = []
            for _ in range(lam):
                if not can_eval():
                    break
                op = choose_op()
                parent = tournament(pop)
                pk = parent["k"]
                s_new = float(np.clip(parent["s"] * np.exp(0.2 * np.random.randn()), 1e-4, 0.5))
                if op == "line":
                    if infeas and elite:
                        p = infeas[np.random.randint(len(infeas))]
                        tgt = elite[np.random.randint(min(len(elite), 3))]
                        kids = line_search(p, tgt)
                        for c_i, child in enumerate(kids):
                            if c_i == len(kids) - 1:
                                credit["line"][1] += 1.0
                                if child["k"] < p["k"]:
                                    credit["line"][0] += 1.0
                            offspring.append(child)
                        continue
                    u = mutate(parent["u"], s_new)
                elif op == "mut":
                    u = mutate(parent["u"], s_new)
                elif op == "de":
                    u = de_child(pop)
                    s_new = parent["s"]
                elif op == "exploit":
                    base = target if target is not None else parent
                    u, s_new = exploit(base)
                elif op == "bisect":
                    if infeas and elite:
                        p = infeas[np.random.randint(len(infeas))]
                        tgt = elite[np.random.randint(min(len(elite), 3))]
                        u = bisect_child(p, tgt)
                        pk = p["k"]
                        s_new = p["s"]
                    else:
                        u = mutate(parent["u"], s_new)
                elif op == "grad":
                    p = infeas[np.random.randint(len(infeas))] if infeas else parent
                    u = grad_step(p, s_new)
                    pk = p["k"]
                else:  # repair
                    p = infeas[np.random.randint(len(infeas))] if infeas else parent
                    u = targeted_repair(p)
                    pk = p["k"]
                child = safe_eval(u, s_new, op, pk)
                if child is not None:
                    offspring.append(child)

            if not offspring:
                break

            combined = pop + offspring
            update_weights(combined)
            refresh(combined)
            combined.sort(key=lambda d: d["k"])
            pop = combined[:mu]

            cur_bf = best["f"]
            if cur_bf < best_f_seen - 1e-12 or pop[0]["k"] < best_k:
                if cur_bf < best_f_seen - 1e-12:
                    best_f_seen = cur_bf
                best_k = pop[0]["k"]
                stall = 0
            else:
                stall += 1

            if stall >= stall_limit and can_eval():
                new_pop = []
                if elite:
                    new_pop.append(elite[0])
                k = 0
                while len(new_pop) < mu and can_eval():
                    if elite and k % 3 != 2:
                        if k % 3 == 0:
                            u, _ = exploit(elite[0])
                        else:
                            u = mutate(elite[0]["u"], 0.4)
                        child = safe_eval(u, 0.4)
                    else:
                        child = safe_eval(random_u(), 0.2)
                    if child is not None:
                        new_pop.append(child)
                    k += 1
                if new_pop:
                    update_weights(new_pop)
                    refresh(new_pop)
                    pop = sorted(new_pop, key=lambda d: d["k"])
                    best_k = pop[0]["k"]
                stall = 0

        return best["f"], best["x"]