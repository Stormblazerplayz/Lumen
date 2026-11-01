"""Lumen NN training: behaviour-cloning warm start from the rules (DAgger), then outcome-based
evolution strategies.

Phase 1, warm start (a few minutes): the student network drives (mixed with the rule brain, less and
less of the rules each round) through fresh random scenarios with noise on its actions for coverage; at
every visited state the rule brain's decision is the label; the network is fitted to those labels. This
gives a driver that is roughly as good as the rules. Warm start from Lumen Drive's rules is the ONLY
place the rules teach the network anything. Never BeamNG's AI.

Phase 2, outcome-based fine-tuning: evolution strategies (antithetic noise, centred ranks, Adam) on
brains.reward(): metres of progress minus incidents. The teacher is now only the outcome, so whatever
the network gains over the rules, it found itself.

usage: py train.py [workers=5] [minutes=60] [es_sigma=0.02]
Checkpoints: lumen_nn_bc.npz (end of phase 1), lumen_nn_best.npz (best check-set reward so far),
lumen_nn_last.npz; train_log.jsonl.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import brains
import sim
import train_es

HERE = os.path.dirname(os.path.abspath(__file__))
NET = brains.MLP()


def lower_priority():
    train_es.lower_priority()


# ---------------------------------------------------------------------------------
# phase 1: DAgger
# ---------------------------------------------------------------------------------
def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def collect(args):
    """Roll out `n` scenarios with the student (theta) mixed with the rules, record (obs, teacher label)."""
    lower_priority()
    theta, seed, n, beta, noise = args
    rng = np.random.default_rng(seed + 7)
    sc = sim.Scenarios(n, seed)
    w = sim.World(sc)
    X, Y = [], []
    for step in range(sim.STEPS):
        obs = w.observe()
        vT_t, dT_t = brains.rules_latched(w)
        share_t = vT_t / np.maximum(1.1 * w.vlim_at(w.es), 1e-3)
        if theta is None or beta >= 1.0:
            vT, dT = vT_t.copy(), dT_t.copy()
        else:
            out = NET.forward(theta[None, :], obs[None, :, :])[0]
            vS, dS = brains.nn_actions(out, w)
            use_t = rng.random(n) < beta
            vT = np.where(use_t, vT_t, vS)
            dT = np.where(use_t, dT_t, dS)
        # noise for coverage: speed multiplicative, offset a slow random walk
        vT = vT * np.exp(rng.standard_normal(n) * noise * 0.5)
        if step == 0:
            w._off = np.zeros(n)
        w._off = 0.97 * w._off + rng.standard_normal(n) * noise * 0.8
        dT = dT + np.clip(w._off, -2.5, 2.5)
        alive = ~w.done
        if step % 2 == 0 and alive.any():
            X.append(obs[alive])
            Y.append(np.stack([logit(share_t[alive]), np.arctanh(np.clip(dT_t[alive] / 4.0, -0.99, 0.99))], axis=1).astype(np.float32))
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
    return np.concatenate(X), np.concatenate(Y)


def forward_cache(theta, X):
    W1, b1, W2, b2, W3, b3 = [a[0] for a in NET.unpack(theta[None, :])]
    h1 = np.tanh(X @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return h1, h2, h2 @ W3 + b3, (W1, b1, W2, b2, W3, b3)


def fit(theta, X, Y, epochs, rng, lr=2e-3, batch=4096):
    """Adam on the mean squared error of the two outputs (the speed logit is weighted double)."""
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    t = 0
    wts = np.array([2.0, 1.0], dtype=np.float32)
    n = len(X)
    last = None
    for ep in range(epochs):
        order = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xb, yb = X[idx], Y[idx]
            h1, h2, out, (W1, b1, W2, b2, W3, b3) = forward_cache(theta, xb)
            err = (out - yb) * wts
            tot += float((err * (out - yb)).sum())
            g_out = 2 * err / len(idx)
            gW3 = h2.T @ g_out
            gb3 = g_out.sum(0)
            g2 = (g_out @ W3.T) * (1 - h2 ** 2)
            gW2 = h1.T @ g2
            gb2 = g2.sum(0)
            g1 = (g2 @ W2.T) * (1 - h1 ** 2)
            gW1 = xb.T @ g1
            gb1 = g1.sum(0)
            g = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2, gW3.ravel(), gb3])
            t += 1
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            theta = theta - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        last = tot / n
    return theta, last


def phase1(pool, nworkers, rounds=6, per_worker=64):
    rng = np.random.default_rng(1)
    theta = NET.init(rng)
    X = np.zeros((0, sim.OBS_DIM), np.float32)
    Y = np.zeros((0, 2), np.float32)
    for r in range(rounds):
        beta = 1.0 if r == 0 else max(0.0, 0.6 - 0.2 * (r - 1))
        t0 = time.time()
        jobs = [(theta if r > 0 else None, 50_000 + r * 100 + k, per_worker, beta, 0.25 if r == 0 else 0.18) for k in range(nworkers)]
        res = pool.map(collect, jobs)
        Xn = np.concatenate([a for a, _ in res])
        Yn = np.concatenate([b for _, b in res])
        X = np.concatenate([X, Xn])
        Y = np.concatenate([Y, Yn])
        if len(X) > 1_500_000:
            keep = rng.choice(len(X), 1_500_000, replace=False)
            X, Y = X[keep], Y[keep]
        theta, loss = fit(theta, X, Y, epochs=8 if r < rounds - 1 else 14, rng=rng)
        rec = {"phase": "bc", "round": r, "beta": beta, "samples": int(len(X)), "loss": round(float(loss), 4), "sec": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
    np.savez(os.path.join(HERE, "lumen_nn_bc.npz"), theta=theta)
    return theta


# ---------------------------------------------------------------------------------
# phase 2: ES
# ---------------------------------------------------------------------------------
def phase2(pool, nworkers, theta, deadline, sigma):
    train_es.SIGMA = sigma
    rng = np.random.default_rng(2026)
    opt = train_es.Adam(NET.n, 0.01)
    log_path = os.path.join(HERE, "train_log.jsonl")
    best_eval = -1e18
    it = 0
    with open(log_path, "a") as log:
        while time.time() < deadline:
            t0 = time.time()
            eps = rng.standard_normal((train_es.PAIRS, NET.n))
            members = np.concatenate([theta + sigma * eps, theta - sigma * eps], axis=0)
            seed = 10_000 + it
            chunks = np.array_split(members, nworkers)
            res = pool.map(train_es.worker, [(c, seed) for c in chunks])
            r = np.concatenate(res)
            ranks = train_es.centred_ranks(r)
            g = ((ranks[:train_es.PAIRS] - ranks[train_es.PAIRS:])[:, None] * eps).sum(axis=0) / (train_es.PAIRS * sigma)
            theta = theta + opt.step(g - 0.002 * theta)
            rec = {"phase": "es", "it": it, "mean": round(float(r.mean()), 1), "max": round(float(r.max()), 1), "sec": round(time.time() - t0, 1)}
            if it % 10 == 0:
                rr, m = train_es.run_brains(theta[None, :], 777, 128)
                km = m["dist"].sum() / 1000
                inc = m["collision"].sum() + m["red"].sum() + m["stopsign"].sum() + m["offroad"].sum() + m["near"].sum()
                rec.update({"eval": round(float(rr[0]), 1), "km": round(float(km), 2), "inc100": round(float(inc / max(km, 1e-9) * 100), 2),
                            "coll": int(m["collision"].sum()), "red": int(m["red"].sum()), "stop": int(m["stopsign"].sum()),
                            "speed": round(float(m["dist"].sum() / max(1, m["time"].sum())), 2)})
                if rr[0] > best_eval:
                    best_eval = float(rr[0])
                    np.savez(os.path.join(HERE, "lumen_nn_best.npz"), theta=theta, it=it, eval=best_eval)
            np.savez(os.path.join(HERE, "lumen_nn_last.npz"), theta=theta, it=it)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            it += 1
    return theta


def main():
    nworkers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 60
    sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    lower_priority()
    deadline = time.time() + minutes * 60
    with Pool(nworkers) as pool:
        if os.path.exists(os.path.join(HERE, "lumen_nn_bc.npz")) and "--redo-bc" not in sys.argv:
            theta = np.load(os.path.join(HERE, "lumen_nn_bc.npz"))["theta"]
            print("warm start loaded from lumen_nn_bc.npz", flush=True)
        else:
            theta = phase1(pool, nworkers)
        # the warm start as the first "best", scored the same way as every later checkpoint
        rr, m = train_es.run_brains(theta[None, :], 777, 128)
        np.savez(os.path.join(HERE, "lumen_nn_best.npz"), theta=theta, it=-1, eval=float(rr[0]))
        print(json.dumps({"phase": "bc-eval", "eval": round(float(rr[0]), 1), "km": round(float(m["dist"].sum() / 1000), 2),
                          "coll": int(m["collision"].sum()), "red": int(m["red"].sum())}), flush=True)
        phase2(pool, nworkers, theta, deadline, sigma)


if __name__ == "__main__":
    main()
