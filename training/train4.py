"""Lumen NN training, run 4 (2026-11-01): derived from train2.py. Behaviour cloning only, with the final guardrails
(curves, lights with a firm yellow rule, stop signs, cars ahead, junction discipline) in the loop and the free-flow
teacher brains.rules_fast. File: run4_bc.npz.

Original run 2 docstring follows.


What changed from run 1 (train.py), and why (every change comes from a measurement):
 * run 1's evolution strategies did not improve on its warm start: Adam at lr 0.01 on 9.7k weights with
   32 noise pairs was mostly a random walk, and its check-set reward stayed at 365 to 385 (rules: 450).
 * measured with the guardrails (brains.guards: curves, lights, stop signs, cars stopped or slower in the
   lane, the same ones nn.lua applies in the game), run 1's networks dropped from 22 to 5 incidents per
   100 km, and every remaining incident was a cross-traffic collision. So the network now trains WITH the
   guardrails in the loop and is asked to learn only the soft part.
 * a network cannot tell a car about to stop at its line from a red-light runner without the car's
   acceleration: it is now an input (91 inputs).
 * collisions are rare (about 2% of episodes), so ES saw almost no signal: training scenarios are
   junction-heavy with 22% runners (sim.Scenarios(hard=True)); the held-out test is the normal mix.
 * ES steps are per layer (small in the first layers, larger at the output) with stronger incident penalties.

Warm start (behaviour cloning, DAgger) from the rule brain with a yellow-light latch, then outcome-based ES.
usage: py train2.py [workers=5] [minutes=25]
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
PAIRS, SCEN = 40, 24
SIGMA_GROUPS = (0.004, 0.006, 0.025)        # first layer, second layer, output layer
LR_GROUPS = (0.0006, 0.001, 0.004)


def reward2(m):
    return (m["dist"] - 1500 * m["collision"] - 120 * m["near"] - 800 * m["red"] - 500 * m["stopsign"]
            - 800 * m["offroad"] - 0.004 * m["jerk2"])


def group_vectors():
    sig, lr = [], []
    for i, sz in enumerate(NET.sizes):
        g = 0 if i < 2 else (1 if i < 4 else 2)
        sig.append(np.full(sz, SIGMA_GROUPS[g]))
        lr.append(np.full(sz, LR_GROUPS[g]))
    return np.concatenate(sig), np.concatenate(lr)


def run_members(params, seed, n_scen, hard):
    """Every member of params (P, n) drives the same n_scen scenarios; returns reward per member and the metrics."""
    P = params.shape[0]
    sc = sim.Scenarios(n_scen, seed, hard=hard)
    w = sim.World(train_es.tile_scenarios(sc, P))
    for _ in range(sim.STEPS):
        obs = w.observe().reshape(P, n_scen, -1)
        out = NET.forward(params, obs).reshape(P * n_scen, 2)
        vT, dT = brains.nn_actions(out, w)
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
    m = w.summary()
    return reward2(m).reshape(P, n_scen).mean(axis=1), m


def es_worker(args):
    train_es.lower_priority()
    params, seed = args
    r, _ = run_members(params, seed, SCEN, True)
    return r


# ---------------------------------------------------------------------------------
# phase 1: DAgger warm start, the student drives with the guardrails
# ---------------------------------------------------------------------------------
def collect(args):
    train_es.lower_priority()
    theta, seed, n, beta, noise = args
    rng = np.random.default_rng(seed + 7)
    sc = sim.Scenarios(n, seed, hard=(seed % 2 == 0))
    w = sim.World(sc)
    X, Y = [], []
    for step in range(sim.STEPS):
        obs = w.observe()
        vT_t, dT_t = brains.rules_fast(w)
        share_t = np.clip(vT_t / np.maximum(1.1 * w.vlim_at(w.es), 1e-3), 0, 1)
        if theta is None or beta >= 1.0:
            vT, dT = vT_t.copy(), dT_t.copy()
        else:
            out = NET.forward(theta[None, :], obs[None, :, :])[0]
            vS, dS = brains.nn_actions(out, w)
            use_t = rng.random(n) < beta
            vT = np.where(use_t, vT_t, vS)
            dT = np.where(use_t, dT_t, dS)
        vT = vT * np.exp(rng.standard_normal(n) * noise * 0.5)
        if step == 0:
            w._off = np.zeros(n)
        w._off = 0.97 * w._off + rng.standard_normal(n) * noise * 0.8
        dT = dT + np.clip(w._off, -2.5, 2.5)
        alive = ~w.done
        if step % 2 == 0 and alive.any():
            X.append(obs[alive])
            Y.append(share_t[alive].astype(np.float32))
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
    """Adam on the speed share error, running too fast counted three times as much as running too slow, and
    on keeping the offset output at zero (the teacher drives in the middle of the lane)."""
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    t = 0
    n = len(X)
    last = None
    for ep in range(epochs):
        order = rng.permutation(n)
        tot = 0.0
        cur = lr * (0.15 + 0.85 * (1 - ep / max(1, epochs - 1)))
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xb, yb = X[idx], Y[idx]
            h1, h2, out, (W1, b1, W2, b2, W3, b3) = forward_cache(theta, xb)
            p = 1 / (1 + np.exp(-out[:, 0]))
            e = p - yb
            we = np.where(e > 0, 3.0, 1.0)
            tot += float((we * e * e).sum() + 0.2 * (out[:, 1] ** 2).sum())
            g_out = np.stack([2 * we * e * p * (1 - p), 0.4 * out[:, 1]], axis=1) / len(idx)
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
            theta = theta - cur * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        last = tot / n
    return theta, last


def check(theta, label):
    """The current network on the check set (seed 777, normal scenarios): reward and incident counts."""
    r, m = run_members(theta[None, :], 777, 128, False)
    km = float(m["dist"].sum() / 1000)
    inc = m["collision"].sum() + m["red"].sum() + m["stopsign"].sum() + m["offroad"].sum() + m["near"].sum()
    return {"eval": round(float(r[0]), 1), "km": round(km, 2), "inc100": round(float(inc / max(km, 1e-9) * 100), 2),
            "coll": int(m["collision"].sum()), "red": int(m["red"].sum()), "stop": int(m["stopsign"].sum()),
            "speed": round(float(m["dist"].sum() / max(1.0, m["time"].sum())), 2)}


def phase1(pool, nworkers, rounds=6, per_worker=80):
    rng = np.random.default_rng(1)
    theta = NET.init(rng)
    X = np.zeros((0, sim.OBS_DIM), np.float32)
    Y = np.zeros(0, np.float32)
    for r in range(rounds):
        beta = 1.0 if r == 0 else max(0.0, 0.6 - 0.2 * (r - 1))
        t0 = time.time()
        jobs = [(theta if r > 0 else None, 60_000 + r * 100 + k, per_worker, beta, 0.25 if r == 0 else 0.18) for k in range(nworkers)]
        res = pool.map(collect, jobs)
        X = np.concatenate([X] + [a for a, _ in res])
        Y = np.concatenate([Y] + [b for _, b in res])
        if len(X) > 1_500_000:
            keep = rng.choice(len(X), 1_500_000, replace=False)
            X, Y = X[keep], Y[keep]
        theta, loss = fit(theta, X, Y, epochs=6 if r < rounds - 1 else 22, rng=rng)
        rec = {"phase": "bc", "round": r, "beta": round(beta, 2), "samples": int(len(X)), "loss": round(float(loss), 4), "sec": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
    np.savez(os.path.join(HERE, "run4_bc.npz"), theta=theta)
    return theta


# ---------------------------------------------------------------------------------
# phase 2: evolution strategies with per-layer steps
# ---------------------------------------------------------------------------------
def phase2(pool, nworkers, theta, deadline):
    sigma, lr = group_vectors()
    rng = np.random.default_rng(2027)
    opt = train_es.Adam(NET.n, 1.0)
    opt.lr = lr
    log_path = os.path.join(HERE, "run2_log.jsonl")
    best = -1e18
    it = 0
    with open(log_path, "a") as log:
        while time.time() < deadline:
            t0 = time.time()
            eps = rng.standard_normal((PAIRS, NET.n))
            members = np.concatenate([theta + sigma * eps, theta - sigma * eps], axis=0)
            res = pool.map(es_worker, [(c, 200_000 + it) for c in np.array_split(members, nworkers)])
            r = np.concatenate(res)
            ranks = train_es.centred_ranks(r)
            g = ((ranks[:PAIRS] - ranks[PAIRS:])[:, None] * eps).sum(axis=0) / PAIRS
            theta = theta + opt.step(g - 0.01 * theta * (sigma / sigma.max()))
            rec = {"phase": "es", "it": it, "mean": round(float(r.mean()), 1), "max": round(float(r.max()), 1), "sec": round(time.time() - t0, 1)}
            if it % 10 == 0:
                rec.update(check(theta, "es"))
                if rec["eval"] > best:
                    best = rec["eval"]
                    np.savez(os.path.join(HERE, "run2_best.npz"), theta=theta, it=it, eval=best)
            np.savez(os.path.join(HERE, "run2_last.npz"), theta=theta, it=it)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            it += 1
    return theta


def main():
    nworkers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    train_es.lower_priority()
    with Pool(nworkers) as pool:
        theta = phase1(pool, nworkers, rounds=7, per_worker=80)
        rec = check(theta, "bc")
        rec["phase"] = "run4-bc-eval"
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
