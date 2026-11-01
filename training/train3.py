"""Lumen NN training, run 3 (2026-11-01). Files: run3_*.npz, run3_log.jsonl.

What run 2 showed: with the guardrails in the loop the network is close to the rules, and the leftover
incidents are junction right-of-way failures (standing in a junction box, entering across cross traffic).
Evolution strategies could not fix them: collisions are about 2% of episodes, the progress term dominates
the gradient, and even a collision-course timer (sim "risk") is as rare as the collisions.

So run 3 teaches the network from a better expert instead: brains.rules_v3 = Lumen Drive's rules, a
committed yellow-light decision, box discipline (never plan a stop inside a junction box) and gap
acceptance against cross traffic (including cars accelerating from their line and cars too fast to
stop). On 512 diagnosis episodes that teacher has 0.88 incidents/100 km against 3.10 for the rules.
DAgger: the student drives (with the guardrails) in junction-heavy and normal scenarios mixed with the
teacher, the teacher labels every visited state, braking decisions are weighted up. The network sees only
what it will see in the game (91 inputs); the teacher also uses simulator ground truth where the rules
do not (cross cars' stop intentions), which the network can only approximate.

The warm start from rules is the main source of the result here, and the report says so. Selection is by
incident rate on a separate set (seed 888), never the held-out test seed.
usage: py train3.py [workers=5] [minutes=22]
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
import train2

HERE = os.path.dirname(os.path.abspath(__file__))
NET = brains.MLP()


def collect(args):
    train_es.lower_priority()
    theta, seed, n, beta, noise = args
    rng = np.random.default_rng(seed + 7)
    sc = sim.Scenarios(n, seed, hard=(seed % 2 == 0))
    w = sim.World(sc)
    X, Y, Wt = [], [], []
    for step in range(sim.STEPS):
        obs = w.observe()
        vT_t, dT_t = brains.rules_v3(w)
        vlim = np.maximum(1.1 * w.vlim_at(w.es), 1e-3)
        share_t = np.clip(vT_t / vlim, 0, 1)
        free = np.minimum(0.95 * w.vlim_at(w.es), brains.V_MAX) / vlim
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
            Wt.append(np.where(share_t[alive] < 0.55 * free[alive], 3.0, 1.0).astype(np.float32))
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
    return np.concatenate(X), np.concatenate(Y), np.concatenate(Wt)


def fit(theta, X, Y, Wt, epochs, rng, lr=2e-3, batch=4096):
    """Adam on the weighted speed-share error (too fast counts 3x too slow; braking samples 3x) and the
    offset output kept at zero."""
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
            xb, yb, wb = X[idx], Y[idx], Wt[idx]
            h1, h2, out, (W1, b1, W2, b2, W3, b3) = train2.forward_cache(theta, xb)
            p = 1 / (1 + np.exp(-out[:, 0]))
            e = p - yb
            we = np.where(e > 0, 3.0, 1.0) * wb
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


def score(theta, seed=888, n=256):
    """incident rate and speed on the selection set (normal scenarios); lower composite is better"""
    r, m = train2.run_members(theta[None, :], seed, n, False)
    km = float(m["dist"].sum() / 1000)
    inc = m["collision"].sum() + m["red"].sum() + m["stopsign"].sum() + m["offroad"].sum() + m["near"].sum()
    inc100 = float(inc / max(km, 1e-9) * 100)
    speed = float(m["dist"].sum() / max(1.0, m["time"].sum()))
    return inc100 + 8.0 * max(0.0, 7.6 - speed), {"inc100": round(inc100, 2), "speed": round(speed, 2), "coll": int(m["collision"].sum()),
                                                  "red": int(m["red"].sum()), "km": round(km, 1), "eval": round(float(r[0]), 1)}


def main():
    nworkers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 22
    train_es.lower_priority()
    deadline = time.time() + minutes * 60
    rng = np.random.default_rng(3)
    theta = NET.init(rng)
    X = np.zeros((0, sim.OBS_DIM), np.float32)
    Y = np.zeros(0, np.float32)
    Wt = np.zeros(0, np.float32)
    best, best_theta = 1e9, theta
    log = open(os.path.join(HERE, "run3_log.jsonl"), "a")
    with Pool(nworkers) as pool:
        r = 0
        while time.time() < deadline:
            beta = 1.0 if r == 0 else max(0.0, 0.7 - 0.2 * (r - 1))
            t0 = time.time()
            jobs = [(theta if r > 0 else None, 70_000 + r * 100 + k, 64, beta, 0.25 if r == 0 else 0.18) for k in range(nworkers)]
            res = pool.map(collect, jobs)
            X = np.concatenate([X] + [a for a, _, _ in res])
            Y = np.concatenate([Y] + [b for _, b, _ in res])
            Wt = np.concatenate([Wt] + [c for _, _, c in res])
            if len(X) > 1_200_000:
                keep = rng.choice(len(X), 1_200_000, replace=False)
                X, Y, Wt = X[keep], Y[keep], Wt[keep]
            theta, loss = fit(theta, X, Y, Wt, epochs=8 if r < 6 else 14, rng=rng)
            sc, info = score(theta)
            rec = {"round": r, "beta": round(beta, 2), "samples": int(len(X)), "loss": round(float(loss), 4), "sec": round(time.time() - t0, 1), "score": round(sc, 2), **info}
            if sc < best and r >= 1:
                best, best_theta = sc, theta.copy()
                np.savez(os.path.join(HERE, "run3_best.npz"), theta=theta, it=r, eval=info["eval"])
                rec["best"] = True
            np.savez(os.path.join(HERE, "run3_last.npz"), theta=theta, it=r)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            r += 1


if __name__ == "__main__":
    main()
