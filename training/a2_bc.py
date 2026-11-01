"""Attempt 2, stage 1 (2026-11-01): behaviour cloning of BOTH heads, with the shield in the loop (DAgger).

 * speed head: the free-flow rule teacher brains.rules_fast (as in train4.py);
 * offset head: the hand-written pass rule (rulepass.py), which asks for the lateral offset that clears whatever blocks
   the lane; whether a request is carried out is the shield's decision, not the teacher's or the student's;
 * observation: the filtered one (sim.observe(): what nn.lua shows the network);
 * rollouts: original, junction-heavy, overtake-heavy, and the two diagnostic classes (standing start with a stop sign
   ahead, stopped cars behind and beside), scenario seeds 60_000+ (never a development or held-out seed).
The hidden layers learned here are then FROZEN for stage 2 (a2_es.py), which optimises only the output layer against the
real objective.

usage: py a2_bc.py [workers=5] [rounds=7]
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
import brains2
import rulepass
import sim
import train_es

HERE = os.path.dirname(os.path.abspath(__file__))
NET = brains.MLP()
CLASSES = ("hard", "plain", "overtake", "overtake", "start_stop", "start_crowd")


def make_scenarios(n, seed):
    c = CLASSES[seed % len(CLASSES)]
    if c == "hard":
        return sim.Scenarios(n, seed, hard=True)
    if c == "plain":
        return sim.Scenarios(n, seed)
    if c == "overtake":
        return sim.Scenarios(n, seed, overtake=True)
    return sim.Scenarios(n, seed, mode=c)


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def collect(args):
    train_es.lower_priority()
    theta, seed, n, beta, noise = args
    rng = np.random.default_rng(seed + 7)
    sc = make_scenarios(n, seed)
    w = sim.World(sc)
    X, Ys, Yo = [], [], []
    ou = np.zeros(n)
    for step in range(sim.STEPS):
        obs = w.observe()
        vT_t, _ = brains.rules_fast(w)
        share_t = np.clip(vT_t / np.maximum(1.1 * w.vlim_at(w.es), 1e-3), 0, 1)
        t_off = rulepass.pass_target(w)
        out_t = np.stack([logit(share_t), np.arctanh(np.clip(t_off / 4.0, -0.99, 0.99))], axis=1)
        if theta is None or beta >= 1.0:
            out = out_t
        else:
            out_s = NET.forward(theta[None, :], obs[None, :, :])[0]
            use_t = rng.random(n) < beta
            out = np.where(use_t[:, None], out_t, out_s)
        ou = 0.97 * ou + rng.standard_normal(n) * noise * 0.25
        out = out + np.stack([rng.standard_normal(n) * noise, np.clip(ou, -0.15, 0.15)], axis=1)
        vT, dT = brains2.nn_actions2(out, w)
        alive = ~w.done
        if step % 2 == 0 and alive.any():
            X.append(obs[alive])
            Ys.append(share_t[alive].astype(np.float32))
            Yo.append((t_off[alive] / 4.0).astype(np.float32))
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
    return np.concatenate(X), np.concatenate(Ys), np.concatenate(Yo)


def forward_cache(theta, X):
    W1, b1, W2, b2, W3, b3 = [a[0] for a in NET.unpack(theta[None, :])]
    h1 = np.tanh(X @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return h1, h2, h2 @ W3 + b3, (W1, b1, W2, b2, W3, b3)


def fit(theta, X, Ys, Yo, epochs, rng, lr=2e-3, batch=4096, lam_off=0.8):
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
            xb, ys, yo = X[idx], Ys[idx], Yo[idx]
            h1, h2, out, (W1, b1, W2, b2, W3, b3) = forward_cache(theta, xb)
            p = 1 / (1 + np.exp(-out[:, 0]))
            e = p - ys
            we = np.where(e > 0, 3.0, 1.0)
            th = np.tanh(out[:, 1])
            eo = th - yo
            wo = lam_off * (1.0 + 2.0 * (np.abs(yo) > 0.02))
            tot += float((we * e * e).sum() + (wo * eo * eo).sum())
            g_out = np.stack([2 * we * e * p * (1 - p), 2 * wo * eo * (1 - th * th)], axis=1) / len(idx)
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


def main():
    nworkers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    per_worker = 48
    train_es.lower_priority()
    rng = np.random.default_rng(11)
    theta = NET.init(rng)
    X = np.zeros((0, sim.OBS_DIM), np.float32)
    Ys = np.zeros(0, np.float32)
    Yo = np.zeros(0, np.float32)
    with Pool(nworkers) as pool:
        for r in range(rounds):
            beta = 1.0 if r == 0 else max(0.0, 0.6 - 0.2 * (r - 1))
            t0 = time.time()
            jobs = [(theta if r > 0 else None, 60_000 + r * 100 + k, per_worker, beta, 0.25 if r == 0 else 0.18) for k in range(nworkers * 2)]
            res = pool.map(collect, jobs)
            X = np.concatenate([X] + [a for a, _, _ in res])
            Ys = np.concatenate([Ys] + [b for _, b, _ in res])
            Yo = np.concatenate([Yo] + [c for _, _, c in res])
            if len(X) > 1_500_000:
                keep = rng.choice(len(X), 1_500_000, replace=False)
                X, Ys, Yo = X[keep], Ys[keep], Yo[keep]
            theta, loss = fit(theta, X, Ys, Yo, epochs=6 if r < rounds - 1 else 24, rng=rng)
            rec = {"phase": "bc", "round": r, "beta": round(beta, 2), "samples": int(len(X)), "pass_rows": float(np.mean(np.abs(Yo) > 0.02)),
                   "loss": round(float(loss), 4), "sec": round(time.time() - t0, 1)}
            print(json.dumps(rec), flush=True)
            np.savez(os.path.join(HERE, "a2_bc.npz"), theta=theta)


if __name__ == "__main__":
    main()
