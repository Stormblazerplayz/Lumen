"""Attempt 2, step C0: can a LINEAR readout from the frozen behaviour-cloned hidden layers express "pass this"?
Collects states while the scripted pass policy (a hand-written rule, human-written, not a model) drives with the shield
in the loop, and fits the output layer's offset head by ridge regression on the rule's offset. Development seeds only."""
import sys
import time

import numpy as np

import brains
import brains2
import sim
import test_shield as T

NET = brains.MLP()
theta0 = np.load("lumen_nn_final.npz")["theta"]
W1, b1, W2, b2, W3, b3 = [a[0] for a in NET.unpack(theta0[None, :])]


def hidden(obs):
    return np.tanh(np.tanh(obs @ W1 + b1) @ W2 + b2)


def collect(N, seed, overtake, every=3):
    sc = sim.Scenarios(N, seed, overtake=overtake)
    w = sim.World(sc)
    H, Y, S = [], [], []
    for step in range(sim.STEPS):
        out = T.scripted_out(w)
        if step % every == 0:
            alive = ~w.done
            obs = w.observe()
            H.append(hidden(obs.astype(np.float64))[alive])
            Y.append(out[alive, 1])
            S.append(out[alive, 0])
        vT, dT = brains2.nn_actions2(out, w)
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
    return np.concatenate(H), np.concatenate(Y)


if __name__ == "__main__":
    t0 = time.time()
    Ha, Ya = collect(192, 300_001, False)
    Hb, Yb = collect(192, 310_001, True)
    H = np.concatenate([Ha, Hb]); Y = np.concatenate([Ya, Yb])
    print(f"collected {len(Y)} states in {time.time() - t0:.0f}s; pass-requested fraction {np.mean(np.abs(Y) > 0.5):.3f}")
    X = np.concatenate([H, np.ones((len(H), 1))], axis=1)
    n_tr = int(len(X) * 0.8)
    idx = np.random.default_rng(0).permutation(len(X))
    tr, te = idx[:n_tr], idx[n_tr:]
    for lam in (1e-3, 1e-1, 1.0, 10.0):
        A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
        beta = np.linalg.solve(A, X[tr].T @ Y[tr])
        pred = X[te] @ beta
        r2 = 1 - np.sum((pred - Y[te]) ** 2) / np.sum((Y[te] - Y[te].mean()) ** 2)
        want = np.abs(Y[te]) > 0.5
        got = np.abs(pred) > 0.6
        tp = np.sum(want & got); fp = np.sum(~want & got); fn = np.sum(want & ~got)
        print(f"lam {lam:7.3f}: R2 {r2:5.3f}  pass requested {want.sum()}, caught {tp} ({100 * tp / max(want.sum(), 1):.0f}%), false alarms {fp} ({100 * fp / max((~want).sum(), 1):.1f}% of the rest)")
    np.save("a2_offset_fit_data.npy", np.concatenate([H, Y[:, None]], axis=1).astype(np.float32))
