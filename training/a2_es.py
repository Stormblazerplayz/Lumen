"""Attempt 2, stage 2 (2026-11-01): optimise the OUTPUT layer only (W3 and b3, 130 parameters) of the behaviour-cloned
network against the real objective, with the shield in the loop.

 * the hidden layers from a2_bc.py are frozen;
 * antithetic evolution strategies on the 130 output parameters, centred ranks, Adam;
 * every candidate of a generation drives the SAME scenarios (common random numbers): 40 per candidate, a mix of the
   junction-heavy original, plain original, overtake-heavy and the two diagnostic classes (training seeds 300_000+);
 * reward = metres - 3000 x collisions - 300 x seconds too close - 2000 x red lights - 1000 x stop signs - 2000 x
   leaving the road - 0.004 x jerk^2 - 1 x seconds in the opposing lane (the goal needs FEWER incidents than Drive);
 * every CHECK generations the current mean is run on development seeds (400_001 original, 450_001 overtake-heavy, 128
   and 64 episodes); the best by the ORIGINAL development reward is kept (a2_es_best.npz, and every checked one as
   a2_es_itNNN.npz); training stops after PATIENCE checks
   without improvement or at the deadline.

usage: py a2_es.py [workers=5] [minutes=20]
"""
import copy
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
import sim
import train_es

HERE = os.path.dirname(os.path.abspath(__file__))
NET = brains.MLP()
N_OUT = 64 * 2 + 2
MIX = (("hard", 12), ("plain", 16), ("overtake", 8), ("start_stop", 2), ("start_crowd", 2))
PAIRS, SIGMA, LR = 8, 0.05, 0.03
CHECK, PATIENCE = 4, 6
ATTRS = ("kap", "vlim", "twoWay", "lanesRight", "lanesLeftSame", "shoulder", "edgeR", "edgeL", "jS", "jKind", "jCycle",
         "jGreen", "jPhase", "jRunner", "kind", "s", "d", "v", "vdes", "L", "W", "trig", "junc", "side", "runner")


def make(cls, n, seed):
    if cls == "hard":
        return sim.Scenarios(n, seed, hard=True)
    if cls == "plain" or cls == "original":
        return sim.Scenarios(n, seed)
    if cls == "overtake":
        return sim.Scenarios(n, seed, overtake=True)
    return sim.Scenarios(n, seed, mode=cls)


def concat(scs):
    big = copy.copy(scs[0])
    big.E = sum(s.E for s in scs)
    for name in ATTRS:
        setattr(big, name, np.concatenate([getattr(s, name) for s in scs], axis=0))
    return big


def reward(m):
    return (m["dist"] - 3000 * m["collision"] - 300 * m["near"] - 2000 * m["red"] - 1000 * m["stopsign"]
            - 2000 * m["offroad"] - 0.004 * m["jerk2"] - 1.0 * m["wronglane"])


def run_members(theta0, outs, mix, seed):
    """every row of outs (P, N_OUT) replaces the output layer of theta0; all drive the same scenarios. Returns the
    reward (P, M) and the metrics reshaped to (P, M)."""
    P = outs.shape[0]
    scs = [make(c, n, seed + 17 * i) for i, (c, n) in enumerate(mix)]
    sc = concat(scs)
    M = sc.E
    w = sim.World(train_es.tile_scenarios(sc, P))
    W1, b1, W2, b2, _, _ = [a[0] for a in NET.unpack(theta0[None, :])]
    W3 = outs[:, :128].reshape(P, 64, 2)
    b3 = outs[:, 128:]
    for _ in range(sim.STEPS):
        obs = w.observe().astype(np.float64)
        h2 = np.tanh(np.tanh(obs @ W1 + b1) @ W2 + b2).reshape(P, M, 64)
        out = (np.matmul(h2, W3) + b3[:, None, :]).reshape(P * M, 2)
        vT, dT = brains2.nn_actions2(out, w)
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
    m = w.summary()
    r = reward(m).reshape(P, M)
    return r, {k: v.reshape(P, M) for k, v in m.items()}


def worker(args):
    train_es.lower_priority()
    theta0, outs, seed = args
    r, _ = run_members(theta0, outs, MIX, seed)
    return r.mean(axis=1)


def dev_check(theta):
    """the mean parameters on development seeds: reward and incident counts"""
    out = theta[None, -N_OUT:]
    r1, m1 = run_members(theta, out, (("plain", 192),), 400_001)
    r2, m2 = run_members(theta, out, (("overtake", 64),), 450_001)
    res = {}
    for tag, r, m in (("orig", r1, m1), ("over", r2, m2)):
        km = m["dist"].sum() / 1000
        inc = (m["collision"] + m["red"] + m["stopsign"] + m["offroad"] + m["near"]).sum()
        res[tag] = {"reward": round(float(r.mean()), 1), "m_ep": round(float(m["dist"].mean()), 1), "inc100": round(float(inc / max(km, 1e-9) * 100), 2),
                    "coll": int(m["collision"].sum())}
    res["score"] = res["orig"]["reward"]           # selection on the ORIGINAL distribution, the one the goal is decided on
    return res


def main():
    nworkers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 20
    start = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "a2_bc.npz")
    deadline = time.time() + minutes * 60
    train_es.lower_priority()
    theta = np.load(start)["theta"].astype(np.float64)
    theta0 = theta.copy()
    out = theta[-N_OUT:].copy()
    rng = np.random.default_rng(2026_11_01)
    opt = train_es.Adam(N_OUT, LR)
    best, since, it = -1e18, 0, 0
    log = open(os.path.join(HERE, "a2_es_log.jsonl"), "a")
    with Pool(nworkers) as pool:
        base = dev_check(np.concatenate([theta0[:-N_OUT], out]))
        print(json.dumps({"it": -1, "dev": base}), flush=True)
        log.write(json.dumps({"it": -1, "dev": base}) + "\n")
        best = base["score"]
        np.savez(os.path.join(HERE, "a2_es_best.npz"), theta=np.concatenate([theta0[:-N_OUT], out]), it=-1, score=best)
        while time.time() < deadline:
            t0 = time.time()
            eps = rng.standard_normal((PAIRS, N_OUT))
            members = np.concatenate([out + SIGMA * eps, out - SIGMA * eps], axis=0)
            seed = 300_001 + 100 * it
            res = pool.map(worker, [(theta0, c, seed) for c in np.array_split(members, nworkers)])
            r = np.concatenate(res)
            ranks = train_es.centred_ranks(r)
            g = ((ranks[:PAIRS] - ranks[PAIRS:])[:, None] * eps).sum(axis=0) / PAIRS
            out = out + opt.step(g - 0.01 * out)
            rec = {"it": it, "mean": round(float(r.mean()), 1), "max": round(float(r.max()), 1), "min": round(float(r.min()), 1), "sec": round(time.time() - t0, 1)}
            full = np.concatenate([theta0[:-N_OUT], out])
            if it % CHECK == CHECK - 1:
                dv = dev_check(full)
                rec["dev"] = dv
                np.savez(os.path.join(HERE, f"a2_es_it{it:03d}.npz"), theta=full, it=it, score=dv["score"])
                if dv["score"] > best:
                    best, since = dv["score"], 0
                    np.savez(os.path.join(HERE, "a2_es_best.npz"), theta=full, it=it, score=best)
                else:
                    since += 1
            np.savez(os.path.join(HERE, "a2_es_last.npz"), theta=full, it=it)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            it += 1
            if since >= PATIENCE:
                print("early stop: no development improvement for", PATIENCE, "checks", flush=True)
                break


if __name__ == "__main__":
    main()
