"""Train Lumen NN with evolution strategies (OpenAI-ES: antithetic noise, centred ranks, Adam).

Every iteration: 32 noise vectors, each tried as +sigma and -sigma (64 brains), each brain drives the
same 16 fresh random scenarios; the update follows the brains that scored better. The score is
reward(): metres of progress minus incidents (collision 300, red light 150, stop sign 100, off road
200, 40 per second too close) and discomfort. Nothing is copied from Lumen Drive or any other
driver: the only teacher is the outcome.

Workers run at below-normal priority so the PC stays usable. Progress goes to train_log.jsonl and
the best parameters to lumen_nn_best.npz after each evaluation round.
usage: py train_es.py [iterations] [workers]"""
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

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS, SCEN, SIGMA, LR = 32, 16, 0.04, 0.03
NET = brains.MLP()


def lower_priority():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
    except Exception:
        pass


def run_brains(params, seed, n_scen):
    """Drive n_scen scenarios with each of params (P, n). Returns per-member mean reward and metrics."""
    P = params.shape[0]
    sc = sim.Scenarios(n_scen, seed)
    # the same scenarios for every member: tile them
    big = tile_scenarios(sc, P)
    w = sim.World(big)
    for _ in range(sim.STEPS):
        obs = w.observe().reshape(P, n_scen, -1)
        out = NET.forward(params, obs).reshape(P * n_scen, 2)
        vT, dT = brains.nn_actions(out, w)
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
    m = w.summary()
    r = brains.reward(m).reshape(P, n_scen).mean(axis=1)
    return r, m


def tile_scenarios(sc, P):
    """P copies of the same E scenarios as one batch of P*E environments."""
    import copy
    big = copy.copy(sc)
    big.E = sc.E * P
    for name in ("kap", "vlim", "twoWay", "lanesRight", "lanesLeftSame", "shoulder", "edgeR", "edgeL", "jS", "jKind", "jCycle",
                 "jGreen", "jPhase", "jRunner", "kind", "s", "d", "v", "vdes", "L", "W", "trig", "junc", "side", "runner"):
        a = getattr(sc, name)
        setattr(big, name, np.concatenate([a] * P, axis=0))
    return big


def worker(args):
    lower_priority()
    params, seed = args
    r, _ = run_brains(params, seed, SCEN)
    return r


class Adam:
    def __init__(self, n, lr, b1=0.9, b2=0.999):
        self.m, self.v, self.t, self.lr, self.b1, self.b2 = np.zeros(n), np.zeros(n), 0, lr, b1, b2

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return self.lr * mh / (np.sqrt(vh) + 1e-8)


def centred_ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    lower_priority()
    rng = np.random.default_rng(2026)
    theta = NET.init(rng)
    opt = Adam(NET.n, LR)
    log_path = os.path.join(HERE, "train_log.jsonl")
    best_eval = -1e18
    with Pool(nworkers) as pool, open(log_path, "a") as log:
        for it in range(iters):
            t0 = time.time()
            eps = rng.standard_normal((PAIRS, NET.n))
            members = np.concatenate([theta + SIGMA * eps, theta - SIGMA * eps], axis=0)
            seed = 10_000 + it
            chunks = np.array_split(members, nworkers)
            res = pool.map(worker, [(c, seed) for c in chunks])
            r = np.concatenate(res)
            ranks = centred_ranks(r)
            g = ((ranks[:PAIRS] - ranks[PAIRS:])[:, None] * eps).sum(axis=0) / (PAIRS * SIGMA)
            theta = theta + opt.step(g - 0.005 * theta)
            rec = {"it": it, "mean": float(r.mean()), "max": float(r.max()), "sec": round(time.time() - t0, 1)}
            # every 10 iterations: the current brain on 64 fixed check scenarios
            if it % 10 == 0 or it == iters - 1:
                rr, m = run_brains(theta[None, :], 777, 64)
                km = m["dist"].sum() / 1000
                rec.update({"eval": float(rr[0]), "km": round(km, 2), "coll": int(m["collision"].sum()), "red": int(m["red"].sum()),
                            "stopsign": int(m["stopsign"].sum()), "off": int(m["offroad"].sum()), "near_s": round(float(m["near"].sum()), 1)})
                if rr[0] > best_eval:
                    best_eval = float(rr[0])
                    np.savez(os.path.join(HERE, "lumen_nn_best.npz"), theta=theta, it=it, eval=best_eval)
            np.savez(os.path.join(HERE, "lumen_nn_last.npz"), theta=theta, it=it)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
