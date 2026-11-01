"""Attempt 4 (README "Attempt 4: PRE-REGISTRATION ADDENDUM"): ONE detached process that trains, selects, runs the held-out
seeds once and stages the weights. Log: a4_train_log.jsonl (one line per generation and per development check) and
a4_train_stdout.log.

  sep-CMA-ES (diagonal covariance, mirrored sampling) over W2, b2, W3, b3 of brains4.Net5 (4485 parameters), warm start
  Net5.from_a2(a2_final.npz); 16 candidates x 64 common scenarios per generation (seeds 320_001 + 100 g); objective
  = the attempt-3 objective; every 5 generations the CMA mean drives development seeds and the best is kept; then
  eval2.py --split heldout5 once; then staged/nn_weights_a4.json with numpy self-test vectors.

usage: py a4_train.py [generations=100] [workers=5] [deadline_minutes=170]
"""
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import brains4
import sim
import train_es
from brains4 import Net5

HERE = os.path.dirname(os.path.abspath(__file__))
LAM = 16
SIGMA0 = 0.5
SCALE_HIDDEN, SCALE_OUT = 0.02, 0.08
MIX = (("hard", 24), ("plain", 20), ("overtake", 12), ("start_stop", 4), ("start_crowd", 4))
CHECK = 5
ATTRS = ("kap", "vlim", "twoWay", "lanesRight", "lanesLeftSame", "shoulder", "edgeR", "edgeL", "jS", "jKind", "jCycle",
         "jGreen", "jPhase", "jRunner", "kind", "s", "d", "v", "vdes", "L", "W", "trig", "junc", "side", "runner")
FROZEN = ("a4_train.py", "brains4.py", "brains2.py", "shield.py", "caution.py", "sim.py", "brains.py", "eval2.py",
          "a3_extra.py", "train_es.py", "a2_final.npz", "lumen_nn_final.npz")


def make(cls, n, seed):
    if cls == "hard":
        return sim.Scenarios(n, seed, hard=True)
    if cls in ("plain", "original"):
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
    return (m["dist"] - 10000 * m["collision"] - 1000 * m["near"] - 6000 * m["red"] - 3000 * m["stopsign"]
            - 6000 * m["offroad"] - 0.004 * m["jerk2"] - 2.0 * m["wronglane"])


def run_members(thetas, mix, seed):
    """every row of thetas (P, n) drives the same scenarios; returns reward (P, M) and the metrics (P, M)"""
    P = thetas.shape[0]
    sc = concat([make(c, n, seed + 17 * i) for i, (c, n) in enumerate(mix)])
    M = sc.E
    w = sim.World(train_es.tile_scenarios(sc, P))
    for _ in range(sim.STEPS):
        obs = w.observe().astype(np.float64).reshape(P, M, -1)
        out = Net5.forward(thetas, obs).reshape(P * M, 5)
        vT, dT = brains4.nn_actions4(out, w)
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
    m = w.summary()
    return reward(m).reshape(P, M), {k: v.reshape(P, M) for k, v in m.items()}


def worker(args):
    train_es.lower_priority()
    thetas, seed = args
    r, _ = run_members(thetas, MIX, seed)
    return r.mean(axis=1)


def dev_worker(args):
    train_es.lower_priority()
    theta, cls, seed, n = args
    r, m = run_members(theta[None, :], ((cls, n),), seed)
    return cls, float(r.sum()), float(m["dist"].sum()), float((m["collision"] + m["red"] + m["stopsign"] + m["offroad"] + m["near"]).sum()), int(m["collision"].sum()), int(m["red"].sum()), n


def dev_check(pool, theta):
    jobs = [(theta, "original", 400_001, 128), (theta, "original", 400_002, 128), (theta, "overtake", 450_001, 128)]
    res = pool.map(dev_worker, jobs)
    out = {}
    for cls, rsum, dist, inc, coll, red, n in res:
        o = out.setdefault(cls, {"r": 0.0, "dist": 0.0, "inc": 0.0, "coll": 0, "red": 0, "n": 0})
        o["r"] += rsum; o["dist"] += dist; o["inc"] += inc; o["coll"] += coll; o["red"] += red; o["n"] += n
    rep = {}
    for cls, o in out.items():
        rep[cls] = {"reward": round(o["r"] / o["n"], 1), "m_ep": round(o["dist"] / o["n"], 1),
                    "inc100": round(o["inc"] / max(o["dist"] / 1000, 1e-9) * 100, 2), "coll": o["coll"], "red": o["red"]}
    rep["score"] = rep["original"]["reward"]
    return rep


class SepCMA:
    """separable CMA-ES (Ros and Hansen 2008) with mirrored sampling; maximises"""

    def __init__(self, n, sigma0, lam, rng):
        self.n, self.lam, self.mu = n, lam, lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1 / np.sum(self.w ** 2)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.ds = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = 4 / (n + 4)
        c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        cmu = min(1 - c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        k = (n + 2) / 3
        self.c1, self.cmu = min(c1 * k, 0.5), min(cmu * k, 0.5)
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = np.zeros(n)
        self.sigma = sigma0
        self.C = np.ones(n)
        self.ps = np.zeros(n)
        self.pc = np.zeros(n)
        self.g = 0
        self.rng = rng

    def ask(self):
        z = self.rng.standard_normal((self.lam // 2, self.n))
        z = np.concatenate([z, -z])
        return self.m + self.sigma * z * np.sqrt(self.C)

    def tell(self, x, f):
        idx = np.argsort(-f)[:self.mu]
        y = (x[idx] - self.m) / self.sigma
        yw = self.w @ y
        self.m = self.m + self.sigma * yw
        zw = yw / np.sqrt(self.C)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * zw
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.g + 1))) / self.chiN < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (self.pc ** 2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (self.w @ (y ** 2)))
        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.g += 1


def export_staged(theta, path):
    """the attempt-4 weights for nn.lua (5 outputs), with numpy outputs for 14 test vectors computed from the rounded
    weights exactly as the game will read them"""
    def r6(a):
        return [float(f"{x:.6g}") for x in np.asarray(a, dtype=np.float64).ravel()]
    parts = Net5.unpack(theta)
    rounded = [np.array(r6(a)).reshape(a.shape) for a in parts]
    sc = sim.Scenarios(32, 31337, hard=True)
    w = sim.World(sc)
    xs = []
    for k in range(400):
        if k in (60, 120, 180, 240, 300) and len(xs) < 10:
            o = w.observe()
            xs.extend([o[0].astype(np.float64), o[7].astype(np.float64)])
        w.step(np.full(32, 8.0), np.zeros(32))
    rng = np.random.default_rng(5)
    xs = xs[:10] + [rng.standard_normal(sim.OBS_DIM) * 0.7 for _ in range(4)]
    W1, b1, W2, b2, W3, b3 = rounded
    tests = []
    for x in xs:
        x = np.array(r6(x))
        y = np.tanh(np.tanh(x @ W1 + b1) @ W2 + b2) @ W3 + b3
        tests.append({"x": [float(v) for v in x], "y": [float(f"{v:.9g}") for v in y]})
    info = {"obs_dim": sim.OBS_DIM, "hidden": 64, "outputs": 5, "checkpoint": "a4_final.npz",
            "outputs_meaning": ["speed share logit", "junction-approach share logit (0.3 + 0.7 sigmoid)", "intent: stay", "intent: pass left", "intent: pass right"],
            "note": "Lumen NN attempt 4 (2026-11-01): needs the attempt-4 pipeline (training/brains4.py, staged/NN_INTEGRATION.md)"}
    out = {"info": info, "w1": r6(W1), "b1": r6(b1), "w2": r6(W2), "b2": r6(b2), "w3": r6(W3), "b3": r6(b3), "selftest": tests}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, separators=(",", ":"))


def log(rec, f):
    rec["t"] = time.strftime("%H:%M:%S")
    line = json.dumps(rec)
    f.write(line + "\n")
    f.flush()
    print(line, flush=True)


def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    minutes = float(sys.argv[3]) if len(sys.argv) > 3 else 170
    deadline = time.time() + minutes * 60
    train_es.lower_priority()
    flog = open(os.path.join(HERE, "a4_train_log.jsonl"), "a")
    hashes = {p: hashlib.sha256(open(os.path.join(HERE, p), "rb").read()).hexdigest()[:16] for p in FROZEN}
    log({"event": "start", "generations": gens, "workers": nworkers, "hashes": hashes}, flog)
    theta0 = Net5.from_a2(np.load(os.path.join(HERE, "a2_final.npz"))["theta"])
    tr = Net5.TRAIN
    ntr = tr.stop - tr.start
    scale = np.full(ntr, SCALE_HIDDEN)
    scale[Net5.sizes[2] + Net5.sizes[3]:] = SCALE_OUT            # W3, b3 (after W2 and b2 inside the slice)
    rng = np.random.default_rng(2026_11_04)
    cma = SepCMA(ntr, SIGMA0, LAM, rng)

    def full(x):
        th = theta0.copy()
        th[tr] = theta0[tr] + scale * x
        return th

    best, best_g = -1e18, -1
    with Pool(nworkers) as pool:
        dv = dev_check(pool, theta0)
        log({"event": "dev", "g": -1, "dev": dv}, flog)
        best = dv["score"]
        np.savez(os.path.join(HERE, "a4_best.npz"), theta=theta0, g=-1, score=best)
        for g in range(gens):
            if time.time() > deadline:
                log({"event": "deadline", "g": g}, flog)
                break
            t0 = time.time()
            xs = cma.ask()
            thetas = np.stack([full(x) for x in xs])
            res = pool.map(worker, [(c, 320_001 + 100 * g) for c in np.array_split(thetas, nworkers)])
            f = np.concatenate(res)
            cma.tell(xs, f)
            rec = {"event": "gen", "g": g, "mean": round(float(f.mean()), 1), "max": round(float(f.max()), 1), "min": round(float(f.min()), 1),
                   "sigma": round(float(cma.sigma), 4), "sec": round(time.time() - t0, 1)}
            if g % CHECK == CHECK - 1:
                th_m = full(cma.m)
                dv = dev_check(pool, th_m)
                rec["dev"] = dv
                np.savez(os.path.join(HERE, f"a4_g{g:03d}.npz"), theta=th_m, g=g, score=dv["score"])
                if dv["score"] > best:
                    best, best_g = dv["score"], g
                    np.savez(os.path.join(HERE, "a4_best.npz"), theta=th_m, g=g, score=best)
            log(rec, flog)
    z = np.load(os.path.join(HERE, "a4_best.npz"))
    np.savez(os.path.join(HERE, "a4_final.npz"), theta=z["theta"], g=z["g"], score=z["score"])
    log({"event": "frozen", "best_g": int(z["g"]), "dev_score": float(z["score"])}, flog)
    export_staged(z["theta"], os.path.join(HERE, "staged", "nn_weights_a4.json"))
    log({"event": "staged", "file": "staged/nn_weights_a4.json"}, flog)
    # the one held-out run (pre-registered arms and seeds)
    with open(os.path.join(HERE, "a4_heldout_stdout.log"), "w", encoding="utf-8") as fo:
        rc = subprocess.call([sys.executable, os.path.join(HERE, "eval2.py"), "--split", "heldout5", "--arms", "drive,a4,a4_noshield,shipped_inlane",
                              "--classes", "original,overtake", "--theta", os.path.join(HERE, "a4_final.npz"), "--workers", str(nworkers),
                              "--out", os.path.join(HERE, "a4_heldout5_results.json")], stdout=fo, stderr=subprocess.STDOUT, cwd=HERE)
    log({"event": "heldout_done", "returncode": rc}, flog)


if __name__ == "__main__":
    main()
