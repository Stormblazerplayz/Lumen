"""Hard scenarios (diagnostic, NOT part of the pre-registered test): a car pulling out from the curb, a car in the next
lane swerving toward us, a car alongside changing into our lane. Storm asked: "Have you tested it through hard
scenarios? Like a car pulling out and a car swerving and a car changing lanes right next to Lumen?" (2026-11-01).

Every brain drives the same 512 episodes per class (fixed seeds, paired), each under the guards and shield it ships with:
  drive    Lumen Drive (brains.rules_v3)
  shipped  the network in src/lua/ge/extensions/lumen/nn_weights.json now (copied to training/old/nn_weights_v1.json),
           attempt-1 guards (brains.nn_actions), the filtered observation nn.lua uses now
  new      the attempt-2 candidate (brains2.nn_actions2: guards, lateral shield, in-lane check)
Measured after the actor starts its manoeuvre: collision (any), smallest box-to-box gap to the actor, smallest time to
collision with it at constant velocities (capped at 10 s), seconds braking harder than 4 m/s2.

usage: py a2_hard.py <candidate.npz> [workers=5] [episodes=512]
"""
import json
import math
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import brains
import brains2
import sim
import train_es

HERE = os.path.dirname(os.path.abspath(__file__))
NET = brains.MLP()
CLASSES = {"pull_out": 970_001, "swerve": 971_001, "side_merge": 972_001}
CHUNK = 128


def load_shipped():
    j = json.load(open(os.path.join(HERE, "old", "nn_weights_v1.json")))
    return np.concatenate([np.array(j[k], dtype=float) for k in ("w1", "b1", "w2", "b2", "w3", "b3")])


def run(task):
    brain, cls, seed, n, cand = task
    train_es.lower_priority()
    sc = sim.Scenarios(n, seed, mode=cls)
    w = sim.World(sc)
    if brain == "drive":
        pol = brains.rules_v3
    elif brain == "shipped":
        th = load_shipped()
        pol = lambda w_: brains.nn_actions(NET.forward(th[None, :], w_.observe()[None, :, :])[0], w_)  # noqa: E731
    else:
        th = np.load(cand)["theta"]
        pol = lambda w_: brains2.nn_actions2(NET.forward(th[None, :], w_.observe()[None, :, :])[0], w_)  # noqa: E731
    for _ in range(sim.STEPS):
        vT, dT = pol(w)
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
    m = w.summary()
    trig = w.hz_t0 >= 0
    return (brain, cls, seed), {"coll": (m["collision"] > 0).astype(float), "trig": trig.astype(float),
                                "gap": np.where(trig, w.hz_mingap, np.nan), "ttc": np.where(trig, np.minimum(w.hz_minttc, 10.0), np.nan),
                                "brake_s": w.hz_brake_s, "brake": (w.hz_brake_s > 0).astype(float)}


def binom_cdf(k, n, p):
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    tot = 0.0
    for i in range(0, k + 1):
        tot += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(1.0, tot)


def clopper_pearson(k, n, alpha=0.05):
    """exact two-sided interval for a binomial proportion"""
    def solve(f, lo=0.0, hi=1.0):
        for _ in range(80):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) >= alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) <= alpha / 2)
    return lower, upper


def paired_ci(a, b, B=10000, seed=3):
    """bootstrap 95% interval of mean(a - b) over episodes (paired); NaNs (actor never started) dropped pairwise"""
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (B, len(a)))
    d = (a[idx] - b[idx]).mean(axis=1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    cand = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    brains_ = ("drive", "shipped", "new")
    tasks = [(b, cls, seed + i, CHUNK, cand) for cls, seed in CLASSES.items() for i in range(n_ep // CHUNK) for b in brains_]
    with Pool(workers) as pool:
        res = pool.map(run, tasks, chunksize=1)
    store = {}
    for (b, cls, seed), out in sorted(res, key=lambda r: r[0][2]):
        d = store.setdefault((b, cls), {})
        for k, v in out.items():
            d.setdefault(k, []).append(v)
    table = {}
    lines = []
    for cls in CLASSES:
        lines.append(f"\n{cls}: {n_ep} episodes per brain, the same cars for every brain")
        lines.append(f"{'brain':<8} {'started':>7} {'collisions':>10} {'rate':>6} {'exact 95% CI':>16} {'mean min gap':>12} {'TTC median':>10} {'TTC 5th pct':>11} {'hard brake':>10} {'s/ep':>5}")
        M = {b: {k: np.concatenate(v) for k, v in store[(b, cls)].items()} for b in brains_}
        for b in brains_:
            m = M[b]
            k, n = int(m["coll"].sum()), len(m["coll"])
            lo, hi = clopper_pearson(k, n)
            gap = np.nanmean(m["gap"])
            ttc = m["ttc"][~np.isnan(m["ttc"])]
            row = {"started": int(m["trig"].sum()), "collisions": k, "n": n, "rate": k / n, "ci": [lo, hi], "mean_min_gap": float(gap),
                   "ttc_median": float(np.median(ttc)) if len(ttc) else None, "ttc_p5": float(np.percentile(ttc, 5)) if len(ttc) else None,
                   "hard_brake_rate": float(m["brake"].mean()), "hard_brake_s": float(m["brake_s"].mean())}
            table.setdefault(cls, {})[b] = row
            lines.append(f"{b:<8} {row['started']:7d} {k:10d} {100 * row['rate']:5.1f}% {100 * lo:6.1f}..{100 * hi:5.1f}% {gap:12.2f} {row['ttc_median']:10.2f} {row['ttc_p5']:11.2f} {100 * row['hard_brake_rate']:9.1f}% {row['hard_brake_s']:5.2f}")
        for b in ("shipped", "new"):
            dc = paired_ci(M[b]["coll"], M["drive"]["coll"])
            dg = paired_ci(M[b]["gap"], M["drive"]["gap"])
            dh = paired_ci(M[b]["brake"], M["drive"]["brake"])
            table[cls][b]["vs_drive"] = {"collision_rate": dc, "min_gap": dg, "hard_brake_rate": dh}
            lines.append(f"  {b} minus Drive: collision rate {100 * dc[0]:+.1f} points (95% CI {100 * dc[1]:+.1f}..{100 * dc[2]:+.1f}), "
                         f"min gap {dg[0]:+.2f} m ({dg[1]:+.2f}..{dg[2]:+.2f}), hard-brake rate {100 * dh[0]:+.1f} points ({100 * dh[1]:+.1f}..{100 * dh[2]:+.1f})")
    print("\n".join(lines))
    with open(os.path.join(HERE, "a2_hard_results.json"), "w") as f:
        json.dump(table, f, indent=1)


if __name__ == "__main__":
    main()
