"""Attempt 3, step 1 (development seeds only): where do the incidents of Drive, the attempt-2 network and the control come
from? Every incident is filed by type (collision with which kind of vehicle, red light, stop sign, off road, seconds
too close) and by the car's phase when it happened:
  in lane      in its lane, no lateral manoeuvre in the last 4 s
  going out    out of its lane (|offset| > 0.45 m) and moving away from the lane centre or holding
  returning    out of its lane and moving back toward the lane centre
  just back    in its lane again, within 4 s of having been out
and, for the diagnostic classes, by the class (start_stop = standing start with a stop sign ahead).

usage: py a3_breakdown.py [workers=5] [pipeline=a2|a3] [theta=a2_final.npz]
"""
import collections
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import brains
import rulepass
import sim
import train_es

HERE = os.path.dirname(os.path.abspath(__file__))
NET = brains.MLP()
NAMES = {0: "lead", 1: "oncoming", 2: "cross", 3: "parked", 4: "cutin", 5: "reverser"}
SEEDS = {"original": [400_001 + i for i in range(4)], "overtake": [450_001 + i for i in range(4)],
         "start_stop": [420_001, 420_002]}


def make(cls, seed, n=256):
    if cls == "original":
        return sim.Scenarios(n, seed)
    if cls == "overtake":
        return sim.Scenarios(n, seed, overtake=True)
    return sim.Scenarios(n, seed, mode=cls)


def run(task):
    arm, cls, seed, pipeline, theta_path = task
    train_es.lower_priority()
    b2 = __import__("brains2_a2" if pipeline == "a2" else "brains2")
    sc = make(cls, seed)
    w = sim.World(sc)
    E = sc.E
    if arm == "drive":
        pol = brains.rules_v3
    elif arm == "control":
        lg = float(np.log(0.95 / 1.1 / (1 - 0.95 / 1.1)))
        pol = lambda w_: b2.nn_actions2(rulepass.rule_out(w_, free_logit=lg), w_)  # noqa: E731
    else:
        th = np.load(theta_path)["theta"]
        pol = lambda w_: b2.nn_actions2(NET.forward(th[None, :], w_.observe()[None, :, :])[0], w_)  # noqa: E731
    last_out = np.full(E, -1e9)
    counts = collections.Counter()
    near_s = collections.Counter()
    for _ in range(sim.STEPS):
        vT, dT = pol(w)
        alive = ~w.done
        out = np.abs(w.ed) > 0.45
        last_out = np.where(out & alive, w.t, last_out)
        moving_back = (w.ed * np.sin(w.epsi)) < 0
        phase = np.where(out, np.where(moving_back, "returning", "going out"), np.where(w.t - last_out < 4.0, "just back", "in lane"))
        before = {k: w.m[k].copy() for k in ("collision", "red", "stopsign", "offroad", "near")}
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        kind = sc.kind
        cross = kind == sim.CROSS
        aL = np.where(cross, sc.W, sc.L)
        aW = np.where(cross, sc.L, sc.W)
        hit = (kind >= 0) & (np.abs(w.agS - w.es[:, None]) < (sim.EGO_L + aL) / 2) & (np.abs(w.agD - w.ed[:, None]) < (sim.EGO_W + aW) / 2)
        for e in np.where(w.m["collision"] > before["collision"])[0]:
            ks = np.where(hit[e])[0]
            who = NAMES.get(int(kind[e, ks[0]]), "other") if len(ks) else "?"
            counts[("collision:" + who, phase[e])] += 1
        for k in ("red", "stopsign", "offroad"):
            for e in np.where(w.m[k] > before[k])[0]:
                counts[(k, phase[e])] += 1
        dn = w.m["near"] - before["near"]
        for e in np.where(dn > 0)[0]:
            near_s[phase[e]] += float(dn[e])
        if w.done.all():
            break
    km = float(w.m["dist"].sum() / 1000)
    return (arm, cls), {"counts": {f"{a}|{b}": v for (a, b), v in counts.items()}, "near_s": dict(near_s), "km": km}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    pipeline = sys.argv[2] if len(sys.argv) > 2 else "a2"
    theta = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "a2_final.npz")
    arms = ("drive", "nn", "control")
    tasks = [(arm, cls, seed, pipeline, theta) for cls, seeds in SEEDS.items() for seed in seeds for arm in arms]
    with Pool(workers) as pool:
        res = pool.map(run, tasks, chunksize=1)
    agg = {}
    for (arm, cls), r in res:
        a = agg.setdefault((arm, cls), {"counts": collections.Counter(), "near_s": collections.Counter(), "km": 0.0})
        a["counts"].update(r["counts"])
        a["near_s"].update(r["near_s"])
        a["km"] += r["km"]
    out = {}
    for cls in SEEDS:
        print(f"\n=== {cls} (development seeds, {len(SEEDS[cls]) * 256} episodes per arm), pipeline {pipeline} ===")
        for arm in arms:
            a = agg[(arm, cls)]
            total = sum(a["counts"].values()) + sum(a["near_s"].values())
            print(f"{arm:<8} {a['km']:7.1f} km, {total:6.1f} incidents ({total / max(a['km'], 1e-9) * 100:.2f} per 100 km)")
            for key, v in sorted(a["counts"].items(), key=lambda kv: -kv[1]):
                what, ph = key.split("|")
                print(f"     {what:<22} {ph:<10} {v:4d}")
            for ph, v in sorted(a["near_s"].items(), key=lambda kv: -kv[1]):
                print(f"     {'near (s)':<22} {ph:<10} {v:6.1f}")
            out[f"{arm}|{cls}"] = {"km": a["km"], "counts": dict(a["counts"]), "near_s": dict(a["near_s"])}
    with open(os.path.join(HERE, f"a3_breakdown_{pipeline}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
