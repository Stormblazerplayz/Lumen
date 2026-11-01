"""Measure the parked-car caution rule (caution.py) for Drive and for the network on a pull-out set and on the original
distribution, development seeds only (pull_out 470_001+, original 400_001+)."""
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

import brains
import brains2
import caution
import sim
import train_es

NET = brains.MLP()


def run(task):
    brain, variant, cls, seed, n = task
    train_es.lower_priority()
    sc = sim.Scenarios(n, seed, mode="pull_out") if cls == "pull_out" else sim.Scenarios(n, seed)
    w = sim.World(sc)
    th = np.load("a2_final.npz")["theta"] if brain == "nn" else None
    for _ in range(sim.STEPS):
        if brain == "drive":
            vT, dT = brains.rules_v3(w)
            if variant != "none":
                vT = np.minimum(vT, caution.cap(w, variant))
        else:
            out = NET.forward(th[None, :], w.observe()[None, :, :])[0]
            brains2.CAUTION = None if variant == "none" else variant
            vT, dT = brains2.nn_actions2(out, w)
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
    m = w.summary()
    km = m["dist"].sum() / 1000
    inc = (m["collision"] + m["red"] + m["stopsign"] + m["offroad"] + m["near"]).sum()
    return task, {"coll_eps": int((m["collision"] > 0).sum()), "n": n, "m_ep": float(m["dist"].mean()), "inc100": float(inc / max(km, 1e-9) * 100),
                  "dist": m["dist"], "coll": (m["collision"] > 0).astype(float)}


if __name__ == "__main__":
    tasks = []
    for brain in ("drive", "nn"):
        for variant in ("none", "flat", "stopping"):
            for i in range(4):
                tasks.append((brain, variant, "pull_out", 470_001 + i, 128))
                tasks.append((brain, variant, "original", 400_001 + i, 256))
    with Pool(5) as pool:
        res = pool.map(run, tasks, chunksize=1)
    agg = {}
    for (brain, variant, cls, seed, n), r in res:
        a = agg.setdefault((brain, variant, cls), {"coll": [], "dist": [], "inc": []})
        a["coll"].append(r["coll"]); a["dist"].append(r["dist"]); a["inc"].append(r["inc100"])
    for cls in ("pull_out", "original"):
        print(f"\n{cls}:")
        for brain in ("drive", "nn"):
            base = np.concatenate(agg[(brain, "none", cls)]["dist"])
            for variant in ("none", "flat", "stopping"):
                a = agg[(brain, variant, cls)]
                c = np.concatenate(a["coll"]); d = np.concatenate(a["dist"])
                print(f"  {brain:<6} {variant:<9} collision episodes {int(c.sum()):4d}/{len(c)} ({100 * c.mean():5.1f}%)  m/ep {d.mean():6.1f} ({100 * (d.mean() / base.mean() - 1):+.1f}% vs no caution)  incidents/100km {np.mean(a['inc']):.2f}")
