"""Collisions of a network under the attempt-2 pipeline, with the shield's mode in the steps before (development seeds)."""
import sys
import numpy as np
import brains, brains2, shield, sim
NET = brains.MLP()
theta = np.load(sys.argv[1])["theta"]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 400_001
overtake = len(sys.argv) > 4 and sys.argv[4] == "o"
NAMES = {0: "lead", 1: "oncoming", 2: "cross", 3: "parked", 4: "cutin", 5: "reverser"}
sc = sim.Scenarios(N, seed, overtake=overtake)
w = sim.World(sc)
rec = {}
orig = shield.shield
def wrapped(w_, vT_plan, dT_plan, rows, cap_fn):
    v, d, m = orig(w_, vT_plan, dT_plan, rows, cap_fn)
    rec.clear(); rec.update({int(r): int(mm) for r, mm in zip(rows, m)})
    return v, d, m
shield.shield = wrapped
hist = {e: [] for e in range(N)}
for step in range(sim.STEPS):
    out = NET.forward(theta[None, :], w.observe()[None, :, :])[0]
    vT, dT = brains2.nn_actions2(out, w)
    for e in np.where(~w.done)[0]:
        hist[e].append((step, round(float(w.ed[e]), 2), round(float(w.ev[e]), 1), round(float(vT[e]), 1), round(float(dT[e]), 2), rec.get(int(e), -1)))
        hist[e] = hist[e][-30:]
    before = w.done.copy()
    w.prevT, w.prevD = vT, dT
    w.step(vT, dT)
    kind = sc.kind
    cross = kind == sim.CROSS
    aL = np.where(cross, sc.W, sc.L); aW = np.where(cross, sc.L, sc.W)
    hit = (kind >= 0) & (np.abs(w.agS - w.es[:, None]) < (sim.EGO_L + aL) / 2) & (np.abs(w.agD - w.ed[:, None]) < (sim.EGO_W + aW) / 2)
    for e in np.where(hit.any(axis=1) & ~before)[0]:
        k = int(np.where(hit[e])[0][0])
        print(f"env {e} step {step}: hit {NAMES[int(kind[e, k])]} agent d {w.agD[e, k]:+.1f} v {w.agV[e, k]:.1f} | ego ed {w.ed[e]:+.2f} v {w.ev[e]:.1f} | twoWay {bool(sc.twoWay[e])}")
        print("    (step, ed, v, vT, dT, shield mode; -1 = not shielded):", hist[e][-16::3])
m = w.summary()
print("collisions", int(m["collision"].sum()), "red", int(m["red"].sum()), "near_s", round(float(m["near"].sum()), 1), "m/ep", round(float(m["dist"].mean()), 1))
