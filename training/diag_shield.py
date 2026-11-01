"""Why does the shielded scripted pass still collide? Prints the state at every collision (development seeds)."""
import sys

import numpy as np

import brains2
import sim
import test_shield as T

N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
overtake = len(sys.argv) > 2 and sys.argv[2] == "o"
use_shield = not (len(sys.argv) > 3 and sys.argv[3] == "noshield")
sc = sim.Scenarios(N, 400_001, overtake=overtake)
w = sim.World(sc)
hist = {e: [] for e in range(N)}
NAMES = {0: "lead", 1: "oncoming", 2: "cross", 3: "parked", 4: "cutin", 5: "reverser"}
printed = 0
modes_hist = np.zeros((N, 8), dtype=int)
for step in range(sim.STEPS):
    out = T.scripted_out(w)
    vT, dT = brains2.nn_actions2(out, w, use_shield=use_shield)
    alive = ~w.done
    w.prevT, w.prevD = vT, dT
    snap = (step, w.es.copy(), w.ed.copy(), w.ev.copy(), vT.copy(), dT.copy(), w.agS.copy(), w.agD.copy(), w.agV.copy())
    for e in np.where(alive)[0]:
        hist[e].append((step, w.es[e], w.ed[e], w.ev[e], vT[e], dT[e]))
        if len(hist[e]) > 40:
            hist[e].pop(0)
    before_done = w.done.copy()
    w.step(vT, dT)
    newly = w.done & ~before_done
    kind = sc.kind
    cross = kind == sim.CROSS
    aL = np.where(cross, sc.W, sc.L)
    aW = np.where(cross, sc.L, sc.W)
    hit = (kind >= 0) & (np.abs(w.agS - w.es[:, None]) < (sim.EGO_L + aL) / 2) & (np.abs(w.agD - w.ed[:, None]) < (sim.EGO_W + aW) / 2)
    for e in np.where(newly & hit.any(axis=1))[0]:
        k = int(np.where(hit[e])[0][0])
        printed += 1
        if printed <= 24:
            h = hist[e]
            print(f"env {e} step {step}: hit {NAMES[int(kind[e, k])]} (L {sc.L[e, k]:.1f}) ego s {w.es[e]:.1f} ed {w.ed[e]:+.2f} v {w.ev[e]:.1f} epsi {w.epsi[e]:+.2f}"
                  f" | agent s {w.agS[e, k]:.1f} d {w.agD[e, k]:+.2f} v {w.agV[e, k]:.1f} | twoWay {bool(sc.twoWay[e])} edges {sc.edgeL[e]:.1f}..{sc.edgeR[e]:.1f}")
            print("      last plans (step, s, ed, v, vT, dT):", " ".join(f"({a},{b:.0f},{c:+.1f},{d_:.1f},{e_:.1f},{f_:+.1f})" for a, b, c, d_, e_, f_ in h[-12::3]))
print("collisions printed:", printed)
