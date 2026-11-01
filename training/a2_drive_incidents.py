"""Where do Lumen Drive's incidents come from (development seeds)? The NN can only beat Drive's incident rate by
avoiding these, so this decides what the training objective must emphasise."""
import numpy as np
import brains, sim
NAMES = {0: "lead", 1: "oncoming", 2: "cross", 3: "parked", 4: "cutin", 5: "reverser"}
tot = {"red": 0, "stop": 0, "near_s": 0.0, "km": 0.0}
causes = {}
red_ctx = []
for seed in (400_001, 400_002, 400_003, 400_004):
    sc = sim.Scenarios(256, seed)
    w = sim.World(sc)
    for step in range(sim.STEPS):
        vT, dT = brains.rules_v3(w)
        w.prevT, w.prevD = vT, dT
        sig = sim.signal_state(w.t, sc)
        s0, v0 = w.es.copy(), w.ev.copy()
        red0 = w.m["red"].copy()
        w.step(vT, dT)
        kind = sc.kind
        cross = kind == sim.CROSS
        aL = np.where(cross, sc.W, sc.L); aW = np.where(cross, sc.L, sc.W)
        hit = (kind >= 0) & (np.abs(w.agS - w.es[:, None]) < (sim.EGO_L + aL) / 2) & (np.abs(w.agD - w.ed[:, None]) < (sim.EGO_W + aW) / 2)
        newc = hit.any(axis=1) & (w.m["collision"] > 0) & ~((w.m["collision"] > 1))
        for e in np.where((w.m["red"] > red0))[0]:
            red_ctx.append((round(float(v0[e]), 1), round(float(w.t), 1)))
        if w.done.all():
            break
    m = w.summary()
    tot["red"] += int(m["red"].sum()); tot["stop"] += int(m["stopsign"].sum()); tot["near_s"] += float(m["near"].sum()); tot["km"] += float(m["dist"].sum() / 1000)
    # collision causes: rerun-free approximation: the kind of the nearest agent at the end for collided episodes
    for e in np.where(m["collision"] > 0)[0]:
        d = np.abs(w.agS[e] - w.es[e]) + np.abs(w.agD[e] - w.ed[e])
        d = np.where(sc.kind[e] >= 0, d, 1e9)
        k = int(np.argmin(d))
        n = NAMES[int(sc.kind[e, k])]
        causes[n] = causes.get(n, 0) + 1
print("Drive on 1024 development episodes:", tot, "collisions by kind:", causes)
print("speed at the step a red was run (m/s, time):", red_ctx[:20])
