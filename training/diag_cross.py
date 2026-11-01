"""Trace cross-traffic and cut-in collisions of the rule brain: who, where, how fast."""
import numpy as np
import sim, brains
N = 512
sc = sim.Scenarios(N, seed=99)
w = sim.World(sc)
rec = {}
hist = {e: [] for e in range(N)}
for k in range(sim.STEPS):
    vT, dT = brains.rules(w)
    alive = ~w.done
    sig = sim.signal_state(w.t, sc)
    w.prevT, w.prevD = vT, dT
    for e in range(N):
        if alive[e]:
            hist[e].append((round(w.t, 1), round(float(w.es[e]), 1), round(float(w.ev[e]), 1), round(float(vT[e]), 1)))
    w.step(vT, dT)
    kind = sc.kind; active = kind >= 0; cross = kind == sim.CROSS
    aL = np.where(cross, sc.W, sc.L); aW = np.where(cross, sc.L, sc.W)
    hit = active & (np.abs(w.agS - w.es[:, None]) < (sim.EGO_L + aL) / 2) & (np.abs(w.agD - w.ed[:, None]) < (sim.EGO_W + aW) / 2)
    for e in np.where(hit.any(axis=1) & alive)[0]:
        j = np.where(hit[e])[0][0]
        if e not in rec and kind[e, j] in (sim.CROSS, sim.CUTIN):
            jj = sc.junc[e, j]
            rec[e] = dict(kind=int(kind[e, j]), t=round(w.t, 1), ego_s=round(float(w.es[e]), 1), ego_v=round(float(w.ev[e]), 1), junc_s=round(float(sc.jS[e, jj]), 1), jkind=int(sc.jKind[e, jj]), light=int(sig[e, jj]),
                          ag_d=round(float(w.agD[e, j]), 1), ag_v=round(float(w.agV[e, j]), 1), runner=bool(sc.runner[e, j]), side=float(sc.side[e, j]))
for e, r in rec.items():
    print(e, r)
    print("    ego (t, s, v, vT):", hist[e][-14::3])
