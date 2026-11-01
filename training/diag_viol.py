"""Trace the first few red-light / stop-sign violations of the rule brain in the simulator."""
import numpy as np
import sim, brains

N = 512
sc = sim.Scenarios(N, seed=99)
w = sim.World(sc)
trace = {e: [] for e in range(N)}
viol = {}
for k in range(sim.STEPS):
    vT, dT = brains.rules(w)
    sig = sim.signal_state(w.t, sc)
    s0 = w.es.copy()
    before = {"red": w.m["red"].copy(), "stop": w.m["stopsign"].copy()}
    w.prevT, w.prevD = vT, dT
    jj = np.argmin(np.where(sc.jS - 6 - (w.es[:, None] + 2.6) > -3, sc.jS - 6 - (w.es[:, None] + 2.6), 1e9), axis=1)
    for e in range(N):
        d = sc.jS[e, jj[e]] - 6 - (w.es[e] + 2.6)
        trace[e].append((round(w.t, 1), round(float(w.es[e]), 1), round(float(w.ev[e]), 2), round(float(vT[e]), 2), int(sig[e, jj[e]]), round(float(d), 1), int(sc.jKind[e, jj[e]])))
    w.step(vT, dT)
    for e in np.where(w.m["red"] > before["red"])[0]:
        viol.setdefault(e, ("red", k))
    for e in np.where(w.m["stopsign"] > before["stop"])[0]:
        viol.setdefault(e, ("stop", k))
print("violation episodes:", len(viol), "red", sum(1 for v in viol.values() if v[0] == "red"), "stop", sum(1 for v in viol.values() if v[0] == "stop"))
shown = {"red": 0, "stop": 0}
for e, (kind, k) in viol.items():
    if shown[kind] >= 3:
        continue
    shown[kind] += 1
    print(f"--- episode {e} {kind} violation at step {k} (t={k*0.1:.1f}); trace (t, s, v, vT, sigstate[0 none 1 green 2 yellow 3 red 4 stopsign], dist front->line, kind):")
    for row in trace[e][max(0, k - 40):k + 3:3]:
        print("   ", row)
