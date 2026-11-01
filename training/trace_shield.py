"""Step-by-step trace of one environment under the scripted pass + shield (development seeds)."""
import sys
import numpy as np
import brains2, shield, sim
import test_shield as T

env = int(sys.argv[1]); N = int(sys.argv[2]); overtake = sys.argv[3] == "o"
t0, t1 = int(sys.argv[4]), int(sys.argv[5])
sc = sim.Scenarios(N, 400_001, overtake=overtake)
w = sim.World(sc)
NAMES = {0: "lead", 1: "oncoming", 2: "cross", 3: "parked", 4: "cutin", 5: "reverser"}
# patch the shield to record the mode per row
rec = {}
orig = shield.shield
def wrapped(w_, vT_plan, dT_plan, rows, cap_fn):
    v, d, m = orig(w_, vT_plan, dT_plan, rows, cap_fn)
    rec["rows"], rec["m"], rec["v"], rec["d"] = rows.copy(), m.copy(), v.copy(), d.copy()
    return v, d, m
shield.shield = wrapped
for step in range(sim.STEPS):
    rec.clear()
    out = T.scripted_out(w)
    vT, dT = brains2.nn_actions2(out, w)
    alive = ~w.done
    if t0 <= step <= t1 and alive[env]:
        mode = "-"
        if "rows" in rec and env in rec["rows"]:
            mode = str(int(rec["m"][list(rec["rows"]).index(env)]))
        near = []
        for k in range(sim.K):
            if sc.kind[env, k] >= 0 and abs(w.agS[env, k] - w.es[env]) < 70 and abs(w.agD[env, k] - w.ed[env]) < 6:
                near.append(f"{NAMES[int(sc.kind[env, k])][:4]}@{w.agS[env, k] - w.es[env]:+.0f}/{w.agD[env, k]:+.1f}/v{w.agV[env, k]:.1f}")
        print(f"{step:3d} s {w.es[env]:6.1f} ed {w.ed[env]:+5.2f} v {w.ev[env]:5.2f} epsi {w.epsi[env]:+.2f} | vT {vT[env]:5.1f} dT {dT[env]:+5.2f} mode {mode} | " + " ".join(near))
    w.prevT, w.prevD = vT, dT
    w.step(vT, dT)
    if not alive[env] or w.done[env] and step > t1:
        break
