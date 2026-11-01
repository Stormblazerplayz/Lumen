"""Where does Lumen NN (with the guardrails) still fail? Collisions and red-light runs with their context.
usage: py diag_nn.py [checkpoint.npz] [episodes=512] [seed=555] [hard]
(seed 555 is a diagnosis set, not the held-out test seed 424242)"""
import sys

import numpy as np

import brains
import sim

NET = brains.MLP()
NAMES = {0: "lead", 1: "oncoming", 2: "cross", 3: "parked", 4: "cutin", 5: "reverser"}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "run2_best.npz"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 555
    hard = len(sys.argv) > 4 and sys.argv[4] == "hard"
    theta = np.load(path)["theta"]
    sc = sim.Scenarios(N, seed, hard=hard)
    w = sim.World(sc)
    events = []
    for k in range(sim.STEPS):
        out = NET.forward(theta[None, :], w.observe()[None])[0]
        vT, dT = brains.nn_actions(out, w)
        alive = ~w.done
        sig = sim.signal_state(w.t, sc)
        red0 = w.m["red"].copy()
        pre = (w.es.copy(), w.ev.copy())
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        kind = sc.kind
        active = kind >= 0
        cross = kind == sim.CROSS
        aL = np.where(cross, sc.W, sc.L)
        aW = np.where(cross, sc.L, sc.W)
        hit = active & (np.abs(w.agS - w.es[:, None]) < (sim.EGO_L + aL) / 2) & (np.abs(w.agD - w.ed[:, None]) < (sim.EGO_W + aW) / 2)
        for e in np.where(hit.any(axis=1) & alive)[0]:
            j = np.where(hit[e])[0][0]
            jj = int(sc.junc[e, j]) if kind[e, j] == sim.CROSS else -1
            events.append(dict(what="collision", ep=e, t=round(w.t, 1), by=NAMES[int(kind[e, j])], ego_v=round(float(pre[1][e]), 1), ag_v=round(float(w.agV[e, j]), 1),
                               ag_acc=round(float(w.agAcc[e, j]), 1), runner=bool(sc.runner[e, j]), light=int(sig[e, jj]) if jj >= 0 else None,
                               vT=round(float(vT[e]), 1), ego_s=round(float(w.es[e]), 0)))
        for e in np.where(w.m["red"] > red0)[0]:
            events.append(dict(what="red-light", ep=e, t=round(w.t, 1), ego_v=round(float(pre[1][e]), 1), vT=round(float(vT[e]), 1)))
    m = w.summary()
    km = m["dist"].sum() / 1000
    print(f"{path}: {km:.1f} km, collisions {int(m['collision'].sum())}, red {int(m['red'].sum())}, stop {int(m['stopsign'].sum())}, near {m['near'].sum():.1f} s, speed {m['dist'].sum() / m['time'].sum():.2f} m/s")
    by = {}
    for ev in events:
        key = (ev["what"], ev.get("by"), ev.get("runner"))
        by[key] = by.get(key, 0) + 1
    for key, n in sorted(by.items(), key=lambda kv: -kv[1]):
        print("  ", n, key)
    for ev in events[:10]:
        print("   ", ev)


if __name__ == "__main__":
    main()
