"""Sideways step response of the simulated ego: full time series so that a model can be fitted (development only)."""
import numpy as np
import sim

def response(v0, target, T=6.0, ed0=0.0, epsi0=0.0):
    sc2 = sim.Scenarios(1, 5)
    sc2.kind[:] = -1
    sc2.jKind[:] = sim.J_NONE
    sc2.kap[:] = 0.0
    sc2.edgeL[:] = -9.0
    sc2.edgeR[:] = 9.0
    w2 = sim.World(sc2)
    w2.ev[:] = v0
    w2.ed[:] = ed0
    w2.epsi[:] = epsi0
    out = [ed0]
    for k in range(int(T / sim.DT)):
        w2.step(np.full(1, v0), np.full(1, target))
        out.append(w2.ed[0])
    return np.array(out)

if __name__ == "__main__":
    for v0 in (2.0, 5.0, 10.0, 15.0, 20.0):
        r = response(v0, -3.4)
        ts = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        print(f"v={v0:4.1f}: " + " ".join(f"{t}s:{r[int(t * 10)]:+.2f}" for t in ts), f" peak lat speed {np.max(np.abs(np.diff(r))) / sim.DT:.2f}", f" overshoot {np.min(r) + 3.4:+.2f}")
