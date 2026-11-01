"""Baseline table: run a brain function over N scenarios, print the metrics and collision causes."""
import sys
import numpy as np
import sim, brains
names = {0: 'lead', 1: 'oncoming', 2: 'cross', 3: 'parked', 4: 'cutin', 5: 'reverser'}

def run(policy, N, seed, verbose=False):
    sc = sim.Scenarios(N, seed)
    w = sim.World(sc)
    hits = {}
    for k in range(sim.STEPS):
        vT, dT = policy(w)
        alive = ~w.done
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        kind = sc.kind; active = kind >= 0; cross = kind == sim.CROSS
        aL = np.where(cross, sc.W, sc.L); aW = np.where(cross, sc.L, sc.W)
        hit = active & (np.abs(w.agS - w.es[:, None]) < (sim.EGO_L + aL) / 2) & (np.abs(w.agD - w.ed[:, None]) < (sim.EGO_W + aW) / 2)
        for e in np.where(hit.any(axis=1) & alive)[0]:
            n = names[int(kind[e, np.where(hit[e])[0][0]])]
            hits[n] = hits.get(n, 0) + 1
    return w.summary(), hits

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    m, hits = run(brains.rules, N, 99)
    print('rules: km %.1f, collisions %d (%s), near %.1f s, red %d, stopsign %d, offroad %d, stopped %.0f%% of time, mean speed %.1f m/s' % (
        m['dist'].sum() / 1000, m['collision'].sum(), hits, m['near'].sum(), m['red'].sum(), m['stopsign'].sum(), m['offroad'].sum(),
        100 * m['blocked'].sum() / max(1, m['time'].sum()), m['dist'].sum() / max(1, m['time'].sum())))
