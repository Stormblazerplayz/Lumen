"""Quick development comparison (single process): Drive vs a network or the rule under the attempt-2 pipeline."""
import sys, time
import numpy as np
import brains, brains2, rulepass, sim, evaluate
NET = brains.MLP()
arm = sys.argv[1]; N = int(sys.argv[2]); seed = int(sys.argv[3]); overtake = sys.argv[4] == "o"
theta = np.load(sys.argv[5])["theta"] if len(sys.argv) > 5 else None
sc = sim.Scenarios(N, seed, overtake=overtake)
w = sim.World(sc)
t0 = time.time()
modes = np.zeros(8); slowed = 0
for _ in range(sim.STEPS):
    if arm == "drive":
        vT, dT = brains.rules_v3(w)
    elif arm == "rule":
        vT, dT = brains2.nn_actions2(rulepass.rule_out(w), w)
    else:
        out = NET.forward(theta[None, :], w.observe()[None, :, :])[0]
        vT, dT = brains2.nn_actions2(out, w)
    if arm != "drive":
        modes += brains2.LAST["mode"]; slowed += brains2.LAST.get("slowed", 0)
    w.prevT, w.prevD = vT, dT
    w.step(vT, dT)
    if w.done.all():
        break
m = w.summary(); st = evaluate.stats(m, N)
print(f"{arm:<6} {'over' if overtake else 'orig'} N={N}: inc/100km {st['per100km']:5.2f} coll {st['collisions']} red {st['red']} stop {st['stopsign']} off {st['offroad']} near {st['near_s']:.1f}s | m/ep {st['m_per_ep']:.0f} speed {st['speed']:.2f} | {time.time() - t0:.0f}s | modes/ep {np.round(modes / N, 1).tolist()} slowed/ep {slowed / N:.1f}")
