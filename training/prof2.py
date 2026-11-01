import time, cProfile, pstats, numpy as np
import brains, brains2, shield, sim
NET = brains.MLP()
theta = np.load("a2_bc.npz")["theta"]
sc = sim.Scenarios(512, 300_123, hard=True)
w = sim.World(sc)
def step():
    out = NET.forward(theta[None, :], w.observe()[None, :, :])[0]
    vT, dT = brains2.nn_actions2(out, w)
    w.prevT, w.prevD = vT, dT
    w.step(vT, dT)
for _ in range(60): step()
t0 = time.time()
pr = cProfile.Profile(); pr.enable()
for _ in range(60): step()
pr.disable()
print(f"512 hard envs: {(time.time() - t0) / 60 * 1000:.0f} ms per step")
pstats.Stats(pr).sort_stats("cumulative").print_stats(12)
