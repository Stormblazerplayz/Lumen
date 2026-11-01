import time, numpy as np
import brains2, shield, sim
import test_shield as T

N = 1024
sc = sim.Scenarios(N, 400_001, overtake=True)
w = sim.World(sc)
tot_shield = 0.0; tot_other = 0.0; rows_sum = 0; calls = 0
orig = shield.shield
def wrapped(*a, **k):
    global tot_shield, rows_sum, calls
    t0 = time.time(); r = orig(*a, **k); tot_shield += time.time() - t0
    rows_sum += len(a[3]); calls += 1
    return r
shield.shield = wrapped
t_all = time.time()
steps = 200
for step in range(steps):
    out = T.scripted_out(w)
    vT, dT = brains2.nn_actions2(out, w)
    w.prevT, w.prevD = vT, dT
    w.step(vT, dT)
tt = time.time() - t_all
print(f"{N} envs x {steps} steps: total {tt:.1f}s, shield {tot_shield:.1f}s ({calls} calls, {rows_sum / max(calls,1):.0f} rows/call); per step {tt / steps * 1000:.0f} ms")
import cProfile, pstats
pr = cProfile.Profile(); pr.enable()
for step in range(30):
    out = T.scripted_out(w)
    vT, dT = brains2.nn_actions2(out, w)
    w.prevT, w.prevD = vT, dT
    w.step(vT, dT)
pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(14)
