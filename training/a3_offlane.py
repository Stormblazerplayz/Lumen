"""Does the in-lane check still predict the car's real sideways position when it starts 2 to 5 m off the lane centre
(the game's take-over in lots and gas stations)? The simulated ego starts at an offset ed0 with a speed v0 and drives
the lane plan (target offset 0, speed v0); its real offset at each of the check's 36 steps is compared with the
envelope the check assumes (shield.lat_envelope with the same speed profile)."""
import numpy as np

import shield
import sim

print("ed0  v0 | real offset after 1, 2, 4, 6 s | envelope [lo, hi] at the same times | worst miss outside it (m)")
for ed0 in (2.0, 3.0, 4.0, 5.0):
    for v0 in (2.0, 5.0, 10.0):
        sc = sim.Scenarios(1, 5)
        sc.kind[:] = -1
        sc.jKind[:] = sim.J_NONE
        sc.kap[:] = 0.0
        sc.edgeL[:] = -9.0
        sc.edgeR[:] = 9.0
        w = sim.World(sc)
        w.ev[:] = v0
        w.ed[:] = ed0
        ts, eds, vs = [], [], []
        for k in range(int(round(shield.H * shield.DTS / sim.DT))):
            w.step(np.full(1, v0), np.zeros(1))
            ts.append((k + 1) * sim.DT)
            eds.append(float(w.ed[0]))
            vs.append(float(w.ev[0]))
        real = np.interp(shield.T, ts, eds)
        vk = np.interp(shield.T, ts, vs)[None, :]
        lo, hi = shield.lat_envelope(vk, np.array([ed0]), np.zeros(1), np.zeros(1), np.zeros(1))
        lo, hi = lo[0], hi[0]
        miss = np.maximum(lo - real, 0) + np.maximum(real - hi, 0)
        idx = [3, 7, 15, 23]
        print(f"{ed0:3.0f} {v0:4.0f} | {' '.join(f'{real[i]:+5.2f}' for i in idx)} | {' '.join(f'[{lo[i]:+.2f},{hi[i]:+.2f}]' for i in idx)} | {miss.max():.2f}")
