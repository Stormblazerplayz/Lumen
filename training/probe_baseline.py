"""Where does Lumen Drive (rules_v3) lose progress, and how fast does the simulated ego move sideways?
(2026-11-01, NN goal "actually better than Drive": measurements taken before any design decision.)
Seeds used here are development seeds (400_000+), never the held-out ones."""
import sys

import numpy as np

import brains
import sim

N = 512
sc = sim.Scenarios(N, 400_001)
w = sim.World(sc)
kind = sc.kind
parked_in_lane = ((kind == sim.PARKED) & (np.abs(sc.d) < 1.6)).any(axis=1)
has_lead = (kind == sim.LEAD).any(axis=1)
two_way = sc.twoWay
stop_why = {"parked": np.zeros(N), "lead": np.zeros(N), "signal": np.zeros(N), "other": np.zeros(N)}
for step in range(sim.STEPS):
    vT, dT = brains.rules_v3(w)
    alive = ~w.done
    w.prevT, w.prevD = vT, dT
    # classify why a standing car stands (v < 0.5) from the world state before stepping
    stand = alive & (w.ev < 0.5)
    rs = w.agS - w.es[:, None]
    inl = (kind >= 0) & (np.abs(w.agD - w.ed[:, None]) < 1.9) & (rs > 0) & (rs < 12)
    parked_ahead = (inl & (kind == sim.PARKED)).any(axis=1)
    lead_ahead = (inl & (kind != sim.PARKED) & (kind != sim.CROSS)).any(axis=1)
    sig = sim.signal_state(w.t, sc)
    line = sc.jS - 6.0
    near_sig = ((line - (w.es[:, None] + 2.6)) < 12) & ((line - (w.es[:, None] + 2.6)) > -2) & (sc.jKind != sim.J_NONE)
    sigw = (near_sig & ((sig == sim.S_RED) | (sig == sim.S_STOPSIGN) | (sig == sim.S_YELLOW))).any(axis=1)
    stop_why["parked"] += (stand & parked_ahead) * sim.DT
    stop_why["lead"] += (stand & ~parked_ahead & lead_ahead) * sim.DT
    stop_why["signal"] += (stand & ~parked_ahead & ~lead_ahead & sigw) * sim.DT
    stop_why["other"] += (stand & ~parked_ahead & ~lead_ahead & ~sigw) * sim.DT
    w.step(vT, dT)
    if w.done.all():
        break
m = w.summary()
dist = m["dist"]
print(f"rules_v3, {N} dev episodes: mean {dist.mean():.0f} m/episode, mean speed {m['dist'].sum() / m['time'].sum():.2f} m/s")
print(f"  episodes with a parked car in the lane: {parked_in_lane.mean() * 100:.0f}%  -> mean {dist[parked_in_lane].mean():.0f} m   without: {dist[~parked_in_lane].mean():.0f} m")
print(f"  episodes with a lead: {has_lead.mean() * 100:.0f}% -> {dist[has_lead].mean():.0f} m   without a lead and without parked: {dist[~has_lead & ~parked_in_lane].mean():.0f} m")
tot = sum(v.sum() for v in stop_why.values())
for k, v in stop_why.items():
    print(f"  standing time, {k:<7}: {v.sum() / N:6.1f} s per episode ({100 * v.sum() / tot:4.1f}% of standing time)")
print(f"  episodes that reached the end of the route: {(w.es > 690).mean() * 100:.0f}%")

# lateral response: step the offset target from 0 to -3.6 and to +1.5 at several speeds
print("\nsideways response of the simulated ego (time to cover 90% of the step, and the peak lateral speed):")
for v0 in (3.0, 6.0, 10.0, 15.0):
    for target in (-3.6, 1.5):
        sc2 = sim.Scenarios(8, 5)
        sc2.kind[:] = -1
        sc2.jKind[:] = sim.J_NONE
        sc2.kap[:] = 0.0
        w2 = sim.World(sc2)
        w2.ev[:] = v0
        t90, peak, prev = None, 0.0, 0.0
        for k in range(120):
            w2.step(np.full(8, v0), np.full(8, target))
            lv = (w2.ed[0] - prev) / sim.DT
            prev = w2.ed[0]
            peak = max(peak, abs(lv))
            if t90 is None and abs(w2.ed[0]) >= 0.9 * abs(target):
                t90 = (k + 1) * sim.DT
        print(f"  v={v0:4.1f} m/s, offset 0 -> {target:+.1f} m: 90% after {t90} s, peak lateral speed {peak:.2f} m/s")
