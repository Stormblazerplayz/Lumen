"""Smoke test and demonstration of the shield (attempt 2): a hand-scripted "pass whatever blocks my lane" policy,
with and without the shield, against Lumen Drive, on development seeds (never the held-out ones)."""
import sys
import time

import numpy as np

import brains
import brains2
import evaluate
import shield
import sim


def scripted_out(w, trigger=55.0):
    """network-style outputs (E, 2): free speed, and an offset into the free side lane when something blocks ours"""
    sc = w.sc
    kind = sc.kind
    rs = w.agS - w.es[:, None]
    rd = w.agD - w.ed[:, None]
    block = ((kind == sim.PARKED) | ((kind == sim.LEAD) & (w.agV < 7.5))) & (np.abs(w.agD) < 2.0) & (rs > -4) & (rs < trigger)
    obstructed = block.any(axis=1)
    side = np.where(sc.twoWay | sc.lanesLeftSame, -3.4, np.where(sc.lanesRight > 0, 3.4, 0.0))
    target = np.where(obstructed, side, 0.0)
    out = np.zeros((w.E, 2))
    out[:, 0] = 6.0
    out[:, 1] = np.arctanh(np.clip(target / 4.0, -0.99, 0.99))
    return out


def policy_scripted(use_shield):
    def pol(w):
        return brains2.nn_actions2(scripted_out(w), w, use_shield=use_shield)
    return pol


def policy_inlane(w):
    out = np.zeros((w.E, 2))
    out[:, 0] = 6.0
    return brains2.nn_actions2(out, w)


def run(policy, N, seed, overtake):
    sc = sim.Scenarios(N, seed, overtake=overtake)
    w = sim.World(sc)
    t0 = time.time()
    lat = 0
    modes = np.zeros(8, dtype=int)
    for _ in range(sim.STEPS):
        vT, dT = policy(w)
        w.prevT, w.prevD = vT, dT
        lat += brains2.LAST["lateral"]
        modes += brains2.LAST["mode"]
        w.step(vT, dT)
        if w.done.all():
            break
    print('   shield modes (net plan / lane / continue / least-bad / gated / stay):', modes.tolist())
    return w, time.time() - t0, lat


def show(name, w, N, secs, lat=None):
    m = w.summary()
    st = evaluate.stats(m, N)
    extra = f"  passedParked {w.passedParked.mean() * 100:4.0f}% passedLead {w.passedLead.mean() * 100:4.0f}% wronglane {m['wronglane'].sum() / N:5.2f}s/ep"
    print(f"{name:<26} inc/100km {st['per100km']:7.2f} (coll {st['collisions']:3d} red {st['red']:2d} stop {st['stopsign']:2d} off {st['offroad']:2d} near {st['near_s']:5.1f}s) "
          f"m/ep {st['m_per_ep']:6.0f} speed {st['speed']:5.2f}{extra}  [{secs:5.1f}s, lateral rows/step {0 if lat is None else lat / sim.STEPS:5.1f}]")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    for overtake in (False, True):
        print(f"--- {'overtake-heavy' if overtake else 'original'} distribution, {N} dev episodes (seed 400001) ---")
        w, secs, _ = run(brains.rules_v3, N, 400_001, overtake) if False else (None, 0, 0)
        # Drive
        sc = sim.Scenarios(N, 400_001, overtake=overtake)
        w = sim.World(sc)
        t0 = time.time()
        for _ in range(sim.STEPS):
            vT, dT = brains.rules_v3(w)
            w.prevT, w.prevD = vT, dT
            w.step(vT, dT)
            if w.done.all():
                break
        show("Drive (rules_v3)", w, N, time.time() - t0)
        w, secs, lat = run(policy_inlane, N, 400_001, overtake)
        show("in-lane, guards2", w, N, secs, lat)
        w, secs, lat = run(policy_scripted(True), N, 400_001, overtake)
        show("scripted pass + shield", w, N, secs, lat)
        w, secs, lat = run(policy_scripted(False), N, 400_001, overtake)
        show("scripted pass, NO shield", w, N, secs, lat)
