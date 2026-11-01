"""Which hard-scenario episodes can be avoided at all? (attempt 3, the parent's request, 2026-11-01)

An ideal driver approaches exactly like Lumen Drive, so the actor starts at the same moment in the same place. From the
first step the actor is moving, it reacts in one of two ways:
  brake        : full braking (target 0 at once; the controller's pedal limit is 6.5 m/s2), staying in its lane
  brake+swerve : full braking and the best of six lateral targets within the drivable edges (the outer edges, +-1 m,
                 +-2 m)
An episode is avoidable when at least one reaction ends without a collision. Used by a2_hard.py to report each brain's
collision rate on the avoidable episodes only."""
import numpy as np

import brains
import sim

OFFSETS = ("left_edge", -2.0, -1.0, 1.0, 2.0, "right_edge")


def _run(cls, seed, n, offset=None):
    sc = sim.Scenarios(n, seed, mode=cls)
    w = sim.World(sc)
    lo = sc.edgeL + sim.EGO_W / 2 + 0.2
    hi = sc.edgeR - sim.EGO_W / 2 - 0.2
    if offset == "left_edge":
        tgt = lo
    elif offset == "right_edge":
        tgt = hi
    elif offset is None:
        tgt = None
    else:
        tgt = np.clip(np.full(n, float(offset)), lo, hi)
    for _ in range(sim.STEPS):
        vT, dT = brains.rules_v3(w)
        react = w.hz_t0 >= 0
        vT = np.where(react, 0.0, vT)
        if tgt is not None:
            dT = np.where(react, tgt, dT)
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
    return (w.m["collision"] > 0), (w.hz_t0 >= 0)


def avoidable(cls, seed, n):
    """per episode: (avoidable by braking, avoidable by braking and the best swerve, the actor started)"""
    coll_b, started = _run(cls, seed, n)
    ok_b = ~coll_b
    ok_s = ok_b.copy()
    for off in OFFSETS:
        c, _ = _run(cls, seed, n, off)
        ok_s |= ~c
    return ok_b, ok_s, started
