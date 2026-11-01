"""A hand-written pass rule (human-written, not a model): where to put the car to get past whatever blocks its lane.

Used three ways in attempt 2, always named as such in the report:
  * the offset teacher of the behaviour-cloning stage (the shield decides whether a request is carried out);
  * the control arm "Drive + a hand-written pass rule" (does the learning add anything beyond rule + shield?);
  * the scripted requester in the shield tests.
It looks at the true state of the simulator (the net sees an observation of it).
"""
import numpy as np

import sim

TRIGGER = 55.0          # start looking for a way round when the blocking vehicle is closer than this (m)
SLOW = 7.5              # a vehicle in front slower than this is something to pass (m/s)
CLEAR = 2.55            # lateral distance between our centre and the blocker's centre that clears both boxes with 0.6 m to spare


def pass_target(w, trigger=TRIGGER, slow=SLOW):
    """(E,) lateral offset to ask for (0 when nothing blocks the lane)"""
    sc = w.sc
    kind = sc.kind
    rs = w.agS - w.es[:, None]
    blocks = ((kind == sim.PARKED) | ((kind == sim.LEAD) & (w.agV < slow))) & (np.abs(w.agD) < 2.0) & (rs > -4.0) & (rs < trigger)
    any_b = blocks.any(axis=1)
    c_left = np.where(blocks, w.agD - CLEAR, 1e9).min(axis=1)             # to pass on the left: left of the leftmost blocker's left edge
    c_right = np.where(blocks, w.agD + CLEAR, -1e9).max(axis=1)
    left_ok = sc.twoWay | sc.lanesLeftSame
    right_ok = sc.lanesRight > 0
    lo = sc.edgeL + sim.EGO_W / 2 + 0.2
    hi = sc.edgeR - sim.EGO_W / 2 - 0.2
    cl = np.maximum(c_left, lo)
    cr = np.minimum(c_right, hi)
    # prefer a same-direction lane on the right when it costs less than a metre more than the opposing lane
    use_right = right_ok & (~left_ok | (np.abs(cr) < np.abs(cl) + 1.0))
    tgt = np.where(use_right, cr, np.where(left_ok, cl, 0.0))
    return np.where(any_b, tgt, 0.0)


def rule_out(w, free_logit=6.0):
    """network-style outputs (E, 2): free speed (the guards cap it), and the pass offset"""
    t = pass_target(w)
    out = np.zeros((w.E, 2))
    out[:, 0] = free_logit
    out[:, 1] = np.arctanh(np.clip(t / 4.0, -0.99, 0.99))
    return out
