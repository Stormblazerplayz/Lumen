"""Parked-car caution (attempt 3, the parent's proposal after the pull-out results): while a stationary car stands within
3.5 m of our lane centre without blocking the lane, ahead of us or alongside, cap our speed. Two variants:

  flat    : 65% of the posted limit while such a car's rear is less than 30 m ahead of our front bumper, and until we
            are past it
  stopping: the speed from which the car stops (5 m/s2, after 0.3 s) 2 m short of that car's rear, i.e.
            v <= -a t_r + sqrt((a t_r)^2 + 2 a (gap - 2)), at least 2 m/s; only while it is ahead (alongside, braking
            no longer helps)

Both apply to Drive and to the network alike (a rule for both brains, not an advantage for one)."""
import numpy as np

import sim
from brains import FRONT, LANE_HALF

RANGE = 30.0
SHARE = 0.65
A_C, T_R, MARGIN = 5.0, 0.3, 2.0
LAT = 3.5


def cap(w, variant):
    """(E,) speed cap, 1e9 where no stationary car is beside the lane ahead of us"""
    sc = w.sc
    kind = sc.kind
    rs = w.agS - w.es[:, None]
    still = (kind >= 0) & (kind != sim.CROSS) & (np.abs(w.agV) <= 1.0)
    # in OUR LANE (the lane's own frame, not where the car is now): that is an obstacle to stop behind or pass, not a car
    # at the curb (a car passing it would otherwise slow down beside it in the opposing lane)
    in_lane = np.abs(w.agD) < LANE_HALF + np.maximum(0, sc.W / 2 - 0.95)
    near_lane = still & ~in_lane & (np.abs(w.agD) < LAT)
    gap = rs - sc.L / 2 - FRONT                          # our front bumper to its rear
    not_passed = rs + sc.L / 2 > -sim.EGO_L / 2           # its front is still ahead of our rear
    rel = near_lane & not_passed & (gap < RANGE)
    if variant == "flat":
        c = np.where(rel.any(axis=1), SHARE * w.vlim_at(w.es), 1e9)
    else:
        ahead = rel & (gap > 0)                            # alongside, braking no longer helps: no cap
        g = np.where(ahead, np.maximum(gap - MARGIN, 0.0), 1e9).min(axis=1)
        c = np.where(g < 1e8, np.maximum(-A_C * T_R + np.sqrt((A_C * T_R) ** 2 + 2 * A_C * g), 2.0), 1e9)
    return c
