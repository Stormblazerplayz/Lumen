"""Attempt 2 (2026-11-01): the NN arm's action pipeline.

nn_actions2(out, w): network outputs -> (target speed, offset).
  * the speed the network asks for is never above the posted limit, and is capped by the SAME guards Lumen Drive uses (curves, lights with the committed
    yellow and Drive's own 2.2 m/s2 "room to stop" test, stop signs, junction discipline, following a moving car or
    stopping behind a stopped one in the lane the plan occupies): `guards2`, which is brains.guards with Drive's
    yellow braking limit and an optional planned offset;
  * an offset within +-0.45 m of the lane centre is plain lane keeping: nothing else is checked;
  * a bigger offset, or a car already out of its lane, is a lateral manoeuvre: the predicted-collision shield
    (shield.py) decides what the car may do (see its docstring).
"""
import numpy as np

import brains
import caution
import shield
import sim
from brains import (A_DEC, A_HARD, FRONT, IDM_A, IDM_B, IDM_S0, IDM_T, LANE_HALF, LINE_GAP, STOP_GAP, V_MAX,
                    _junction_discipline)

YELLOW_DECEL = 2.2          # Drive's own (guard.lua): the NN arm of attempt 2 uses the same "room to stop" test


def guards2(w, off=None, update_latch=True, lead=True, lead_onc=True, with_stop=False, yellow_decel=None):
    """brains.guards with Drive's yellow braking limit; `off` (E,) is the lateral position the car plans to occupy
    (default: where it is), which decides which vehicle counts as "in my lane". Returns the speed cap (and, with
    with_stop, the distance to the point it must stop at: 1e9 when there is none)."""
    sc, E = w.sc, w.E
    s, v = w.es, w.ev
    ahead = np.arange(0, 120, 3.0)
    kap = np.abs(w.kap_at(s[:, None] + ahead[None, :]))
    vcurve = np.sqrt(sim.A_LAT / np.maximum(kap, 1e-4))
    vsafe = np.minimum(np.sqrt(vcurve ** 2 + 2 * A_DEC * ahead[None, :]).min(axis=1), V_MAX)
    stopS = np.full(E, 1e9)
    sig = sim.signal_state(w.t, sc)
    for j in range(sim.J):
        line = sc.jS[:, j] - 6.0
        lineS = line - FRONT - LINE_GAP
        before = (s + FRONT) < line + 0.5
        room = (lineS - s) > v * v / (2 * (YELLOW_DECEL if yellow_decel is None else yellow_decel)) + 1
        red = before & (sig[:, j] == sim.S_RED)
        yellow_now = before & (sig[:, j] == sim.S_YELLOW) & room
        if update_latch:
            w.latchStop[:, j] = np.where(sig[:, j] == sim.S_GREEN, False, w.latchStop[:, j] | yellow_now)
            w.latchStop[:, j] &= before
            yellow = yellow_now | (w.latchStop[:, j] & (sig[:, j] == sim.S_YELLOW))
        else:
            yellow = yellow_now | (w.latchStop[:, j] & before & (sig[:, j] == sim.S_YELLOW))
        stopsign = before & (sig[:, j] == sim.S_STOPSIGN)
        done = (w.stopped_at[:, j] >= 0) & (w.t - w.stopped_at[:, j] >= 1.6)
        hold = red | yellow | (stopsign & ~done)
        stopS = np.where(hold & (lineS < stopS), lineS, stopS)
    kind = sc.kind
    active = kind >= 0
    cross = kind == sim.CROSS
    rs = w.agS - s[:, None]
    base = w.ed if off is None else off
    rd = w.agD - base[:, None]
    along = np.where(kind == sim.ONCOMING, -w.agV, np.where(cross, 0.0, w.agV))
    inlane = active & ~cross & (np.abs(rd) < LANE_HALF + np.maximum(0, sc.W / 2 - 0.95)) & (rs > 1) & (rs < 150)
    inlane &= (kind != sim.ONCOMING) | ((np.abs(rd) < 1.0) & lead_onc)
    if lead:
        gap = np.where(inlane, rs - sc.L / 2 - FRONT, 1e9)
    else:
        gap = np.full((E, sim.K), 1e9)
    li = np.argmin(gap, axis=1)
    lg = gap[np.arange(E), li]
    lv = np.maximum(0, along[np.arange(E), li])
    has = lg < 1e8
    stoppedLead = has & (lv < 0.5)
    stopS = np.where(stoppedLead & (s + lg - STOP_GAP < stopS), s + lg - STOP_GAP, stopS)
    v0 = np.maximum(np.minimum(vsafe, 1.05 * w.vlim_at(s)), 1)
    sstar = IDM_S0 + np.maximum(0, v * IDM_T + v * (v - lv) / (2 * np.sqrt(IDM_A * IDM_B)))
    acc = IDM_A * (1 - (v / v0) ** 4 - (sstar / np.maximum(lg, 0.5)) ** 2)
    idmCap = np.where(has & ~stoppedLead, np.maximum(0, v + np.clip(acc, -A_HARD, IDM_A) * 0.6), 1e9)
    stopS = _junction_discipline(w, stopS)
    lead_d = 1.0 + 0.6 * v
    vcap = np.minimum(vsafe, idmCap)
    vcap = np.minimum(vcap, np.sqrt(2 * A_DEC * np.maximum(0, stopS - s - lead_d)))
    vcap = np.where(stopS - s < 0.3, 0.0, vcap)
    if with_stop:
        return vcap, np.where(stopS < 1e8, stopS - s, 1e9)
    return vcap


class RowView:
    """Some environments of a World, as far as guards2 needs them (the caps the shield asks for are computed for its
    rows only, not for the whole batch)."""
    kap_at = sim.World.kap_at
    vlim_at = sim.World.vlim_at

    class _Sc:
        pass

    def __init__(self, w, rows):
        sc = self._Sc()
        for name in ("kap", "vlim", "jS", "jKind", "jCycle", "jGreen", "jPhase", "kind", "L", "W", "side", "junc", "edgeL"):
            setattr(sc, name, getattr(w.sc, name)[rows])
        self.sc = sc
        self.E = len(rows)
        self.t = w.t
        for name in ("es", "ev", "ed", "agS", "agD", "agV", "agAcc", "latchStop", "stopped_at"):
            setattr(self, name, getattr(w, name)[rows])


LAST = {"lateral": 0, "mode": np.zeros(8, dtype=int), "slowed": 0}
INLANE_SHIELD = True        # the in-lane check (shield.shield_inlane): Drive's 4 s prediction, as part of the shield
CAUTION = "flat"            # attempt 3: the parked-car caution rule (caution.py) on the network's speed ("stopping", None: other variants)


def nn_actions2(out, w, use_shield=True, vfree_share=1.1):
    E = w.E
    share = 1 / (1 + np.exp(-out[:, 0]))
    vlim = w.vlim_at(w.es)
    # never above the posted limit: the output's range reaches 1.1 x the limit (attempt 1's layout), and a "better"
    # that comes from speeding is not better (2026-11-01)
    vT_net = np.minimum(np.minimum(share * vfree_share * vlim, vlim), V_MAX)
    if CAUTION:
        vT_net = np.minimum(vT_net, caution.cap(w, CAUTION))
    sc = w.sc
    lo = sc.edgeL + sim.EGO_W / 2 + 0.2
    hi = sc.edgeR - sim.EGO_W / 2 - 0.2
    dT_c = np.clip(np.tanh(out[:, 1]) * 4.0, lo, hi)
    # commitment (2026-11-01): once the shield has accepted a pass, a weaker request to the same side keeps the accepted
    # offset (a network that wavers between -2.7 and -1.5 m otherwise aborts every pass it starts and stops behind)
    commit = getattr(w, "a2_commit", None)
    if commit is None:
        commit = w.a2_commit = np.zeros(E)
    keep = (commit != 0) & (np.sign(dT_c) == np.sign(commit)) & (np.abs(dT_c) < np.abs(commit)) & (np.abs(dT_c) > shield.OUT_LANE)
    dT_c = np.where(keep, commit, dT_c)
    v_lane, d_lane = guards2(w, with_stop=True)          # the guards at the car's own lateral position (updates the latch)
    vT = np.minimum(vT_net, v_lane)
    lat_req = (np.abs(dT_c - w.ed) > shield.OUT_LANE) | (np.abs(w.ed) > shield.OUT_LANE) | (np.abs(dT_c) > shield.OUT_LANE)
    lat_req &= ~w.done
    dT = np.clip(dT_c, -shield.OUT_LANE, shield.OUT_LANE)
    if not use_shield:
        # ablation: the network's offset goes straight to the controller, the lane guards still cap the speed
        dT = dT_c
        return vT, dT
    rows = np.where(lat_req)[0]
    modes = np.zeros(8, dtype=int)
    if INLANE_SHIELD:
        keep = np.where(~lat_req & ~w.done)[0]
        vI, mI = shield.shield_inlane(w, vT, d_lane, keep)
        vT[keep] = vI
        modes[6] = int((mI == 6).sum())
        modes[7] = int((mI == 7).sum())
        LAST["slowed"] = int((vI < vT_net[keep] - 1e-9).sum())
    if len(rows):
        def cap_fn(r, offs):
            return guards2(RowView(w, r), off=np.asarray(offs, dtype=float), update_latch=False, lead_onc=False, with_stop=True)
        vS, dS, mode = shield.shield(w, vT_net, dT_c, rows, cap_fn)
        vT[rows], dT[rows] = vS, dS
        commit[rows] = np.where((mode == 0) | (mode == 2), dS, 0.0)
        for m_ in range(6):
            modes[m_] = int((mode == m_).sum())
    commit[~lat_req] = 0.0
    LAST["lateral"] = len(rows)
    LAST["mode"] = modes
    return vT, dT
