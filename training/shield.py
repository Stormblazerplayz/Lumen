"""Predicted-collision shield for lateral manoeuvres (attempt 2, 2026-11-01). Python reference; nn.lua mirrors it.

A network that may choose its own lateral offset can steer into trouble. The shield decides, every step, whether the
plan (target speed, lateral offset) the network asked for is one the car may execute. The first version held the
requested offset for the whole horizon and fell back to "stop where you are" when that failed; that parked the car in
the opposing lane (12 of 14 collisions on development seeds, 2026-11-01). A manoeuvre has an END, so the check is on
the whole manoeuvre:

  * the plan is "go to the offset, stay there until the vehicles that block my own lane are behind me, then come back
    to the middle of the lane" (a pass). The hold time is computed from the predicted positions of those vehicles and
    the car's own speed profile, not given by the network;
  * the car is rolled forward H x 0.25 s (9 s) under the plan: speed toward min(f x requested, the lane guards) at
    <= 1.6 m/s2, sideways with a transport delay and a lag, as an ENVELOPE of a fast and a slow model (a measured
    spread of the simulated controller, widened), so the checked box is as wide as the uncertainty;
  * every other vehicle is rolled forward from what the car measures (position, velocity, acceleration for 2 s,
    size): stopping at zero speed, a sideways drift for those moving sideways;
  * the plan is allowed at the highest speed of f = 1.0, 0.8, 0.6, 0.4, 0.25, 0.1 times the requested speed for which
    no predicted overlap occurs over the 9 s;
  * if the requested plan fails, the car continues the manoeuvre it is already in (same rule, current offset), then
    tries the lane plan (offset 0 now), then takes whichever candidate stays clear longest (never "stop where you are"
    unless that is the best of them);
  * a NEW lateral manoeuvre is not started within 50 m + 4 s x speed of a junction or stop line; one already under
    way is never aborted into a trap (it is checked like any other), and no target leaves the road.
"""
import numpy as np

import sim

H = 36
DTS = 0.25
T = np.arange(1, H + 1) * DTS
A_ACC, A_PLAN_DEC, STOP_DECEL = 1.6, 3.0, 2.2
M_S0, M_SV, M_D = 0.6, 0.06, 0.35           # margins around the boxes: along 0.6 m + 0.06 v, across 0.35 m
M_ONC_T = 0.6                                # attempt 3: an oncoming car's box is also longer by 0.6 s of closing speed
AGENT_ACC_T = 2.0                            # a vehicle keeps its measured acceleration for 2 s, then its speed
AGENT_LAT_T = 2.5                            # a car moving sideways is assumed to keep doing so for 2.5 s
EGO_L, EGO_W = sim.EGO_L, sim.EGO_W
SPEED_TRIES = (1.0, 0.8, 0.6, 0.4, 0.25, 0.1)
J_GATE_START, J_GATE_T = 50.0, 4.0           # no NEW lateral manoeuvre within 50 m + 4 s of speed of a junction or stop line
RANGE = 200.0                                # the car sees vehicles this far along its path (nn.lua scans map.objects)
OUT_LANE = 0.45                              # |offset| below this is "in the lane": no shield, the lane guards decide
AGENT_W, CROSS_LEN = 2.1, 5.0                # what the car assumes about a vehicle's other dimension
KS = 12                                      # the nearest vehicles that are considered (of at most 20)
CLUSTER_GAP = 22.0                           # vehicles of my lane closer together than this are passed as one group
RET_M0, RET_MV = 1.0, 0.15                   # the rear of the car must be this far (+ per m/s) past a vehicle before the return starts
LANE_BAND = 2.5                              # a vehicle whose centre is closer than this to my lane's middle blocks it
CLEAR_M = 0.6                                # a pass clears the group's outermost edge by this much (m)
CK_MAX = 24                                  # a pass must have the group behind it within 6 s, so the return is inside the 9 s checked
# sideways models: (delay s, time constant s, overshoot gain, drift from the current heading, reference speed m/s). The
# response runs on "equivalent time": real time while the car moves at the reference speed or faster, slower below it, and
# not at all when stopped (a car cannot steer standing still).
FAST_MODEL = (0.20, 0.33, 1.12, 0.8, 3.0)
SLOW_MODEL = (0.40, 0.55, 1.00, 0.0, 10.0)
MODELS = (FAST_MODEL, SLOW_MODEL)


def view(w, rows):
    """the KS nearest visible vehicles for the environments `rows`, as the car measures them (relative position along
    the path and across, absolute lateral position, velocity along and across, acceleration, extents)"""
    sc = w.sc
    kind = sc.kind[rows]
    active = kind >= 0
    cross = kind == sim.CROSS
    onc = kind == sim.ONCOMING
    rs = w.agS[rows] - w.es[rows, None]
    ad = w.agD[rows]
    rd = ad - w.ed[rows, None]
    vs = np.where(onc, -w.agV[rows], np.where(cross, 0.0, w.agV[rows]))
    vd = w.agVd[rows]
    acc = np.clip(w.agAcc[rows], -7.0, 2.0)
    aL = np.where(cross, sc.W[rows], sc.L[rows])               # extent along the path
    aW = np.where(cross, CROSS_LEN, AGENT_W)                   # extent across (not observed: assumed)
    vis = active & (np.abs(rs) < RANGE) & (np.abs(rd) < 30)
    # a crossing car that waits for its light or sign is predicted to stop at its line, as Drive predicts it
    # (drive.lua predict(): stopAt); the line is 2 m outside our road
    held = cross & w.agHeld[rows]
    held_d = np.maximum(0.0, np.abs(-sc.side[rows] * (np.abs(sc.edgeL[rows, None]) + 2.0) - ad) - 1.0)
    key = np.where(vis, np.abs(rs), 1e9)
    idx = np.argsort(key, axis=1)[:, :KS]

    def g(a):
        return np.take_along_axis(a, idx, axis=1)
    return {"rs": g(rs), "rd": g(rd), "ad": g(ad), "vs": g(vs), "vd": g(vd), "acc": g(acc), "aL": g(aL), "aW": g(aW),
            "cross": g(cross), "onc": g(onc), "vis": g(vis), "held": g(held), "held_d": g(held_d)}


def predict_agents(ag):
    """predicted displacement of each vehicle at each step: (n, KS, H) along and across"""
    t = T[None, None, :]
    vs, acc = ag["vs"], ag["acc"]
    sgn = np.sign(vs)
    a_s = acc * sgn
    tstop = np.where(a_s * vs < 0, -vs / np.where(a_s == 0, 1e-9, a_s), 1e9)
    t_lim = np.where(a_s < 0, 1e9, AGENT_ACC_T)[..., None]               # braking goes on until the stop; speeding up lasts 2 s
    tacc = np.minimum(np.minimum(t, t_lim), tstop[..., None])
    v_end = vs[..., None] + a_s[..., None] * tacc
    trav_s = vs[..., None] * tacc + 0.5 * a_s[..., None] * tacc * tacc + v_end * (t - tacc)
    trav_s = np.where(np.abs(vs)[..., None] < 0.4, 0.0, trav_s)
    vd, cross = ag["vd"], ag["cross"]
    sgd = np.sign(vd)
    a_d = np.where(cross, ag["acc"] * sgd, 0.0)
    tstop_d = np.where(a_d * vd < 0, -vd / np.where(a_d == 0, 1e-9, a_d), 1e9)
    ttd = np.where(cross[..., None], np.minimum(t, tstop_d[..., None]), np.minimum(t, AGENT_LAT_T))
    trav_d = vd[..., None] * ttd + 0.5 * a_d[..., None] * ttd * ttd
    trav_d = np.where(np.abs(vd)[..., None] < 0.3, 0.0, trav_d)
    lim = ag["held_d"][..., None]
    trav_d = np.where(ag["held"][..., None], np.clip(trav_d, -lim, lim), trav_d)
    return trav_s, trav_d


def speed_profile(v0, vT, dstop):
    """progress and speed of the ego at each step: toward vT at <= 1.6 m/s2 (braking at 3 m/s2), and never faster than
    lets it stop (2.2 m/s2) at the point the guards say it must stop at, dstop metres ahead (1e9 when none): (n, H) each"""
    n = len(v0)
    v = v0.copy()
    ds = np.zeros(n)
    prog = np.empty((n, H))
    vk = np.empty((n, H))
    for k in range(H):
        tgt = np.minimum(vT, np.sqrt(2 * STOP_DECEL * np.maximum(0.0, dstop - ds - (1.0 + 0.6 * v))))
        dv = np.where(tgt > v, np.minimum(A_ACC * DTS, tgt - v), -np.minimum(A_PLAN_DEC * DTS, v - tgt))
        vn = v + dv
        ds = ds + 0.5 * (v + vn) * DTS
        v = vn
        prog[:, k] = ds
        vk[:, k] = v
    return prog, vk


def lane_obstacles(ag):
    """vehicles that block my own lane and that I have not yet passed (their front edge is ahead of my rear)"""
    return (ag["vis"] & ~ag["onc"] & ~ag["cross"] & (np.abs(ag["ad"]) < LANE_BAND)
            & (ag["rs"] + ag["aL"] / 2 > -EGO_L / 2))


def cluster_mask(ag, lane_obs):
    """the group of lane-blocking vehicles to pass in one go: the nearest one and those within CLUSTER_GAP of the group's front"""
    rs, aL = ag["rs"], ag["aL"]
    n, m = rs.shape
    rear = np.where(lane_obs, rs - aL / 2, 1e9)
    front = rs + aL / 2
    order = np.argsort(rear, axis=1)
    inc = np.zeros((n, m), dtype=bool)
    ar = np.arange(n)
    front_c = np.full(n, -1e9)
    for j in range(m):
        i = order[:, j]
        r = rear[ar, i]
        ok = (r < 1e8) & ((front_c < -1e8) | (r - front_c < CLUSTER_GAP))
        inc[ar, i] = ok
        front_c = np.where(ok, np.maximum(front_c, front[ar, i]), front_c)
    return inc


def clear_step(prog, vk, ag, trav_s, cluster):
    """first step at which the rear of the ego is past the front of every vehicle of the group (+ margin): H if never"""
    front = ag["rs"][..., None] + trav_s + (ag["aL"] / 2)[..., None]
    rear_ego = prog[:, None, :] - EGO_L / 2 - (RET_M0 + RET_MV * vk)[:, None, :]
    ok = (rear_ego > front) | ~cluster[..., None]
    allclear = ok.all(axis=1)
    return np.where(allclear.any(axis=1), allclear.argmax(axis=1), H)


def lat_envelope(vk, ed, vd0, dA, T_h):
    """lowest and highest lateral position of the ego's centre at each step under: command dA now, 0 from T_h on.
    vk (n, H) is the ego's speed profile (the response stalls when it stops); T_h is in real seconds."""
    t = T[None, :]
    xs = []
    for delay, tau, gain, drift, vref in MODELS:
        rate = np.minimum(vk, vref) / vref
        # equivalent time at the end of each step (and at the return command, mapped through the same clock)
        teq = np.cumsum(rate * DTS, axis=1)
        steps_h = np.clip(T_h / DTS, 0.0, H)                        # fractional step index of the return command
        lo_i = np.minimum(steps_h.astype(int), H - 1)
        cum = np.concatenate([np.zeros((len(vk), 1)), teq], axis=1)  # cum[:, k] = equivalent time at the START of step k
        frac = steps_h - lo_i
        t_ret = cum[np.arange(len(vk)), lo_i] + frac * (cum[np.arange(len(vk)), lo_i + 1] - cum[np.arange(len(vk)), lo_i])
        t_ret = np.where(T_h >= 1e2, 1e3, t_ret)

        def S(tt, delay=delay, tau=tau):
            z = np.maximum(tt - delay, 0.0) / tau
            return 1.0 - (1.0 + z) * np.exp(-z)
        x = ed[:, None] + gain * ((dA - ed)[:, None] * S(teq) - dA[:, None] * S(teq - t_ret[:, None]))
        if drift:
            x = x + vd0[:, None] * drift * (1.0 - np.exp(-t / drift))
        xs.append(x)
    return np.minimum(xs[0], xs[1]), np.maximum(xs[0], xs[1])


def first_conflict(prog, vk, lo, hi, ed, ag, trav_s, trav_d):
    """first step at which the ego's (inflated) box overlaps a vehicle's predicted box: H when none does"""
    mid = (lo + hi) / 2 - ed[:, None]
    half = (hi - lo) / 2
    ps = ag["rs"][..., None] + trav_s - prog[:, None, :]
    pd = ag["rd"][..., None] + trav_d - mid[:, None, :]
    hs = (EGO_L + ag["aL"])[..., None] / 2 + M_S0 + M_SV * vk[:, None, :]
    hs = hs + np.where(ag["onc"][..., None], M_ONC_T * (np.abs(ag["vs"])[..., None] + vk[:, None, :]), 0.0)
    hd = (EGO_W + ag["aW"])[..., None] / 2 + M_D + half[:, None, :]
    hit = ag["vis"][..., None] & (np.abs(ps) < hs) & (np.abs(pd) < hd)
    anyhit = hit.any(axis=1)
    return np.where(anyhit.any(axis=1), anyhit.argmax(axis=1), H)


def shield(w, vT_plan, dT_plan, rows, cap_fn):
    """For the environments `rows` that asked for a lateral manoeuvre (or are out of their lane): returns
    (speed, offset, mode) with mode 0 = the requested plan, 1 = lane plan (back to the middle), 2 = continue the
    manoeuvre the car is in, 3 = none stayed clear: the one that stays clear longest, 4 = the lane plan because the junction gate refused a new
    manoeuvre, 5 = none stayed clear and the car is in its lane: it stays there.
    cap_fn(rows, offsets) gives the lane guards' speed cap for a car at those lateral positions."""
    sc = w.sc
    n = len(rows)
    ar = np.arange(n)
    v0 = w.ev[rows]
    ed = w.ed[rows]
    vd0 = w.ev[rows] * np.sin(w.epsi[rows])
    lo_e = sc.edgeL[rows] + EGO_W / 2 + 0.2
    hi_e = sc.edgeR[rows] - EGO_W / 2 - 0.2
    dA = np.clip(dT_plan[rows], lo_e, hi_e)
    vA = vT_plan[rows]
    ag = view(w, rows)
    trav_s, trav_d = predict_agents(ag)
    lane_obs = lane_obstacles(ag)
    cluster = cluster_mask(ag, lane_obs)
    needs = cluster.any(axis=1)
    # the SIZE of a pass is the shield's (2026-11-01): the network chooses whether to pass and on which side; a request
    # smaller than the offset that clears the group (its outermost edge + half the car + 0.6 m) is widened to it. A
    # regression head averages "pass" and "stay" into offsets too small to clear anything, and the car then stopped behind.
    edge_l = np.where(cluster, ag["ad"] - ag["aW"] / 2, 1e9).min(axis=1)
    edge_r = np.where(cluster, ag["ad"] + ag["aW"] / 2, -1e9).max(axis=1)
    want = needs & (np.abs(dA) > OUT_LANE)
    dA = np.where(want & (dA < 0), np.minimum(dA, edge_l - EGO_W / 2 - CLEAR_M), dA)
    dA = np.where(want & (dA > 0), np.maximum(dA, edge_r + EGO_W / 2 + CLEAR_M), dA)
    dA = np.clip(dA, lo_e, hi_e)
    # the junction gate
    line = sc.jS[rows] - 6.0
    dist = line - (w.es[rows, None] + 2.6)
    valid = (dist > -2.0) & (sc.jKind[rows] != sim.J_NONE)
    near = np.where(valid, dist, 1e9).min(axis=1)
    out_now = np.abs(ed) > OUT_LANE
    allowed = out_now | (near > J_GATE_START + J_GATE_T * v0)       # a car already out may always finish; a new pass needs room
    zeros = np.zeros(n)
    caps = {"net": cap_fn(rows, dA), "cont": cap_fn(rows, np.clip(ed, lo_e, hi_e)), "lane": cap_fn(rows, zeros)}
    cap_v = {k: c[0] for k, c in caps.items()}
    cap_s = {k: c[1] for k, c in caps.items()}

    best_first = np.full(n, -1)
    out_v = np.zeros(n)
    out_d = ed.copy()
    out_m = np.full(n, 3)
    resolved = np.zeros(n, dtype=bool)
    F = np.array(SPEED_TRIES)
    nf = len(F)

    ahead_in_lane = ~ag["onc"] & ~ag["cross"] & (np.abs(ag["ad"]) < LANE_BAND) & (ag["rs"] - ag["aL"] / 2 > EGO_L / 2)

    def ladder(sel, name, d_target, hold, mode):
        """all speed factors of one plan at once for the rows `sel` (the same result as trying them one by one:
        the highest factor that stays clear wins; otherwise the one that stays clear longest, the earliest on a tie)"""
        if not len(sel):
            return
        m = len(sel)
        rep = np.repeat(sel, nf)
        f = np.tile(F, m)
        vcmd = np.minimum(f * vA[rep], cap_v[name][rep])
        sub = {k: a[rep] for k, a in ag.items()}
        if not hold:
            # attempt 3: only a car still in its lane leaves the car ahead to the guards; one returning from a pass
            # checks it (2026-11-01 dev breakdown: collisions with the car ahead while returning)
            sub["vis"] = sub["vis"] & ~(ahead_in_lane[rep] & ~out_now[rep, None])
        ts, td = trav_s[rep], trav_d[rep]
        prog, vk = speed_profile(v0[rep], vcmd, cap_s[name][rep])
        if hold:
            ck = clear_step(prog, vk, sub, ts, cluster[rep])
            T_h = np.where(ck < H, T[np.minimum(ck, H - 1)], 1e3)
        else:
            T_h = np.zeros(len(rep))
        dt_rep = d_target[rep]
        lo, hi = lat_envelope(vk, ed[rep], vd0[rep], dt_rep, T_h)
        fc = first_conflict(prog, vk, lo, hi, ed[rep], sub, ts, td)
        if hold:
            # a pass that is not over well inside the horizon is not known to be safe (2026-11-01: crawling passes
            # were caught by oncoming cars that came into range after the check)
            fc = np.where(ck > CK_MAX, np.minimum(fc, H - 1), fc)
        V = vcmd.reshape(m, nf)
        FC = fc.reshape(m, nf)
        ok = FC >= H
        any_ok = ok.any(axis=1)
        j_ok = np.argmax(ok, axis=1)
        j_best = np.argmax(FC, axis=1)
        j = np.where(any_ok, j_ok, j_best)
        fc_j = FC[np.arange(m), j]
        better = fc_j > best_first[sel]
        upd = sel[better]
        best_first[upd] = fc_j[better]
        out_v[upd] = V[np.arange(m), j][better]
        out_d[upd] = (dt_rep.reshape(m, nf)[:, 0] if hold else np.zeros(m))[better]
        out_m[upd] = mode
        resolved[sel[any_ok]] = True

    # 1. the requested manoeuvre (a pass), if there is something in my lane to pass
    ladder(np.where(needs & allowed & ~resolved)[0], "net", dA, True, 0)
    # 2. the manoeuvre the car is already in: keep its offset until clear
    cont = np.clip(ed, lo_e, hi_e)
    ladder(np.where(needs & allowed & out_now & ~resolved)[0], "cont", cont, True, 2)
    # 3. back to the middle of the lane
    ladder(np.where(~resolved)[0], "lane", zeros, False, 1)
    # nothing stayed clear: a car still in its lane stays in it under the lane guards; one that is out takes the candidate
    # that stays clear longest
    stay = ~resolved & ~out_now
    out_v = np.where(stay, np.minimum(vA, cap_v["lane"]), out_v)
    out_d = np.where(stay, 0.0, out_d)
    mode = np.where(resolved, out_m, np.where(stay, 5, 3))
    mode = np.where(~allowed & (mode == 1), 4, mode)
    return out_v, out_d, mode


def shield_inlane(w, vT, dstop, rows, ahead_rel=False):
    """The check for a car that stays in its lane (no lateral manoeuvre): the same roll-out of the lane plan against the
    vehicles the lane guards do not handle (crossing traffic, cars moving sideways such as cut-ins, cars beside the lane;
    not the car ahead in the lane, which the guards follow or stop behind, not oncoming cars in their lane, nothing
    behind). Drive has a 4 s version of this inside its plan; the network's guards do not. Returns (speed, mode) with
    mode 6 = clear at f x the speed, 7 = nothing clear: the speed that stays clear longest."""
    n = len(rows)
    out_v = vT[rows].copy()
    mode = np.full(n, 6)
    if not n:
        return out_v, mode
    ag = view(w, rows)
    if ahead_rel:
        # the car the guards follow is the one in the lane the car is in NOW (their lead test, relative to its offset): a
        # car far off its lane centre (the game's take-over in a lot) must check one in the lane centre itself
        lane_half = 1.9 + np.maximum(0.0, ag["aW"] / 2 - 0.95)
        inlane_ahead = ~ag["onc"] & ~ag["cross"] & (np.abs(ag["rd"]) < lane_half) & (ag["rs"] - ag["aL"] / 2 > EGO_L / 2)
    else:
        inlane_ahead = ~ag["onc"] & ~ag["cross"] & (np.abs(ag["ad"]) < LANE_BAND) & (ag["rs"] - ag["aL"] / 2 > EGO_L / 2)
    behind = ag["rs"] + ag["aL"] / 2 < -EGO_L / 2
    relevant = ag["vis"] & ~ag["onc"] & ~inlane_ahead & ~behind & (ag["cross"] | (np.abs(ag["vd"]) >= 0.3) | (np.abs(ag["rd"]) < 4.0))
    ag["vis"] = relevant
    act = np.where(relevant.any(axis=1))[0]
    if not len(act):
        return out_v, mode
    sub = {k: a[act] for k, a in ag.items()}
    trav_s, trav_d = predict_agents(sub)
    v0 = w.ev[rows][act]
    ed = w.ed[rows][act]
    vd0 = v0 * np.sin(w.epsi[rows][act])
    vA = vT[rows][act]
    ds = dstop[rows][act]
    m = len(act)
    F = np.array(SPEED_TRIES)
    nf = len(F)
    rep = np.repeat(np.arange(m), nf)
    vcmd = np.tile(F, m) * vA[rep]
    prog, vk = speed_profile(v0[rep], vcmd, ds[rep])
    lo, hi = lat_envelope(vk, ed[rep], vd0[rep], np.zeros(len(rep)), np.zeros(len(rep)))
    fc = first_conflict(prog, vk, lo, hi, ed[rep], {k: a[rep] for k, a in sub.items()}, trav_s[rep], trav_d[rep])
    V = vcmd.reshape(m, nf)
    FC = fc.reshape(m, nf)
    ok = FC >= H
    done = ok.any(axis=1)
    j = np.where(done, np.argmax(ok, axis=1), np.argmax(FC, axis=1))
    best_v = V[np.arange(m), j]
    out_v[act] = best_v
    mode[act] = np.where(done, 6, 7)
    return out_v, mode
