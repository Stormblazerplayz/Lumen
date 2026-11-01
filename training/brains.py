"""The two brains in the simulator.

rules(world)   Lumen Drive's decisions, ported from lua/ge/extensions/lumen/drive.lua with the same
               constants: posted limit x 0.95 and curve speeds with braking ahead, red and yellow
               lights, stop signs (full stop, 1.6 s, then go), the intelligent driver model behind a
               moving car, a stop behind a stopped one, and a 4 s prediction of everyone else that
               yields at the first predicted overlap. It always keeps to its lane (offset 0).
MLP            Lumen NN: observation (sim.OBS_DIM) -> 64 -> 64 -> 2, tanh. Output 1 is the target
               speed as a share of 1.1 x the posted limit, output 2 the sideways offset (+-4 m).
               Many parameter sets are evaluated at once: weights carry a leading member axis.
"""
import numpy as np

import sim

A_DEC, A_HARD, LIMIT_SHARE, V_MAX = 2.2, 6.0, 0.95, 25.0
FRONT, LINE_GAP, STOP_GAP = 2.6, 1.2, 2.5
IDM_T, IDM_S0, IDM_A, IDM_B = 1.5, 2.5, 1.6, 2.2
LANE_HALF = 1.9


def _time_to_cover(v, d, a=1.6):
    """seconds to cover d metres from speed v while accelerating at a"""
    return (-v + np.sqrt(v * v + 2 * a * np.maximum(d, 0.0))) / a


def _junction_discipline(w, stopS):
    """Box discipline and gap acceptance (rules_v3). Returns the stop point with the junction rules applied."""
    sc, E = w.sc, w.E
    s, v = w.es, w.ev
    cross = sc.kind == sim.CROSS
    a = np.clip(w.agAcc, -7.0, 2.0)
    vc = np.maximum(w.agV, 0.0)
    dd = np.abs(w.agD - w.ed[:, None])
    zone = (sim.EGO_W + sc.L) / 2 + 0.4                     # half-extent in d: a cross car's length lies along d
    toward = sc.side * (w.ed[:, None] - w.agD) > -zone
    dz = np.maximum(dd - zone, 0.0)
    stops_short = (a < -1.0) & (vc ** 2 / (2 * np.maximum(-a, 1e-3)) < dz - 1.0)
    veff = np.maximum(vc + np.maximum(a, 0.0) * 1.0, 0.5)
    t_in = dz / veff
    t_out = t_in + 2 * zone / veff
    sz = (sim.EGO_L + sc.W) / 2 + 0.4                       # our s-extent around a cross car
    d_in = np.maximum(w.agS - sz - s[:, None], 0.0)
    d_out = np.maximum(w.agS + sz - s[:, None], 0.0)
    te_in = _time_to_cover(v[:, None], d_in)
    te_out = _time_to_cover(v[:, None], d_out)
    for j in range(sim.J):
        exists = sc.jKind[:, j] != sim.J_NONE
        line = sc.jS[:, j] - 6.0
        lineS = line - FRONT - LINE_GAP
        before = (s + FRONT) < line - 0.3
        in_box_stop = (stopS > lineS + 0.2) & (stopS < sc.jS[:, j] + 7.5) & (stopS < 1e8)
        stopS = np.where(exists & before & in_box_stop, lineS, stopS)
        mine = cross & (sc.junc == j)
        threat = mine & toward & ~stops_short & ((vc > 0.4) | (a > 0.5))
        conflict = threat & (t_in - GAP_MARGIN < te_out) & (te_in - GAP_MARGIN < t_out) & (d_out > 0)
        hold = conflict.any(axis=1) & exists & before
        stopS = np.where(hold & (lineS < stopS), lineS, stopS)
    return stopS


def rules(w, latch=False, junction=False, share=None):
    """Lumen Drive's decisions (drive.lua plan()). latch=True is the same brain but committed to a yellow
    decision (once it chose to stop it does not flip to "go" as the light turns red): the stronger rule
    baseline that the game does not have yet."""
    sc, E = w.sc, w.E
    s, v = w.es, w.ev
    # free speed: the limit, and curves within braking reach
    ahead = np.arange(0, 120, 3.0)
    kap = np.abs(w.kap_at(s[:, None] + ahead[None, :]))
    vcurve = np.sqrt(sim.A_LAT / np.maximum(kap, 1e-4))
    vfree = np.minimum(w.vlim_at(s) * (LIMIT_SHARE if share is None else share), V_MAX)
    vfree = np.minimum(vfree, np.sqrt(vcurve ** 2 + 2 * A_DEC * ahead[None, :]).min(axis=1))
    stopS = np.full(E, 1e9)
    # lights and stop signs
    sig = sim.signal_state(w.t, sc)
    for j in range(sim.J):
        line = sc.jS[:, j] - 6.0
        lineS = line - FRONT - LINE_GAP
        before = (s + FRONT) < line + 0.5
        room = (lineS - s) > v * v / (2 * A_DEC) + 1
        red = before & (sig[:, j] == sim.S_RED)
        yellow_now = before & (sig[:, j] == sim.S_YELLOW) & room
        if latch:
            w.latchStop[:, j] = np.where(sig[:, j] == sim.S_GREEN, False, w.latchStop[:, j] | yellow_now)
            w.latchStop[:, j] &= before
            yellow = yellow_now | (w.latchStop[:, j] & (sig[:, j] == sim.S_YELLOW))
        else:
            yellow = yellow_now
        stopsign = before & (sig[:, j] == sim.S_STOPSIGN)
        # a stop sign is done once it has stood still at the line for 1.6 s
        done = (w.stopped_at[:, j] >= 0) & (w.t - w.stopped_at[:, j] >= 1.6)
        hold = red | yellow | (stopsign & ~done)
        # drive.lua keeps a signal in force until the front bumper is 0.5 m past its line (the
        # `before` test above); a second "line not yet passed" test here released the hold 0.2 m
        # short of the line and the car rolled through at 1 m/s (found 2026-11-01 by tracing violations)
        stopS = np.where(hold & (lineS < stopS), lineS, stopS)
    # vehicles: the one ahead in our lane, and everyone's predicted path
    kind = sc.kind
    active = kind >= 0
    cross = kind == sim.CROSS
    rs = w.agS - s[:, None]
    rd = w.agD - w.ed[:, None]
    along = np.where(kind == sim.ONCOMING, -w.agV, np.where(cross, 0.0, w.agV))
    inlane = active & ~cross & (np.abs(rd) < LANE_HALF + np.maximum(0, sc.W / 2 - 0.95)) & (rs > 1) & (rs < 150)
    inlane &= (kind != sim.ONCOMING) | (np.abs(rd) < 1.0)
    gap = np.where(inlane, rs - sc.L / 2 - FRONT, 1e9)
    li = np.argmin(gap, axis=1)
    lg = gap[np.arange(E), li]
    lv = np.maximum(0, along[np.arange(E), li])
    has = lg < 1e8
    stoppedLead = has & (lv < 0.5)
    # lg is the gap from our front bumper to its rear already (the same fix as drive.lua, 2026-11-01)
    stopS = np.where(stoppedLead & (s + lg - STOP_GAP < stopS), s + lg - STOP_GAP, stopS)
    v0 = np.maximum(vfree, 1)
    sstar = IDM_S0 + np.maximum(0, v * IDM_T + v * (v - lv) / (2 * np.sqrt(IDM_A * IDM_B)))
    acc = IDM_A * (1 - (v / v0) ** 4 - (sstar / np.maximum(lg, 0.5)) ** 2)
    idmCap = np.where(has & ~stoppedLead, np.maximum(0, v + np.clip(acc, -A_HARD, IDM_A) * 0.6), 1e9)
    prof = np.minimum(vfree, idmCap)
    # our own timeline over the next 4 s (drive.lua egoTimeline: accelerate at most 1.6 m/s2 toward the
    # profile, braking-limited stops), against every other vehicle's predicted path (predict()):
    # constant acceleration for v t + a t^2 / 2, a car under 0.4 m/s stays put, a cross car waiting at its
    # line for a light or sign is predicted to stop there
    a_ag = np.clip(w.agAcc, -7.0, 2.0)
    held_d = np.maximum(0.0, np.abs(-sc.side * (np.abs(sc.edgeL[:, None]) + 2.0) - w.agD) - 1.0)
    conflictS = np.full(E, 1e9)
    vk, sk = v.copy(), s.copy()
    for k in range(1, 17):
        t = k * 0.25
        tgt = np.minimum(prof, np.sqrt(2 * A_DEC * np.maximum(0, stopS - sk)))
        vn = np.where(tgt > vk, np.minimum(tgt, vk + 1.6 * 0.25), tgt)
        sk = sk + 0.5 * (vk + vn) * 0.25
        vk = vn
        trav = w.agV * t + 0.5 * a_ag * t * t
        trav = np.where((a_ag < 0) & (w.agV + a_ag * t < 0), w.agV ** 2 / (-2 * np.minimum(a_ag, -1e-3)), trav)
        trav = np.where(np.abs(w.agV) < 0.4, 0.0, np.maximum(0.0, trav))
        trav = np.where(cross & w.agHeld, np.minimum(trav, held_d), trav)
        ps = np.where(kind == sim.ONCOMING, w.agS - trav, np.where(cross, w.agS, w.agS + trav))
        pd = np.where(cross, w.agD + sc.side * trav, w.agD)
        aL = np.where(cross, sc.W, sc.L)
        aW = np.where(cross, sc.L, sc.W)
        hit = active & ~inlane & (np.abs(ps - sk[:, None]) < (sim.EGO_L + aL) / 2 + 0.5 + 0.05 * v[:, None]) &             (np.abs(pd - w.ed[:, None]) < (sim.EGO_W + aW) / 2 + 0.4)
        any_hit = hit.any(axis=1) & (conflictS > 1e8)
        conflictS = np.where(any_hit, sk, conflictS)
    room = conflictS - s
    canStop = room > v * v / (2 * A_HARD) * 0.7
    stopS = np.where((conflictS < 1e8) & canStop & (conflictS - 2.5 < stopS), conflictS - 2.5, stopS)
    if junction:
        stopS = _junction_discipline(w, stopS)
    # the controller reads the plan's speed 1 + 0.6 v metres ahead (speedAhead in lumenRobotaxi.lua),
    # so braking starts that much earlier; the plan's own profile is the braking-limited one
    lead = 1.0 + 0.6 * v
    vT = np.minimum(prof, np.sqrt(2 * A_DEC * np.maximum(0, stopS - s - lead)))
    vT = np.where(stopS - s < 0.3, 0.0, vT)
    return vT, np.zeros(E)


def rules_latched(w):
    return rules(w, latch=True)


def rules_fast(w):
    """The free-flow teacher for the network (the guardrails supply stops, junctions and following): Lumen Drive's
    speed choice at the posted limit instead of 95% of it, with a committed yellow decision."""
    return rules(w, latch=True, junction=False, share=1.0)


def rules_v3(w):
    """The rule teacher for the network: Lumen Drive's rules, a committed yellow decision, and junction discipline."""
    return rules(w, latch=True, junction=True)


class MLP:
    """Batched two-hidden-layer tanh network; params is (members, n_params)."""

    def __init__(self, obs_dim=sim.OBS_DIM, hidden=64):
        self.shapes = [(obs_dim, hidden), (hidden,), (hidden, hidden), (hidden,), (hidden, 2), (2,)]
        self.sizes = [int(np.prod(sh)) for sh in self.shapes]
        self.n = sum(self.sizes)

    def init(self, rng):
        parts = []
        for sh in self.shapes:
            if len(sh) == 2:
                parts.append(rng.standard_normal(sh).ravel() * (1.0 / np.sqrt(sh[0])))
            else:
                parts.append(np.zeros(sh))
        p = np.concatenate(parts)
        return p

    def unpack(self, params):
        out, i = [], 0
        for sh, sz in zip(self.shapes, self.sizes):
            out.append(params[:, i:i + sz].reshape((params.shape[0],) + sh))
            i += sz
        return out

    def forward(self, params, obs):
        """params (P, n), obs (P, M, obs_dim) -> (P, M, 2)"""
        W1, b1, W2, b2, W3, b3 = self.unpack(params)
        h = np.tanh(np.matmul(obs, W1) + b1[:, None, :])
        h = np.tanh(np.matmul(h, W2) + b2[:, None, :])
        return np.matmul(h, W3) + b3[:, None, :]


SHIELD = True
GAP_MARGIN = 0.5            # seconds of margin in the junction gap test (nn.lua mirrors it)
YELLOW_DECEL = 3.5          # the guardrails stop for a yellow they can stop for at this braking (the rules use 2.2)
JUNCTION_SHIELD = True       # box discipline and gap acceptance are part of the guardrails (nn.lua mirrors them)


def guards(w, latch=True):
    """The guardrails Lumen NN shares with Lumen Drive (nn.lua implements the same): the curves ahead,
    red and yellow lights, stop signs, and cars stopped or slower in our lane. Returns the highest speed
    they allow. Lumen Drive's soft decisions (free speed, predicting cross traffic, lane position) are NOT
    in here: those are the network's job."""
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
        room = (lineS - s) > v * v / (2 * YELLOW_DECEL) + 1
        red = before & (sig[:, j] == sim.S_RED)
        yellow_now = before & (sig[:, j] == sim.S_YELLOW) & room
        if latch:
            w.latchStop[:, j] = np.where(sig[:, j] == sim.S_GREEN, False, w.latchStop[:, j] | yellow_now)
            w.latchStop[:, j] &= before
            yellow = yellow_now | (w.latchStop[:, j] & (sig[:, j] == sim.S_YELLOW))
        else:
            yellow = yellow_now
        stopsign = before & (sig[:, j] == sim.S_STOPSIGN)
        done = (w.stopped_at[:, j] >= 0) & (w.t - w.stopped_at[:, j] >= 1.6)
        hold = red | yellow | (stopsign & ~done)
        stopS = np.where(hold & (lineS < stopS), lineS, stopS)
    kind = sc.kind
    active = kind >= 0
    cross = kind == sim.CROSS
    rs = w.agS - s[:, None]
    rd = w.agD - w.ed[:, None]
    along = np.where(kind == sim.ONCOMING, -w.agV, np.where(cross, 0.0, w.agV))
    inlane = active & ~cross & (np.abs(rd) < LANE_HALF + np.maximum(0, sc.W / 2 - 0.95)) & (rs > 1) & (rs < 150)
    inlane &= (kind != sim.ONCOMING) | (np.abs(rd) < 1.0)
    gap = np.where(inlane, rs - sc.L / 2 - FRONT, 1e9)
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
    if JUNCTION_SHIELD:
        stopS = _junction_discipline(w, stopS)
    lead = 1.0 + 0.6 * v
    vcap = np.minimum(vsafe, idmCap)
    vcap = np.minimum(vcap, np.sqrt(2 * A_DEC * np.maximum(0, stopS - s - lead)))
    return np.where(stopS - s < 0.3, 0.0, vcap)


def nn_actions(out, w, shield=None):
    """network outputs (E, 2) -> target speed, offset (the guardrails above cap the speed)"""
    share = 1 / (1 + np.exp(-out[:, 0]))
    vT = np.minimum(share * 1.1 * w.vlim_at(w.es), V_MAX)
    if SHIELD if shield is None else shield:
        vT = np.minimum(vT, guards(w))
    dT = np.tanh(out[:, 1]) * 4.0
    return vT, dT


def reward(m):
    """one number per episode: progress, minus incidents and discomfort"""
    return (m["dist"] - 500 * m["collision"] - 60 * m["near"] - 300 * m["red"] - 200 * m["stopsign"]
            - 300 * m["offroad"] - 0.004 * m["jerk2"])
