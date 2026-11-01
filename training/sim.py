"""Lumen NN training simulator (Frontier Studios).

A fast, vectorised driving world for training and judging the Lumen's driving decisions. It is
built around what the Lumen sees in BeamNG (lua/ge/extensions/lumen/drive.lua and nn.lua): its
own lane ahead (curvature, speed limit, road edges), the next stop line (light or stop sign), and
the vehicles around it, all in road coordinates (s along the lane, d to the right of it). The
brain chooses two things ten times a second, exactly as in game: a target speed and a sideways
offset in the road. A copy of the in-game controller (pure pursuit on the rear axle, a speed loop
with the pedals' limits) turns those into motion, so what is learned here is decisions, not the
physics of this toy world.

Every episode is a fresh random 700 m stretch of road: curves, speed limits, two-way or one-way,
up to three junctions (traffic lights with real cycles, stop signs, or none), lead traffic driven
by IDM, oncoming cars, cross traffic (a few run their red), cars cutting in, cars that suddenly
reverse, and stopped cars blocking the lane. The same events happen in every evaluation run with
the same seed, so two brains can be compared on identical traffic.

Units: metres, seconds, m/s. Arrays are (E,) per environment or (E, K) per agent slot.
"""
import numpy as np

DT = 0.1
STEPS = 600                    # 60 s episodes
ROUTE_LEN = 700
GRID = 1.0                     # m between road samples
NG = int(ROUTE_LEN / GRID) + 120
K = 20                         # agent slots per environment
J = 3                          # junction slots
LANE = 3.6
EGO_L, EGO_W = 4.8, 1.9
WHEELBASE, REAR = 3.0, 1.18
A_LAT = 2.2

REVERSER_P = 0.0
BEHIND_DROP, SHOW_RD_STILL, SHOW_RD_MOVING = -1.0, 2.6, 4.5     # which vehicles the network is shown (mirrors nn.lua)
OBS_FILTER = True                                                 # False reproduces the observation before 2026-11-01 (evaluation of old networks only)
# agent kinds
LEAD, ONCOMING, CROSS, PARKED, CUTIN, REVERSER = range(6)
# hazard actors of the diagnostic classes pull_out / swerve / side_merge (Storm, 2026-11-01: a car pulling out, a car
# swerving, a car changing lanes right next to Lumen)
PULLOUT, SWERVER, MERGER = 6, 7, 8
# junction kinds
J_NONE, J_SIGNAL, J_STOP, J_OPEN = range(4)
# signal states seen by the ego
S_NONE, S_GREEN, S_YELLOW, S_RED, S_STOPSIGN = range(5)


class Scenarios:
    """Random roads and traffic for E environments, reproducible from one seed."""

    def __init__(self, E, seed, hard=False, overtake=False, mode=None):
        """hard=True is for training only: at least one junction, 3 to 5 cross cars at each and 22% of them
        running their red (5% in the held-out test), so the rare cross-traffic events carry enough signal."""
        rng = np.random.default_rng(seed)
        self.hard = hard
        self.overtake = overtake
        self.mode = mode
        self.hazard = False
        self.E = E
        self.rng = rng
        s = np.arange(NG) * GRID
        # --- road: curvature in segments (straights, sweepers, tight bends) ---
        kap = np.zeros((E, NG))
        vlim = np.zeros((E, NG))
        for e in range(E):
            pos = 0
            while pos < NG:
                L = int(rng.uniform(40, 160))
                kind = rng.random()
                if kind < 0.45:
                    k = 0.0
                elif kind < 0.85:
                    k = rng.choice([-1, 1]) / rng.uniform(80, 500)
                else:
                    k = rng.choice([-1, 1]) / rng.uniform(25, 80)
                kap[e, pos:pos + L] = k
                pos += L
            # smooth the curvature changes over about 15 m (roads have transitions)
            ker = np.ones(15) / 15
            kap[e] = np.convolve(np.pad(kap[e], 7, mode="edge"), ker, mode="valid")[:NG]
            pos = 0
            while pos < NG:
                L = int(rng.uniform(150, 400))
                vlim[e, pos:pos + L] = rng.choice([11.1, 13.9, 15.6, 19.4, 22.2, 25.0], p=[0.15, 0.3, 0.2, 0.15, 0.1, 0.1])
                pos += L
        self.kap, self.vlim = kap, vlim
        self.twoWay = rng.random(E) < 0.7                     # opposite lane on the left
        self.lanesRight = (rng.random(E) < 0.25).astype(float)  # a same-direction lane on the right (one-way multi-lane or a wide road)
        self.lanesLeftSame = (~self.twoWay) & (rng.random(E) < 0.6)  # one-way: a same-direction lane on the left
        self.shoulder = rng.uniform(0.0, 1.8, E) * (rng.random(E) < 0.6)
        # drivable edges relative to our lane centre (d positive = right)
        self.edgeR = LANE / 2 + self.lanesRight * LANE + self.shoulder
        self.edgeL = -(LANE / 2 + np.where(self.twoWay | self.lanesLeftSame, LANE, 0.0) + 0.3)
        # --- junctions ---
        self.jS = np.full((E, J), 1e9)
        self.jKind = np.zeros((E, J), dtype=int)
        self.jCycle = np.zeros((E, J))        # full cycle length
        self.jGreen = np.zeros((E, J))        # our green duration
        self.jPhase = np.zeros((E, J))        # cycle offset
        self.jRunner = np.zeros((E, J))       # probability a cross car runs its red
        for e in range(E):
            n = rng.integers(0, 2) if overtake else rng.integers(1 if hard else 0, J + 1)
            spots = np.sort(rng.uniform(90, ROUTE_LEN - 60, n))
            for j, sj in enumerate(spots):
                if j > 0 and sj - spots[j - 1] < 80:
                    continue
                self.jS[e, j] = sj
                kind = rng.choice([J_SIGNAL, J_STOP, J_OPEN], p=[0.55, 0.3, 0.15])
                self.jKind[e, j] = kind
                g = rng.uniform(12, 30)
                r = rng.uniform(12, 30)
                self.jGreen[e, j] = g
                self.jCycle[e, j] = g + 3.0 + r
                self.jPhase[e, j] = rng.uniform(0, g + 3.0 + r)
                self.jRunner[e, j] = 0.22 if hard else 0.05
        # --- agents ---
        self.kind = np.full((E, K), -1)
        self.s = np.zeros((E, K))
        self.d = np.zeros((E, K))
        self.v = np.zeros((E, K))        # speed along its own direction of travel
        self.vdes = np.zeros((E, K))
        self.L = np.full((E, K), 4.6)
        self.W = np.full((E, K), 1.9)
        self.trig = np.full((E, K), 1e9)  # cut-in gap / reverse time / cross start time
        self.junc = np.zeros((E, K), dtype=int)
        self.side = np.zeros((E, K))      # cross traffic: +1 from the right, -1 from the left
        self.runner = np.zeros((E, K), dtype=bool)
        for e in range(E):
            k = 0
            # lead traffic in our lane
            n = rng.integers(0, 3) if overtake else rng.integers(0, 4)
            pos = rng.uniform(25, 60)
            for _ in range(n):
                if k >= K:
                    break
                if overtake:
                    # slow movers (3 to 7 m/s): the traffic a policy can pass
                    self._add(e, k, LEAD, pos, 0.0, rng.uniform(3, 7), 0.9)
                else:
                    self._add(e, k, LEAD, pos, 0.0, rng.uniform(6, 14), rng.uniform(0.7, 1.05))
                # Reversing cars are handled by the shared back-away maneuver in the game (robotaxi.lua
                # updateEvasion), not by either brain, so the simulator leaves them out.
                if rng.random() < REVERSER_P:
                    self.kind[e, k] = REVERSER
                    self.trig[e, k] = rng.uniform(8, 40)
                pos += rng.uniform(30, 120)
                k += 1
            # oncoming cars
            if self.twoWay[e]:
                for _ in range(rng.integers(0, 6)):
                    if k >= K:
                        break
                    self._add(e, k, ONCOMING, rng.uniform(60, ROUTE_LEN + 300), -LANE, rng.uniform(8, 18), 1.0)
                    k += 1
            # stopped cars blocking the lane (a breakdown, a car parked across it)
            if overtake:
                if rng.random() < 0.65:
                    for _ in range(rng.integers(1, 3)):
                        if k >= K:
                            break
                        self._add(e, k, PARKED, rng.uniform(110, ROUTE_LEN - 80), rng.choice([0.0, 0.3, -0.3, 0.6, -0.5, 1.1]), 0.0, 0.0)
                        k += 1
            elif rng.random() < 0.25 and k < K:
                self._add(e, k, PARKED, rng.uniform(120, ROUTE_LEN - 80), rng.choice([0.0, 0.6, -0.4, 1.1]), 0.0, 0.0)
                k += 1
            # cars cutting in from the next lane
            if (self.lanesRight[e] > 0 or self.lanesLeftSame[e]) and rng.random() < 0.6:
                for _ in range(rng.integers(1, 3)):
                    if k >= K:
                        break
                    side = LANE if (self.lanesRight[e] > 0 and (not self.lanesLeftSame[e] or rng.random() < 0.5)) else -LANE
                    self._add(e, k, CUTIN, rng.uniform(-30, 120), side, rng.uniform(8, 16), 1.0)
                    self.trig[e, k] = rng.uniform(10, 25)         # gap ahead of us at which it moves over
                    k += 1
            # cross traffic at junctions
            for j in range(J):
                if self.jS[e, j] > 1e8 or self.jKind[e, j] == J_NONE:
                    continue
                for _ in range(rng.integers(3 if hard else 2, 6 if hard else 5)):
                    if k >= K:
                        break
                    side = rng.choice([-1.0, 1.0])
                    self._add(e, k, CROSS, self.jS[e, j] + side * (LANE / 2), 0.0, rng.uniform(7, 14), 1.0)
                    self.junc[e, k] = j
                    self.side[e, k] = side
                    self.trig[e, k] = rng.uniform(0, 50)        # when it reaches the junction area
                    self.runner[e, k] = rng.random() < self.jRunner[e, j]
                    # cross cars start off to the side of the road
                    self.d[e, k] = -side * rng.uniform(25, 60)
                    k += 1
        if mode is not None:
            self._diagnostic(mode)

    def _diagnostic(self, mode):
        """Diagnostic classes the game showed (2026-11-01, "Only the NN sat there"); the road is kept, traffic is replaced.
        start_stop : standing start, a stop sign 50 to 70 m ahead, cross traffic at it, oncoming cars, and 3 to 5 cars parked
                     beside the road or in lots (4 to 7.5 m from the lane centre) ahead and behind;
        start_crowd: standing start with a stopped queue right behind (7 and 13 m), a car parked just beside (2.4 to 3.6 m),
                     parked cars at the roadside, one or two slow leads ahead and oncoming traffic; no junction."""
        rng, E = self.rng, self.E
        self.kind[:] = -1
        self.jS[:] = 1e9
        self.jKind[:] = J_NONE
        if mode in ("pull_out", "swerve", "side_merge"):
            return self._hazards(mode)
        for e in range(E):
            k = 0
            if mode == "start_stop":
                sj = 13.6 + rng.uniform(50, 70)
                self.jS[e, 0] = sj
                self.jKind[e, 0] = J_STOP
                g = rng.uniform(12, 30)
                r = rng.uniform(12, 30)
                self.jGreen[e, 0], self.jCycle[e, 0], self.jPhase[e, 0] = g, g + 3.0 + r, rng.uniform(0, g + 3.0 + r)
                self.jRunner[e, 0] = 0.05
                for _ in range(rng.integers(2, 5)):
                    side = rng.choice([-1.0, 1.0])
                    self._add(e, k, CROSS, sj + side * (LANE / 2), 0.0, rng.uniform(7, 14), 1.0)
                    self.junc[e, k] = 0
                    self.side[e, k] = side
                    self.trig[e, k] = rng.uniform(0, 50)
                    self.runner[e, k] = rng.random() < 0.05
                    self.d[e, k] = -side * rng.uniform(25, 60)
                    k += 1
                for _ in range(rng.integers(3, 6)):
                    side = rng.choice([-1.0, 1.0]) if self.twoWay[e] else 1.0
                    self._add(e, k, PARKED, rng.uniform(-10, 130), side * rng.uniform(4.0, 7.5), 0.0, 0.0)
                    k += 1
            else:
                for off in (7.0, 13.0):
                    self._add(e, k, PARKED, 5.0 - off, 0.0, 0.0, 0.0)
                    k += 1
                self._add(e, k, PARKED, 5.0 + rng.uniform(-3, 8), rng.choice([2.4, 3.0, 3.6]), 0.0, 0.0)
                k += 1
                for _ in range(rng.integers(2, 5)):
                    side = rng.choice([-1.0, 1.0]) if self.twoWay[e] else 1.0
                    self._add(e, k, PARKED, rng.uniform(-10, 130), side * rng.uniform(4.0, 7.5), 0.0, 0.0)
                    k += 1
                pos = rng.uniform(35, 70)
                for _ in range(rng.integers(1, 3)):
                    self._add(e, k, LEAD, pos, 0.0, rng.uniform(4, 10), rng.uniform(0.8, 1.0))
                    k += 1
                    pos += rng.uniform(30, 80)
            if self.twoWay[e]:
                for _ in range(rng.integers(0, 4)):
                    self._add(e, k, ONCOMING, rng.uniform(60, ROUTE_LEN + 300), -LANE, rng.uniform(8, 18), 1.0)
                    k += 1

    def _hazards(self, mode):
        """One hazard actor per episode, on a road without junctions (fixed seeds; diagnostic, not pre-registered):
        pull_out  : a car parked at the curb (2.4 to 3.2 m right of our lane centre) pulls into our lane when we are 12 to
                    35 m behind it: it accelerates at 2 to 3 m/s2 from standstill while its offset goes to 0 over 1.5 to
                    3 s; limit 11.1 to 15.6 m/s, so we arrive at 8 to 15 m/s; oncoming traffic on two-way roads
        swerve    : a same-direction car in the lane to our left (3.5 m) keeps pace 0 to 15 m ahead of us (half the
                    episodes) or right beside us (-3 to +3 m), then swerves 1.0 to 1.8 m toward our lane and back over
                    1.5 to 2.5 s; a one-way road with a lane on the left
        side_merge: a car alongside (-4 to +6 m, 3.5 m to the left or right) at our speed +-2 m/s changes into our lane
                    over 2 to 3 s"""
        rng, E = self.rng, self.E
        self.hazard = True
        self.hz_slot = np.zeros(E, dtype=int)
        self.hz_a, self.hz_T, self.hz_amp, self.hz_dv = np.zeros(E), np.ones(E), np.zeros(E), np.zeros(E)
        self.hz_rs, self.hz_t, self.hz_vdes, self.hz_dlane = np.zeros(E), np.zeros(E), np.zeros(E), np.zeros(E)
        self.hz_variant = np.zeros(E, dtype=int)
        for e in range(E):
            k = 0
            if mode == "pull_out":
                self.vlim[e, :] = rng.choice([11.1, 13.9, 15.6])
                self.lanesRight[e] = 0.0
                self.shoulder[e] = 2.0
                d0 = rng.uniform(2.4, 3.2)
                self._add(e, k, PULLOUT, rng.uniform(90, 160), d0, 0.0, 0.0)
                self.L[e, k] = rng.uniform(4.2, 5.2)
                self.hz_slot[e] = k
                self.hz_rs[e] = rng.uniform(12, 35)
                self.hz_a[e] = rng.uniform(2, 3)
                self.hz_T[e] = rng.uniform(1.5, 3.0)
                self.hz_vdes[e] = rng.uniform(8, 12)
                k += 1
                if self.twoWay[e]:
                    for _ in range(rng.integers(0, 4)):
                        self._add(e, k, ONCOMING, rng.uniform(60, ROUTE_LEN + 300), -LANE, rng.uniform(8, 18), 1.0)
                        k += 1
            else:
                self.vlim[e, :] = rng.choice([13.9, 15.6, 19.4])
                self.twoWay[e] = False
                side = -1.0 if mode == "swerve" else rng.choice([-1.0, 1.0])
                self.lanesLeftSame[e] = side < 0
                self.lanesRight[e] = 1.0 if side > 0 else 0.0
                kind = SWERVER if mode == "swerve" else MERGER
                self._add(e, k, kind, 5.0 + rng.uniform(15, 35), side * 3.5, 0.0, 0.0)
                self.hz_slot[e] = k
                self.hz_dlane[e] = side * 3.5
                self.hz_t[e] = rng.uniform(12, 30)
                if mode == "swerve":
                    self.hz_variant[e] = int(rng.random() < 0.5)
                    self.hz_rs[e] = rng.uniform(-3, 3) if self.hz_variant[e] else rng.uniform(0, 15)
                    self.hz_amp[e] = rng.uniform(1.0, 1.8)
                    self.hz_T[e] = rng.uniform(1.5, 2.5)
                else:
                    self.hz_rs[e] = rng.uniform(-4, 6)
                    self.hz_dv[e] = rng.uniform(-2, 2)
                    self.hz_T[e] = rng.uniform(2, 3)
                k += 1
            self.edgeR[e] = LANE / 2 + self.lanesRight[e] * LANE + self.shoulder[e]
            self.edgeL[e] = -(LANE / 2 + (LANE if (self.twoWay[e] or self.lanesLeftSame[e]) else 0.0) + 0.3)

    def _add(self, e, k, kind, s, d, v, vdesFactor):
        self.kind[e, k] = kind
        self.s[e, k] = s
        self.d[e, k] = d
        self.v[e, k] = v
        self.vdes[e, k] = v if vdesFactor <= 0 else max(v, 0.1) * vdesFactor + 2
        self.L[e, k] = self.rng.uniform(4.2, 5.4) if kind != PARKED else self.rng.uniform(4.2, 7.5)
        self.W[e, k] = self.rng.uniform(1.8, 2.1)


def signal_state(t, sc, e_idx=None):
    """Our light at each junction at time t: (E, J) of S_GREEN / S_YELLOW / S_RED, or S_STOPSIGN."""
    ph = np.mod(t + sc.jPhase, np.maximum(sc.jCycle, 1e-6))
    st = np.where(ph < sc.jGreen, S_GREEN, np.where(ph < sc.jGreen + 3.0, S_YELLOW, S_RED))
    st = np.where(sc.jKind == J_SIGNAL, st, np.where(sc.jKind == J_STOP, S_STOPSIGN, S_NONE))
    return st


class World:
    """E environments stepping together. reset() from Scenarios, step(target_speed, offset)."""

    def __init__(self, sc):
        self.sc = sc
        E = sc.E
        self.E = E
        self.t = 0.0
        self.es = np.full(E, 5.0)            # ego s (reference node)
        self.ed = np.zeros(E)                # ego lateral offset from our lane centre
        self.epsi = np.zeros(E)              # heading relative to the lane
        self.ev = np.full(E, 0.0)
        self.eacc = np.zeros(E)
        self.steer = np.zeros(E)
        self.prevT = np.zeros(E)
        self.prevD = np.zeros(E)
        self.stopped_at = np.full((E, J), -1.0)   # time the ego came to a stop before each stop sign
        self.done = np.zeros(E, dtype=bool)
        self.agS, self.agD, self.agV = sc.s.copy(), sc.d.copy(), sc.v.copy()
        self.agMoved = np.zeros((E, K), dtype=bool)
        self.agAcc = np.zeros((E, K))                # smoothed acceleration of each agent (what drive.lua estimates)
        self.agHeld = np.zeros((E, K), dtype=bool)   # cross cars waiting at their line because it is not their turn
        self.agVd = np.zeros((E, K))                 # sideways speed of each agent (right positive): crossing cars and cut-ins
        self.agVdes = sc.vdes.copy()                 # desired speeds (a hazard actor that has merged gets one)
        self.hazard = getattr(sc, "hazard", False)
        if self.hazard:
            self.hz_t0 = np.full(E, -1.0)            # when the hazard started (-1: not yet)
            self.hz_v = np.zeros(E)
            self.hz_d0 = np.zeros(E)
            self.hz_inlane = np.zeros(E, dtype=bool)  # a pull-out or merger that has finished drives in our lane (IDM)
            self.hz_mingap = np.full(E, 1e9)         # after the start: smallest box-to-box distance to the actor, m
            self.hz_minttc = np.full(E, 1e9)         # ... smallest time to collision with it at constant velocities, s
            self.hz_brake_s = np.zeros(E)            # ... seconds braking harder than 4 m/s2
        self.passedParked = np.zeros(E, dtype=bool)  # attempt 2 statistics: the ego got past a parked car / a slow lead
        self.passedLead = np.zeros(E, dtype=bool)
        self.latchStop = np.zeros((E, J), dtype=bool)         # the rules brain: committed to stopping for a yellow
        self.cutting = np.zeros((E, K))
        self.reversing = np.zeros((E, K), dtype=bool)
        # metrics
        self.m = {k: np.zeros(E) for k in ("dist", "time", "collision", "near", "red", "stopsign", "offroad",
                                           "jerk2", "latmax", "blocked", "minttc_events", "speedsum", "risk", "boxstop", "wronglane", "dist20")}
        self.lastAcc = np.zeros(E)

    # --- road lookups on the grid ---
    def road(self, arr, s):
        i = np.clip((s / GRID).astype(int), 0, NG - 1)
        return arr[np.arange(arr.shape[0])[:, None] if arr.ndim == 2 and np.ndim(s) == 2 else np.arange(arr.shape[0]), i]

    def kap_at(self, s):
        i = np.clip((s / GRID).astype(int), 0, NG - 1)
        return np.take_along_axis(self.sc.kap, i if i.ndim == 2 else i[:, None], axis=1).reshape(i.shape)

    def vlim_at(self, s):
        i = np.clip((s / GRID).astype(int), 0, NG - 1)
        return np.take_along_axis(self.sc.vlim, i if i.ndim == 2 else i[:, None], axis=1).reshape(i.shape)

    # --- agents ---
    def _agents_step(self, sig):
        sc, E = self.sc, self.E
        kind = sc.kind
        s, d, v = self.agS, self.agD, self.agV
        t = self.t
        active = kind >= 0
        # leads, reversers and cut-ins (once in our lane) follow IDM on whoever is ahead in our lane
        lane_mask = active & ((kind == LEAD) | (kind == REVERSER) | ((kind == CUTIN) & (np.abs(d) < 1.0)))
        if self.hazard:
            lane_mask |= (np.arange(K)[None, :] == sc.hz_slot[:, None]) & self.hz_inlane[:, None]
        # the nearest thing ahead of each lane agent in our lane: other agents or the ego
        ahead_gap = np.full((E, K), 1e9)
        ahead_v = np.zeros((E, K))
        inlane_s = np.where(lane_mask | ((kind == PARKED) & (np.abs(d) < 1.6)), s, 1e9)
        for k in range(K):
            ds = inlane_s - s[:, k:k + 1]
            ds = np.where(ds > 0.1, ds - (sc.L + sc.L[:, k:k + 1]) / 2, 1e9)
            j = np.argmin(ds, axis=1)
            g = ds[np.arange(E), j]
            vv = np.where(kind[np.arange(E), j] == PARKED, 0.0, v[np.arange(E), j])
            ego_gap = np.where(self.es > s[:, k], self.es - s[:, k] - (EGO_L + sc.L[:, k]) / 2, 1e9)
            use_ego = ego_gap < g
            ahead_gap[:, k] = np.where(use_ego, ego_gap, g)
            ahead_v[:, k] = np.where(use_ego, self.ev, vv)
        # lights: lane agents stop at a red/yellow light ahead (and at stop signs, briefly)
        stop_gap = np.full((E, K), 1e9)
        for j in range(J):
            js = sc.jS[:, j:j + 1] - 6.0
            dist = js - s - sc.L / 2
            red = (sig[:, j:j + 1] == S_RED) | ((sig[:, j:j + 1] == S_YELLOW) & (dist > v * v / 6 + 2))
            stopsign = sig[:, j:j + 1] == S_STOPSIGN
            hold = (red | stopsign) & (dist > -0.5)
            stop_gap = np.where(hold & (dist < stop_gap), np.maximum(dist, 0.0), stop_gap)
        gap = np.minimum(ahead_gap, stop_gap)
        vahead = np.where(stop_gap < ahead_gap, 0.0, ahead_v)
        v0 = np.maximum(self.agVdes, 1.0)
        sstar = 2.0 + np.maximum(0, v * 1.3 + v * (v - vahead) / (2 * np.sqrt(1.5 * 2.5)))
        acc = 1.5 * (1 - (v / v0) ** 4 - (sstar / np.maximum(gap, 0.3)) ** 2)
        acc = np.clip(acc, -7.0, 1.5)
        # stop-sign agents go again after a moment (they creep through)
        acc = np.where((stop_gap < 0.8) & (v < 0.3) & (np.mod(t + s, 6.0) < 2.0) & (ahead_gap > 10), 0.8, acc)
        # reversers: at their trigger time they stop and back up for a few seconds
        rev_now = (kind == REVERSER) & (t >= sc.trig) & (t < sc.trig + 6.0)
        self.reversing = rev_now
        newv = np.where(lane_mask, np.maximum(0.0, v + acc * DT), v)
        newv = np.where(rev_now, np.maximum(-1.8, v - 2.5 * DT), newv)
        s_new = np.where(lane_mask | rev_now, s + newv * DT, s)
        # cut-ins: drive in their lane; when a gap ahead of the ego opens to their trigger, move over
        cut = active & (kind == CUTIN)
        gap_to_ego = s - self.es[:, None]
        # a cutter needs a real gap (4 m between bodies) and is not much slower than us: an unavoidable
        # side-swipe is not a test of anything (2026-11-01 sim audit)
        start_cut = cut & (np.abs(d) > 1.0) & (gap_to_ego - (EGO_L + sc.L) / 2 > 4.0) & (gap_to_ego < sc.trig) & (self.cutting == 0) & (v >= self.ev[:, None] - 1.5)
        self.cutting = np.where(start_cut, 1.0, self.cutting)
        lat_rate = np.sign(-d) * LANE / 2.5
        d_new = np.where(cut & (self.cutting > 0) & (np.abs(d) > 0.02), d + lat_rate * DT, d)
        d_new = np.where(cut & (np.abs(d_new) < 0.05), 0.0, d_new)
        s_new = np.where(cut & (np.abs(d) >= 1.0), s + v * DT, s_new)
        # oncoming: straight at their speed towards us
        onc = active & (kind == ONCOMING)
        s_new = np.where(onc, s - v * DT, s_new)
        # cross traffic: waits for its turn (our red = their green), then crosses; runners ignore lights
        cr = active & (kind == CROSS)
        jidx = sc.junc
        their_go = np.zeros((E, K), dtype=bool)
        for j in range(J):
            sj = sig[:, j:j + 1]
            here = jidx == j
            go = (sj == S_RED) | (sj == S_NONE) | ((sj == S_STOPSIGN) & (np.mod(t + 3 * j, 8.0) < 4.0))
            their_go = np.where(here, go, their_go)
        self.agHeld = cr & ~their_go
        go = cr & (t >= sc.trig) & (their_go | sc.runner)
        # distance from the edge of our road: they stop at their line (about 2 m outside our road)
        line = -sc.side * (np.abs(sc.edgeL[:, None]) + 2.0)
        before_line = (d - line) * sc.side < 0
        waiting = cr & (t >= sc.trig) & ~go & before_line & (np.abs(d - line) < 1.0)
        dv = np.where(go | ~before_line, sc.vdes, np.where(waiting, 0.0, sc.vdes))
        dv = np.where(cr & (t < sc.trig), 0.0, dv)
        # approach and stop at the line when it is not their turn
        approach = cr & (t >= sc.trig) & ~go & before_line
        dist_line = np.abs(line - d)
        dv = np.where(approach, np.minimum(sc.vdes, np.sqrt(2 * 3.0 * np.maximum(dist_line - 1.0, 0))), dv)
        # cross cars accelerate at 2.5 m/s2 and brake at 4 (2026-11-01: they used to jump from standstill to
        # 15 m/s in one step, which nobody could avoid); before their start time they are not there yet
        dvs = np.clip(dv, v - 4.0 * DT, v + 2.5 * DT)
        dvs = np.where(cr & (t < sc.trig), 0.0, dvs)
        d_new = np.where(cr, d + sc.side * dvs * DT, d_new)
        newv = np.where(cr, dvs, newv)
        self.agVd = np.where(cr, sc.side * dvs, np.where(cut & (self.cutting > 0) & (np.abs(d) > 0.02), lat_rate, 0.0))
        if self.hazard:
            s_new, d_new, newv = self._hazard_step(s, d, v, s_new, d_new, newv)
        raw_acc = np.where(active, (newv - v) / DT, 0.0)
        self.agAcc = self.agAcc + (raw_acc - self.agAcc) * min(1.0, 2 * DT)
        self.agS, self.agD, self.agV = s_new, d_new, newv

    def _hazard_step(self, s, d, v, s_new, d_new, newv):
        """the hazard actor of each environment (one slot): pull out, swerve, or merge (see Scenarios._hazards)"""
        sc, E, t = self.sc, self.E, self.t
        ar = np.arange(E)
        k = sc.hz_slot
        kind = sc.kind[ar, k]
        s0, d0, v0 = s[ar, k], d[ar, k], v[ar, k]
        rs = s0 - self.es
        ns, nd, nv, nvd = s0.copy(), d0.copy(), v0.copy(), np.zeros(E)
        handled = ~self.hz_inlane
        # pull-out: parked until we are hz_rs metres behind it
        po = kind == PULLOUT
        start = po & (self.hz_t0 < 0) & (rs > 0) & (rs <= sc.hz_rs)
        self.hz_d0 = np.where(start, d0, self.hz_d0)
        self.hz_t0 = np.where(start, t, self.hz_t0)
        act = po & (self.hz_t0 >= 0) & handled
        tau = np.maximum(t - self.hz_t0, 0.0)
        x = np.clip(tau / sc.hz_T, 0.0, 1.0)
        v_po = np.minimum(sc.hz_vdes, sc.hz_a * (tau + DT))
        nv = np.where(act, v_po, np.where(po & handled, 0.0, nv))
        ns = np.where(act, s0 + v_po * DT, ns)
        nd = np.where(act, self.hz_d0 * (1 - (3 * x * x - 2 * x ** 3)), nd)
        nvd = np.where(act & (x < 1), -self.hz_d0 * 6 * x * (1 - x) / sc.hz_T, nvd)
        fin_po = act & (x >= 1)
        # swerver and merger: keep pace with us at hz_rs until the event
        sm = (kind == SWERVER) | (kind == MERGER)
        pace = sm & (self.hz_t0 < 0)
        v_tgt = np.clip(self.ev + 0.5 * (sc.hz_rs - rs), 0.0, 30.0)
        v_p = np.maximum(0.0, np.clip(v_tgt, v0 - 2.5 * DT, v0 + 2.5 * DT))
        nv = np.where(pace, v_p, nv)
        ns = np.where(pace, s0 + v_p * DT, ns)
        nd = np.where(pace, sc.hz_dlane, nd)
        start = pace & (t >= sc.hz_t) & (np.abs(rs - sc.hz_rs) < 1.5) & (self.ev > 4.0)
        self.hz_v = np.where(start, np.where(kind == MERGER, self.ev + sc.hz_dv, v_p), self.hz_v)
        self.hz_t0 = np.where(start, t, self.hz_t0)
        ev_on = sm & (self.hz_t0 >= 0) & handled
        tau = np.maximum(t - self.hz_t0, 0.0)
        nv = np.where(ev_on, self.hz_v, nv)
        ns = np.where(ev_on, s0 + self.hz_v * DT, ns)
        sw = ev_on & (kind == SWERVER)
        toward = -np.sign(sc.hz_dlane)
        xs = np.clip(tau / sc.hz_T, 0.0, 1.0)
        nd = np.where(sw, sc.hz_dlane + toward * sc.hz_amp * np.sin(np.pi * xs), nd)
        nvd = np.where(sw & (xs < 1), toward * sc.hz_amp * np.pi / sc.hz_T * np.cos(np.pi * xs), nvd)
        mg = ev_on & (kind == MERGER)
        nd = np.where(mg, sc.hz_dlane * (1 - (3 * xs * xs - 2 * xs ** 3)), nd)
        nvd = np.where(mg & (xs < 1), -sc.hz_dlane * 6 * xs * (1 - xs) / sc.hz_T, nvd)
        fin_mg = mg & (xs >= 1)
        # finished pull-outs and merges drive on in our lane (IDM), at their own speed
        fin = fin_po | fin_mg
        self.agVdes[ar, k] = np.where(fin_po, sc.hz_vdes, np.where(fin_mg, self.hz_v, self.agVdes[ar, k]))
        nd = np.where(fin, 0.0, nd)
        self.hz_inlane |= fin
        s_new, d_new, newv = s_new.copy(), d_new.copy(), newv.copy()
        s_new[ar, k] = np.where(handled, ns, s_new[ar, k])
        d_new[ar, k] = np.where(handled, nd, d_new[ar, k])
        newv[ar, k] = np.where(handled, nv, newv[ar, k])
        self.agVd[ar, k] = np.where(handled, nvd, self.agVd[ar, k])
        return s_new, d_new, newv

    def _hazard_metrics(self, alive):
        sc, E = self.sc, self.E
        ar = np.arange(E)
        k = sc.hz_slot
        on = alive & (self.hz_t0 >= 0)
        ds = self.agS[ar, k] - self.es
        dd = self.agD[ar, k] - self.ed
        gs = np.abs(ds) - (EGO_L + sc.L[ar, k]) / 2
        gd = np.abs(dd) - (EGO_W + sc.W[ar, k]) / 2
        gap = np.sqrt(np.maximum(gs, 0) ** 2 + np.maximum(gd, 0) ** 2)
        self.hz_mingap = np.where(on, np.minimum(self.hz_mingap, gap), self.hz_mingap)
        # time to collision with the actor, both at their current velocities (its sideways speed included)
        us = self.agV[ar, k] - self.ev * np.cos(self.epsi)
        ud = self.agVd[ar, k] - self.ev * np.sin(self.epsi)
        hs = (EGO_L + sc.L[ar, k]) / 2
        hd = (EGO_W + sc.W[ar, k]) / 2

        def interval(p, u, h):
            safe = np.where(np.abs(u) < 1e-6, 1e-6, u)
            t1, t2 = (-p - h) / safe, (-p + h) / safe
            lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
            inside = np.abs(p) < h
            still = np.abs(u) < 1e-6
            lo = np.where(still, np.where(inside, -1e9, 1e9), lo)
            hi = np.where(still, np.where(inside, 1e9, -1e9), hi)
            return lo, hi
        ls, hs_ = interval(ds, us, hs)
        ld, hd_ = interval(dd, ud, hd)
        start = np.maximum(np.maximum(ls, ld), 0.0)
        end = np.minimum(hs_, hd_)
        ttc = np.where(start < end, start, 1e9)
        self.hz_minttc = np.where(on, np.minimum(self.hz_minttc, ttc), self.hz_minttc)
        self.hz_brake_s += (on & (self.eacc < -4.0)) * DT

    # --- the ego: a copy of the in-game controller following (target speed, offset) ---
    def _ego_step(self, vT, dT):
        sc = self.sc
        v = self.ev
        # the plan's forward limit shared by both brains (drive.lua and nn.lua): the speed target 1 + 0.6 v
        # metres ahead cannot exceed what 1.6 m/s2 reaches from the current speed
        vT = np.minimum(vT, np.sqrt(v * v + 2 * 1.6 * (1.0 + 0.6 * v)) + 0.5)
        # target offset within the drivable road
        dT = np.clip(dT, sc.edgeL + EGO_W / 2 + 0.2, sc.edgeR - EGO_W / 2 - 0.2)
        look = np.clip(2.5 + 0.5 * v + 0.025 * v * v, 3.5, 40)
        # pure pursuit from the rear axle onto the offset line, in road coordinates
        kap = self.kap_at(self.es)
        lat_err = dT - (self.ed - np.sin(self.epsi) * REAR)
        # the target point's direction relative to the lane, including the road's own curvature
        alpha = np.arctan2(lat_err + 0.5 * kap * look * look * np.sign(look), look) - self.epsi
        wheel = np.arctan(2 * np.sin(alpha) * WHEELBASE / look)
        lim = np.arctan(WHEELBASE * 6.0 / np.maximum(1, v * v))
        wheel = np.clip(wheel, -np.minimum(lim, 0.52), np.minimum(lim, 0.52))
        # steering rate (FILTER_AI caps it at about 4 input units a second)
        self.steer += np.clip(wheel - self.steer, -4 * 0.52 * DT, 4 * 0.52 * DT)
        # speed loop with the pedals' limits
        acc = np.clip(1.4 * (vT - v), -6.5, 2.4)
        acc = np.where((vT < 0.05) & (v < 0.4), -v / DT, acc)
        self.eacc = self.eacc + (acc - self.eacc) * 0.5
        v2 = np.maximum(0.0, v + self.eacc * DT)
        # kinematic bicycle in road coordinates
        yaw_rate = v2 * np.tan(self.steer) / WHEELBASE
        sdot = v2 * np.cos(self.epsi) / np.maximum(0.3, 1 - self.ed * kap)
        self.es = self.es + sdot * DT
        self.ed = self.ed + v2 * np.sin(self.epsi) * DT
        self.epsi = self.epsi + (yaw_rate - kap * sdot) * DT
        self.ev = v2
        return dT

    def step(self, vT, dT):
        sc = self.sc
        E = self.E
        alive = ~self.done
        sig = signal_state(self.t, sc)
        s0 = self.es.copy()
        dT = self._ego_step(np.where(alive, vT, 0.0), dT)
        self._agents_step(sig)
        self.t += DT
        m = self.m
        prog = np.where(alive, self.es - s0, 0.0)
        m["dist"] += prog
        m["dist20"] += np.where(self.t < 20.0, prog, 0.0)
        m["time"] += np.where(alive, DT, 0.0)
        m["speedsum"] += np.where(alive, self.ev, 0.0)
        jerk = (self.eacc - self.lastAcc) / DT
        self.lastAcc = self.eacc.copy()
        m["jerk2"] += np.where(alive, jerk * jerk * DT, 0.0)
        m["latmax"] = np.maximum(m["latmax"], np.where(alive, np.abs(self.ev ** 2 * np.tan(self.steer) / WHEELBASE), 0.0))
        # --- collisions (boxes in road coordinates) and near misses ---
        kind = sc.kind
        active = kind >= 0
        cross = kind == CROSS
        aL = np.where(cross, sc.W, sc.L)
        aW = np.where(cross, sc.L, sc.W)
        ds = np.abs(self.agS - self.es[:, None])
        dd = np.abs(self.agD - self.ed[:, None])
        hit = active & (ds < (EGO_L + aL) / 2) & (dd < (EGO_W + aW) / 2)
        near = active & (ds < (EGO_L + aL) / 2 + 0.6) & (dd < (EGO_W + aW) / 2 + 0.35) & ~hit & (self.ev[:, None] > 1.5)
        collided = hit.any(axis=1) & alive
        m["collision"] += collided
        m["near"] += (near.any(axis=1) & alive & ~collided) * DT        # seconds spent too close
        # off the road
        off = (self.ed + EGO_W / 2 > sc.edgeR + 0.25) | (self.ed - EGO_W / 2 < sc.edgeL - 0.25)
        m["offroad"] += off & alive
        # red lights and stop signs: the front bumper crossing the stop line
        front0 = s0 + 2.6
        front1 = self.es + 2.6
        for j in range(J):
            line = sc.jS[:, j] - 6.0
            crossing = (front0 < line) & (front1 >= line) & alive
            m["red"] += crossing & (sig[:, j] == S_RED)
            # stop signs: it must have stood still within 8 m of the line
            near_line = (line - front1 > -0.5) & (line - front1 < 8.0) & (self.ev < 0.3)
            # the FIRST time it stood still at the line (a later refresh would restart its wait)
            self.stopped_at[:, j] = np.where(near_line & (sig[:, j] == S_STOPSIGN) & (self.stopped_at[:, j] < 0), self.t, self.stopped_at[:, j])
            m["stopsign"] += crossing & (sig[:, j] == S_STOPSIGN) & (self.stopped_at[:, j] < 0)
        # blocked: standing still with no light and nothing driving ahead of it (a stopped car in the lane)
        m["blocked"] += (alive & (self.ev < 0.3)) * DT
        # attempt 2: passing statistics and time spent in the opposing lane on a two-way road
        self.passedParked |= ((kind == PARKED) & (self.es[:, None] > self.agS + 2.0)).any(axis=1) & alive
        self.passedLead |= ((kind == LEAD) & (self.es[:, None] > self.agS + 2.0)).any(axis=1) & alive
        m["wronglane"] += (alive & sc.twoWay & (self.ed < -1.2)) * DT
        # dense safety signals for training: time on a collision course, and time stopped inside a junction box
        m["risk"] += ((self._ttc() < 1.5) & alive) * DT
        inbox = np.zeros(E, dtype=bool)
        for j in range(J):
            inbox |= (self.es > sc.jS[:, j] - 6.0) & (self.es < sc.jS[:, j] + 5.0)
        m["boxstop"] += (inbox & (self.ev < 1.0) & alive) * DT
        if self.hazard:
            self._hazard_metrics(alive)
        self.done = self.done | collided | off | (self.es > ROUTE_LEN - 5)
        return prog, collided, off

    def _ttc(self):
        """Time to the first overlap of our box with each vehicle's, everything at its current velocity;
        the smallest over vehicles, in seconds (1e9 when none within reach). Boxes overlapping now give 0."""
        sc = self.sc
        kind = sc.kind
        active = kind >= 0
        cross = kind == CROSS
        ego_vs = self.ev * np.cos(self.epsi)
        ego_vd = self.ev * np.sin(self.epsi)
        vs = np.where(kind == ONCOMING, -self.agV, np.where(cross, 0.0, self.agV))
        vd = np.where(cross, sc.side * self.agV, 0.0)
        ps = self.agS - self.es[:, None]
        pd = self.agD - self.ed[:, None]
        us = vs - ego_vs[:, None]
        ud = vd - ego_vd[:, None]
        aL = np.where(cross, sc.W, sc.L)
        aW = np.where(cross, sc.L, sc.W)
        hs = (EGO_L + aL) / 2
        hd = (EGO_W + aW) / 2

        def interval(p, u, h):
            safe = np.where(np.abs(u) < 1e-6, 1e-6, u)
            t1 = (-p - h) / safe
            t2 = (-p + h) / safe
            lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
            inside = np.abs(p) < h
            still = np.abs(u) < 1e-6
            lo = np.where(still, np.where(inside, -1e9, 1e9), lo)
            hi = np.where(still, np.where(inside, 1e9, -1e9), hi)
            return lo, hi

        ls, hs_ = interval(ps, us, hs)
        ld, hd_ = interval(pd, ud, hd)
        start = np.maximum(np.maximum(ls, ld), 0.0)
        end = np.minimum(hs_, hd_)
        ttc = np.where(active & (start < end), start, 1e9)
        return ttc.min(axis=1)

    # --- what the brain sees (the same features lua/ge/extensions/lumen/nn.lua computes in game) ---
    def observe(self):
        sc, E = self.sc, self.E
        v = self.ev
        obs = []
        ahead = np.array([5, 10, 20, 30, 45, 60, 80, 100], dtype=float)
        sk = self.es[:, None] + ahead[None, :]
        kap = self.kap_at(sk)
        obs.append(np.clip(kap * 50, -3, 3))
        # the speed curves allow, braking in time, and the limit
        vcurve = np.sqrt(A_LAT / np.maximum(np.abs(kap), 1e-4))
        vsafe = np.sqrt(vcurve ** 2 + 2 * 2.2 * ahead[None, :])
        obs.append(np.minimum(vsafe.min(axis=1), 30)[:, None] / 20)
        obs.append(self.vlim_at(self.es)[:, None] / 20)
        obs.append(np.stack([v / 20, self.ed / 2, self.epsi * 2, self.prevT / 20, self.prevD / 2], axis=1))
        obs.append(np.stack([(sc.edgeR - self.ed) / 4, (self.ed - sc.edgeL) / 4, sc.twoWay.astype(float), sc.shoulder / 2], axis=1))
        # next stop line
        sig = signal_state(self.t, sc)
        line = sc.jS - 6.0
        dist = line - (self.es[:, None] + 2.6)
        valid = (dist > -2.0) & (sc.jKind != J_NONE)
        dist = np.where(valid, dist, 1e9)
        j = np.argmin(dist, axis=1)
        dn = dist[np.arange(E), j]
        st = np.where(dn < 1e8, sig[np.arange(E), j], S_NONE)
        st = np.where(dn > 150, S_NONE, st)
        onehot = np.zeros((E, 5))
        onehot[np.arange(E), st] = 1
        obs.append(np.clip(np.where(dn < 1e8, dn, 150) / 100, -0.1, 1.5)[:, None])
        obs.append(onehot)
        waited = np.where(st == S_STOPSIGN, np.clip((self.t - self.stopped_at[np.arange(E), j]) / 3, 0, 1) * (self.stopped_at[np.arange(E), j] >= 0), 0)
        obs.append(waited[:, None])
        # the 8 nearest vehicles: relative position, velocity, size, crossing or not.
        # WHAT THE NETWORK IS SHOWN (2026-11-01, after the game showed the net stopping for any parked car near the road):
        # the same filter as nn.lua features(): nothing entirely behind the car (rs + half length < -1 m), moving crossing
        # vehicles as before, stationary vehicles (speed <= 1 m/s) only within 2.6 m of the lane centre, moving
        # non-crossing ones within 4.5 m; a stationary vehicle is never flagged as crossing and has no sideways speed.
        kind = sc.kind
        active = kind >= 0
        crossk = kind == CROSS
        rs = self.agS - self.es[:, None]
        rd = self.agD - self.ed[:, None]
        moving = np.abs(self.agV) > 1.0
        cross = crossk & moving
        behind = rs + sc.L / 2 < BEHIND_DROP
        shown = ~behind & (cross | (np.abs(rd) <= np.where(moving, SHOW_RD_MOVING, SHOW_RD_STILL)))
        if not OBS_FILTER:
            shown = np.ones_like(shown)
            cross = crossk
        vs = np.where(kind == ONCOMING, -self.agV, np.where(crossk, 0.0, self.agV))
        vd = np.where(cross, sc.side * self.agV, 0.0)
        far = (~active) | (~shown) | (np.abs(rs) > 80) | (np.abs(rd) > 30)
        dist2 = np.where(far, 1e18, rs * rs + (rd * 1.5) ** 2)
        order = np.argsort(dist2, axis=1)[:, :8]
        idx = np.arange(E)[:, None]
        present = dist2[idx, order] < 1e17
        feats = np.stack([rs[idx, order] / 50, rd[idx, order] / 5, (vs[idx, order] - v[:, None]) / 15, vd[idx, order] / 10,
                          np.where(crossk[idx, order], sc.W[idx, order], sc.L[idx, order]) / 5, cross[idx, order].astype(float),
                          np.clip(self.agAcc[idx, order] / 5, -2, 2), present.astype(float)], axis=2)
        feats = np.where(present[:, :, None], feats, 0.0)
        obs.append(feats.reshape(E, -1))
        obs.append(np.clip((ROUTE_LEN - self.es) / 200, 0, 2)[:, None])
        return np.concatenate(obs, axis=1).astype(np.float32)

    def summary(self):
        return {k: v.copy() for k, v in self.m.items()}


OBS_DIM = 8 + 1 + 1 + 5 + 4 + 1 + 5 + 1 + 64 + 1       # 8 vehicles x 8 features (acceleration added 2026-11-01)
