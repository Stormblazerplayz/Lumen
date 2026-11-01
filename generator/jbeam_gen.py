"""Lumen soft-body physics (jbeam), generated from the same shape as the meshes.

Our own skeleton, not derived from any vanilla or third-party vehicle:
  * body: a lattice of stations along y and levels in height (sill, mid, canopy
    line), roof rails held by the A and C pillars plus hidden mid-roof supports
    (there is no B pillar), crumple zones at the nose and tail
  * suspension: double wishbone at all four corners; front tie rods are steering
    hydros, rear toe links are fixed beams; coilovers with preload and bump stops
  * wheels: pressureWheels with our tyre/hub parameters (255/40 R21)
  * powertrain: dual electric motors -> reduction -> open differentials -> wheels,
    82 kWh battery
  * panels: doors, hood, tailgate and bumpers as their own node groups, attached
    with breakable beams (doors fixed in v1)

Frame: +x left, +y rearward, +z up. Node positions are placed on/inside the real
body surface (shape.F) so flexbody meshes always have nodes around them.
"""
import json
import os

import numpy as np

import details
import shape
from shape import WHEEL_Z, Y_AXLE_F, Y_AXLE_R

FLT_MAX = "FLT_MAX"


def sx(y, z):
    """Body surface half-width at (y, z)."""
    return float(details.surface_x(np.array([y]), np.array([z]), 1.0)[0])


class Part:
    def __init__(self, name, slot_type, title, value=1000):
        self.name = name
        self.d = {"information": {"authors": "Frontier Studios (generated)", "name": title, "value": value}, "slotType": slot_type}
        self.nodes = []      # (id, x, y, z, opts)
        self.beams = []      # (id1, id2, opts)
        self.node_groups = {}

    def node(self, nid, x, y, z, **opts):
        if os.environ.get("LUMEN_DEBUG_NOSELF") and "selfCollision" in opts:
            opts["selfCollision"] = False
        self.nodes.append((nid, round(float(x), 4), round(float(y), 4), round(float(z), 4), opts))
        return nid

    def beam(self, a, b, **opts):
        self.beams.append((a, b, opts))

    def to_dict(self):
        d = dict(self.d)
        if self.nodes:
            rows = [["id", "posX", "posY", "posZ"]]
            for nid, x, y, z, o in self.nodes:
                rows.append([nid, x, y, z, o] if o else [nid, x, y, z])
            d["nodes"] = rows
        if self.beams:
            rows = [["id1:", "id2:"]]
            for a, b, o in self.beams:
                rows.append([a, b, o] if o else [a, b])
            d["beams"] = rows
        return d


# ---- beam property presets (applied per beam as inline options) ---------------------------------
STRUCT = {"beamSpring": 3001000, "beamDamp": 130, "beamDeform": 90000, "beamStrength": FLT_MAX}
CRUMPLE = {"beamSpring": 1501000, "beamDamp": 90, "beamDeform": 26000, "beamStrength": FLT_MAX}
ROOF = {"beamSpring": 2201000, "beamDamp": 110, "beamDeform": 55000, "beamStrength": FLT_MAX}
SUPPORT = {"beamSpring": 1201000, "beamDamp": 80, "beamDeform": 40000, "beamStrength": FLT_MAX}
PANEL = {"beamSpring": 801000, "beamDamp": 120, "beamDeform": 9000, "beamStrength": FLT_MAX}
ATTACH = {"beamSpring": 1501000, "beamDamp": 220, "beamDeform": 15000, "beamStrength": 60000}
# braces from a panel to body nodes off the panel's plane, so doors cannot flap in and out and the
# hood and tailgate cannot flap up and down. 2026-10-31, Storm: "kinda not welded together. The
# doors and hood is wobbling": every v1 attachment beam lay roughly in the panel's own plane.
BRACE = {"beamSpring": 1201000, "beamDamp": 220, "beamDeform": 15000, "beamStrength": 60000}
# Stiffness is capped by explicit integration at 2000 Hz: keep sqrt(sum k / m) * dt under ~1.3 per node.
# (v1 used 9e6/12e6 here; knuckle nodes reached w*dt 2.4 and the car exploded on spawn, 2026-10-31.)
ARM = {"beamSpring": 3501000, "beamDamp": 150, "beamDeform": 120000, "beamStrength": FLT_MAX}
KNUCKLE = {"beamSpring": 3001000, "beamDamp": 120, "beamDeform": 200000, "beamStrength": FLT_MAX}
PICKUP = {"beamSpring": 2001000, "beamDamp": 110, "beamDeform": 90000, "beamStrength": FLT_MAX}

BODY_NODE = {"group": "lumen_body", "nodeMaterial": "|NM_METAL", "frictionCoef": 0.5, "collision": True, "selfCollision": True}

STATIONS = [("n", -2.18), ("f", -1.94), ("t", -1.62), ("w", -1.05), ("s", -0.40),
            ("b", 0.32), ("d", 0.75), ("c", 1.12), ("r", 1.62), ("q", 2.00), ("e", 2.30)]
ROOF_Y = [("k1", -0.72), ("k2", 0.00), ("k3", 0.72), ("k4", 1.40)]


def body_part():
    p = Part("lumen_body", "lumen_body", "Lumen Body", 20000)
    W = {}
    for st, y in STATIONS:
        end = st in ("n", "e")
        wt = 6.0 if not end else 4.0
        # L1 sill / lower structure
        z1 = 0.26 if not end else 0.32
        x1 = min(sx(y, z1 + 0.06) - 0.10, 0.86)
        for side, sg in (("l", 1), ("r", -1)):
            p.node(f"{st}1{side}", sg * x1, y, z1, nodeWeight=wt, **BODY_NODE)
        W[(st, 1)] = True
        # L2 mid: front/rear structures and the door hinge stations
        if st in ("n", "f", "t", "w", "c", "r", "q", "e"):
            z2 = 0.60
            x2 = sx(y, z2) - 0.08
            for side, sg in (("l", 1), ("r", -1)):
                p.node(f"{st}2{side}", sg * x2, y, z2, nodeWeight=wt, **BODY_NODE)
            W[(st, 2)] = True
        # L3 canopy line / hood edge
        if st in ("n", "f", "t"):
            z3 = float(shape.top_z(0.70, y)) - 0.05
            x3 = min(sx(y, z3 - 0.02) - 0.08, 0.72)
        elif st in ("w", "c", "r", "q", "e"):
            z3 = float(shape.BELT(y)) - 0.04
            x3 = sx(y, z3) - 0.07
        else:
            z3 = None
        if z3 is not None:
            for side, sg in (("l", 1), ("r", -1)):
                p.node(f"{st}3{side}", sg * x3, y, z3, nodeWeight=wt * 1.05, **BODY_NODE)
            W[(st, 3)] = True
        # floor centre
        if st in ("w", "s", "b", "d", "c"):
            p.node(f"{st}0", 0.0, y, 0.20, nodeWeight=9.0, **BODY_NODE)
            W[(st, 0)] = True
    for kn, y in ROOF_Y:
        zr = float(shape.top_z(0.56, y)) - 0.06
        for side, sg in (("l", 1), ("r", -1)):
            p.node(f"{kn}{side}", sg * 0.56, y, zr, nodeWeight=4.2, **BODY_NODE)
        # 5.2 kg (was 4.2): the door braces land here, keep w*dt under the 1.3 comfort line
        p.node(f"{kn}m", 0.0, y, float(shape.top_z(0.0, y)) - 0.07, nodeWeight=5.2, **BODY_NODE)
    # reference "up" node and a rigidifier above the floor
    p.node("rup", 0.0, 0.32, 0.80, nodeWeight=3.0, group="lumen_body", collision=False, selfCollision=False)

    st_names = [s for s, _ in STATIONS]

    def has(st, lv):
        return W.get((st, lv), False)

    def b(a, c, preset):
        p.beam(a, c, **preset)

    # longitudinal, lateral, vertical and diagonals
    for i, st in enumerate(st_names):
        preset_here = CRUMPLE if st in ("n", "e") else STRUCT
        for lv in (1, 2, 3):
            if has(st, lv):
                b(f"{st}{lv}l", f"{st}{lv}r", preset_here)                       # lateral
                for side in "lr":
                    for lv2 in (1, 2, 3):                                        # verticals in the same station
                        if lv2 > lv and has(st, lv2):
                            b(f"{st}{lv}{side}", f"{st}{lv2}{side}", preset_here)
        if i + 1 < len(st_names):
            nx = st_names[i + 1]
            preset = CRUMPLE if (st in ("n",) or nx in ("e",)) else STRUCT
            for lv in (1, 2, 3):
                for side in "lr":
                    if has(st, lv) and has(nx, lv):
                        b(f"{st}{lv}{side}", f"{nx}{lv}{side}", preset)          # longitudinal
                    for lv2 in (1, 2, 3):                                        # side-face diagonals
                        if lv2 != lv and has(st, lv) and has(nx, lv2) and abs(lv2 - lv) == 1:
                            b(f"{st}{lv}{side}", f"{nx}{lv2}{side}", preset)
                if has(st, lv) and has(nx, lv):                                  # floor/top cross diagonals
                    b(f"{st}{lv}l", f"{nx}{lv}r", preset)
                    b(f"{st}{lv}r", f"{nx}{lv}l", preset)
    # floor centre nodes
    cen = [s for s in st_names if has(s, 0)]
    for i, st in enumerate(cen):
        for side in "lr":
            b(f"{st}0", f"{st}1{side}", STRUCT)
        if i + 1 < len(cen):
            b(f"{st}0", f"{cen[i + 1]}0", STRUCT)
            for side in "lr":
                b(f"{st}0", f"{cen[i + 1]}1{side}", STRUCT)
                b(f"{cen[i + 1]}0", f"{st}1{side}", STRUCT)
    # roof: rails, cross members, diagonals, pillars and hidden mid supports
    kn = [k for k, _ in ROOF_Y]
    for i, k in enumerate(kn):
        b(f"{k}l", f"{k}m", ROOF)
        b(f"{k}m", f"{k}r", ROOF)
        b(f"{k}l", f"{k}r", ROOF)
        if i + 1 < len(kn):
            for side in "lr":
                b(f"{k}{side}", f"{kn[i + 1]}{side}", ROOF)
            b(f"{k}m", f"{kn[i + 1]}m", ROOF)
            b(f"{k}l", f"{kn[i + 1]}m", ROOF)
            b(f"{k}r", f"{kn[i + 1]}m", ROOF)
            b(f"{k}m", f"{kn[i + 1]}l", ROOF)
            b(f"{k}m", f"{kn[i + 1]}r", ROOF)
    for side in "lr":
        b(f"w3{side}", f"k1{side}", ROOF)          # A pillar
        b(f"w2{side}", f"k1{side}", SUPPORT)
        b(f"c3{side}", f"k4{side}", ROOF)          # C pillar
        b(f"r3{side}", f"k4{side}", ROOF)
        b(f"c2{side}", f"k4{side}", SUPPORT)
        b(f"w3{side}", f"k2{side}", SUPPORT)
        b(f"c3{side}", f"k3{side}", SUPPORT)
        # hidden supports through the cabin (no B pillar)
        b(f"b1{side}", f"k2{side}", SUPPORT)
        b(f"d1{side}", f"k3{side}", SUPPORT)
        b(f"s1{side}", f"k2{side}", SUPPORT)
        b(f"b1{side}", f"k3{side}", SUPPORT)
    b("k1m", "w3l", SUPPORT)
    b("k1m", "w3r", SUPPORT)
    b("k4m", "r3l", SUPPORT)
    b("k4m", "r3r", SUPPORT)
    for k in kn:
        b(f"{k}m", "rup", SUPPORT)
    for n_ in ("b0", "s0", "d0", "b1l", "b1r"):
        b("rup", n_, SUPPORT)
    # torsion: cross diagonals at the pillar stations
    for st in ("w", "c", "t", "r"):
        if has(st, 3):
            b(f"{st}1l", f"{st}3r", STRUCT)
            b(f"{st}1r", f"{st}3l", STRUCT)
    p.W = W
    for nid, x, y, z, _ in p.nodes:
        _BODY_POS[nid] = (x, y, z)
    return p


MAX_STEER_DEG = 32.0          # inner-wheel lock the steering hydros are sized for


def _intersect(p1, p2, p3, p4):
    """Intersection of the 2D lines p1-p2 and p3-p4, or None if they are parallel."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def tie_rod_pivot(l_in, l_out, u_in, u_out, t_out):
    """Front-view (x, z) inner pivot for a tie rod or toe link that adds no bump steer: on the
    line through the two arms' inner pivots, and on the line from the outer tie rod end to the
    arms' instant centre. 2026-10-31: with the pivot 5 cm inboard of this spot and the springs
    sagging 8 cm (see precompressionRange), the parked car sat at 4 deg total front toe-out and
    2.3 deg rear toe-in, scrubbed all four tyres, and crawled stop-start on an empty freeway."""
    ic = _intersect(l_in, l_out, u_in, u_out)
    if ic is None:   # parallel arms: the link runs parallel to them
        far = (t_out[0] - (l_out[0] - l_in[0]) * 100, t_out[1] - (l_out[1] - l_in[1]) * 100)
    else:
        far = ic
    return _intersect(t_out, far, l_in, u_in)


def corner(p, which, side):
    """Double wishbone corner. which: 'F' or 'R', side: 'l' (+x) or 'r' (-x)."""
    sg = 1.0 if side == "l" else -1.0
    yA = Y_AXLE_F if which == "F" else Y_AXLE_R
    fr = 1.0 if which == "F" else -1.0          # sign so 'ahead of the axle' is -y for the front
    pre = which.lower()
    hub = f"lumen_hub_{which}{side.upper()}"
    W = WHEEL_Z
    half = shape.TRACK / 2
    # hub axis: outer node first (node1), inner node (node2); midpoint = track/2
    ax_out = p.node(f"{pre}w1{side}{side}", sg * (half + 0.13), yA, W, nodeWeight=5.0, group=[hub, f"lumen_knuckle_{which}{side.upper()}"], collision=True, selfCollision=False, nodeMaterial="|NM_METAL")
    ax_in = p.node(f"{pre}w1{side}", sg * (half - 0.13), yA, W, nodeWeight=5.0, group=[hub, f"lumen_knuckle_{which}{side.upper()}"], collision=True, selfCollision=False, nodeMaterial="|NM_METAL")
    kg = {"group": f"lumen_knuckle_{which}{side.upper()}", "collision": True, "selfCollision": False, "nodeMaterial": "|NM_METAL"}
    lbj = p.node(f"{pre}h1{side}", sg * (half - 0.08), yA, 0.17, nodeWeight=6.0, **kg)
    ubj = p.node(f"{pre}h2{side}", sg * (half - 0.14), yA, 0.62, nodeWeight=5.0, **kg)
    tre = p.node(f"{pre}h3{side}", sg * (half - 0.10), yA + 0.13 * fr, 0.24, nodeWeight=4.0, **kg)
    arm = p.node(f"{pre}h5{side}", sg * (half - 0.13), yA - 0.12 * fr, 0.37, nodeWeight=4.0, **kg)
    # body pickups
    bg = dict(BODY_NODE)
    la1 = p.node(f"{pre}x1{side}", sg * 0.40, yA - 0.22 * fr, 0.20, nodeWeight=5.0, **bg)
    la2 = p.node(f"{pre}x2{side}", sg * 0.40, yA + 0.22 * fr, 0.20, nodeWeight=5.0, **bg)
    ua1 = p.node(f"{pre}x3{side}", sg * 0.50, yA - 0.18 * fr, 0.66, nodeWeight=4.0, **bg)
    ua2 = p.node(f"{pre}x4{side}", sg * 0.50, yA + 0.14 * fr, 0.66, nodeWeight=4.0, **bg)
    rx, rz = tie_rod_pivot((0.40, 0.20), (half - 0.08, 0.17), (0.50, 0.66), (half - 0.14, 0.62), (half - 0.10, 0.24))
    rack = p.node(f"{pre}s1{side}", sg * rx, yA + 0.13 * fr, rz, nodeWeight=5.0, **bg)
    top = p.node(f"{pre}s2{side}", sg * 0.58, yA, 0.86, nodeWeight=4.0, **bg)
    # knuckle: rigid cluster
    kn = [ax_out, ax_in, lbj, ubj, tre, arm]
    for i in range(len(kn)):
        for j in range(i + 1, len(kn)):
            p.beam(kn[i], kn[j], **KNUCKLE)
    # arms
    for a, c in ((lbj, la1), (lbj, la2), (ubj, ua1), (ubj, ua2)):
        p.beam(a, c, **ARM)
    # coilover: spring + damper, and a bump stop. The spring's precompressionRange is set in
    # build() from the car's real mass distribution, so the car sits at its design ride height.
    # (BeamNG has no "beamPreload": stage2.lua passes only beamPrecompression or
    # precompressionRange to the physics, so the v1 preload key was ignored and the springs sagged.)
    k = 38000 if which == "F" else 46000
    coil = {"beamSpring": k, "beamDamp": 3200, "beamDeform": 120000, "beamStrength": FLT_MAX, "beamType": "|NORMAL", "dampCutoffHz": 500}
    p.beam(lbj, top, **coil)
    coil = p.beams[-1][2]                 # the stored options (beam() copies), updated by build()
    p.beam(lbj, top, beamType="|BOUNDED", beamSpring=0, beamDamp=0, beamLimitSpring=1500000, beamLimitDamp=4000,
           beamDeform=200000, beamStrength=FLT_MAX, longBoundRange=0.10, shortBoundRange=0.10, boundZone=0.02)
    # toe: front = steering hydro (in "hydros"), rear = fixed link
    if which == "R":
        p.beam(tre, rack, **ARM)
    # geometry the build needs later: spring axis angle, and the steering arm and rod lengths
    lbj_p, top_p = np.array([half - 0.08, 0.0, 0.17]), np.array([0.58, 0.0, 0.86])
    axis = top_p - lbj_p
    kp_x = (half - 0.08) + ((half - 0.14) - (half - 0.08)) * (0.24 - 0.17) / (0.62 - 0.17)
    return {"ax_out": ax_out, "ax_in": ax_in, "arm": arm, "tre": tre, "rack": rack, "top": top,
            "pickups": [la1, la2, ua1, ua2, rack, top], "coil": coil, "k": k,
            "spring_cos": float(axis[2] / np.linalg.norm(axis)),
            "steer_arm": float(np.hypot(kp_x - (half - 0.10), 0.13)), "rod": float((half - 0.10) - rx)}


_BODY_POS = {}


def near_nodes(pos, names):
    """Names sorted by distance from pos (body node positions cached by body_part)."""
    return sorted(names, key=lambda n: np.linalg.norm(np.asarray(_BODY_POS[n]) - np.asarray(pos)))


def suspension_part(which):
    p = Part(f"lumen_suspension_{which}", f"lumen_suspension_{which}", f"Lumen {'Front' if which == 'F' else 'Rear'} Suspension", 3000)
    c = {side: corner(p, which, side) for side in "lr"}
    # tie the body pickups into the body lattice
    near = {"F": ["t1", "t2", "t3", "f1", "f2", "w1", "w2"], "R": ["r1", "r2", "r3", "q1", "q2", "c1", "c2"]}[which]
    for side in "lr":
        for pk in c[side]["pickups"]:
            # the three nearest body nodes carry each pickup
            pk_pos = [n for n in p.nodes if n[0] == pk][0][1:4]
            for nb in near_nodes(pk_pos, [f"{n}{side}" for n in near])[:3]:
                p.beam(pk, nb, **PICKUP)
        p.beam(c[side]["pickups"][0], c[side]["pickups"][1], **PICKUP)
    p.beam(c["l"]["rack"], c["r"]["rack"], **PICKUP)
    p.beam(c["l"]["top"], c["r"]["top"], **PICKUP)
    p.coils = {side: c[side] for side in "lr"}
    if which == "F":
        # steering: tie rods as hydros (right turn = +1: left rod lengthens, right rod shortens),
        # sized so full lock turns the wheel about MAX_STEER_DEG. The direction lives in the SIGN
        # of "factor": nodeBeam.lua derives inLimit/outLimit from |factor| and overwrites
        # inputFactor with sign(factor). v1 had factor +0.16 on both rods with inputFactor +-1,
        # so both rods lengthened together and the front wheels pointed up to 72 deg apart at
        # full lock (measured 2026-10-31); that is how the first test car pinned itself on a gate.
        f = round(c["l"]["steer_arm"] * float(np.sin(np.radians(MAX_STEER_DEG))) / c["l"]["rod"], 3)
        p.d["hydros"] = [["id1:", "id2:"],
                         {"beamSpring": 9001000, "beamDamp": 200, "beamDeform": 80000, "beamStrength": FLT_MAX},
                         [c["l"]["rack"], c["l"]["tre"], {"inputSource": "steering", "inRate": 3.0, "outRate": 3.0, "factor": f}],
                         [c["r"]["rack"], c["r"]["tre"], {"inputSource": "steering", "inRate": 3.0, "outRate": 3.0, "factor": -f}]]
    # wheels
    w = []
    w.append(["name", "hubGroup", "group", "node1:", "node2:", "nodeS", "nodeArm:", "wheelDir"])
    w.append({"selfCollision": False, "collision": True, "hasTire": True, "enableHubcaps": False,
              "radius": shape.TIRE_RADIUS, "tireWidth": 0.255, "hubRadius": 0.267, "hubWidth": 0.23, "numRays": 18,
              "wheelOffset": 0.0, "pressurePSI": 42, "nodeWeight": 0.24, "hubNodeWeight": 0.55,
              "nodeMaterial": "|NM_RUBBER", "hubNodeMaterial": "|NM_METAL", "hubFrictionCoef": 0.5, "frictionCoef": 1.0,
              "slidingFrictionCoef": 1.0, "stribeckExponent": 1.5, "treadCoef": 0.5, "noLoadCoef": 1.6,
              "loadSensitivitySlope": 0.000158, "fullLoadCoef": 0.5, "softnessCoef": 0.8, "triangleCollision": False, "dragCoef": 5,
              "enableTireReinfBeams": False, "enableTireLbeams": True, "enableTireSideReinfBeams": False,
              "enableTreadReinfBeams": True, "enableTirePeripheryReinfBeams": True,
              "wheelSideBeamSpring": 46000, "wheelSideBeamDamp": 18, "wheelSideBeamSpringExpansion": 480000, "wheelSideBeamDampExpansion": 45,
              "wheelSideTransitionZone": 0.12, "wheelSideBeamPrecompression": 0.985,
              "wheelReinfBeamSpring": 45000, "wheelReinfBeamDamp": 190, "wheelReinfBeamDampCutoffHz": 500, "wheelReinfBeamPrecompression": 0.98,
              "wheelTreadBeamSpring": 52000, "wheelTreadBeamDamp": 110, "wheelTreadBeamDampCutoffHz": 500, "wheelTreadBeamPrecompression": 0.98,
              "wheelTreadReinfBeamSpring": 200000, "wheelTreadReinfBeamDamp": 60, "wheelTreadReinfBeamDampCutoffHz": 500, "wheelTreadReinfBeamPrecompression": 0.98,
              "wheelPeripheryBeamSpring": 45000, "wheelPeripheryBeamDamp": 35, "wheelPeripheryBeamDampCutoffHz": 500, "wheelPeripheryBeamPrecompression": 0.98,
              "wheelPeripheryReinfBeamSpring": 130000, "wheelPeripheryReinfBeamDamp": 35, "wheelPeripheryReinfBeamDampCutoffHz": 500, "wheelPeripheryReinfBeamPrecompression": 0.98,
              "hubTreadBeamSpring": 1001000, "hubTreadBeamDamp": 10, "hubPeripheryBeamSpring": 1001000, "hubPeripheryBeamDamp": 10,
              "hubSideBeamSpring": 1551000, "hubSideBeamDamp": 10, "hubBeamDeform": 43600, "hubBeamStrength": 78000,
              "wheelSideBeamDeform": 17000, "wheelSideBeamStrength": 22000, "wheelTreadBeamDeform": 12000, "wheelTreadBeamStrength": 16000,
              "wheelPeripheryBeamDeform": 55000, "wheelPeripheryBeamStrength": 55000,
              "brakeTorque": 2400 if which == "F" else 1600, "parkingTorque": 0 if which == "F" else 1500,
              "brakeSpring": 10, "enableABS": True, "enableBrakeThermals": False})
    for side, wd in (("l", -1), ("r", 1)):
        nm = f"{which}{side.upper()}"
        w.append([nm, f"wheel_{nm}", f"tire_{nm}", c[side]["ax_out"], c[side]["ax_in"], 9999, c[side]["arm"], wd])
    p.d["pressureWheels"] = w
    # wheel and tyre meshes
    fb = [["mesh", "[group]:", "nonFlexMaterials"]]
    half = shape.TRACK / 2
    yA = Y_AXLE_F if which == "F" else Y_AXLE_R
    for side, sg in (("l", 1), ("r", -1)):
        nm = f"{which}{side.upper()}"
        pos = {"pos": {"x": round(sg * half, 4), "y": yA, "z": WHEEL_Z}, "rot": {"x": 0, "y": 0, "z": 0 if sg > 0 else 180}, "scale": {"x": 1, "y": 1, "z": 1}}
        fb.append(["rtx_wheel", [f"wheel_{nm}", f"lumen_hub_{nm}"], [], pos])
        fb.append(["rtx_tire", [f"tire_{nm}", f"wheel_{nm}"], [], pos])
        fb.append([f"rtx_liner_{which.lower()}{side}", ["lumen_body"]])
    p.d["flexbodies"] = fb
    # powertrain spindles for this axle
    p.d["powertrain"] = [["type", "name", "inputName", "inputIndex"],
                         ["shaft", f"spindle{which}L", f"differential_{which}", 1, {"connectedWheel": f"{which}L", "friction": 1.2, "dynamicFriction": 0.0035}],
                         ["shaft", f"spindle{which}R", f"differential_{which}", 2, {"connectedWheel": f"{which}R", "friction": 1.2, "dynamicFriction": 0.0035}]]
    return p


def motor_part(which, power_kw, torque_nm):
    p = Part(f"lumen_motor_{which}", f"lumen_motor_{which}", f"Lumen {'Front' if which == 'F' else 'Rear'} Motor ({power_kw} kW)", 6000)
    yA = Y_AXLE_F if which == "F" else Y_AXLE_R
    pre = "m" + which.lower()
    mg = {"group": f"lumen_motor_{which}", "nodeMaterial": "|NM_METAL", "collision": True, "selfCollision": False, "frictionCoef": 0.5}
    ids = []
    for dx in (-0.22, 0.22):
        for dy in (-0.14, 0.14):
            ids.append(p.node(f"{pre}{'l' if dx > 0 else 'r'}{'a' if dy < 0 else 'b'}", dx, yA + dy, 0.34, nodeWeight=18.0, **mg))
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            p.beam(ids[i], ids[j], beamSpring=4001000, beamDamp=250, beamDeform=100000, beamStrength=FLT_MAX)
    # mounts to the suspension pickups and body
    lv = which.lower()
    for nid in ids:
        side = "l" if nid[2] == "l" else "r"
        for tgt in (f"{lv}x1{side}", f"{lv}x2{side}", f"{lv}s1{side}"):
            p.beam(nid, tgt, beamSpring=1501000, beamDamp=300, beamDeform=40000, beamStrength=90000)
    rpm = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000], float)
    base = power_kw * 1000 / torque_nm * 60 / (2 * np.pi)          # rpm where constant torque meets constant power
    tq = np.where(rpm <= base, torque_nm, power_kw * 1000 / np.maximum(rpm, 1) * 60 / (2 * np.pi))
    tq[-1] = tq[-1] * 0.6
    name = "frontMotor" if which == "F" else "rearMotor"
    p.d["powertrain"] = [["type", "name", "inputName", "inputIndex"],
                         ["electricMotor", name, "dummy", 0],
                         ["torsionReactor", f"reduction{which}", name, 1],
                         ["differential", f"differential_{which}", f"reduction{which}", 1, {"diffType": "open", "uiName": f"{'Front' if which == 'F' else 'Rear'} Differential", "defaultVirtualInertia": 0.25}]]
    p.d[name] = {"torque": [["rpm", "torque"]] + [[int(r), round(float(t), 1)] for r, t in zip(rpm, tq)],
                 "maxRPM": 18000, "inertia": 0.08, "friction": 2, "dynamicFriction": 0.0002, "electricalEfficiency": 0.95,
                 "torqueReactionNodes:": [ids[0], ids[1], ids[2]], "energyStorage": "mainBattery", "maximumWantedRegenTorque": 260,
                 "uiName": f"{'Front' if which == 'F' else 'Rear'} Motor", "soundConfig": f"soundConfig{which}"}
    p.d[f"soundConfig{which}"] = {"sampleName": "ElectricMotor_02", "mainGain": -6 if which == "F" else -4}
    p.d[f"reduction{which}"] = {"gearRatio": 9.0, "torqueReactionNodes:": [ids[0], ids[1], ids[3]]}
    p.d[f"differential_{which}"] = {"friction": 2.5, "dynamicFriction": 0.0012, "torqueLossCoef": 0.015}
    return p


def battery_part():
    p = Part("lumen_battery", "lumen_battery", "Lumen 82 kWh Battery", 18000)
    bg = {"group": "lumen_battery", "nodeMaterial": "|NM_METAL", "collision": False, "selfCollision": False}
    ids = []
    for y in (-0.70, 0.10, 0.90):
        for sg in (1, -1):
            ids.append(p.node(f"bt{'l' if sg > 0 else 'r'}{int((y + 1) * 10)}", sg * 0.42, y, 0.22, nodeWeight=95.0, **bg))
    for nid in ids:
        y = [n for n in p.nodes if n[0] == nid][0][2]
        side = "l" if "l" in nid[2] else "r"
        near = min([s for s in STATIONS if s[0] in ("w", "s", "b", "d", "c")], key=lambda s: abs(s[1] - y))[0]
        for tgt in (f"{near}1{side}", f"{near}0"):
            p.beam(nid, tgt, beamSpring=2001000, beamDamp=250, beamDeform=60000, beamStrength=FLT_MAX)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            p.beam(ids[i], ids[j], beamSpring=2001000, beamDamp=200, beamDeform=60000, beamStrength=FLT_MAX)
    p.d["energyStorage"] = [["type", "name"], ["electricBattery", "mainBattery"]]
    p.d["variables"] = [["name", "type", "unit", "category", "default", "min", "max", "title", "description"],
                        ["$fuel", "range", "kWh", "Chassis", 82, 0, 82, "Battery Level", "Initial battery charge", {"stepDis": 0.5}]]
    p.d["mainBattery"] = {"energyType": "electricEnergy", "batteryCapacity": 82, "startingCapacity": "$fuel"}
    return p


def panel_part(name, slot, title, group, pts, attach, meshes, weight=4.0, normal=None):
    """A panel with its own nodes (pts: {id: (x,y,z)}), fully interconnected, attached to body
    nodes with breakable beams. normal: the panel's rough outward direction; each panel node is
    also braced to the two nearby body nodes that lie most out of the panel's plane."""
    p = Part(name, slot, title, 600)
    ng = {"group": group, "nodeMaterial": "|NM_METAL", "collision": True, "selfCollision": True, "frictionCoef": 0.5}
    ids = list(pts.keys())
    for nid in ids:
        x, y, z = pts[nid]
        p.node(nid, x, y, z, nodeWeight=weight, **ng)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            p.beam(ids[i], ids[j], **PANEL)
    for a, bnodes in attach.items():
        for bn in bnodes:
            p.beam(a, bn, **ATTACH)
    if normal is not None:
        nrm = np.asarray(normal, float) / np.linalg.norm(normal)
        body = [n for n in _BODY_POS if n != "rup"]
        for nid in ids:
            pos = np.asarray(pts[nid], float)
            near = near_nodes(pos, body)[:10]
            def offplane(bn):
                d = np.asarray(_BODY_POS[bn]) - pos
                return abs(float(d @ nrm)) / (np.linalg.norm(d) + 1e-9)
            picks = [bn for bn in sorted(near, key=offplane, reverse=True) if bn not in attach.get(nid, [])][:2]
            for bn in picks:
                if offplane(bn) > 0.35:
                    p.beam(nid, bn, **BRACE)
    p.d["flexbodies"] = [["mesh", "[group]:", "nonFlexMaterials"]] + [[m, [group]] for m in meshes]
    return p


def door_parts():
    parts = []
    belt = lambda y: float(shape.BELT(y))
    for side, sg in (("L", 1), ("R", -1)):
        s = side.lower()
        # front door: hinged at the front (w station), latches at the door split
        y0, y1 = -1.04, 0.29
        pts = {}
        for tag, y, z in (("a", y0, 0.36), ("b", y1, 0.36), ("c", y0 + 0.02, belt(y0) - 0.03), ("d", y1, belt(y1) - 0.03),
                          ("e", -0.55, float(shape.top_z(0.6, -0.55)) - 0.12), ("f", 0.22, float(shape.top_z(0.6, 0.22)) - 0.10)):
            pts[f"df{tag}{s}"] = (sg * (sx(y, z) - 0.04), y, z)
        attach = {f"dfa{s}": [f"w1{s[0]}", f"w2{s[0]}"], f"dfc{s}": [f"w2{s[0]}", f"w3{s[0]}"],
                  f"dfb{s}": [f"b1{s[0]}"], f"dfd{s}": [f"k2{s[0]}"], f"dfe{s}": [f"k1{s[0]}"], f"dff{s}": [f"k2{s[0]}"]}
        attach = {k: [n if n[-1] in "lr" else n for n in v] for k, v in attach.items()}
        parts.append(panel_part(f"lumen_door_F{side}", f"lumen_door_F{side}", f"Front {'Left' if side == 'L' else 'Right'} Door",
                                f"lumen_door_F{side}", pts, _fix_side(attach, s), [f"rtx_door_f{s}", f"rtx_glass_door_f{s}", f"rtx_campod_{s}"], weight=3.6,
                                normal=(sg, 0, 0)))
        # rear door: hinged at the back (c station), latches at the door split
        y0, y1 = 0.35, 1.08
        pts = {}
        for tag, y, z in (("a", y1, 0.36), ("b", y0, 0.36), ("c", y1, belt(y1) - 0.03), ("d", y0, belt(y0) - 0.03),
                          ("e", 0.95, float(shape.top_z(0.6, 0.95)) - 0.12), ("f", 0.42, float(shape.top_z(0.6, 0.42)) - 0.10)):
            pts[f"dr{tag}{s}"] = (sg * (sx(y, z) - 0.04), y, z)
        attach = {f"dra{s}": ["c1", "c2"], f"drc{s}": ["c2", "c3"], f"drb{s}": ["b1"], f"drd{s}": ["k2"], f"dre{s}": ["k4"], f"drf{s}": ["k3"]}
        parts.append(panel_part(f"lumen_door_R{side}", f"lumen_door_R{side}", f"Rear {'Left' if side == 'L' else 'Right'} Door",
                                f"lumen_door_R{side}", pts, _fix_side(attach, s), [f"rtx_door_r{s}", f"rtx_glass_door_r{s}"], weight=3.6,
                                normal=(sg, 0, 0)))
    return parts


def _fix_side(attach, s):
    """Append the side letter to body node names given without one (c1 -> c1l)."""
    return {k: [n if n.endswith(("l", "r")) else n + s for n in v] for k, v in attach.items()}


def hood_part():
    tz = lambda x, y: float(shape.top_z(x, y))
    pts = {"hda": (0.66, -2.12, tz(0.66, -2.12) - 0.03), "hdb": (-0.66, -2.12, tz(0.66, -2.12) - 0.03),
           "hdc": (0.62, -1.50, tz(0.62, -1.50) - 0.03), "hdd": (-0.62, -1.50, tz(0.62, -1.50) - 0.03),
           "hde": (0.0, -1.82, tz(0.0, -1.82) - 0.03)}
    attach = {"hdc": ["t3l", "w3l"], "hdd": ["t3r", "w3r"], "hda": ["n3l", "f3l"], "hdb": ["n3r", "f3r"], "hde": ["f3l", "f3r"]}
    return panel_part("lumen_hood", "lumen_hood", "Hood", "lumen_hood", pts, attach, ["rtx_hood"], weight=3.0, normal=(0, -0.25, 1))


def tailgate_part():
    tz = lambda x, y: float(shape.top_z(x, y))
    pts = {"tga": (0.48, 1.60, tz(0.48, 1.60) - 0.05), "tgb": (-0.48, 1.60, tz(0.48, 1.60) - 0.05),
           "tgc": (0.62, 2.28, 0.70), "tgd": (-0.62, 2.28, 0.70), "tge": (0.0, 2.33, 1.02), "tgf": (0.0, 2.05, tz(0.0, 2.05) - 0.05)}
    attach = {"tga": ["k4l"], "tgb": ["k4r"], "tgc": ["e2l", "q2l"], "tgd": ["e2r", "q2r"], "tge": ["e3l", "e3r"], "tgf": ["q3l", "q3r"]}
    return panel_part("lumen_tailgate", "lumen_tailgate", "Tailgate", "lumen_tailgate", pts, attach, ["rtx_tailgate"], weight=3.5, normal=(0, 0.7, 0.7))


def bumper_parts():
    fpts = {"bfa": (0.74, -2.14, 0.40), "bfb": (-0.74, -2.14, 0.40), "bfc": (0.0, -2.27, 0.42),
            "bfd": (0.78, -2.10, 0.66), "bfe": (-0.78, -2.10, 0.66), "bff": (0.0, -2.28, 0.66)}
    fatt = {"bfa": ["n1l", "n2l"], "bfb": ["n1r", "n2r"], "bfc": ["n1l", "n1r"], "bfd": ["n2l", "n3l"], "bfe": ["n2r", "n3r"], "bff": ["n2l", "n2r"]}
    rpts = {"bra": (0.74, 2.26, 0.40), "brb": (-0.74, 2.26, 0.40), "brc": (0.0, 2.36, 0.42),
            "brd": (0.78, 2.22, 0.62), "bre": (-0.78, 2.22, 0.62), "brf": (0.0, 2.38, 0.62)}
    ratt = {"bra": ["e1l", "q1l"], "brb": ["e1r", "q1r"], "brc": ["e1l", "e1r"], "brd": ["e2l", "q2l"], "bre": ["e2r", "q2r"], "brf": ["e2l", "e2r"]}
    return [panel_part("lumen_bumper_F", "lumen_bumper_F", "Front Fascia", "lumen_bumper_F", fpts, fatt, ["rtx_bumper_f"], weight=2.5),
            panel_part("lumen_bumper_R", "lumen_bumper_R", "Rear Bumper", "lumen_bumper_R", rpts, ratt, ["rtx_bumper_r"], weight=2.5)]


BODY_MESHES = ["rtx_body", "rtx_canopy", "rtx_glass_quarter_l", "rtx_glass_quarter_r", "rtx_rocker_l", "rtx_rocker_r",
               "rtx_taillight_l", "rtx_taillight_r", "rtx_fender_l", "rtx_fender_r"]
INTERIOR_MESHES = ["rtx_dash", "rtx_consoles", "rtx_floor", "rtx_seat_fl", "rtx_seat_fr", "rtx_seat_rl", "rtx_seat_rr"]

GLOWMAP = {
    "rtx_drl": {"simpleFunction": {"ignition": 0.3, "lowbeam": 0.3, "highbeam": 0.3}, "off": "rtx_drl", "on": "rtx_drl_on", "on_intense": "rtx_drl_on_intense"},
    "rtx_headlight": {"simpleFunction": {"lowbeam": 0.49, "highbeam": 1.0}, "off": "rtx_headlight", "on": "rtx_headlight_on", "on_intense": "rtx_headlight_on_intense"},
    "rtx_signal_fl": {"simpleFunction": {"signal_L": 1.0}, "off": "rtx_signal_fl", "on": "rtx_signal_on"},
    "rtx_signal_fr": {"simpleFunction": {"signal_R": 1.0}, "off": "rtx_signal_fr", "on": "rtx_signal_on"},
    "rtx_tail_l": {"simpleFunction": {"ignition": 0.3, "brakelights": 0.5, "signal_L": 0.5}, "off": "rtx_tail_l", "on": "rtx_tail_on", "on_intense": "rtx_tail_on_intense"},
    "rtx_tail_r": {"simpleFunction": {"ignition": 0.3, "brakelights": 0.5, "signal_R": 0.5}, "off": "rtx_tail_r", "on": "rtx_tail_on", "on_intense": "rtx_tail_on_intense"},
    "rtx_tailbar": {"simpleFunction": {"ignition": 0.3, "brakelights": 0.5}, "off": "rtx_tailbar", "on": "rtx_tail_on", "on_intense": "rtx_tail_on_intense"},
    "rtx_halo": {"simpleFunction": {"rtx_autonomy": 1.0}, "off": "rtx_halo", "on": "rtx_halo_on"},
    "rtx_ambient": {"simpleFunction": {"ignition": 1.0}, "off": "rtx_ambient", "on": "rtx_ambient_on"},
    "rtx_screen_dash": {"simpleFunction": {"ignition": 1.0}, "off": "rtx_screen_off", "on": "rtx_screen_dash"},
    "rtx_screen_rear": {"simpleFunction": {"ignition": 1.0}, "off": "rtx_screen_off", "on": "rtx_screen_rear"},
}


def main_part():
    d = {"information": {"authors": "Frontier Studios (generated)", "name": "Lumen Robotaxi", "value": 72000},
         "slotType": "main",
         "slots": [["type", "default", "description"],
                   ["lumen_body", "lumen_body", "Body", {"coreSlot": True}],
                   ["paint_design", "lumen_livery_robotaxi", "Livery"],
                   ["lumen_sensors", "lumen_sensors", "Autonomy Sensors"]],
         "glowMap": GLOWMAP,
         "variables": [["name", "type", "unit", "category", "default", "min", "max", "title", "description"],
                       ["$brakestrength", "range", "", "Brakes", 1, 0.6, 1.0, "Brake Force Multiplier", "Scales the overall brake torque", {"minDis": 60, "maxDis": 100}]]}
    return d


def body_dict(body):
    d = body.to_dict()
    d["slots"] = [["type", "default", "description"],
                  ["lumen_suspension_F", "lumen_suspension_F", "Front Suspension", {"coreSlot": True}],
                  ["lumen_suspension_R", "lumen_suspension_R", "Rear Suspension", {"coreSlot": True}],
                  ["lumen_battery", "lumen_battery", "Battery", {"coreSlot": True}],
                  ["lumen_motor_R", "lumen_motor_R", "Rear Motor", {"coreSlot": True}],
                  ["lumen_motor_F", "lumen_motor_F", "Front Motor"],
                  ["lumen_door_FL", "lumen_door_FL", "Front Left Door"], ["lumen_door_FR", "lumen_door_FR", "Front Right Door"],
                  ["lumen_door_RL", "lumen_door_RL", "Rear Left Door"], ["lumen_door_RR", "lumen_door_RR", "Rear Right Door"],
                  ["lumen_hood", "lumen_hood", "Hood"], ["lumen_tailgate", "lumen_tailgate", "Tailgate"],
                  ["lumen_bumper_F", "lumen_bumper_F", "Front Fascia"], ["lumen_bumper_R", "lumen_bumper_R", "Rear Bumper"],
                  ["lumen_interior", "lumen_interior", "Interior"],
                  ["soundscape_horn", "soundscape_horn_4", "Horn"]]
    d["flexbodies"] = [["mesh", "[group]:", "nonFlexMaterials"]] + [[m, ["lumen_body"]] for m in BODY_MESHES]
    d["refNodes"] = [["ref:", "back:", "left:", "up:", "leftCorner:", "rightCorner:"], ["b0", "d0", "b1l", "rup", "n2l", "n2r"]]
    d["cameraExternal"] = {"distance": 6.0, "distanceMin": 3.2, "offset": {"x": 0.0, "y": 0.3, "z": 0.55}, "fov": 65}
    d["cameraChase"] = {"distance": 6.2, "distanceMin": 3.0, "offset": {"x": 0.0, "y": 0.3, "z": 1.3}, "defaultRotation": {"x": 0, "y": -12, "z": 0}, "fov": 65}
    d["camerasInternal"] = [["type", "x", "y", "z", "fov", "id1:", "id2:", "id3:", "id4:", "id5:", "id6:"],
                            {"nodeWeight": 1.5}, {"selfCollision": False}, {"collision": False}, {"beamSpring": 60000, "beamDamp": 300},
                            # one camera per seat (Storm: "I can't sit in any seat I want"); onboard.<name> in game
                            ["driver", 0.38, -0.22, 1.13, 74, "s1l", "s1r", "w3l", "w3r", "k1l", "k1r", {"beamDeform": 5001000, "beamStrength": FLT_MAX}],
                            ["front_passenger", -0.38, -0.22, 1.13, 74, "s1l", "s1r", "w3l", "w3r", "k1l", "k1r", {"beamDeform": 5001000, "beamStrength": FLT_MAX}],
                            ["rear_left", 0.38, 0.86, 1.13, 76, "d1l", "d1r", "c3l", "c3r", "k3l", "k3r", {"beamDeform": 5001000, "beamStrength": FLT_MAX}],
                            ["rear_right", -0.38, 0.86, 1.13, 76, "d1l", "d1r", "c3l", "c3r", "k3l", "k3r", {"beamDeform": 5001000, "beamStrength": FLT_MAX}]]
    d["controller"] = [["fileName"], ["vehicleController", {}], ["lumenRobotaxi", {}]]
    # defaultRegenStrength 0: one-pedal driving starts off. The electricMotor shift logic
    # otherwise defaults to its strongest level, which turns any throttle under ~25 % into regen
    # braking; the AI does not know that and crawled stop-start (2026-10-31). Players can still
    # raise the regen level in game.
    d["vehicleController"] = {"shiftLogicName": "electricMotor", "motorNames": ["rearMotor", "frontMotor"], "topSpeedLimitReverse": 15,
                              "onePedalRegenCoef": 0.6, "defaultRegenStrength": 0}
    d["sounds"] = {"cabinFilterCoef": 0.14, "windNode": "k1m"}
    if not os.environ.get("LUMEN_DEBUG_NOTRI"):
        d["triangles"] = body_triangles(body)
    return d


def body_triangles(body):
    """Collision/aero skin over the outer body lattice (sides, nose, tail, roof)."""
    W = body.W
    st = [s for s, _ in STATIONS]
    tris = [["id1:", "id2:", "id3:"], {"triangleType": "NORMALTYPE", "dragCoef": 9}]
    for i in range(len(st) - 1):
        a, b_ = st[i], st[i + 1]
        for l1, l2 in ((1, 2), (2, 3), (1, 3)):
            for side in "lr":
                if W.get((a, l1)) and W.get((a, l2)) and W.get((b_, l1)) and W.get((b_, l2)):
                    q = [f"{a}{l1}{side}", f"{b_}{l1}{side}", f"{b_}{l2}{side}", f"{a}{l2}{side}"]
                    if side == "r":
                        q = q[::-1]
                    tris.append([q[0], q[1], q[2]])
                    tris.append([q[0], q[2], q[3]])
    k = [kn for kn, _ in ROOF_Y]
    for i in range(len(k) - 1):
        tris.append([f"{k[i]}l", f"{k[i]}m", f"{k[i + 1]}m"])
        tris.append([f"{k[i]}l", f"{k[i + 1]}m", f"{k[i + 1]}l"])
        tris.append([f"{k[i]}m", f"{k[i]}r", f"{k[i + 1]}r"])
        tris.append([f"{k[i]}m", f"{k[i + 1]}r", f"{k[i + 1]}m"])
    return tris


def sensors_part():
    return {"information": {"authors": "Frontier Studios (generated)", "name": "Autonomy Sensor Suite", "value": 9000},
            "slotType": "lumen_sensors",
            "flexbodies": [["mesh", "[group]:", "nonFlexMaterials"], ["rtx_lidar", ["lumen_body"]]]}


def interior_part():
    """Cabin: the interior flexbodies plus the steering wheel and pedals as props. The props
    hang off three light nodes at the wheel centre (cockpit.frame_nodes) that give BeamNG the
    prop frame: X = left, Y = up in the wheel plane, Z = down the column."""
    import cockpit
    p = Part("lumen_interior", "lumen_interior", "Lumen Cabin", 4000)
    ng = {"nodeWeight": 2.0, "collision": False, "selfCollision": False, "group": "lumen_cockpit"}
    frame = cockpit.frame_nodes()
    for nid, xyz in frame.items():
        p.node(nid, *xyz, **ng)
    ids = list(frame)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            p.beam(ids[i], ids[j], **PANEL)
    for nid, xyz in frame.items():
        for nb in near_nodes(xyz, list(_BODY_POS))[:6]:
            p.beam(nid, nb, **PANEL)
    d = p.to_dict()
    d["flexbodies"] = [["mesh", "[group]:", "nonFlexMaterials"]] + [[m, ["lumen_body"]] for m in INTERIOR_MESHES]
    zero = {"x": 0, "y": 0, "z": 0}
    # Measured in game (2026-10-31, prop:getLiveTransformWorld): a prop is NOT placed at its idRef
    # node; without baseTranslationGlobal it sits at the vehicle origin (the first wheel was under
    # the floor). The frame maps mesh X to idX and mesh Z to idY, so meshes built with Z down the
    # column need a +90 deg base rotation about X to stand up. Steering right turned the wheel
    # anticlockwise with z = +1, hence z = -1.
    up = {"x": 90, "y": 0, "z": 0}
    wx, wy, wz = (float(v) for v in cockpit.WHEEL_C)
    props = [["func", "mesh", "idRef:", "idX:", "idY:", "baseRotation", "rotation", "translation", "min", "max", "offset", "multiplier"],
             # electrics.steering is the wheel angle in degrees (450 at full lock)
             ["steering", "rtx_steerwheel", "strw", "strwx", "strwy", up, {"x": 0, "y": 0, "z": -1}, zero, -1080, 1080, 0, 1,
              {"baseTranslationGlobal": {"x": wx, "y": wy, "z": wz}}]]
    for func, mesh, key, deg in (("throttle", "rtx_pedal_gas", "gas", -16), ("brake", "rtx_pedal_brake", "brake", -12)):
        x, y, z = cockpit.PEDAL_PIVOTS[key]
        props.append([func, mesh, "strw", "strwx", "strwy", up, {"x": deg, "y": 0, "z": 0}, zero, 0, 1, 0, 1,
                      {"baseTranslationGlobal": {"x": x, "y": y, "z": z}}])
    d["props"] = props
    return d


def livery_parts():
    robotaxi = {"information": {"authors": "Frontier Studios (generated)", "name": "Robotaxi Livery", "value": 300},
                "slotType": "paint_design",
                "flexbodies": [["mesh", "[group]:", "nonFlexMaterials"],
                               ["rtx_decal_l", ["lumen_door_FL"]],
                               ["rtx_decal_r", ["lumen_door_FR"]]]}
    return {"lumen_livery_robotaxi": robotaxi}


def static_spring_ranges(files, susp):
    """Sets each coilover spring's precompressionRange so it carries its corner's static sprung
    load at the design ride height. Mass comes from every node BeamNG will create, including the
    pressureWheel tyre and hub nodes; the axle split comes from the centre of gravity."""
    pos, pts = {}, []
    for data in files.values():
        for part in data.values():
            defaults = {}
            for row in part.get("nodes", [])[1:]:
                if isinstance(row, dict):
                    defaults.update(row)
                    continue
                o = dict(defaults)
                if len(row) > 4 and isinstance(row[4], dict):
                    o.update(row[4])
                g = o.get("group", "")
                pos[row[0]] = (row[1], row[2], row[3])
                pts.append((float(o.get("nodeWeight", 25.0)), row[2], g if isinstance(g, list) else [g]))
    for data in files.values():
        for part in data.values():
            defaults = {}
            for row in part.get("pressureWheels", [])[1:]:
                if isinstance(row, dict):
                    defaults.update(row)
                    continue
                rays = int(defaults.get("numRays", 16))
                w = 2 * rays * (float(defaults.get("nodeWeight", 0)) + float(defaults.get("hubNodeWeight", 0)))
                pts.append((w, pos[row[3]][1], ["wheel_" + row[0]]))
    mass = sum(w for w, _, _ in pts)
    ycg = sum(w * y for w, y, _ in pts) / mass
    springs = {}
    for which, part in susp.items():
        share = (Y_AXLE_R - ycg) if which == "F" else (ycg - Y_AXLE_F)
        share /= (Y_AXLE_R - Y_AXLE_F)
        for side in "lr":
            tag = f"{which}{side.upper()}"
            unsprung = sum(w for w, _, g in pts if f"lumen_knuckle_{tag}" in g or f"wheel_{tag}" in g)
            c = part.coils[side]
            force = (mass * share / 2 - unsprung) * 9.81 / c["spring_cos"]
            c["coil"]["precompressionRange"] = round(force / c["k"], 4)
            springs[tag] = (round(force), c["coil"]["precompressionRange"])
    return {"mass": round(mass, 1), "cg_y": round(ycg, 3), "springs": springs}


def build(out_dir):
    body = body_part()
    files = {}
    files["lumen.jbeam"] = {"lumen": main_part(), "lumen_body": body_dict(body), "lumen_sensors": sensors_part(),
                            "lumen_interior": interior_part(), **livery_parts()}
    sF, sR = suspension_part("F"), suspension_part("R")
    files["lumen_suspension.jbeam"] = {sF.name: sF.to_dict(), sR.name: sR.to_dict()}
    mR, mF, bat = motor_part("R", 250, 420), motor_part("F", 160, 280), battery_part()
    files["lumen_powertrain.jbeam"] = {x.name: x.to_dict() for x in (mR, mF, bat)}
    panels = door_parts() + [hood_part(), tailgate_part()] + bumper_parts()
    files["lumen_panels.jbeam"] = {x.name: x.to_dict() for x in panels}
    # the coil option dicts are shared with the rows above, so this lands in the files
    load = static_spring_ranges(files, {"F": sF, "R": sR})
    os.makedirs(out_dir, exist_ok=True)
    for fn, data in files.items():
        with open(os.path.join(out_dir, fn), "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=1)
    # quick sanity: every beam endpoint exists somewhere
    all_nodes = set()
    all_beams = []
    for data in files.values():
        for part in data.values():
            for row in part.get("nodes", [])[1:]:
                if isinstance(row, list):
                    all_nodes.add(row[0])
            for row in part.get("beams", [])[1:]:
                if isinstance(row, list):
                    all_beams.append((row[0], row[1]))
            for row in part.get("hydros", [])[1:]:
                if isinstance(row, list):
                    all_beams.append((row[0], row[1]))
    missing = sorted({n for a, b_ in all_beams for n in (a, b_) if n not in all_nodes})
    worst = stability(files)
    return {"nodes": len(all_nodes), "beams": len(all_beams), "missing": missing, "worst_w_dt": worst[:3], **load}


def stability(files, dt=0.0005):
    """Per-node explicit-integration margin: sqrt(sum of attached spring rates / mass) * dt.
    Above ~2 a node diverges; the build refuses anything above 1.4."""
    from collections import defaultdict
    w, ksum = {}, defaultdict(float)
    for data in files.values():
        for part in data.values():
            for row in part.get("nodes", [])[1:]:
                if isinstance(row, list):
                    w[row[0]] = (row[4] if len(row) > 4 else {}).get("nodeWeight", 25.0)
            for sec in ("beams", "hydros"):
                props = {}
                for row in part.get(sec, [])[1:]:
                    if isinstance(row, dict):
                        props.update(row)
                        continue
                    o = dict(props)
                    o.update(row[2] if len(row) > 2 else {})
                    k = float(o.get("beamSpring", 0))
                    if o.get("beamType") == "|BOUNDED":
                        k = max(k, float(o.get("beamLimitSpring", 0)))
                    ksum[row[0]] += k
                    ksum[row[1]] += k
    res = sorted(((round(float(np.sqrt(ksum[n] / w[n]) * dt), 3), n) for n in w), reverse=True)
    if res and res[0][0] > 1.4:
        raise SystemExit("jbeam would be unstable: %s" % res[:8])
    return res


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mod", "vehicles", "lumen")
    print(build(out))
