"""Lumen panel, glass, trim and lamp layout on the implicit body (our own design).

Every boundary is the zero set of a field on (position V, unit normal N), so seams
are smooth curves on the real surface. Labels with a side suffix use BeamNG's
convention: _L is +x (left), _R is -x (right).

Look: everything above the canopy line is black glass split by thin gloss-black
seams; a turquoise autonomy halo runs just below the glass; razor light bars front
and rear; front doors hinge at the front, rear doors at the back, no B pillar.
(The earlier Model Y layout is kept in archive/modely/.)
"""
import numpy as np

import shape
from shape import Pchip, Y_AXLE_F, Y_AXLE_R, WHEEL_Z, ARCH_RADIUS

# ---- lines (metres, vehicle frame) -----------------------------------------------------
FD_SEAM = lambda z: -1.090 + 0.060 * (np.clip(z, 0.3, 1.1) - 0.31)     # front door leading edge
DOOR_SPLIT = 0.320                                                       # front/rear doors meet (no B pillar)
RD_LINE = lambda z: 1.120 - 0.020 * (1.1 - np.clip(z, 0.8, 1.1)) / 0.3   # rear door trailing edge (upper)
RD_ARC_R = ARCH_RADIUS + 0.060                                           # ...then it follows the rear arch
DOOR_BOTTOM = 0.312
ROCKER_TOP = 0.292
HOOD_X = Pchip([-2.35, -2.05, -1.80, -1.55, -1.38], [0.830, 0.835, 0.815, 0.775, 0.735])
COWL = lambda ax: -1.430 + 0.12 * (np.clip(ax, 0, 0.8) / 0.75) ** 2     # windshield base
CANOPY_TG = 1.560                                                        # canopy glass: body / tailgate split
TG_X = Pchip([0.60, 0.80, 1.00, 1.10, 1.25], [0.720, 0.720, 0.700, 0.660, 0.560])
TG_BOTTOM = 0.660
RB_SIDE_Y = 1.960
FB_SIDE = lambda z: -1.960 + 0.0 * np.asarray(z, float)                 # front fascia rear edge on the side
LIGHTBAR = (0.688, 0.714)
HEADLAMP = (0.604, 0.646, 0.555, 0.785)                                  # z0, z1, ax0, ax1
TAILBAR = (1.030, 1.068)
HALO = 0.020                                                             # halo strip height below the canopy line
SEAM = 0.010                                                             # half-width of the black seams between panes
LAT_SIDE = 0.64                                                          # side glass vs canopy roof (|n_x|)
GLASS_SPLITS = (DOOR_SPLIT, 1.120)                                       # door glass splits (y)
FACE_BAND = 0.588                                                        # gloss-black face band under the light bar
TAIL_BAND = 0.930                                                        # gloss-black tail band under the light bar
LOWER_ACCENT = 0.400                                                     # gloss-black lower body between the arches
PAINT_LABELS = ("body", "hood", "bumper_F", "bumper_R", "tailgate",
                "fender_L", "fender_R", "door_F_L", "door_F_R", "door_R_L", "door_R_R")


def belt(y):
    return shape.BELT(y)


def lat(N):
    return np.abs(N[:, 0])


def arch_dist(V, y_axle):
    return np.hypot(V[:, 1] - y_axle, V[:, 2] - WHEEL_Z)


def cuts():
    ax = lambda V: np.abs(V[:, 0])
    side = lambda V, N: ax(V) > 0.45
    top = lambda V, N: N[:, 2] > 0.25
    canopy = lambda V: (V[:, 1] > -1.60) & (V[:, 1] < 2.35)
    above = lambda V: V[:, 2] > belt(V[:, 1]) - 0.03
    C = []
    C.append(("arch_F", lambda V, N: ARCH_RADIUS - arch_dist(V, Y_AXLE_F), lambda V, N: side(V, N) & (V[:, 2] < 0.95)))
    C.append(("arch_R", lambda V, N: ARCH_RADIUS - arch_dist(V, Y_AXLE_R), lambda V, N: side(V, N) & (V[:, 2] < 0.95)))
    btw = lambda V: (V[:, 1] > Y_AXLE_F + 0.30) & (V[:, 1] < Y_AXLE_R - 0.30)
    C.append(("rocker", lambda V, N: V[:, 2] - ROCKER_TOP, lambda V, N: side(V, N) & btw(V)))
    C.append(("door_bottom", lambda V, N: V[:, 2] - DOOR_BOTTOM, lambda V, N: side(V, N) & btw(V)))
    # canopy line and the halo strip under it
    C.append(("belt", lambda V, N: V[:, 2] - belt(V[:, 1]), lambda V, N: side(V, N) & canopy(V)))
    C.append(("halo", lambda V, N: V[:, 2] - (belt(V[:, 1]) - HALO), lambda V, N: side(V, N) & canopy(V)))
    # glass panes: side glass vs canopy roof, with seams
    C.append(("lat_a", lambda V, N: lat(N) - LAT_SIDE, lambda V, N: above(V) & canopy(V)))
    C.append(("lat_b", lambda V, N: lat(N) - (LAT_SIDE - 0.035), lambda V, N: above(V) & canopy(V)))
    for i, ys in enumerate(GLASS_SPLITS):
        C.append((f"gs{i}a", (lambda ys: lambda V, N: V[:, 1] - ys)(ys), lambda V, N: above(V) & side(V, N)))
        C.append((f"gs{i}b", (lambda ys: lambda V, N: V[:, 1] - (ys + SEAM * 2))(ys), lambda V, N: above(V) & side(V, N)))
    C.append(("tg_a", lambda V, N: V[:, 1] - CANOPY_TG, lambda V, N: (V[:, 2] > 0.95) & (V[:, 1] > 1.3)))
    C.append(("tg_b", lambda V, N: V[:, 1] - (CANOPY_TG + SEAM * 2), lambda V, N: (V[:, 2] > 0.95) & (V[:, 1] > 1.3)))
    # door seams below the canopy
    below = lambda V: V[:, 2] < belt(V[:, 1]) + 0.005
    C.append(("fd_seam", lambda V, N: V[:, 1] - FD_SEAM(V[:, 2]), lambda V, N: side(V, N) & below(V) & (V[:, 1] > -1.4) & (V[:, 1] < -0.7)))
    C.append(("split", lambda V, N: V[:, 1] - DOOR_SPLIT, lambda V, N: side(V, N) & below(V)))
    C.append(("rd_line", lambda V, N: V[:, 1] - RD_LINE(V[:, 2]), lambda V, N: side(V, N) & below(V) & (V[:, 2] > 0.78) & (V[:, 1] > 0.8)))
    C.append(("rd_arc", lambda V, N: RD_ARC_R - arch_dist(V, Y_AXLE_R), lambda V, N: side(V, N) & (V[:, 2] < 0.82) & (V[:, 2] > 0.25)))
    # hood, cowl halo
    C.append(("hood_side", lambda V, N: np.abs(V[:, 0]) - HOOD_X(V[:, 1]), lambda V, N: top(V, N) & (V[:, 1] < -1.25) & (V[:, 1] > -2.40)))
    C.append(("cowl", lambda V, N: V[:, 1] - COWL(np.abs(V[:, 0])), lambda V, N: top(V, N) & (V[:, 1] > -1.75) & (V[:, 1] < -1.1)))
    C.append(("cowl_h", lambda V, N: V[:, 1] - (COWL(np.abs(V[:, 0])) - HALO * 0.8), lambda V, N: top(V, N) & (V[:, 1] > -1.75) & (V[:, 1] < -1.1)))
    # tailgate, rear bumper
    C.append(("tg_side", lambda V, N: np.abs(V[:, 0]) - TG_X(V[:, 2]), lambda V, N: (V[:, 1] > 1.9) & (V[:, 2] > TG_BOTTOM - 0.05) & (V[:, 2] < 1.3)))
    C.append(("tg_bottom", lambda V, N: V[:, 2] - TG_BOTTOM, lambda V, N: V[:, 1] > 1.80))
    C.append(("rb_side", lambda V, N: V[:, 1] - RB_SIDE_Y, lambda V, N: side(V, N) & (V[:, 2] < TG_BOTTOM + 0.01) & (V[:, 1] > 1.7)))
    # front fascia
    C.append(("lb_top", lambda V, N: V[:, 2] - LIGHTBAR[1], lambda V, N: V[:, 1] < -1.80))
    C.append(("lb_bot", lambda V, N: V[:, 2] - LIGHTBAR[0], lambda V, N: V[:, 1] < -1.80))
    C.append(("fb_side", lambda V, N: V[:, 1] - FB_SIDE(V[:, 2]), lambda V, N: side(V, N) & (V[:, 2] < LIGHTBAR[1] + 0.01) & (V[:, 1] < -1.7)))
    z0, z1, a0, a1 = HEADLAMP
    for nm, f in (("hl_t", lambda V, N: V[:, 2] - z1), ("hl_b", lambda V, N: V[:, 2] - z0),
                  ("hl_i", lambda V, N: np.abs(V[:, 0]) - a0), ("hl_o", lambda V, N: np.abs(V[:, 0]) - a1)):
        C.append((nm, f, lambda V, N: (V[:, 1] < -1.9) & (V[:, 2] > 0.55) & (V[:, 2] < 0.70)))
    C.append(("intake_t", lambda V, N: V[:, 2] - 0.340, lambda V, N: V[:, 1] < -1.95))
    C.append(("intake_b", lambda V, N: V[:, 2] - 0.205, lambda V, N: V[:, 1] < -1.95))
    C.append(("intake_s", lambda V, N: np.abs(V[:, 0]) - 0.700, lambda V, N: (V[:, 1] < -1.95) & (V[:, 2] < 0.36)))
    # rear light bar
    C.append(("tail_t", lambda V, N: V[:, 2] - TAILBAR[1], lambda V, N: V[:, 1] > 1.95))
    C.append(("tail_b", lambda V, N: V[:, 2] - TAILBAR[0], lambda V, N: V[:, 1] > 1.95))
    C.append(("tail_e", lambda V, N: np.abs(V[:, 0]) - 0.640, lambda V, N: (V[:, 1] > 2.1) & (V[:, 2] > 1.0) & (V[:, 2] < 1.1)))
    C.append(("tail_s", lambda V, N: V[:, 1] - 2.040, lambda V, N: side(V, N) & (V[:, 2] > 1.0) & (V[:, 2] < 1.1)))
    C.append(("face_b", lambda V, N: V[:, 2] - FACE_BAND, lambda V, N: (V[:, 1] < -1.85)))
    C.append(("face_s", lambda V, N: np.abs(V[:, 0]) - 0.845, lambda V, N: (V[:, 1] < -1.85) & (V[:, 2] > 0.57) & (V[:, 2] < 0.73)))
    C.append(("tail_band", lambda V, N: V[:, 2] - TAIL_BAND, lambda V, N: (V[:, 1] > 2.0)))
    C.append(("tail_bs", lambda V, N: np.abs(V[:, 0]) - 0.880, lambda V, N: (V[:, 1] > 2.0) & (V[:, 2] > 0.90) & (V[:, 2] < 1.08)))
    C.append(("lower", lambda V, N: V[:, 2] - LOWER_ACCENT, lambda V, N: (np.abs(V[:, 0]) > 0.45) & (V[:, 1] > Y_AXLE_F) & (V[:, 1] < Y_AXLE_R)))
    C.append(("diffuser", lambda V, N: V[:, 2] - 0.380, lambda V, N: V[:, 1] > 2.0))
    C.append(("diff_s", lambda V, N: np.abs(V[:, 0]) - 0.800, lambda V, N: (V[:, 1] > 2.0) & (V[:, 2] < 0.41)))
    return C


def classify(c, n):
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    ax = np.abs(x)
    L = np.where(x >= 0, "_L", "_R")
    la = np.abs(n[:, 0])
    bl = belt(y)
    lab = np.full(len(c), "body", dtype=object)

    def put(mask, name, sided=False):
        if not mask.any():
            return
        lab[mask] = np.char.add(name, L[mask]).astype(object) if sided else name

    sidez = ax > 0.45
    between = (y > Y_AXLE_F + 0.30) & (y < Y_AXLE_R - 0.30)
    hole = sidez & (z < 0.95) & ((arch_dist(c, Y_AXLE_F) < ARCH_RADIUS) | (arch_dist(c, Y_AXLE_R) < ARCH_RADIUS))
    canopy_y = (y > -1.60) & (y < 2.35)
    above = z > bl

    # paint panels
    put((y < -1.70) & (z < LIGHTBAR[1]) & (y < FB_SIDE(z)), "bumper_F")
    hood = (n[:, 2] > 0.25) & (y < COWL(ax) - HALO * 0.8) & (ax < HOOD_X(y)) & (z >= LIGHTBAR[1] - 1e-4) & (y > -2.45)
    put(hood, "hood")
    put(sidez & (y < FD_SEAM(z)) & (y >= FB_SIDE(z) - 1e-4) & ~hood & (z < bl) & (y > -2.4), "fender", True)
    put(sidez & between & (z >= DOOR_BOTTOM) & (z < bl) & (y >= FD_SEAM(z)) & (y < DOOR_SPLIT), "door_F", True)
    behind_rd = ((z >= 0.78) & (y > RD_LINE(z))) | ((z < 0.82) & (arch_dist(c, Y_AXLE_R) < RD_ARC_R))
    put(sidez & (z >= DOOR_BOTTOM) & (z < bl) & (y >= DOOR_SPLIT) & ~behind_rd & (y < 1.45), "door_R", True)
    tg_face = (y > 1.9) & (z >= TG_BOTTOM) & (ax < TG_X(z)) & ~above
    put(tg_face, "tailgate")
    put((z < TG_BOTTOM) & (y > RB_SIDE_Y) & ~hole, "bumper_R")

    # canopy: everything above the canopy line is glass, split by black seams
    glass = above & canopy_y
    side_glass = glass & sidez & (la >= LAT_SIDE)
    lat_seam = glass & (la >= LAT_SIDE - 0.035) & (la < LAT_SIDE)
    roof_glass = glass & ~side_glass & ~lat_seam
    s0, s1 = GLASS_SPLITS
    seam_y = side_glass & (((y >= s0) & (y < s0 + 2 * SEAM)) | ((y >= s1) & (y < s1 + 2 * SEAM)))
    tg_seam = glass & (y >= CANOPY_TG) & (y < CANOPY_TG + 2 * SEAM)
    in_tg = glass & (y >= CANOPY_TG + 2 * SEAM)
    put(roof_glass & (y < CANOPY_TG), "glass_canopy")
    put(roof_glass & in_tg, "glass_tailgate")
    put(side_glass & (y < s0), "glass_door_F", True)
    put(side_glass & (y >= s0 + 2 * SEAM) & (y < s1), "glass_door_R", True)
    put(side_glass & (y >= s1 + 2 * SEAM) & (y < CANOPY_TG), "glass_quarter", True)
    put(side_glass & in_tg, "glass_tailgate")
    put(lat_seam & (y < s0), "seam_door_F", True)
    put(lat_seam & (y >= s0) & (y < s1), "seam_door_R", True)
    put(lat_seam & (y >= s1), "seam_body")
    put(seam_y & (y < s1), "seam_door_R", True)
    put(seam_y & (y >= s1), "seam_body")
    put(tg_seam, "seam_body")

    # autonomy halo just below the canopy line (and across the cowl)
    in_doors = (y >= FD_SEAM(z)) & (y < 1.45) & ~behind_rd
    halo = sidez & canopy_y & (z < bl) & (z >= bl - HALO)
    put(halo & (y >= FD_SEAM(z)) & (y < DOOR_SPLIT), "halo_door_F", True)
    put(halo & (y >= DOOR_SPLIT) & ~behind_rd & (y < 1.45), "halo_door_R", True)
    put(halo & ~in_doors, "halo_body")
    cowl_halo = (n[:, 2] > 0.25) & (y >= COWL(ax) - HALO * 0.8) & (y < COWL(ax)) & (ax < 0.78)
    put(cowl_halo, "halo_body")

    # black trims
    put(sidez & between & (z < ROCKER_TOP), "trim_rocker", True)
    put((y < -1.95) & (z > 0.205) & (z < 0.340) & (ax < 0.700), "trim_intake")
    put((y > 2.0) & (z < 0.380) & (ax < 0.800), "trim_diffuser")

    # lamps
    lb = (y < -1.80) & (z > LIGHTBAR[0]) & (z < LIGHTBAR[1])
    put(lb & (ax < 0.640), "lamp_drl")
    put(lb & (ax >= 0.640), "lamp_signal_F", True)
    z0, z1, a0, a1 = HEADLAMP
    put((y < -1.9) & (z > z0) & (z < z1) & (ax > a0) & (ax < a1), "lamp_head", True)
    tail = (z > TAILBAR[0]) & (z < TAILBAR[1]) & (y > 2.04)
    put(tail & (ax < 0.640), "lamp_tailbar")
    put(tail & (ax >= 0.640), "lamp_tail", True)

    # gloss-black bands painted onto whatever panel they cross (same mesh, black material)
    paint = np.isin(lab, PAINT_LABELS)
    face = (y < -1.85) & (z >= FACE_BAND) & (z < LIGHTBAR[0]) & (ax < 0.845)
    tailb = (y > 2.0) & (z >= TAIL_BAND) & (z < TAILBAR[0]) & (ax < 0.880)
    lower = sidez & (y > Y_AXLE_F) & (y < Y_AXLE_R) & (z < LOWER_ACCENT)
    black = paint & (face | tailb | lower)
    if black.any():
        lab[black] = np.char.add(lab[black].astype(str), "_black").astype(object)

    under = (n[:, 2] < -0.55) & (z < 0.40)
    lab[under & np.isin(lab, ["body", "bumper_F", "bumper_R", "body_black", "bumper_F_black", "bumper_R_black"])] = "trim_under"
    lab[hole] = "hole"
    return lab


PAINT = "rtx_paint"
BLACK = "rtx_black_gloss"
SATIN = "rtx_black_satin"
GLASS = "rtx_glass"


def panel_specs():
    """label -> (mesh name, material, gap, flange, offset)."""
    S = {}
    for side in ("L", "R"):
        s = side.lower()
        S[f"fender_{side}"] = (f"rtx_fender_{s}", PAINT, 0.002, 0.008, 0.0)
        S[f"door_F_{side}"] = (f"rtx_door_f{s}", PAINT, 0.002, 0.010, 0.0)
        S[f"door_R_{side}"] = (f"rtx_door_r{s}", PAINT, 0.002, 0.010, 0.0)
        S[f"glass_door_F_{side}"] = (f"rtx_glass_door_f{s}", GLASS, 0.0, 0.0, -0.003)
        S[f"glass_door_R_{side}"] = (f"rtx_glass_door_r{s}", GLASS, 0.0, 0.0, -0.003)
        S[f"glass_quarter_{side}"] = (f"rtx_glass_quarter_{s}", GLASS, 0.0, 0.0, -0.003)
        S[f"seam_door_F_{side}"] = (f"rtx_door_f{s}", BLACK, 0.0, 0.0, 0.0)
        S[f"seam_door_R_{side}"] = (f"rtx_door_r{s}", BLACK, 0.0, 0.0, 0.0)
        S[f"halo_door_F_{side}"] = (f"rtx_door_f{s}", "rtx_halo", 0.0, 0.0, 0.0)
        S[f"halo_door_R_{side}"] = (f"rtx_door_r{s}", "rtx_halo", 0.0, 0.0, 0.0)
        S[f"trim_rocker_{side}"] = (f"rtx_rocker_{s}", BLACK, 0.002, 0.006, 0.0)
        S[f"lamp_signal_F_{side}"] = ("rtx_bumper_f", f"rtx_signal_f{s}", 0.0, 0.0, 0.0)
        S[f"lamp_head_{side}"] = ("rtx_bumper_f", "rtx_headlight", 0.0, 0.0, -0.004)
        S[f"lamp_tail_{side}"] = (f"rtx_taillight_{s}", f"rtx_tail_{s}", 0.0, 0.0, 0.0)
    S.update({
        "body": ("rtx_body", PAINT, 0.002, 0.008, 0.0),
        "hood": ("rtx_hood", PAINT, 0.002, 0.010, 0.0),
        "bumper_F": ("rtx_bumper_f", PAINT, 0.002, 0.008, 0.0),
        "bumper_R": ("rtx_bumper_r", PAINT, 0.002, 0.008, 0.0),
        "tailgate": ("rtx_tailgate", PAINT, 0.002, 0.010, 0.0),
        "glass_canopy": ("rtx_canopy", GLASS, 0.0, 0.0, -0.002),
        "glass_tailgate": ("rtx_tailgate", GLASS, 0.0, 0.0, -0.002),
        "seam_body": ("rtx_body", BLACK, 0.0, 0.0, 0.0),
        "halo_body": ("rtx_body", "rtx_halo", 0.0, 0.0, 0.0),
        "trim_intake": ("rtx_bumper_f", SATIN, 0.0, 0.0, -0.012),
        "trim_diffuser": ("rtx_bumper_r", SATIN, 0.0, 0.0, 0.0),
        "trim_under": ("rtx_body", "rtx_underbody", 0.0, 0.0, 0.0),
        "lamp_drl": ("rtx_bumper_f", "rtx_drl", 0.0, 0.0, 0.0),
        "lamp_tailbar": ("rtx_tailgate", "rtx_tailbar", 0.0, 0.0, 0.0),
    })
    for lab in PAINT_LABELS:
        mesh = S[lab][0]
        S[lab + "_black"] = (mesh, BLACK, 0.002, 0.006, 0.0)
    return S
