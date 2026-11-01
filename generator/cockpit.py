"""Driver's cockpit and cabin details for the Lumen (Frontier Studios).

Storm, 2026-10-31: "make sure the car has a fully decorated interior and steering wheel".

The steering wheel and the pedals are props: BeamNG turns them with electrics.steering,
.throttle and .brake, so the robotaxi visibly steers and works its pedals while the AI drives.
Prop meshes are built around their own pivot in the prop frame that BeamNG builds from the
strw / strwx / strwy nodes (see frame_nodes and jbeam_gen.interior_part):
    local X = vehicle left, local Y = up in the wheel plane, local Z = down the column
    towards the dash.
Everything else is built in vehicle space (x left, y back, z up) and rides on the body like
the rest of the cabin.
"""
import numpy as np

import details
import meshkit
import shape

# Wheel centre (driver sits at x 0.38). 2026-11-01: z 0.950 -> 0.920 so the rim top sits about 7
# degrees below the driver's eye line instead of 3.5 (the view-blockage analysis, out/interior).
WHEEL_C = np.array([0.380, -0.600, 0.920])
COLUMN_DEG = 22.0                              # column angle below horizontal
WHEEL_RX, WHEEL_RY = 0.180, 0.172              # rim centreline half-sizes (a soft squircle, flat bottom)
WHEEL_FLAT = 0.80                              # the flat bottom sits at this share of RY below the centre
RIM_AX, RIM_RAD = 0.0150, 0.0175               # rim cross-section semi-axes: along the column, radial
PEDAL_PIVOTS = {"gas": (0.290, -1.020, 0.640), "brake": (0.445, -1.020, 0.650)}

_TH = np.radians(COLUMN_DEG)
AX_X = np.array([1.0, 0.0, 0.0])
AX_Z = np.array([0.0, -np.cos(_TH), -np.sin(_TH)])     # down the column, towards the dash
AX_Y = np.cross(AX_Z, AX_X)                             # up in the wheel plane
FRAME = np.stack([AX_X, AX_Y, AX_Z], axis=1)            # vehicle = FRAME @ local

LEATHER, METAL, GLOSS, DARK = "rtx_leather", "rtx_metal_trim", "rtx_black_gloss", "rtx_interior_dark"


def frame_nodes():
    """Reference nodes for the props: centre, +X and +Y of the wheel frame (vehicle space)."""
    return {"strw": WHEEL_C, "strwx": WHEEL_C + 0.12 * AX_X, "strwy": WHEEL_C + 0.12 * AX_Y}


# ------------------------------------------------------------------------------------------
# small builders
# ------------------------------------------------------------------------------------------
def _orient(P, N, F):
    """Wind every triangle so its geometric normal agrees with the vertex normals."""
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    vn = N[F].sum(1)
    F = F.copy()
    flip = (fn * vn).sum(1) < 0
    F[flip] = F[flip][:, ::-1]
    return F


def _solid(material, center, axes, R=None, e1=0.3, e2=0.3, nu=32, nv=16):
    P, N, F = details.superellipsoid((0, 0, 0), axes, e1=e1, e2=e2, nu=nu, nv=nv)
    R = np.eye(3) if R is None else np.asarray(R, float)
    m = meshkit.Mesh("part")
    m.add(P @ R.T + np.asarray(center, float), meshkit.normalize(N @ R.T), np.zeros((len(P), 2)), F, material)
    return m


def _tube(path, radius, material, nv=14, closed=True, up=(0.0, 0.0, 1.0), materials=None):
    """Tube around a polyline. radius: scalar or per point. materials: optional per-segment
    list (len = number of path segments) to split the faces between materials."""
    nv = meshkit.lod(nv, 6)
    path = np.asarray(path, float)
    n = len(path)
    r = np.broadcast_to(np.asarray(radius, float), (n,)).copy()
    T = (np.roll(path, -1, 0) - np.roll(path, 1, 0)) if closed else np.gradient(path, axis=0)
    T = meshkit.normalize(T)
    ref = np.asarray(up, float)
    Nn = ref - T * (T @ ref)[:, None]
    weak = np.linalg.norm(Nn, axis=1) < 1e-6
    if weak.any():
        alt = np.array([0.0, 1.0, 0.0])
        Nn[weak] = alt - T[weak] * (T[weak] @ alt)[:, None]
    Nn = meshkit.normalize(Nn)
    B = np.cross(T, Nn)
    ang = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    ring = np.cos(ang)[None, :, None] * Nn[:, None, :] + np.sin(ang)[None, :, None] * B[:, None, :]
    P = (path[:, None, :] + ring * r[:, None, None]).reshape(-1, 3)
    N = ring.reshape(-1, 3)
    F = _orient(P, N, meshkit.grid_faces(n, nv, wrap_cols=True, wrap_rows=closed))
    m = meshkit.Mesh("tube")
    uv = np.zeros((len(P), 2))
    if materials is None:
        m.add(P, N, uv, F, material)
    else:
        seg = F.min(1) // nv                       # row (path segment) of each face
        if closed:
            seg = np.where(F.max(1) // nv - F.min(1) // nv > 1, n - 1, seg)
        mats = np.asarray(materials)
        for mat in np.unique(mats):
            sel = mats[seg] == mat
            if sel.any():
                m.add(P, N, uv, F[sel], str(mat))
    return m


def _rim_tube(path, ax, rad, nv=20, ntile=None):
    """Closed tube with an elliptical cross-section (ax along the wheel axis, rad radial) and real
    UVs: u along the path (ntile whole tiles, with a duplicated seam row so the texture does not
    smear across the last quad), v around the section (0 = the side facing the column, 0.5 =
    facing the driver, 0.75 = the inner edge). ax and rad are scalars or per point."""
    nv = meshkit.lod(nv, 6)
    path = np.asarray(path, float)
    n = len(path)
    ax = np.broadcast_to(np.asarray(ax, float), (n,))
    rad = np.broadcast_to(np.asarray(rad, float), (n,))
    T = meshkit.normalize(np.roll(path, -1, 0) - np.roll(path, 1, 0))
    Nn = np.tile([0.0, 0.0, 1.0], (n, 1))                    # the wheel axis (path lies in its XY plane)
    B = np.cross(T, Nn)                                      # outward radial
    ang = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    ang = np.append(ang, 2 * np.pi)                          # duplicated seam column
    ca, sa = np.cos(ang), np.sin(ang)
    ring = ca[None, :, None] * Nn[:, None, :] * ax[:, None, None] + sa[None, :, None] * B[:, None, :] * rad[:, None, None]
    P = path[:, None, :] + ring                              # (n, nv+1, 3)
    # outward normal of an ellipse: scale by the inverse semi-axes
    Nrm = ca[None, :, None] * Nn[:, None, :] / ax[:, None, None] + sa[None, :, None] * B[:, None, :] / rad[:, None, None]
    Nrm = Nrm / np.linalg.norm(Nrm, axis=2, keepdims=True)
    seg = np.linalg.norm(np.roll(path, -1, 0) - path, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])              # n + 1 values, the last is the full length
    L = s[-1]
    nt = ntile or max(1, int(round(L / 0.12)))
    u = s / L * nt
    P = np.concatenate([P, P[:1]], axis=0)                   # duplicated seam row
    Nrm = np.concatenate([Nrm, Nrm[:1]], axis=0)
    UV = np.stack(np.meshgrid(u, np.linspace(0, 1, nv + 1), indexing="ij"), axis=-1).reshape(-1, 2)
    Pf, Nf = P.reshape(-1, 3), Nrm.reshape(-1, 3)
    F = _orient(Pf, Nf, meshkit.grid_faces(n + 1, nv + 1))
    m = meshkit.Mesh("rim")
    m.add(Pf, Nf, UV, F, "rtx_leather_stitch")
    return m


def _circle(center, normal, radius, n=40):
    """Points on a circle in the plane with the given normal."""
    normal = meshkit.normalize(np.asarray(normal, float))
    a = np.cross(normal, [0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.cross(normal, [1.0, 0.0, 0.0])
    a = meshkit.normalize(a)
    b = np.cross(normal, a)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.asarray(center, float) + radius * (np.cos(t)[:, None] * a + np.sin(t)[:, None] * b)


def _roof_in(x, y):
    """Height of the inside of the glass roof at (x, y)."""
    return float(shape.top_z(x, y)) - 0.012


# ------------------------------------------------------------------------------------------
# props (built around their pivots, in the prop frame)
# ------------------------------------------------------------------------------------------
def _rim_centreline(n=168):
    """The rim's centreline: a soft squircle with a flat bottom (smooth max against a floor)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    e = 2.0 / 2.6
    x = WHEEL_RX * np.sign(c) * np.abs(c) ** e
    y = WHEEL_RY * np.sign(s) * np.abs(s) ** e
    lo, k = -WHEEL_FLAT * WHEEL_RY, 0.010
    d = y - lo
    y = lo + 0.5 * (d + np.sqrt(d * d + k * k))
    return t, np.c_[x, y, np.zeros_like(x)]


def _rim_frame(path, i):
    """Tangent, outward radial and axis direction of the rim at point i."""
    T = meshkit.normalize(path[(i + 1) % len(path)] - path[i - 1])
    Z = np.array([0.0, 0.0, 1.0])
    B = np.cross(T, Z)
    return T, B, Z


def steering_wheel(name="rtx_steerwheel"):
    """Flat-bottom sport wheel (2026-11-01 redo, Storm: "the steering wheel needs to look much
    nicer"): an oval-section leather rim, fuller at the grips and slimmer top and bottom, with
    stitching along the inner and outer seams (rtx_leather_stitch has the texture), thumb rests,
    two brushed-metal spokes with control pads and buttons, and a dished hub with a Frontier
    badge ring. Prop frame: local -Z faces the driver, +Z is down the column."""
    m = meshkit.Mesh(name)
    t, path = _rim_centreline()
    g = 1.0 + 0.15 * np.cos(t) ** 2 - 0.05 * np.sin(t) ** 2
    m.add_mesh(_rim_tube(path, RIM_AX * g, RIM_RAD * g, nv=20))
    # thumb rests on the driver-facing side, at 2 and 10 o'clock
    for deg in (35.0, 145.0):
        i = int(np.argmin(np.abs((np.degrees(t) - deg + 180) % 360 - 180)))
        T, B, Z = _rim_frame(path, i)
        R = np.stack([B, T, Z], axis=1)
        c = path[i] - B * 0.0045 - Z * (RIM_AX * g[i] + 0.0006)
        m.add_mesh(_solid(LEATHER, c, (0.0120, 0.0300, 0.0042), R=R, e1=0.5, e2=0.5, nu=24, nv=10))
    # a turquoise centre mark at 12 o'clock
    m.add_mesh(_solid("rtx_ambient", (0.0, path[np.argmax(path[:, 1])][1], -RIM_AX * g[np.argmax(path[:, 1])] - 0.0004),
                      (0.0075, 0.0021, 0.0007), e1=0.4, e2=0.4, nu=16, nv=6))
    # two spokes from the hub to the rim at 3 and 9 o'clock, drooping about 15 degrees
    ang = np.radians(-15.2)
    for sx in (1.0, -1.0):
        a_ = ang if sx > 0 else np.pi - ang
        R = np.array([[np.cos(a_), -np.sin(a_), 0.0], [np.sin(a_), np.cos(a_), 0.0], [0.0, 0.0, 1.0]])
        ctr = np.array([sx * 0.1195, -0.020, 0.010])
        m.add_mesh(_solid(METAL, ctr, (0.0655, 0.0150, 0.0058), R=R, e1=0.30, e2=0.35, nu=32, nv=12))
        # control pad on the driver-facing side, and two buttons on it
        m.add_mesh(_solid(GLOSS, ctr + np.array([0.0, 0.0, -0.0034]), (0.0390, 0.0108, 0.0016), R=R, e1=0.25, e2=0.25, nu=24, nv=8))
        for k in (-0.016, 0.016):
            along = R @ np.array([k, 0.0, 0.0])
            m.add_mesh(_solid(METAL, ctr + along + np.array([0.0, 0.0, -0.0050]), (0.0072, 0.0060, 0.0016), R=R, e1=0.5, e2=0.5, nu=14, nv=6))
            m.add_mesh(_solid("rtx_ambient", ctr + along + np.array([0.0, 0.0, -0.0063]), (0.0022, 0.0022, 0.0004), e1=1.0, e2=1.0, nu=10, nv=4))
    # hub: dished towards the dash, with a metal ring and a badge facing the driver
    m.add_mesh(_solid(GLOSS, (0.0, 0.0, 0.020), (0.064, 0.052, 0.022), e1=0.25, e2=0.25, nu=36, nv=16))
    m.add_mesh(_tube(_circle((0.0, 0.0, -0.0026), (0, 0, 1), 0.037), 0.0019, METAL, nv=10, closed=True, up=(0, 0, 1)))
    m.add_mesh(_solid(METAL, (0.0, 0.0, -0.0032), (0.0270, 0.0270, 0.0012), e1=1.0, e2=1.0, nu=36, nv=8))
    m.add_mesh(_solid("rtx_ambient", (0.0, 0.0, -0.0043), (0.0105, 0.0105, 0.0006), e1=1.0, e2=1.0, nu=24, nv=6))
    return m


def pedal(name, kind):
    """Pedal hanging from its pivot (origin) along -Y, pad facing the driver (-Z)."""
    m = meshkit.Mesh(name)
    m.add_mesh(_solid(DARK, (0.0, -0.085, 0.0), (0.009, 0.090, 0.007), e1=0.3, e2=0.3, nu=16, nv=8))
    if kind == "gas":
        pad_c, pad_a = (0.0, -0.178, -0.010), (0.030, 0.052, 0.007)
    else:
        pad_c, pad_a = (0.0, -0.166, -0.010), (0.049, 0.035, 0.008)
    m.add_mesh(_solid(METAL, pad_c, pad_a, e1=0.25, e2=0.25, nu=24, nv=10))
    for k in (-1, 0, 1):   # rubber grip strips
        c = (pad_c[0], pad_c[1] + k * pad_a[1] * 0.55, pad_c[2] - pad_a[2] - 0.0008)
        m.add_mesh(_solid("rtx_black_satin", c, (pad_a[0] * 0.82, 0.0035, 0.0012), e1=0.3, e2=0.3, nu=12, nv=6))
    return m


# ------------------------------------------------------------------------------------------
# static details (vehicle space)
# ------------------------------------------------------------------------------------------
def column(m):
    """Column shroud from the wheel into the dash, and two slim stalks."""
    m.add_mesh(_solid(DARK, WHEEL_C + AX_Z * 0.170, (0.044, 0.040, 0.118), R=FRAME, e1=0.30, e2=0.35, nu=28, nv=14))
    for sx in (1.0, -1.0):
        c = WHEEL_C + AX_Z * 0.075 + AX_X * sx * 0.095 + AX_Y * 0.010
        m.add_mesh(_solid(GLOSS, c, (0.052, 0.0065, 0.0065), R=FRAME, e1=0.4, e2=0.4, nu=16, nv=8))


def vents(m):
    """Three slim air vents along the back edge of the dash wing."""
    for xc, half_w in ((0.640, 0.075), (-0.640, 0.075), (0.0, 0.140)):
        m.add_mesh(_solid(DARK, (xc, -0.781, 0.836), (half_w, 0.006, 0.017), e1=0.2, e2=0.2, nu=24, nv=10))
        for dz in (-0.0075, 0.0, 0.0075):
            m.add_mesh(_solid(METAL, (xc, -0.7745, 0.836 + dz), (half_w * 0.94, 0.0028, 0.0014), e1=0.3, e2=0.3, nu=16, nv=6))


def header_y():
    """Header position: where the windshield flattens to about 20 degrees. The Lumen's glass
    sweeps from 28 degrees at the cowl to 8 at the crown (measured from shape.top_z), so this
    lands about 0.28 m ahead of and 0.30 m above the front occupants' eyes."""
    ys = np.linspace(-1.30, 0.0, 131)
    zs = np.array([float(shape.top_z(0.0, y)) for y in ys])
    slope = np.gradient(zs, ys)
    flat = np.where(np.abs(slope) < 0.36)[0]
    return float(ys[flat[0]]) if len(flat) else -0.50


def roof_structure(m):
    """Header bar at the top of the windshield (visors, digital mirror) and a roof spine with
    reading lights under the glass canopy."""
    yh = header_y()
    xs = np.linspace(-0.60, 0.60, 25)
    m.add_mesh(_tube([(x, yh, _roof_in(x, yh) - 0.022) for x in xs], 0.015, GLOSS, nv=12, closed=False, up=(0, 1, 0)))
    ys = np.linspace(yh, 1.38, 40)
    m.add_mesh(_tube([(0.0, y, _roof_in(0.0, y) - 0.020) for y in ys], 0.012, GLOSS, nv=12, closed=False, up=(1, 0, 0)))
    for yp in (-0.25, 0.72):     # reading light pods: two lenses each
        z = _roof_in(0.0, yp) - 0.030
        m.add_mesh(_solid(GLOSS, (0.0, yp, z), (0.060, 0.036, 0.012), e1=0.3, e2=0.3, nu=24, nv=10))
        for sx in (1.0, -1.0):
            m.add_mesh(_solid("rtx_drl", (sx * 0.028, yp, z - 0.011), (0.017, 0.017, 0.0022), e1=1.0, e2=1.0, nu=18, nv=8))
    # (2026-11-01: the sun visors are gone: they sat on the glass in front of the front occupants,
    # and a black-glass canopy has nothing to shade.)
    # digital rear-view mirror on a short stem
    zm = _roof_in(0.0, yh) - 0.090
    m.add_mesh(_solid(GLOSS, (0.0, yh + 0.020, zm + 0.045), (0.010, 0.010, 0.030), e1=0.5, e2=0.5, nu=12, nv=8))
    m.add_mesh(_solid(GLOSS, (0.0, yh + 0.030, zm), (0.118, 0.013, 0.036), e1=0.2, e2=0.2, nu=28, nv=10))
    m.add_mesh(_solid("rtx_screen_off", (0.0, yh + 0.0435, zm), (0.108, 0.0015, 0.029), e1=0.15, e2=0.15, nu=24, nv=8))


def _rotx(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _surf_x(door_mesh, y, z, default):
    """|x| of the door card's surface at (y, z), from the card vertices that interior.door_trim kept."""
    cv = getattr(door_mesh, "card_v", None)
    if cv is None:
        return default
    for r in (0.05, 0.10, 0.20):
        sel = (np.abs(cv[:, 2] - z) < r * 0.8) & (np.abs(cv[:, 1] - y) < r)
        if sel.sum() >= 3:
            return float(np.median(np.abs(cv[sel, 0])))
    return default


def door_details(door_mesh):
    """Everything on a door card (call after interior.door_trim): speaker, pull handle with an inside
    release lever, a metal trim line between the fabric and the dark roll, a switch pod on the
    armrest (windows and locks), and a map pocket in the lower rear of the card."""
    if "rtx_paint" not in door_mesh.groups:
        return
    used = np.unique(door_mesh.groups["rtx_paint"])
    ys, xs = door_mesh.v[used][:, 1], door_mesh.v[used][:, 0]
    y0, y1 = np.percentile(ys, 12), np.percentile(ys, 88)
    side = 1.0 if np.median(xs) > 0 else -1.0
    xin = np.median(np.abs(xs)) - 0.080          # the card surface (door_trim pushes the skin in 8 cm)
    front = door_mesh.name.endswith(("fl", "fr"))
    ymid = 0.5 * (y0 + y1)
    yc, zc = y0 + 0.30 * (y1 - y0), 0.435
    normal = (-side, 0.0, 0.0)
    door_mesh.add_mesh(_solid("rtx_speaker", (side * (xin - 0.004), yc, zc), (0.004, 0.066, 0.066), e1=1.0, e2=1.0, nu=32, nv=12))
    door_mesh.add_mesh(_tube(_circle((side * (xin - 0.006), yc, zc), normal, 0.069), 0.0045, METAL, nv=10, closed=True, up=normal))
    # pull handle and, above it, the release lever on a dark base plate
    door_mesh.add_mesh(_solid(METAL, (side * (xin - 0.050), ymid - 0.03, 0.730), (0.011, 0.105, 0.012), e1=0.4, e2=0.4, nu=20, nv=8))
    sx = _surf_x(door_mesh, ymid + 0.05, 0.790, xin)
    door_mesh.add_mesh(_solid(DARK, (side * (sx - 0.004), ymid + 0.05, 0.790), (0.006, 0.060, 0.017), e1=0.3, e2=0.3, nu=20, nv=8))
    door_mesh.add_mesh(_solid(METAL, (side * (sx - 0.014), ymid + 0.045, 0.790), (0.008, 0.050, 0.0065), e1=0.4, e2=0.4, nu=20, nv=8))
    # metal trim line along the top of the upper fabric insert
    yy = np.linspace(y0 + 0.03, y1 - 0.03, 18)
    path = [(side * (_surf_x(door_mesh, y, 0.862, xin) - 0.003), y, 0.862) for y in yy]
    door_mesh.add_mesh(_tube(path, 0.0036, METAL, nv=8, closed=False, up=(1, 0, 0)))
    # switch pod on the armrest (front: windows and locks; rear: windows)
    ay = y0 + (0.15 if front else 0.12) * (y1 - y0)
    ax = side * (xin - 0.035)
    n_keys = 4 if front else 2
    hl = 0.016 * n_keys + 0.014
    door_mesh.add_mesh(_solid(GLOSS, (ax, ay + hl - 0.014, 0.6885), (0.030, hl, 0.0075), e1=0.25, e2=0.25, nu=24, nv=8))
    for k in range(n_keys):
        ky = ay + 0.016 * k + 0.008
        door_mesh.add_mesh(_solid(METAL, (ax, ky, 0.6985), (0.0105, 0.0065, 0.0030), e1=0.35, e2=0.35, nu=14, nv=6))
        door_mesh.add_mesh(_solid("rtx_ambient", (ax, ky, 0.7018), (0.0030, 0.0030, 0.0004), e1=1.0, e2=1.0, nu=10, nv=4))
    # map pocket in the rear half of the lower card, behind the speaker
    py_ = y0 + 0.73 * (y1 - y0)
    sxp = _surf_x(door_mesh, py_, 0.480, xin)
    hlp = 0.20 * (y1 - y0)
    door_mesh.add_mesh(_solid(DARK, (side * (sxp - 0.016), py_, 0.480), (0.018, hlp, 0.042), e1=0.30, e2=0.30, nu=28, nv=10))
    door_mesh.add_mesh(_solid("rtx_black_satin", (side * (sxp - 0.0335), py_, 0.505), (0.0025, hlp * 0.92, 0.022), e1=0.3, e2=0.3, nu=24, nv=8))


def center_display(m):
    """Slim floating ride display between the front seats, on a stem and tilted up 20 degrees at
    the occupants, with a metal control bar (climate keys and two knobs) under it. It sits about 12
    to 23 degrees below the driver's eye line and well to the right, so it is out of the road view."""
    import interior
    R = _rotx(20.0)
    c0 = np.array([0.0, -0.845, 0.955])
    loc = lambda p: c0 + R @ np.asarray(p, float)         # display-frame point -> vehicle space
    m.add_mesh(_solid(GLOSS, c0, (0.176, 0.008, 0.062), R=R, e1=0.12, e2=0.12, nu=36, nv=10))
    scr = interior.screen_face(0.166, 0.053, "rtx_screen_dash").transformed(R, loc((0.0, 0.0085, 0.0)))
    m.add_mesh(scr)
    m.add_mesh(_solid(GLOSS, (0.0, -0.868, 0.893), (0.014, 0.010, 0.034), e1=0.5, e2=0.5, nu=14, nv=8))
    m.add_mesh(_solid(METAL, loc((0.0, 0.003, -0.078)), (0.176, 0.0085, 0.0135), R=R, e1=0.25, e2=0.25, nu=32, nv=8))
    Rk = R @ _rotx(-90.0)                                   # a superellipsoid's z axis -> toward the occupants
    for sx_ in (-0.148, 0.148):
        m.add_mesh(_solid(METAL, loc((sx_, 0.012, -0.078)), (0.0150, 0.0150, 0.0100), R=Rk, e1=0.45, e2=1.0, nu=28, nv=8))
        m.add_mesh(_solid("rtx_ambient", loc((sx_, 0.0225, -0.078)), (0.0020, 0.0020, 0.0004), R=Rk, e1=1.0, e2=1.0, nu=10, nv=4))
    for k in range(-2, 3):
        m.add_mesh(_solid(GLOSS, loc((k * 0.040, 0.0095, -0.078)), (0.0140, 0.0050, 0.0072), R=R, e1=0.35, e2=0.35, nu=14, nv=6))
        m.add_mesh(_solid("rtx_ambient", loc((k * 0.040, 0.0147, -0.078)), (0.0035, 0.0006, 0.0035), R=R, e1=1.0, e2=1.0, nu=10, nv=4))


def cowl_pad(m):
    """Dark scuttle top between the dash and the windshield base. The view analysis (out/interior)
    found a gap there: rays passing over the dash and under the glass base hit the hood's
    underside, so the cabin looked into the engine bay. The pad is a height field that rises from
    the dash top to the glass base and stays 3.5 cm under the outer skin everywhere."""
    xs = np.linspace(-0.86, 0.86, 45)
    ys = np.linspace(-1.50, -1.04, 21)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    # rises from the dash top and meets the glass lip (front edge y -1.43 at the centre, -1.31 at
    # x 0.8), staying 1.4 cm under the outer skin (the inner skins sit 1 cm and 3 mm below it)
    lin = 0.862 + np.clip((-1.04 - Y) / 0.36, 0.0, 1.0) ** 0.8 * 0.115
    Z = np.minimum(lin, shape.top_z(X, Y) - 0.014)
    P = np.stack([X, Y, Z], -1).reshape(-1, 3)
    F = meshkit.grid_faces(len(xs), len(ys))
    N = meshkit.vertex_normals(P, F)
    if N[:, 2].mean() < 0:
        F = F[:, ::-1]
        N = -N
    m.add(P, N, np.zeros((len(P), 2)), F, DARK)


def console_details(m):
    """Cupholders, a drive selector and a rear climate screen."""
    for yc, zc in ((-0.285, 0.6315), (0.860, 0.6715)):
        for sx in (1.0, -1.0):
            c = (sx * 0.047, yc, zc)
            m.add_mesh(_solid(DARK, c, (0.036, 0.036, 0.0025), e1=1.0, e2=1.0, nu=28, nv=8))
            m.add_mesh(_tube(_circle((c[0], c[1], zc + 0.001), (0, 0, 1), 0.038), 0.0035, METAL, nv=8, closed=True))
    m.add_mesh(_solid(METAL, (0.0, -0.520, 0.646), (0.026, 0.026, 0.016), e1=1.0, e2=0.25, nu=28, nv=10))
    m.add_mesh(_solid("rtx_ambient", (0.0, -0.520, 0.6625), (0.0265, 0.0265, 0.0016), e1=1.0, e2=1.0, nu=28, nv=8))
    tilt = np.radians(28)
    R = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    m.add_mesh(_solid(GLOSS, (0.0, 0.405, 0.705), (0.078, 0.006, 0.042), R=R, e1=0.2, e2=0.2, nu=24, nv=8))
    m.add_mesh(_solid("rtx_screen_rear", (0.0, 0.410, 0.708), (0.070, 0.0012, 0.035), R=R, e1=0.12, e2=0.12, nu=20, nv=6))


def floor_details(m):
    """Floor mats and seatbelt buckles."""
    for x in (0.38, -0.38):
        m.add_mesh(_solid("rtx_mat", (x, -0.78, 0.3285), (0.205, 0.235, 0.0035), e1=0.2, e2=0.15, nu=28, nv=6))
        m.add_mesh(_solid("rtx_mat", (x, 0.28, 0.3285), (0.205, 0.215, 0.0035), e1=0.2, e2=0.15, nu=28, nv=6))
    for xs, ys in ((0.380, -0.32), (-0.380, -0.32), (0.380, 0.70), (-0.380, 0.70)):
        inboard = -np.sign(xs)
        m.add_mesh(_solid(GLOSS, (xs + inboard * 0.262, ys + 0.10, 0.575), (0.010, 0.022, 0.034), e1=0.3, e2=0.3, nu=16, nv=8))


def cockpit(meshes):
    """Add the wheel, pedals and every cabin detail to the mesh set."""
    meshes["rtx_steerwheel"] = steering_wheel()
    meshes["rtx_pedal_gas"] = pedal("rtx_pedal_gas", "gas")
    meshes["rtx_pedal_brake"] = pedal("rtx_pedal_brake", "brake")
    column(meshes["rtx_dash"])
    center_display(meshes["rtx_dash"])
    cowl_pad(meshes["rtx_dash"])
    vents(meshes["rtx_dash"])
    roof_structure(meshes["rtx_dash"])
    console_details(meshes["rtx_consoles"])
    floor_details(meshes["rtx_floor"])
    for door in ("rtx_door_fl", "rtx_door_fr", "rtx_door_rl", "rtx_door_rr"):
        door_details(meshes[door])
    return meshes


# names of the meshes that are props (in the DAE, not flexbodies)
PROP_MESHES = ("rtx_steerwheel", "rtx_pedal_gas", "rtx_pedal_brake")
