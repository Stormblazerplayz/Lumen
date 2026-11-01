"""Lumen lounge cabin: no steering wheel (it drives itself).

Four sculpted lounge seats (bolsters, stitch channels, dark shells, floating
headrests), a slim floating dash carrying a pillar-to-pillar curved ride display,
screens on the backs of the front seats for the rear passengers, floating consoles,
door cards with armrests, carpeted floor, and turquoise ambient light strips.
Everything sits under the black glass canopy, so the cabin reads from outside too.

Package (Lumen, 3.0 m wheelbase): floor on the battery at z 0.30, front seats at
y -0.32, rear seats at y +0.70, dash at y -0.92.
"""
import numpy as np

import details
import meshkit

SEAT, STITCH, SHELL = "rtx_seat", "rtx_seat_stitch", "rtx_interior_dark"
FABRIC = "rtx_fabric"


def smoothstep(a, b, t):
    s = np.clip((np.asarray(t, float) - a) / (b - a), 0.0, 1.0)
    return s * s * (3 - 2 * s)


def rot_x(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def solid(name, material, center, axes, pitch=0.0, e1=0.35, e2=0.30, nu=40, nv=22):
    P, N, F = details.superellipsoid((0, 0, 0), axes, e1=e1, e2=e2, nu=nu, nv=nv)
    R = rot_x(pitch)
    m = meshkit.Mesh(name)
    m.add(P @ R.T + np.asarray(center, float), N @ R.T, np.zeros((len(P), 2)), F, material)
    return m


def pad(name, a, b, c, bolster=0.030, channels=6, waist=0.0, nu=72, nv=36):
    """One sculpted seat pad in local space: x across (half width a), y along the
    pad (half length b), z out of the sitting face (half thickness c). The +z face
    gets raised side bolsters, a soft centre dip and stitch channels across y;
    the -z face is the dark shell."""
    P, N, F = details.superellipsoid((0, 0, 0), (a, b, c), e1=0.22, e2=0.30, nu=nu, nv=nv)
    yb = (P[:, 1] / b + 1) / 2                       # 0..1 along the pad
    if waist:
        k = 1 - waist * smoothstep(0.62, 0.80, yb) * (1 - smoothstep(0.86, 0.98, yb))
        P[:, 0] *= k
    xa = np.abs(P[:, 0]) / a
    w = smoothstep(0.05, 0.85, P[:, 2] / c)          # only the sitting face moves
    raise_ = bolster * smoothstep(0.52, 0.84, xa)
    dip = -0.010 * (1 - smoothstep(0.30, 0.62, xa))
    ph = (yb * channels) % 1.0
    groove = -0.0045 * np.exp(-((ph - 0.5) / 0.05) ** 2) * (1 - smoothstep(0.42, 0.52, xa)) * smoothstep(0.08, 0.16, yb) * (1 - smoothstep(0.84, 0.92, yb))
    P[:, 2] += w * (raise_ + dip + groove)
    N = meshkit.vertex_normals(P, F)
    out = P[:, 2] > 0
    # materials per triangle: shell on the back, stitch in the grooves, seat elsewhere
    cz = P[F, 2].mean(1)
    cx = np.abs(P[F, 0]).mean(1) / a
    cph = ((yb[F].mean(1)) * channels) % 1.0
    in_groove = (np.abs(cph - 0.5) < 0.06) & (cx < 0.45) & (cz > 0.4 * c)
    # by the highest corner, not the centroid: both triangles of a grid quad touch both rings,
    # so the shell edge follows a ring cleanly (centroids split each quad and drew a white
    # sawtooth along every seat base, seen in game 2026-10-31)
    shell = P[F, 2].max(1) < -0.15 * c
    m = meshkit.Mesh(name)
    uv = np.zeros((len(P), 2))
    m.add(P, N, uv, F[shell], SHELL)
    m.add(P, N, uv, F[~shell & in_groove], STITCH)
    m.add(P, N, uv, F[~shell & ~in_groove], SEAT)
    return m


def lounge_seat(name, x, y, rear=False):
    """Cushion + reclined backrest + floating headrest + pedestal, at (x, y)."""
    m = meshkit.Mesh(name)
    # cushion: local z up, local y towards the back; front edge slightly raised
    cush = pad(name, 0.255, 0.255, 0.060, bolster=0.034, channels=5)
    m.add_mesh(cush.transformed(rot_x(-6), (x, y, 0.540)))
    # backrest: local z points forward (-y), local y up; recline 22 deg (rear 25). The FRONT seats
    # are a little slimmer, lower and with a narrower headrest than the rear ones, so passengers
    # behind them see more of the road (2026-11-01 view analysis: the front seats blocked 39 % of
    # the rear passengers' 100 x 50 degree view).
    rec = 25 if rear else 22
    ba, bb, bz = (0.250, 0.300, 0.880) if rear else (0.220, 0.272, 0.866)
    back = pad(name, ba, bb, 0.058, bolster=0.040, channels=7)
    R = rot_x(90 - rec)
    m.add_mesh(back.transformed(R, (x, y + 0.235 + 0.10, bz)))
    # floating headrest
    ha, hb, hz = (0.135, 0.085, 1.215) if rear else (0.098, 0.074, 1.176)
    hr = pad(name, ha, hb, 0.045, bolster=0.012, channels=0)
    m.add_mesh(hr.transformed(rot_x(90 - rec + 6), (x, y + 0.235 + 0.10 + 0.23 * np.sin(np.radians(rec)) + 0.03, hz)))
    # pedestal
    m.add_mesh(solid(name, SHELL, (x, y + 0.03, 0.410), (0.17, 0.20, 0.075), e1=0.25, e2=0.25))
    # turquoise accent along the pedestal sides
    for dx in (-0.172, 0.172):
        m.add_mesh(solid(name, "rtx_ambient", (x + dx, y + 0.03, 0.405), (0.003, 0.17, 0.004), e1=0.3, e2=0.3, nu=16, nv=8))
    return m


def seatback_screen(name, x, y, dz=0.014):
    """10-inch screen on the back of a front seat, facing the rear passengers (+y). dz is how far the
    front backrest was lowered (0.880 -> 0.866): the rear face at the screen's height moved back by
    dz * tan(recline), and the screen moves with it (it was buried in the pad after the seat change)."""
    m = meshkit.Mesh(name)
    rec = 22
    R = rot_x(-rec)
    c = np.array([x, y + 0.235 + 0.10 + 0.075 + dz * np.tan(np.radians(rec)), 0.930])
    m.add_mesh(solid(name, "rtx_black_gloss", (0, 0, 0), (0.118, 0.008, 0.074), e1=0.12, e2=0.12).transformed(R, c))
    m.add_mesh(screen_face(0.110, 0.055, "rtx_screen_rear").transformed(R, c + R @ np.array([0, 0.0105, 0])))
    return m


def screen_face(hw, hh, material, curve_r=None):
    """Screen quad facing +y with 0..1 UVs reading left-to-right for its viewer
    (whose right is -x); optionally curved around a vertical axis."""
    nu = 24 if curve_r else 2
    u = np.linspace(0, 1, nu)
    xs = (0.5 - u) * 2 * hw        # u=0 at +hw (viewer's left), u=1 at -hw (viewer's right)
    ys = np.zeros_like(xs)
    if curve_r:
        ang = xs / curve_r
        xs = curve_r * np.sin(ang)
        ys = -(curve_r - curve_r * np.cos(ang))   # ends wrap towards the viewer's front
    P, UV = [], []
    for zv, v in ((-hh, 0.0), (hh, 1.0)):
        for xi, yi, ui in zip(xs, ys, u):
            P.append((xi, -yi, zv))
            UV.append((ui, v))
    P = np.array(P)
    UV = np.array(UV)
    F = meshkit.grid_faces(2, nu)
    N = meshkit.vertex_normals(P, F)
    if (N[:, 1] < 0).mean() > 0.5:
        F = F[:, ::-1]
        N = -N
    m = meshkit.Mesh("screen")
    m.add(P, N, UV, F, material)
    return m


def clip_z(v, n, uv, faces, z0, z1):
    """The part of a triangle set with z0 <= z <= z1, with triangles cut exactly at the planes
    (positions, normals and UVs interpolated). Splitting by centroid left a sawtooth border along
    every zone line on the door cards. Returns (V, N, UV, F) with the original vertices first."""
    zt = v[faces][:, :, 2]
    full = ((zt >= z0) & (zt <= z1)).all(1)
    none = (zt.max(1) < z0) | (zt.min(1) > z1)
    V, N, UV, F = [v], [n], [uv], [faces[full]]
    base = len(v)
    for t in np.nonzero(~full & ~none)[0]:
        poly = [(v[i], n[i], uv[i]) for i in faces[t]]
        for plane, keep_ge in ((z0, True), (z1, False)):
            out = []
            for a, b in zip(poly, poly[1:] + poly[:1]):
                za, zb = a[0][2], b[0][2]
                ina = za >= plane if keep_ge else za <= plane
                inb = zb >= plane if keep_ge else zb <= plane
                if ina:
                    out.append(a)
                if ina != inb:
                    f = (plane - za) / (zb - za)
                    out.append(tuple(pa + (pb - pa) * f for pa, pb in zip(a, b)))
            poly = out
            if len(poly) < 3:
                break
        if len(poly) >= 3:
            idx = []
            for p in poly:
                V.append(p[0][None]); N.append(p[1][None]); UV.append(p[2][None])
                idx.append(base)
                base += 1
            for k in range(1, len(poly) - 1):
                F.append(np.array([[idx[0], idx[k], idx[k + 1]]]))
    return np.vstack(V), np.vstack(N), np.vstack(UV), np.vstack(F)


def split_z(mesh, zsplit, upper, lower):
    """Re-assign a single-material mesh's triangles by centroid height: upper above zsplit."""
    mat, faces = next(iter(mesh.groups.items()))
    zc = mesh.v[faces].mean(1)[:, 2]
    mesh.groups = {upper: faces[zc >= zsplit], lower: faces[zc < zsplit]}
    mesh.groups = {k: v for k, v in mesh.groups.items() if len(v)}
    return mesh


def dash():
    """Slim two-tone dash. 2026-11-01: the old pillar-to-pillar ride display stood across the
    driver's road view (the blockage analysis in out/interior measured it at 43 % of the forward
    view) and is gone. The dash is now a dark soft-touch top (no glare in the glass) over a light
    fabric fascia, split by a thin metal seam; the small floating ride display and the climate
    keys are added by cockpit.center_display(), low and between the front seats."""
    m = meshkit.Mesh("rtx_dash")
    wing = solid("w", "rtx_interior_trim", (0, -0.935, 0.830), (0.760, 0.150, 0.042), pitch=-4, e1=0.22, e2=0.18, nu=64, nv=18)
    m.add_mesh(split_z(wing, 0.815, SHELL, "rtx_interior_trim"))
    m.add_mesh(solid("w2", "rtx_interior_trim", (0, -1.000, 0.770), (0.740, 0.120, 0.050), e1=0.25, e2=0.2, nu=48, nv=14))
    m.add_mesh(solid("seam", "rtx_metal_trim", (0, -0.7815, 0.8135), (0.620, 0.0030, 0.0017), e1=0.3, e2=0.2, nu=48, nv=6))
    m.add_mesh(solid("amb", "rtx_ambient", (0, -0.792, 0.790), (0.740, 0.004, 0.004), e1=0.3, e2=0.2, nu=40, nv=8))
    # knee/footwell bulkhead
    m.add_mesh(solid("fw", "rtx_carpet", (0, -1.10, 0.52), (0.74, 0.05, 0.22), e1=0.25, e2=0.25))
    return m


def consoles():
    m = meshkit.Mesh("rtx_consoles")
    # front: floating bridge between the front seats
    m.add_mesh(solid("fc", "rtx_interior_trim", (0, -0.42, 0.600), (0.110, 0.42, 0.030), e1=0.2, e2=0.2))
    m.add_mesh(solid("fcl", SHELL, (0, -0.42, 0.470), (0.080, 0.36, 0.100), e1=0.25, e2=0.25))
    m.add_mesh(solid("fca", "rtx_ambient", (0, -0.42, 0.567), (0.112, 0.40, 0.003), e1=0.2, e2=0.2, nu=24, nv=8))
    m.add_mesh(solid("pad", "rtx_black_gloss", (0, -0.62, 0.632), (0.075, 0.13, 0.004), e1=0.15, e2=0.15))
    # rear: armrest console between the rear seats
    m.add_mesh(solid("rc", "rtx_interior_trim", (0, 0.66, 0.640), (0.105, 0.30, 0.030), e1=0.2, e2=0.2))
    m.add_mesh(solid("rcl", SHELL, (0, 0.66, 0.480), (0.085, 0.27, 0.140), e1=0.25, e2=0.25))
    m.add_mesh(solid("rca", "rtx_ambient", (0, 0.66, 0.607), (0.107, 0.28, 0.003), e1=0.2, e2=0.2, nu=24, nv=8))
    return m


def floor():
    m = meshkit.Mesh("rtx_floor")
    m.add_mesh(solid("f", "rtx_carpet", (0, -0.05, 0.300), (0.760, 1.10, 0.025), e1=0.15, e2=0.15))
    m.add_mesh(solid("cf", "rtx_carpet", (0, 1.70, 0.460), (0.700, 0.55, 0.020), e1=0.15, e2=0.15))
    m.add_mesh(solid("sb", SHELL, (0, 1.16, 0.700), (0.700, 0.035, 0.260), e1=0.2, e2=0.2))   # cargo bulkhead
    return m


def door_trim(door_mesh, ambient_z):
    """Door card from the door's own skin pushed inwards, plus an armrest bar and
    a turquoise ambient strip."""
    if "rtx_paint" not in door_mesh.groups:
        return
    f = door_mesh.groups["rtx_paint"]
    used = np.unique(f)
    remap = -np.ones(len(door_mesh.v), int)
    remap[used] = np.arange(len(used))
    v = door_mesh.v[used] - door_mesh.n[used] * 0.080
    faces = remap[f][:, ::-1]
    door_mesh.card_v = v                         # the card surface, for cockpit.door_details
    # zones by height: dark kick panel, fabric insert, dark armrest band, fabric upper insert, dark roll
    for z0, z1, mat in ((-9.0, 0.50, SHELL), (0.50, 0.62, FABRIC), (0.62, 0.70, SHELL), (0.70, 0.86, FABRIC), (0.86, 9.0, SHELL)):
        V, N_, UV, F_ = clip_z(v, -door_mesh.n[used], door_mesh.uv[used], faces, z0, z1)
        if len(F_):
            door_mesh.add(V, N_, UV, F_, mat)
    # armrest bar and ambient strip, following the card along y
    ys = door_mesh.v[used][:, 1]
    xs = door_mesh.v[used][:, 0]
    y0, y1 = np.percentile(ys, 12), np.percentile(ys, 88)
    side = 1.0 if np.median(xs) > 0 else -1.0
    xin = np.median(np.abs(xs)) - 0.080
    door_mesh.add_mesh(solid("arm", SHELL, (side * (xin - 0.035), 0.5 * (y0 + y1), 0.655), (0.040, 0.5 * (y1 - y0), 0.030), e1=0.3, e2=0.3))
    door_mesh.add_mesh(solid("amb", "rtx_ambient", (side * (xin - 0.004), 0.5 * (y0 + y1), ambient_z), (0.003, 0.5 * (y1 - y0), 0.003), e1=0.3, e2=0.3, nu=24, nv=8))


def cabin(meshes):
    meshes["rtx_dash"] = dash()
    meshes["rtx_consoles"] = consoles()
    meshes["rtx_floor"] = floor()
    for name, x, y, rear in (("rtx_seat_fl", 0.380, -0.32, False), ("rtx_seat_fr", -0.380, -0.32, False),
                             ("rtx_seat_rl", 0.380, 0.70, True), ("rtx_seat_rr", -0.380, 0.70, True)):
        meshes[name] = lounge_seat(name, x, y, rear)
    meshes["rtx_seat_fl"].add_mesh(seatback_screen("rtx_seat_fl", 0.380, -0.32))
    meshes["rtx_seat_fr"].add_mesh(seatback_screen("rtx_seat_fr", -0.380, -0.32))
    for door in ("rtx_door_fl", "rtx_door_fr", "rtx_door_rl", "rtx_door_rr"):
        door_trim(meshes[door], ambient_z=0.900)
    import cockpit          # steering wheel, pedals and the cabin details (Storm, 2026-10-31)
    cockpit.cockpit(meshes)
    return meshes
