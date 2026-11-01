"""Wheel and tyre meshes in wheel-local space (axis = x, outer face towards +x).

Wheel: 21x9 aero disc (our own design for Lumen): a revolved rim barrel and lip and
a nearly flush face built as a polar height map with six long swept slots, plus a
centre cap. Tyre: 255/40 R21 revolved profile with rounded shoulders, a slight
sidewall bulge and four circumferential grooves.
"""
import numpy as np

import meshkit

RIM_R = 0.2667          # 21 in / 2
TIRE_R = 0.369          # 255/40 R21
TIRE_W = 0.255
RIM_W = 0.229           # 9 in
N_SEG = 96


def tire_mesh(name="rtx_tire"):
    w2 = TIRE_W / 2
    bead = RIM_R + 0.012
    # profile from inner bead, over the tread, to the outer bead: (radius, x)
    prof = []
    # inner sidewall (x = -w2 side)
    for t in np.linspace(0, 1, 9):
        r = bead + (TIRE_R - 0.03 - bead) * t
        bulge = 0.010 * np.sin(np.pi * t)
        prof.append((r, -(w2 - 0.012) - bulge))
    # inner shoulder
    for a in np.linspace(np.pi, np.pi / 2, 7)[1:]:
        prof.append((TIRE_R - 0.03 + 0.03 * np.sin(a), -(w2 - 0.03) + 0.03 * np.cos(a)))
    # tread with four grooves
    xs = np.linspace(-(w2 - 0.03), (w2 - 0.03), 41)
    grooves = [-0.07, -0.025, 0.025, 0.07]
    for x in xs[1:-1]:
        depth = sum(0.006 * np.exp(-((x - g) / 0.004) ** 2) for g in grooves)
        crown = 0.002 * (1 - (x / w2) ** 2)
        prof.append((TIRE_R - depth + crown - 0.002, x))
    # outer shoulder
    for a in np.linspace(np.pi / 2, 0, 7)[:-1]:
        prof.append((TIRE_R - 0.03 + 0.03 * np.sin(a), (w2 - 0.03) + 0.03 * np.cos(a)))
    for t in np.linspace(1, 0, 9):
        r = bead + (TIRE_R - 0.03 - bead) * t
        bulge = 0.010 * np.sin(np.pi * t)
        prof.append((r, (w2 - 0.012) + bulge))
    v, n, uv, f = meshkit.revolve(np.array(prof), N_SEG)
    m = meshkit.Mesh(name)
    m.add(v, n, uv, f, "rtx_tire")
    return m


def _face_heightmap(r, th):
    """Blade mask (0 gap .. 1 blade) on the wheel face, ten swept blades."""
    k = 6
    r0, r1 = 0.075, 0.252
    s = np.clip((r - r0) / (r1 - r0), 0, 1)
    sweep = 1.35 * s ** 1.15                 # long sweep: reads as motion even when parked
    phase = (th - sweep) * k / (2 * np.pi)
    frac = phase - np.floor(phase)
    width = 0.86 - 0.10 * s                  # mostly solid disc, slim slots
    d = np.abs(frac - 0.5) * 2               # 0 at blade centre
    blade = 1.0 - np.clip((d - width) / 0.07, 0, 1)
    blade = blade * (r > r0 - 0.002) * (r < r1 + 0.004)
    return blade


def wheel_mesh(name="rtx_wheel"):
    m = meshkit.Mesh(name)
    # barrel + lip (outer face at +x): (radius, x)
    lip_x = RIM_W / 2
    prof = [(0.214, -lip_x + 0.01), (RIM_R - 0.005, -lip_x + 0.006), (RIM_R + 0.006, -lip_x), (RIM_R + 0.008, -lip_x + 0.012),
            (RIM_R - 0.012, -lip_x + 0.03), (RIM_R - 0.015, 0.0), (RIM_R - 0.012, lip_x - 0.03),
            (RIM_R + 0.008, lip_x - 0.008), (RIM_R + 0.009, lip_x + 0.002), (RIM_R + 0.002, lip_x + 0.008),
            (RIM_R - 0.012, lip_x + 0.004)]
    v, n, uv, f = meshkit.revolve(np.array(prof), N_SEG)
    m.add(v, n, uv, f, "rtx_rim")
    # face: polar grid with blades raised over a recessed dark base
    # 32 x 420 resolved the six swept slots finer than a game camera needs, but the slots spiral across the rings, so the
    # radial step matters most: at 16 rings (14 mm) their edges turned saw-toothed in the renders. 28 x 280 keeps the
    # steps under 8 mm (the face is the most visible detail on the car) at about half the triangles.
    nr, nt = max(28, meshkit.lod(32, 10)), max(280, meshkit.lod(420, 120))
    rr = np.linspace(0.045, RIM_R - 0.010, nr)
    tt = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    R, T = np.meshgrid(rr, tt, indexing="ij")
    blade = _face_heightmap(R, T)
    dish = 0.010 * (1 - (R / RIM_R) ** 2)    # nearly flush aero face
    X = lip_x - 0.012 - 0.022 + dish + 0.014 * blade
    P = np.stack([X, R * np.cos(T), R * np.sin(T)], -1).reshape(-1, 3)
    faces = meshkit.grid_faces(nr, nt, wrap_cols=True)
    # outward = +x: fix winding by checking one normal
    fn = meshkit.face_normals(P, faces)
    if fn[:, 0].mean() < 0:
        faces = faces[:, ::-1]
    N = meshkit.vertex_normals(P, faces)
    tri_blade = blade.reshape(-1)[faces].mean(1) > 0.5
    uvp = np.c_[R.reshape(-1), T.reshape(-1)]
    m.add(P, N, uvp, faces[tri_blade], "rtx_rim")
    m.add(P, N, uvp, faces[~tri_blade], "rtx_rim_dark")
    # centre cap
    cap = [(0.0, lip_x - 0.004), (0.025, lip_x - 0.006), (0.040, lip_x - 0.012), (0.046, lip_x - 0.02)]
    v, n, uv, f = meshkit.revolve(np.array(cap), 48)
    m.add(v, n, uv, f, "rtx_rim_dark")
    # brake disc + caliper hint behind the face
    disc = [(0.06, -0.02), (0.190, -0.02), (0.190, -0.045), (0.06, -0.045)]
    v, n, uv, f = meshkit.revolve(np.array(disc), 64)
    m.add(v, n, uv, f, "rtx_brake")
    return m
