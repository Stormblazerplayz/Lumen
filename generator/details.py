"""Exterior detail meshes: wheel-well liners, mirrors, flush handles, side
repeaters, headlamp units, panel back faces. Small parts are projected onto
the real body surface (shape.F) so they sit exactly on it.
"""
import numpy as np

import meshkit
import shape
from shape import ARCH_RADIUS, WHEEL_Z, Y_AXLE_F, Y_AXLE_R


def surface_x(y, z, side=1.0, x0=0.15, x1=1.25, iters=40):
    """|x| of the body surface at (y, z) by bisection along x (side surfaces)."""
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    lo = np.full(y.shape, x0)
    hi = np.full(y.shape, x1)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = shape.F(np.stack([mid * side, y, z], -1))
        inside = f < 0
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    return 0.5 * (lo + hi)


def side_patch(name, material, y0, y1, z0, z1, side, nu=40, nv=10, lift=0.0015, shape_fn=None):
    """A patch on the body side spanning [y0,y1] x [z0,z1], lifted off the paint.
    shape_fn(u, v) -> bool keeps only cells inside a shape (rounded slots etc.)."""
    nu, nv = meshkit.lod(nu, 4), meshkit.lod(nv, 2)
    u = np.linspace(0, 1, nu)
    v = np.linspace(0, 1, nv)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    Y = y0 + (y1 - y0) * U
    Z = z0 + (z1 - z0) * Vv
    X = surface_x(Y, Z, side)
    P = np.stack([X * side, Y, Z], -1).reshape(-1, 3)
    N = meshkit.normalize(shape.grad(P))
    P = P + N * lift
    F = meshkit.grid_faces(nu, nv)
    fn = meshkit.face_normals(P, F)
    if (fn * N[F[:, 0]]).sum(-1).mean() < 0:
        F = F[:, ::-1]
    if shape_fn is not None:
        cu = U.reshape(-1)[F].mean(1)
        cv = Vv.reshape(-1)[F].mean(1)
        F = F[shape_fn(cu, cv)]
    uv = np.c_[U.reshape(-1), Vv.reshape(-1)]
    if side < 0:
        uv[:, 0] = 1 - uv[:, 0]   # read left-to-right on the right side too (front is on the right)
    m = meshkit.Mesh(name)
    m.add(P, N, uv, F, material)
    return m


def rounded_slot(power=8):
    """Inside test for a long slot with rounded ends in unit (u, v) space."""
    def inside(u, v):
        return (np.abs(2 * u - 1) ** power + np.abs(2 * v - 1) ** 2) <= 1.0
    return inside


def wheel_liner(name, y_axle, side, material="rtx_black_satin"):
    """Arch liner: a cylinder shell over the wheel from the body side inwards, plus
    the inner wall as an annular sector. Normals face the wheel. Clipped above the
    floor line so nothing hangs below the car."""
    R = ARCH_RADIUS + 0.012
    z_floor = 0.215
    # angles where the arch circle is above the floor line
    lim = np.degrees(np.arcsin(np.clip((z_floor - WHEEL_Z) / R, -1, 1)))
    phis = np.linspace(np.radians(lim), np.radians(180 - lim), meshkit.lod(64, 16))   # 0 = rearward, 90 = up
    xin = 0.50
    nx = meshkit.lod(14, 4)
    rows = []
    for ph in phis:
        y = y_axle + R * np.cos(ph)
        z = WHEEL_Z + R * np.sin(ph)
        xo = surface_x(np.array([y]), np.array([z]), side)[0] - 0.006
        xs = np.linspace(xin, max(xo, xin + 0.02), nx)
        rows.append([(x * side, y, z) for x in xs])
    P = np.array(rows).reshape(-1, 3)
    F = meshkit.grid_faces(len(phis), nx)
    c = np.stack([P[:, 0], np.full(len(P), y_axle), np.full(len(P), WHEEL_Z)], -1)
    N = meshkit.normalize(c - P)
    fn = meshkit.face_normals(P, F)
    if (fn * N[F[:, 0]]).sum(-1).mean() < 0:
        F = F[:, ::-1]
    m = meshkit.Mesh(name)
    m.add(P, N, np.zeros((len(P), 2)), F, material)
    # inner wall: annular sector between the tyre's reach and the liner, above the floor
    rr = np.linspace(0.30, R, 6)
    W = np.array([[(xin * side, y_axle + r * np.cos(ph), max(WHEEL_Z + r * np.sin(ph), z_floor)) for r in rr] for ph in phis]).reshape(-1, 3)
    Fw = meshkit.grid_faces(len(phis), len(rr))
    Nw = np.tile([side * 1.0, 0.0, 0.0], (len(W), 1))
    fn = meshkit.face_normals(W, Fw)
    if (fn * Nw[Fw[:, 0]]).sum(-1).mean() < 0:
        Fw = Fw[:, ::-1]
    m.add(W, Nw, np.zeros((len(W), 2)), Fw, material)
    return m


def superellipsoid(center, axes, e1=0.35, e2=0.35, nu=40, nv=24):
    """Rounded box-like solid. Returns (P, N, F) with outward normals."""
    nu, nv = meshkit.lod(nu, 8), meshkit.lod(nv, 4)
    cx, cy, cz = center
    a, b, c = axes
    u = np.linspace(-np.pi, np.pi, nu, endpoint=False)
    v = np.linspace(-np.pi / 2, np.pi / 2, nv)
    U, V = np.meshgrid(u, v, indexing="ij")
    sgn = lambda t: np.sign(t)
    f = lambda t, e: sgn(t) * np.abs(t) ** e
    X = a * f(np.cos(V), e1) * f(np.cos(U), e2)
    Y = b * f(np.cos(V), e1) * f(np.sin(U), e2)
    Z = c * f(np.sin(V), e1)
    P = np.stack([X + cx, Y + cy, Z + cz], -1).reshape(-1, 3)
    F = meshkit.grid_faces(nu, nv, wrap_cols=False, wrap_rows=True)
    # grid is (nu rows wrapping) x (nv cols): rebuild with rows=u (wrap), cols=v
    F = meshkit.grid_faces(nu, nv, wrap_rows=True)
    a_, b_, c_ = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    area = np.linalg.norm(np.cross(b_ - a_, c_ - a_), axis=1)
    F = F[area > 1e-12]
    N = meshkit.vertex_normals(P, F)
    out = P - np.array(center)
    if (N * out).sum(-1).mean() < 0:
        F = F[:, ::-1]
        N = -N
    return P, N, F


def mirror(name, side):
    """Body-colour mirror housing on a short stalk, glass facing rearward."""
    m = meshkit.Mesh(name)
    cx, cy, cz = 0.975 * side, -0.705, 1.090
    P, N, F = superellipsoid((cx, cy, cz), (0.090, 0.052, 0.068), e1=0.45, e2=0.55)
    # glass = rear-facing triangles
    fn = meshkit.normalize(meshkit.face_normals(P, F))
    rear = fn[:, 1] > 0.55
    m.add(P, N, np.zeros((len(P), 2)), F[~rear], "rtx_paint")
    m.add(P, N, np.zeros((len(P), 2)), F[rear], "rtx_mirror")
    # stalk from the door corner to the housing
    sx = surface_x(np.array([-0.76]), np.array([1.035]), side)[0]
    P2, N2, F2 = superellipsoid(((sx + 0.04) * side, -0.745, 1.045), (0.055, 0.035, 0.022), e1=0.6, e2=0.6, nu=24, nv=12)
    m.add(P2, N2, np.zeros((len(P2), 2)), F2, "rtx_black_gloss")
    return m


def door_handle(name, side, y0, y1, zc):
    return side_patch(name, "rtx_black_gloss", y0, y1, zc - 0.012, zc + 0.012, side, nu=40, nv=6, lift=0.0012,
                      shape_fn=rounded_slot())


def repeater(name, side):
    return side_patch(name, "rtx_black_gloss", -1.105, -1.035, 0.845, 0.875, side, nu=16, nv=6, lift=0.002,
                      shape_fn=lambda u, v: v < 1.0 - 0.6 * u)   # wedge, pointing back


def headlamp(name, side):
    """Slim headlamp lens in the corner pod (main low/high beam)."""
    m = meshkit.Mesh(name)
    # sample the front corner surface along a horizontal strip
    xs = np.linspace(0.735, 0.885, 18)
    zs = np.linspace(0.575, 0.610, 5)
    pts = []
    for z in zs:
        for x in xs:
            # find y on the surface by bisection along y (front face)
            lo, hi = -1.6, -2.6
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                f = shape.F(np.array([[x * side, mid, z]]))[0]
                if f < 0:
                    lo = mid
                else:
                    hi = mid
            pts.append((x * side, 0.5 * (lo + hi), z))
    P = np.array(pts)
    N = meshkit.normalize(shape.grad(P))
    P = P + N * 0.002
    F = meshkit.grid_faces(len(zs), len(xs))
    fn = meshkit.face_normals(P, F)
    if (fn * N[F[:, 0]]).sum(-1).mean() < 0:
        F = F[:, ::-1]
    m.add(P, N, np.zeros((len(P), 2)), F, "rtx_headlight")
    return m


def back_faces(mesh, material="rtx_inner", depth=0.010, source_material="rtx_paint"):
    """Give a paint panel an inner skin so it reads as a solid panel from inside
    and through shut lines."""
    if source_material not in mesh.groups:
        return mesh
    f = mesh.groups[source_material]
    used = np.unique(f)
    remap = -np.ones(len(mesh.v), int)
    remap[used] = np.arange(len(used))
    v = mesh.v[used] - mesh.n[used] * depth
    n = -mesh.n[used]
    mesh.add(v, n, mesh.uv[used], remap[f][:, ::-1], material)
    return mesh


def camera_pod(name, side):
    """Aero camera pod on the front door corner (replaces mirrors on a robotaxi)."""
    m = meshkit.Mesh(name)
    y, z = -1.000, 1.005
    sx = surface_x(np.array([y]), np.array([z]), side)[0]
    P, N, F = superellipsoid(((sx + 0.050) * side, y + 0.02, z + 0.012), (0.058, 0.070, 0.026), e1=0.55, e2=0.45, nu=36, nv=18)
    fn = meshkit.normalize(meshkit.face_normals(P, F))
    lens = fn[:, 1] > 0.75
    m.add(P, N, np.zeros((len(P), 2)), F[~lens], "rtx_black_gloss")
    m.add(P, N, np.zeros((len(P), 2)), F[lens], "rtx_lens")
    return m


def lidar_crown(name):
    """Low sensor crown on the front of the roof: black housing, glossy lens band."""
    import shape
    m = meshkit.Mesh(name)
    y = -0.30
    zt = float(shape.top_z(0.0, y))
    P, N, F = superellipsoid((0.0, y, zt + 0.020), (0.300, 0.120, 0.048), e1=0.40, e2=0.35, nu=56, nv=22)
    fn = meshkit.normalize(meshkit.face_normals(P, F))
    band = (fn[:, 1] < -0.55) & (np.abs(fn[:, 2]) < 0.45)
    m.add(P, N, np.zeros((len(P), 2)), F[~band], "rtx_black_gloss")
    m.add(P, N, np.zeros((len(P), 2)), F[band], "rtx_lens")
    # turquoise status line along the crown's base
    P2, N2, F2 = superellipsoid((0.0, y, zt - 0.014), (0.302, 0.122, 0.006), e1=0.3, e2=0.35, nu=56, nv=6)
    m.add(P2, N2, np.zeros((len(P2), 2)), F2, "rtx_halo")
    return m


def corner_sensor(name, side, y, z):
    sx = surface_x(np.array([y]), np.array([z]), side)[0]
    P, N, F = superellipsoid(((sx - 0.004) * side, y, z), (0.018, 0.018, 0.018), e1=0.8, e2=0.8, nu=20, nv=12)
    m = meshkit.Mesh(name)
    m.add(P, N, np.zeros((len(P), 2)), F, "rtx_lens")
    return m
