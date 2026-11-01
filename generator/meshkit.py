"""Minimal triangle-mesh toolkit for the robotaxi generator (numpy only).

A Mesh holds per-vertex positions, normals and UVs, plus triangles grouped by
material name. Everything is in BeamNG vehicle space: +x left, +y rearward, +z up.
"""
import os

import numpy as np

# Level of detail (2026-11-01, Storm: "WHY IS BEAMNG USING SO MUCH STUFF?"). The Lumen was 434,491 triangles (41 MB of
# DAE) and BeamNG keeps a deformable copy of every car's mesh, so 20 Lumens filled the PC's memory and the 6 GB card and
# textures stopped loading. DETAIL scales every tessellation count of the small parts (revolves, superellipsoids, tubes,
# patches); BODY_DETAIL scales the body skin grid, whose panel cuts need more care. 1.0 rebuilds exactly the old meshes.
DETAIL = float(os.environ.get("LUMEN_DETAIL", "0.35"))
BODY_DETAIL = float(os.environ.get("LUMEN_BODY_DETAIL", "0.6"))


def lod(n, lo):
    """a tessellation count scaled by DETAIL, never below lo"""
    return max(lo, int(round(n * DETAIL)))


class Mesh:
    def __init__(self, name):
        self.name = name
        self.v = np.zeros((0, 3))
        self.n = np.zeros((0, 3))
        self.uv = np.zeros((0, 2))
        self.groups = {}  # material -> (M,3) int array

    # ---- building ------------------------------------------------------------
    def add(self, v, n, uv, faces, material):
        """Append vertices and triangles (faces index into the given v)."""
        v = np.asarray(v, float).reshape(-1, 3)
        n = np.asarray(n, float).reshape(-1, 3)
        uv = np.asarray(uv, float).reshape(-1, 2)
        faces = np.asarray(faces, int).reshape(-1, 3)
        if len(faces) == 0:
            return self
        base = len(self.v)
        self.v = np.vstack([self.v, v])
        self.n = np.vstack([self.n, n])
        self.uv = np.vstack([self.uv, uv])
        f = faces + base
        if material in self.groups:
            self.groups[material] = np.vstack([self.groups[material], f])
        else:
            self.groups[material] = f
        return self

    def add_mesh(self, other, material_map=None):
        for mat, f in other.groups.items():
            used = np.unique(f)
            remap = -np.ones(len(other.v), int)
            remap[used] = np.arange(len(used))
            self.add(other.v[used], other.n[used], other.uv[used], remap[f], (material_map or {}).get(mat, mat))
        return self

    def triangle_count(self):
        return int(sum(len(f) for f in self.groups.values()))

    def vertex_count(self):
        return len(self.v)

    # ---- transforms ------------------------------------------------------------
    def transformed(self, R=None, t=None, name=None):
        m = Mesh(name or self.name)
        R = np.eye(3) if R is None else np.asarray(R, float)
        t = np.zeros(3) if t is None else np.asarray(t, float)
        m.v = self.v @ R.T + t
        m.n = normalize(self.n @ R.T)
        m.uv = self.uv.copy()
        flip = np.linalg.det(R) < 0
        m.groups = {k: (f[:, ::-1].copy() if flip else f.copy()) for k, f in self.groups.items()}
        return m

    def mirrored_x(self, name=None):
        """Mirror across the centre plane (left <-> right), keeping faces outward."""
        return self.transformed(np.diag([-1.0, 1.0, 1.0]), name=name)

    def compact(self):
        """Drop vertices not referenced by any triangle."""
        if not self.groups:
            return self
        used = np.unique(np.concatenate([f.ravel() for f in self.groups.values()]))
        remap = -np.ones(len(self.v), int)
        remap[used] = np.arange(len(used))
        self.v, self.n, self.uv = self.v[used], self.n[used], self.uv[used]
        self.groups = {k: remap[f] for k, f in self.groups.items()}
        return self

    def bbox(self):
        return self.v.min(axis=0), self.v.max(axis=0)


def normalize(a, eps=1e-12):
    a = np.asarray(a, float)
    l = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(l, eps)


def face_normals(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return np.cross(b - a, c - a)


def vertex_normals(v, f):
    """Area-weighted vertex normals."""
    fn = face_normals(v, f)
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return normalize(n)


def grid_faces(rows, cols, wrap_cols=False, wrap_rows=False):
    """Two triangles per quad of a rows x cols vertex grid (row-major indices)."""
    r = np.arange(rows - (0 if wrap_rows else 1))
    c = np.arange(cols - (0 if wrap_cols else 1))
    R, C = np.meshgrid(r, c, indexing="ij")
    R, C = R.ravel(), C.ravel()
    R1 = (R + 1) % rows
    C1 = (C + 1) % cols
    a = R * cols + C
    b = R * cols + C1
    c_ = R1 * cols + C1
    d = R1 * cols + C
    return np.vstack([np.c_[a, b, c_], np.c_[a, c_, d]])


def revolve(profile_rz, segments, axis_x0=0.0, side=1, u_scale=1.0, closed=True, smooth=True):
    """Revolve a profile around the x axis (wheel axis).

    profile_rz: (K,2) array of (radius, x_offset) points; x_offset is measured
    outward (towards +x for side=1). Returns (v, n, uv, faces) in local wheel space
    where the wheel axis is x and the wheel centre is the origin.
    """
    segments = lod(segments, 16)
    prof = np.asarray(profile_rz, float)
    K = len(prof)
    ang = np.linspace(0, 2 * np.pi, segments + 1)[:-1] if closed else np.linspace(0, 2 * np.pi, segments)
    S = len(ang)
    r = prof[:, 0][:, None]
    xo = prof[:, 1][:, None] * side
    v = np.zeros((K, S, 3))
    v[..., 0] = xo
    v[..., 1] = r * np.cos(ang)[None, :]
    v[..., 2] = r * np.sin(ang)[None, :]
    v = v.reshape(-1, 3)
    # profile tangent -> normal (in the r-x plane), then rotate around the axis
    d = np.gradient(prof, axis=0)
    # outward normal of the profile curve: rotate tangent (dr, dx) by -90 deg in (r, x)
    nr = d[:, 1] * side
    nx = -d[:, 0] * side
    ln = np.hypot(nr, nx) + 1e-12
    nr, nx = nr / ln, nx / ln
    n = np.zeros((K, S, 3))
    n[..., 0] = nx[:, None]
    n[..., 1] = nr[:, None] * np.cos(ang)[None, :]
    n[..., 2] = nr[:, None] * np.sin(ang)[None, :]
    n = n.reshape(-1, 3)
    uv = np.zeros((K, S, 2))
    uv[..., 0] = (np.arange(S) / S)[None, :] * u_scale
    uv[..., 1] = (np.arange(K) / max(1, K - 1))[:, None]
    uv = uv.reshape(-1, 2)
    faces = grid_faces(K, S, wrap_cols=closed)
    if side < 0:
        faces = faces[:, ::-1]
    return v, n, uv, faces
