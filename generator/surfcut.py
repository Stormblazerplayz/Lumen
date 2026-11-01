"""Cut the body skin into panels along exact seam lines.

The skin is a watertight triangle mesh lying on shape.F = 0. A seam is the zero
set of a scalar field defined on (position, normal); cutting splits every triangle
that the zero set crosses, sharing the new vertices between neighbours so the
mesh stays watertight. New vertices are pulled back onto the surface and get
their normals from the surface gradient, so seams are smooth curves, not stairs.

After all cuts every triangle lies in exactly one cell of the arrangement, and a
classifier assigns it to a panel or material. Panels then get a shut-line gap
(boundary pulled in along the surface) and an inward flange so seams read as thin
dark lines, as on a real car.
"""
import numpy as np

import meshkit
import shape


class SurfMesh:
    def __init__(self, V, F):
        self.V = np.asarray(V, float)
        self.F = np.asarray(F, int)
        self.N = surface_normals(self.V)

    @classmethod
    def from_grid(cls, P):
        R, C, _ = P.shape
        V = P.reshape(-1, 3)
        F = meshkit.grid_faces(R, C, wrap_cols=True)
        # drop degenerate faces at the poles of the end caps
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        F = F[area > 1e-10]
        m = cls(V, F)
        fn = meshkit.face_normals(m.V, m.F)
        if (fn * m.N[m.F[:, 0]]).sum(-1).mean() < 0:
            m.F = m.F[:, ::-1].copy()
        m.weld()
        return m

    def weld(self, tol=1e-6):
        """Merge coincident vertices (the cap poles produce many)."""
        key = np.round(self.V / tol).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel()
        self.V = self.V[first]
        self.N = self.N[first]
        F = inv[self.F]
        ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        self.F = F[ok]

    def cut(self, field, mask=None):
        """Split triangles crossed by field(V, N) = 0. mask(V, N) -> bool per vertex
        limits the cut to triangles whose vertices are all inside the mask."""
        vals = field(self.V, self.N).astype(float)
        vals = np.where(np.abs(vals) < 1e-7, 1e-7, vals)
        pos = vals > 0
        P = pos[self.F]
        npos = P.sum(1)
        cross = (npos == 1) | (npos == 2)
        if mask is not None:
            mk = mask(self.V, self.N)
            cross &= mk[self.F].all(1)
        idx = np.nonzero(cross)[0]
        if len(idx) == 0:
            return 0
        T = self.F[idx]
        Pt = P[idx]
        # lone vertex: the one whose sign differs from the other two
        lone = np.where(npos[idx] == 1, np.argmax(Pt, axis=1), np.argmin(Pt, axis=1))
        order = (lone[:, None] + np.arange(3)[None, :]) % 3
        Tr = np.take_along_axis(T, order, axis=1)  # (a lone, b, c), orientation kept
        a, b, c = Tr[:, 0], Tr[:, 1], Tr[:, 2]
        # unique crossing edges
        e1 = np.sort(np.c_[a, b], axis=1)
        e2 = np.sort(np.c_[a, c], axis=1)
        edges = np.vstack([e1, e2])
        n0 = len(self.V)
        key = edges[:, 0].astype(np.int64) * (n0 + 1) + edges[:, 1]
        ukey, inv = np.unique(key, return_inverse=True)
        inv = inv.ravel()
        ue = np.c_[ukey // (n0 + 1), ukey % (n0 + 1)]
        va, vb = vals[ue[:, 0]], vals[ue[:, 1]]
        t = va / (va - vb)
        newp = self.V[ue[:, 0]] + t[:, None] * (self.V[ue[:, 1]] - self.V[ue[:, 0]])
        newp = shape.project(newp, iters=3)
        self.V = np.vstack([self.V, newp])
        self.N = np.vstack([self.N, surface_normals(newp)])
        pab = n0 + inv[:len(idx)]
        pac = n0 + inv[len(idx):]
        new_tris = np.vstack([np.c_[a, pab, pac], np.c_[pab, b, c], np.c_[pab, c, pac]])
        keep = np.ones(len(self.F), bool)
        keep[idx] = False
        self.F = np.vstack([self.F[keep], new_tris])
        return len(idx)

    def centroid_fields(self):
        c = self.V[self.F].mean(axis=1)
        n = meshkit.normalize(self.N[self.F].mean(axis=1))
        return c, n


def surface_normals(V):
    g = shape.grad(V)
    return meshkit.normalize(g)


def boundary_loops_edges(F):
    """Boundary edges (directed, as they appear in their triangle) of a triangle set."""
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key_fwd = e[:, 0].astype(np.int64) * (F.max() + 1) + e[:, 1]
    key_rev = e[:, 1].astype(np.int64) * (F.max() + 1) + e[:, 0]
    s = set(key_fwd.tolist())
    is_b = np.array([k not in s for k in key_rev.tolist()])
    tri_of_edge = np.tile(np.arange(len(F)), 3)
    return e[is_b], tri_of_edge[is_b]


def panel_mesh(sm, tri_idx, name, material, gap=0.002, flange=0.007, offset=0.0,
               flange_material=None, uv_fn=None):
    """Extract triangles as a Mesh.

    gap    : pull the boundary in along the surface (half of the shut line)
    flange : depth of an inward return along -normal at the boundary (paint seams)
    offset : move the region along its normal (negative = recessed: glass, trims);
             a wall then connects the recessed edge back up to the surface so no
             crack opens at grazing angles
    """
    F = sm.F[tri_idx]
    m = meshkit.Mesh(name)
    if len(F) == 0:
        return m
    used = np.unique(F)
    remap = -np.ones(len(sm.V), int)
    remap[used] = np.arange(len(used))
    V = sm.V[used].copy()
    N = sm.N[used].copy()
    F = remap[F]
    bedges, btri = boundary_loops_edges(F)
    bverts = np.unique(bedges) if len(bedges) else np.zeros(0, int)
    inward = np.zeros_like(V)
    if len(bedges):
        # inward direction per boundary vertex: towards the opposite corner of its
        # triangle, perpendicular to the edge and tangent to the surface
        Ft = F[btri]
        i, j = bedges[:, 0], bedges[:, 1]
        k = np.where((Ft[:, 0] != i) & (Ft[:, 0] != j), Ft[:, 0], np.where((Ft[:, 1] != i) & (Ft[:, 1] != j), Ft[:, 1], Ft[:, 2]))
        ed = meshkit.normalize(V[j] - V[i])
        to_k = V[k] - 0.5 * (V[i] + V[j])
        d = to_k - ed * (to_k * ed).sum(1, keepdims=True)
        nrm = meshkit.normalize(N[i] + N[j])
        d = meshkit.normalize(d - nrm * (d * nrm).sum(1, keepdims=True))
        np.add.at(inward, i, d)
        np.add.at(inward, j, d)
        inward[bverts] = meshkit.normalize(inward[bverts])
    if gap > 0 and len(bverts):
        V[bverts] = shape.project(V[bverts] + inward[bverts] * gap, iters=2)
    surf_b = V[bverts].copy()
    if offset:
        V = V + N * offset
    m.add(V, N, uv_fn(V) if uv_fn else np.zeros((len(V), 2)), F, material)
    if len(bedges) == 0:
        return m
    nb = len(bverts)
    loc = -np.ones(len(V), int)
    loc[bverts] = np.arange(nb)
    a, b = loc[bedges[:, 0]], loc[bedges[:, 1]]
    if flange > 0:
        top = V[bverts]
        bot = top - N[bverts] * flange + inward[bverts] * (0.5 * flange)
        fn_ = meshkit.normalize(-inward[bverts] + 0.15 * N[bverts])
        quads = np.vstack([np.c_[b, a, a + nb], np.c_[b, a + nb, b + nb]])
        m.add(np.vstack([top, bot]), np.vstack([fn_, fn_]), np.zeros((2 * nb, 2)), quads, flange_material or material)
    if offset < 0:
        # wall from the recessed edge (a, b) up to the surface edge (a', b')
        low = V[bverts]
        wall_n = meshkit.normalize(inward[bverts] + 0.15 * N[bverts])
        quads = np.vstack([np.c_[a, b, b + nb], np.c_[a, b + nb, a + nb]])
        m.add(np.vstack([low, surf_b]), np.vstack([wall_n, wall_n]), np.zeros((2 * nb, 2)), quads, flange_material or material)
    return m
