"""Sample the implicit body (shape.F) into a structured quad grid.

Rays leave an internal axis (nose-side point A to tail-side point B): hemispherical
fans at both ends cover the nose and tail, rays in the x-z plane cover the middle.
The grid is then re-spaced by arc length along rows and columns and pulled back
onto the surface, so triangles are near-uniform (about 1.5 cm) everywhere.

Grid layout: P[row, col] with rows running nose -> tail and cols running around the
body (col angle 0 = straight up, increasing towards +x/left, wrapping).
"""
import numpy as np

import shape

AXIS_A = np.array([0.0, -1.85, 0.62])
AXIS_B = np.array([0.0, 1.95, 0.80])


def _march(origins, dirs, t_max=3.5):
    """First inside->outside crossing of F along each ray (sphere tracing + bisection)."""
    n = len(origins)
    t = np.zeros(n)
    done = np.zeros(n, bool)
    t_lo = np.zeros(n)
    t_hi = np.full(n, np.nan)
    for _ in range(200):
        act = ~done
        if not act.any():
            break
        p = origins[act] + dirs[act] * t[act, None]
        f = shape.F(p)
        out = f > 0
        idx = np.nonzero(act)[0]
        hit = idx[out]
        t_hi[hit] = t[hit]
        done[hit] = True
        go = idx[~out]
        t_lo[go] = t[go]
        t[go] = t[go] + np.maximum(0.8 * -f[~out], 0.004)
        over = t > t_max
        if over.any():
            bad = np.nonzero(over & ~done)[0]
            t_hi[bad] = t_max
            done[bad] = True
    for _ in range(16):
        tm = 0.5 * (t_lo + t_hi)
        f = shape.F(origins + dirs * tm[:, None])
        out = f > 0
        t_hi = np.where(out, tm, t_hi)
        t_lo = np.where(out, t_lo, tm)
    return origins + dirs * (0.5 * (t_lo + t_hi))[:, None]


def raw_grid(n_cap=36, n_mid=300, n_theta=288):
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    sx, cz = np.sin(th), np.cos(th)
    rows_o, rows_d = [], []
    # front cap: phi 0 (straight ahead) .. just short of 90 deg
    for phi in np.linspace(0, np.pi / 2, n_cap, endpoint=False):
        d = np.stack([np.sin(phi) * sx, np.full_like(sx, -np.cos(phi)), np.sin(phi) * cz], -1)
        rows_o.append(np.repeat(AXIS_A[None], n_theta, 0))
        rows_d.append(d)
    # middle stations
    for s in np.linspace(0, 1, n_mid):
        o = AXIS_A + s * (AXIS_B - AXIS_A)
        d = np.stack([sx, np.zeros_like(sx), cz], -1)
        rows_o.append(np.repeat(o[None], n_theta, 0))
        rows_d.append(d)
    # rear cap: just short of 90 deg .. 0 (straight back)
    for phi in np.linspace(np.pi / 2, 0, n_cap + 1)[1:]:
        d = np.stack([np.sin(phi) * sx, np.full_like(sx, np.cos(phi)), np.sin(phi) * cz], -1)
        rows_o.append(np.repeat(AXIS_B[None], n_theta, 0))
        rows_d.append(d)
    O = np.concatenate(rows_o)
    D = np.concatenate(rows_d)
    P = _march(O, D)
    return P.reshape(-1, n_theta, 3)


CURVE_WEIGHT = 0.03  # metres of extra "length" per radian of turning: more samples at creases


def _resample_polyline(P, n_out, closed):
    """Re-sample each polyline along axis 1 by arc length plus a turning-angle term,
    so sharp features (spoiler lip, cowl, roof rail) get denser sampling. P: (L, K, 3)."""
    if closed:
        Q = np.concatenate([P, P[:, :1]], axis=1)
    else:
        Q = P
    d = np.diff(Q, axis=1)
    seg = np.linalg.norm(d, axis=-1)
    dn = d / np.maximum(seg[..., None], 1e-12)
    turn = np.zeros_like(seg)
    cosang = np.clip((dn[:, 1:] * dn[:, :-1]).sum(-1), -1.0, 1.0)
    ang = np.arccos(cosang)
    # split each vertex's turning angle between its two neighbouring segments
    turn[:, 1:] += 0.5 * ang
    turn[:, :-1] += 0.5 * ang
    seg = seg + CURVE_WEIGHT * turn
    s = np.concatenate([np.zeros((len(Q), 1)), np.cumsum(seg, axis=1)], axis=1)
    total = s[:, -1:]
    if closed:
        targets = np.linspace(0, 1, n_out, endpoint=False)[None, :] * total
    else:
        targets = np.linspace(0, 1, n_out)[None, :] * total
    out = np.empty((len(Q), n_out, 3))
    for i in range(len(Q)):
        for k in range(3):
            out[i, :, k] = np.interp(targets[i], s[i], Q[i, :, k])
    return out


def relax(P, passes=2, weight_rows=None):
    """Even out spacing: arc-length re-sample rings (closed) then columns (open),
    re-projecting onto the surface after each pass."""
    R, C, _ = P.shape
    for _ in range(passes):
        P = _resample_polyline(P, C, closed=True)
        P = shape.project(P.reshape(-1, 3)).reshape(R, C, 3)
        Pt = np.transpose(P, (1, 0, 2))
        Pt = _resample_polyline(Pt, R, closed=False)
        P = np.transpose(Pt, (1, 0, 2))
        P = shape.project(P.reshape(-1, 3)).reshape(R, C, 3)
    return P


def build(n_cap=36, n_mid=300, n_theta=288, passes=2):
    P = raw_grid(n_cap, n_mid, n_theta)
    P = relax(P, passes)
    return P


def normals(P):
    g = shape.grad(P.reshape(-1, 3))
    g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)
    return g.reshape(P.shape)
