"""Small numpy rasteriser for checking generated meshes without the game.

Vectorised: every triangle tests the pixels of its bounding box (up to a cap),
the nearest fragment wins via a depth sort. Shading is a studio look meant to
reveal surface quality: sky/ground hemisphere, a key light, and reflections of
soft horizontal stripes (the usual "zebra" check for car surfaces).
"""
import numpy as np
from PIL import Image

MATERIAL_COLORS = {
    "paint": (0.22, 0.24, 0.27),
    "black": (0.03, 0.03, 0.035),
    "glass": (0.06, 0.07, 0.08),
    "trim": (0.05, 0.05, 0.055),
    "light": (0.85, 0.85, 0.82),
    "red": (0.55, 0.03, 0.03),
    "rim": (0.62, 0.63, 0.65),
    "tire": (0.035, 0.035, 0.035),
    "decal": (0.80, 0.68, 0.42),
}


def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return eye, r, u, f


def render(meshes, eye, target, width=1200, height=520, fov_deg=None, ortho_height=None,
           bg=(0.97, 0.97, 0.97), material_color=None, max_box=12):
    """meshes: list of Mesh. fov_deg for perspective, ortho_height (m) for orthographic."""
    eye, r, u, f = look_at(eye, target)
    zbuf = np.full((height, width), np.inf)
    nbuf = np.zeros((height, width, 3))
    cbuf = np.zeros((height, width, 3))
    pbuf = np.zeros((height, width, 3))
    for m in meshes:
        for mat, faces in m.groups.items():
            col = np.array((material_color or {}).get(mat) or MATERIAL_COLORS.get(mat.split("_")[-1], None)
                           or MATERIAL_COLORS.get(mat, (0.5, 0.5, 0.5)))
            V = m.v - eye
            cam = np.stack([V @ r, V @ u, V @ f], -1)
            if ortho_height:
                s = height / ortho_height
                sx = width / 2 + cam[:, 0] * s
                sy = height / 2 - cam[:, 1] * s
            else:
                fl = (height / 2) / np.tan(np.radians(fov_deg) / 2)
                z = np.maximum(cam[:, 2], 1e-3)
                sx = width / 2 + cam[:, 0] * fl / z
                sy = height / 2 - cam[:, 1] * fl / z
            depth = cam[:, 2]
            tri = faces
            if not ortho_height:
                # near-plane cull: triangles reaching behind the camera would project
                # to garbage (no clipping in this preview rasteriser)
                tri = tri[(depth[tri] > 0.04).all(axis=1)]
            x0, x1, x2 = sx[tri[:, 0]], sx[tri[:, 1]], sx[tri[:, 2]]
            y0, y1, y2 = sy[tri[:, 0]], sy[tri[:, 1]], sy[tri[:, 2]]
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            ok = np.abs(area) > 1e-9
            tri, x0, x1, x2, y0, y1, y2, area = tri[ok], x0[ok], x1[ok], x2[ok], y0[ok], y1[ok], y2[ok], area[ok]
            bx0 = np.floor(np.minimum(np.minimum(x0, x1), x2)).astype(int)
            by0 = np.floor(np.minimum(np.minimum(y0, y1), y2)).astype(int)
            bx1 = np.ceil(np.maximum(np.maximum(x0, x1), x2)).astype(int)
            by1 = np.ceil(np.maximum(np.maximum(y0, y1), y2)).astype(int)
            vis = (bx1 >= 0) & (by1 >= 0) & (bx0 < width) & (by0 < height)
            sel = np.nonzero(vis)[0]
            bw = np.minimum(bx1 - bx0 + 1, max_box)
            bh = np.minimum(by1 - by0 + 1, max_box)
            big = sel[(bx1[sel] - bx0[sel] + 1 > max_box) | (by1[sel] - by0[sel] + 1 > max_box)]
            small = np.setdiff1d(sel, big)
            for group, box in ((small, max_box), (big, None)):
                if len(group) == 0:
                    continue
                if box is None:
                    # rare large triangles: process one at a time
                    for t in group:
                        _raster(t, bx0[t], by0[t], bx1[t] - bx0[t] + 1, by1[t] - by0[t] + 1,
                                x0, x1, x2, y0, y1, y2, area, tri, depth, m, col, width, height, zbuf, nbuf, cbuf, pbuf)
                    continue
                oy, ox = np.mgrid[0:box, 0:box]
                ox, oy = ox.ravel(), oy.ravel()
                T = group
                px = bx0[T][:, None] + ox[None, :]
                py = by0[T][:, None] + oy[None, :]
                inb = (ox[None, :] < bw[T][:, None]) & (oy[None, :] < bh[T][:, None]) & (px >= 0) & (py >= 0) & (px < width) & (py < height)
                cx, cy = px + 0.5, py + 0.5
                w0 = ((x1[T][:, None] - cx) * (y2[T][:, None] - cy) - (x2[T][:, None] - cx) * (y1[T][:, None] - cy)) / area[T][:, None]
                w1 = ((x2[T][:, None] - cx) * (y0[T][:, None] - cy) - (x0[T][:, None] - cx) * (y2[T][:, None] - cy)) / area[T][:, None]
                w2 = 1 - w0 - w1
                inside = inb & (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
                ti, pi = np.nonzero(inside)
                if len(ti) == 0:
                    continue
                tt = T[ti]
                a0, a1, a2 = w0[ti, pi], w1[ti, pi], w2[ti, pi]
                d = a0 * depth[tri[tt, 0]] + a1 * depth[tri[tt, 1]] + a2 * depth[tri[tt, 2]]
                X, Y = px[ti, pi], py[ti, pi]
                order = np.argsort(-d)  # far first, near last (last write wins)
                X, Y, d, tt, a0, a1, a2 = X[order], Y[order], d[order], tt[order], a0[order], a1[order], a2[order]
                closer = d < zbuf[Y, X]
                X, Y, d, tt, a0, a1, a2 = X[closer], Y[closer], d[closer], tt[closer], a0[closer], a1[closer], a2[closer]
                n = a0[:, None] * m.n[tri[tt, 0]] + a1[:, None] * m.n[tri[tt, 1]] + a2[:, None] * m.n[tri[tt, 2]]
                p = a0[:, None] * m.v[tri[tt, 0]] + a1[:, None] * m.v[tri[tt, 1]] + a2[:, None] * m.v[tri[tt, 2]]
                zbuf[Y, X] = d
                nbuf[Y, X] = n
                cbuf[Y, X] = col
                pbuf[Y, X] = p
    return shade(zbuf, nbuf, cbuf, pbuf, eye, bg, key=key_light(eye, target))


def key_light(eye, target):
    """Key light from above and slightly behind the camera, so the visible side is lit."""
    d = np.asarray(eye, float) - np.asarray(target, float)
    d /= np.linalg.norm(d)
    k = d + np.array([0.0, 0.0, 1.1])
    return k / np.linalg.norm(k)


def _raster(t, bx, by, bw, bh, x0, x1, x2, y0, y1, y2, area, tri, depth, m, col, width, height, zbuf, nbuf, cbuf, pbuf):
    ys, xs = np.mgrid[max(by, 0):min(by + bh, height), max(bx, 0):min(bx + bw, width)]
    if xs.size == 0:
        return
    cx, cy = xs + 0.5, ys + 0.5
    w0 = ((x1[t] - cx) * (y2[t] - cy) - (x2[t] - cx) * (y1[t] - cy)) / area[t]
    w1 = ((x2[t] - cx) * (y0[t] - cy) - (x0[t] - cx) * (y2[t] - cy)) / area[t]
    w2 = 1 - w0 - w1
    ins = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
    if not ins.any():
        return
    a0, a1, a2 = w0[ins], w1[ins], w2[ins]
    X, Y = xs[ins], ys[ins]
    d = a0 * depth[tri[t, 0]] + a1 * depth[tri[t, 1]] + a2 * depth[tri[t, 2]]
    closer = d < zbuf[Y, X]
    X, Y, d, a0, a1, a2 = X[closer], Y[closer], d[closer], a0[closer], a1[closer], a2[closer]
    zbuf[Y, X] = d
    nbuf[Y, X] = a0[:, None] * m.n[tri[t, 0]] + a1[:, None] * m.n[tri[t, 1]] + a2[:, None] * m.n[tri[t, 2]]
    pbuf[Y, X] = a0[:, None] * m.v[tri[t, 0]] + a1[:, None] * m.v[tri[t, 1]] + a2[:, None] * m.v[tri[t, 2]]
    cbuf[Y, X] = col


def shade(zbuf, nbuf, cbuf, pbuf, eye, bg, key=None):
    H, W = zbuf.shape
    img = np.empty((H, W, 3))
    img[:] = bg
    hit = np.isfinite(zbuf)
    n = nbuf[hit]
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    p = pbuf[hit]
    view = p - eye
    view /= np.linalg.norm(view, axis=-1, keepdims=True)
    # two-sided: flip normals facing away from the camera
    back = (n * view).sum(-1) > 0
    n[back] *= -1
    c = cbuf[hit]
    if key is None:
        key = np.array([0.35, 0.55, 0.76])
        key /= np.linalg.norm(key)
    diff = np.clip((n * key).sum(-1), 0, 1)
    hemi = 0.55 + 0.45 * n[:, 2]
    refl = view - 2 * (view * n).sum(-1, keepdims=True) * n
    # studio environment: bright soft stripes above the horizon, darker floor
    elev = refl[:, 2]
    stripes = 0.5 + 0.5 * np.cos(elev * 14.0)
    env = np.where(elev > 0, 0.35 + 0.55 * stripes * np.clip(elev * 3, 0, 1), 0.10 + 0.05 * (1 + elev))
    fres = 0.04 + 0.96 * (1 - np.clip(-(view * n).sum(-1), 0, 1)) ** 5
    spec = np.clip((refl * key).sum(-1), 0, 1) ** 60
    lum = c * (0.20 * hemi + 0.75 * diff)[:, None] + (env * (0.10 + 0.6 * fres))[:, None] + 0.5 * spec[:, None]
    img[hit] = np.clip(lum, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


def side_by_side(images, out, pad=8):
    w = sum(i.width for i in images) + pad * (len(images) - 1)
    h = max(i.height for i in images)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for i in images:
        canvas.paste(i, (x, 0))
        x += i.width + pad
    canvas.save(out)
    return out


def stacked(images, out, pad=8):
    w = max(i.width for i in images)
    h = sum(i.height for i in images) + pad * (len(images) - 1)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for i in images:
        canvas.paste(i, (0, y))
        y += i.height + pad
    canvas.save(out)
    return out
