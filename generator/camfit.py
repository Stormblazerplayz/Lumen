"""Fit the cameras of Storm's front and rear reference renders.

The references are perspective renders, so overlaying them needs their camera.
Our skin already matches the side view, so its top outline (roof, pillars, hood,
fenders, spoiler) is trusted: search camera distance, height, pitch and focal
length until the projected top outline of our skin matches the reference's top
outline column by column (mirror columns excluded). Vertex splatting is enough
for an outline, so thousands of candidates take seconds.

Writes ref/cameras.json, used by preview.py.
"""
import json
import os

import numpy as np
from PIL import Image

import skin

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "ref")


def ref_top_outline(path):
    """Top outline per column, and the body's extent measured on lower-body rows only
    (the bottom rows hold the ground shadow, which spans the whole image)."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(float)
    car = np.abs(im - 247).max(axis=2) > 25
    H, W = car.shape
    top = np.where(car.any(axis=0), np.argmax(car, axis=0), H).astype(float)
    x0, x1 = W, 0
    for v in range(int(0.55 * H), int(0.92 * H)):
        xs = np.nonzero(car[v])[0]
        if len(xs):
            x0, x1 = min(x0, xs.min()), max(x1, xs.max())
    return top, W, H, x0, x1


def project(P, D, h, pitch, f, W, H, rear):
    # camera on the centre line, looking along +y (front view) or -y (rear view)
    s = -1.0 if rear else 1.0
    cam = np.array([0.0, -s * D, h])
    fwd = np.array([0.0, s * np.cos(pitch), -np.sin(pitch)])
    right = np.array([s * -1.0, 0.0, 0.0]) if not rear else np.array([1.0, 0.0, 0.0])
    # image right for a front view is the car's left (+x)... keep a right-handed basis
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    V = P - cam
    z = V @ fwd
    x = V @ right
    y = V @ up
    u = W / 2 + f * x / z
    v = H / 2 - f * y / z
    return u, v


def outline_from_points(u, v, W, H):
    top = np.full(W, float(H))
    col = np.floor(u).astype(int)
    ok = (col >= 0) & (col < W) & (v >= -50)
    np.minimum.at(top, col[ok], v[ok])
    return top


def fit(ref_name, rear):
    top_ref, W, H, bx0, bx1 = ref_top_outline(os.path.join(REF, ref_name))
    P = skin.build(n_cap=20, n_mid=150, n_theta=160, passes=1).reshape(-1, 3)
    # exclude the mirror columns: where the reference outline jumps up outside the body
    best = None
    for D in np.arange(4.0, 16.01, 0.25):
        for h in np.arange(0.6, 1.81, 0.05):
            for pitch_deg in np.arange(-4, 24.01, 1.0):
                pitch = np.radians(pitch_deg)
                # focal length: scale so the body's projected width matches the reference
                u, v = project(P, D, h, pitch, 1000.0, W, H, rear)
                width_ours = (u.max() - u.min())
                width_ref = bx1 - bx0
                f = 1000.0 * width_ref / width_ours
                u, v = project(P, D, h, pitch, f, W, H, rear)
                top = outline_from_points(u, v, W, H)
                # compare in the central 70% of the body width (mirrors live at the sides)
                c0 = bx0 + 0.15 * width_ref
                c1 = bx1 - 0.15 * width_ref
                sel = np.arange(int(c0), int(c1))
                err = np.median(np.abs(top[sel] - top_ref[sel])) + 0.3 * np.mean(np.abs(top[sel] - top_ref[sel]))
                if best is None or err < best[0]:
                    best = (err, D, h, pitch_deg, f)
    return best, W, H


def main():
    out = {}
    for name, rear in (("storm_1_front.png", False), ("storm_4_rear.png", True)):
        (err, D, h, pitch, f), W, H = fit(name, rear)
        out[name] = {"distance": D, "height": h, "pitch_deg": pitch, "focal_px": f, "rear": rear, "err_px": err, "size": [W, H]}
        print(name, "D=%.2f h=%.2f pitch=%.0f f=%.0f  err=%.1fpx" % (D, h, pitch, f, err))
    with open(os.path.join(REF, "cameras.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
