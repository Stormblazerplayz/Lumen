"""Trace Storm's side reference render into vehicle coordinates.

The reference images are perspective renders, so one pixel scale does not fit the
whole car (fitting the wheelbase alone gives a 4.45 m car instead of 4.797 m).
Horizontal positions are therefore mapped piecewise-linearly between four anchors
whose vehicle positions are known from Tesla's owner's manual (2025 Model Y,
Premium): rear bumper end, rear axle, front axle, front bumper end. Vertical
positions are mapped linearly between the hub height and the roof top.

Vehicle frame (BeamNG): +y is rearward, -x is the right side, +z is up, ground z=0,
wheelbase centred on y=0.

Output: ref/side_trace.json and ref/side_trace_overlay.png (for a visual check).
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "ref")

# Tesla owner's manual, Model Y (2025) Premium: wheelbase 2890, front overhang 896
# (with plate bracket), rear overhang 1011, height 1624 mm.
WB, OH_F, OH_R, HEIGHT = 2.890, 0.896, 1.011, 1.624
Y_FRONT_AXLE, Y_REAR_AXLE = -WB / 2, WB / 2
Y_FRONT_END, Y_REAR_END = Y_FRONT_AXLE - OH_F, Y_REAR_AXLE + OH_R
TIRE_R = 0.356  # 255/45 R19: 19*25.4/2 + 0.45*255 mm


def fit_rim(lum, sat, x0, x1, y0, y1, rmax=90):
    """Least-squares circle through the outer edge of the silver rim."""
    H, W = lum.shape
    sub = (lum[y0:y1, x0:x1] > 150) & (sat[y0:y1, x0:x1] < 30)
    ys, xs = np.nonzero(sub)
    cx, cy = xs.mean() + x0, ys.mean() + y0
    r = 0.0
    for _ in range(8):
        pts = []
        for a in np.linspace(-np.pi, np.pi, 720, endpoint=False):
            rr = np.arange(15, rmax)
            px = (cx + rr * np.cos(a)).astype(int)
            py = (cy + rr * np.sin(a)).astype(int)
            ok = (px >= x0) & (px < x1) & (py >= y0) & (py < min(y1, H))
            px, py = px[ok], py[ok]
            if len(px) == 0:
                continue
            b = (lum[py, px] > 150) & (sat[py, px] < 30)
            if b.any():
                k = np.nonzero(b)[0].max()
                pts.append((px[k], py[k]))
        pts = np.array(pts, float)
        for _ in range(3):
            A = np.c_[2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))]
            sol, *_ = np.linalg.lstsq(A, (pts ** 2).sum(1), rcond=None)
            cx, cy = sol[0], sol[1]
            r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
            res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
            pts = pts[res < max(2.0, np.percentile(res, 75))]
    return cx, cy, r


def main():
    path = os.path.join(REF, "storm_2_side_right.png")  # front of the car is on the right
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    H, W, _ = im.shape
    lum = im.mean(axis=2)
    sat = im.max(axis=2) - im.min(axis=2)
    car = np.abs(im - 247.0).max(axis=2) > 25

    rear = fit_rim(lum, sat, 130, 360, 340, H)
    front = fit_rim(lum, sat, 900, 1120, 340, H)
    xs = np.nonzero(car.any(axis=0))[0]
    u_rear_end, u_front_end = float(xs.min()), float(xs.max())

    # u (image x) -> vehicle y, piecewise linear through the four anchors
    U = np.array([u_rear_end, rear[0], front[0], u_front_end])
    Y = np.array([Y_REAR_END, Y_REAR_AXLE, Y_FRONT_AXLE, Y_FRONT_END])

    def u_to_y(u):
        return np.interp(u, U, Y)

    def y_to_u(y):
        return np.interp(-np.asarray(y), -Y, U)

    v_hub = (rear[1] + front[1]) / 2
    v_roof = float(np.nonzero(car.any(axis=1))[0].min())
    sv = (HEIGHT - TIRE_R) / (v_hub - v_roof)

    def v_to_z(v):
        return TIRE_R + (v_hub - np.asarray(v)) * sv

    def z_to_v(z):
        return v_hub - (np.asarray(z) - TIRE_R) / sv

    # top silhouette: first car pixel from the top in each column
    top = []
    for u in range(int(u_rear_end), int(u_front_end) + 1):
        col = car[:, u]
        if col.any():
            v = int(np.argmax(col))
            top.append((float(u_to_y(u)), float(v_to_z(v))))

    # paint vs black: classify pixels for the side features
    paint = car & (lum > 70) & (lum < 175) & (sat < 40)
    black = car & (lum < 45)

    # beltline: in each column between the A pillar and the C pillar, the lowest
    # black (window seal / glass frame) pixel that sits above a long run of paint.
    belt = []
    for u in range(300, 860):
        col_p = paint[:, u]
        col_b = black[:, u]
        # scan from z=0.8 up to z=1.4 for the first transition paint -> black
        v_lo, v_hi = int(z_to_v(0.85)), int(z_to_v(1.45))
        for v in range(v_lo, v_hi, -1):
            if col_b[v] and col_p[v + 1:v + 25].mean() > 0.8:
                belt.append((float(u_to_y(u)), float(v_to_z(v))))
                break

    out = {
        "source": "ref/storm_2_side_right.png",
        "anchors_px": {"rear_end": u_rear_end, "rear_axle": rear[0], "front_axle": front[0], "front_end": u_front_end},
        "rim_px": {"rear": [rear[0], rear[1], rear[2]], "front": [front[0], front[1], front[2]]},
        "vertical_scale_m_per_px": sv,
        "hub_v": v_hub, "roof_v": v_roof,
        "top": top,
        "belt": belt,
    }
    with open(os.path.join(REF, "side_trace.json"), "w") as f:
        json.dump(out, f, indent=1)

    # overlay for a visual check
    vis = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(vis)
    for (y, z) in top[::3]:
        d.point((float(y_to_u(y)), float(z_to_v(z))), fill=(255, 0, 0))
    for (y, z) in belt[::2]:
        d.point((float(y_to_u(y)), float(z_to_v(z))), fill=(0, 200, 255))
    for c in (rear, front):
        d.ellipse((c[0] - c[2], c[1] - c[2], c[0] + c[2], c[1] + c[2]), outline=(0, 255, 0))
    vis.save(os.path.join(REF, "side_trace_overlay.png"))

    print("anchors px", U.round(1), "vertical m/px %.5f" % sv)
    print("top samples (y, z):")
    for i in range(0, len(top), max(1, len(top) // 30)):
        print("  %.3f %.3f" % top[i])
    print("belt samples:", len(belt))
    for i in range(0, len(belt), max(1, len(belt) // 12)):
        print("  %.3f %.3f" % belt[i])


if __name__ == "__main__":
    main()
