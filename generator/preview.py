"""Build the body skin and render comparison images against Storm's references.

    py preview.py [quick]

Writes into ../out/:
  body_side_overlay.png   our orthographic right side blended over the rectified reference
  body_views.png          front 3/4, rear 3/4, front, rear
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image

import meshkit
import render
import shape
import skin

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
REF = os.path.join(HERE, "..", "ref")
os.makedirs(OUT, exist_ok=True)

# metric frame for side comparisons: y from -2.55 to 2.65, z from -0.05 to 1.75
PX_PER_M = 240
PREVIEW_COLORS = {"paint": (0.50, 0.52, 0.55)}
Y0, Y1, Z0, Z1 = -2.55, 2.65, -0.05, 1.75


def skin_mesh(P, name="skin"):
    R, C, _ = P.shape
    m = meshkit.Mesh(name)
    v = P.reshape(-1, 3)
    n = skin.normals(P).reshape(-1, 3)
    uv = np.zeros((len(v), 2))
    f = meshkit.grid_faces(R, C, wrap_cols=True)
    # faces must wind outward: check one and flip all if needed
    fn = meshkit.face_normals(v, f)
    if (fn * n[f[:, 0]]).sum(-1).mean() < 0:
        f = f[:, ::-1]
    m.add(v, n, uv, f, "paint")
    return m


def rectified_reference():
    t = json.load(open(os.path.join(REF, "side_trace.json")))
    im = np.asarray(Image.open(os.path.join(REF, "storm_2_side_right.png")).convert("RGB"))
    A = t["anchors_px"]
    U = np.array([A["rear_end"], A["rear_axle"], A["front_axle"], A["front_end"]])
    Yk = np.array([shape.Y_REAR, shape.Y_AXLE_R, shape.Y_AXLE_F, shape.Y_FRONT])
    W = int((Y1 - Y0) * PX_PER_M)
    H = int((Z1 - Z0) * PX_PER_M)
    # output pixel -> vehicle (y, z): image right = front (-y)
    xs = np.arange(W) + 0.5
    ys = np.arange(H) + 0.5
    yv = Y1 - xs / PX_PER_M
    zv = Z1 - ys / PX_PER_M
    u = np.interp(-yv, -Yk, U)
    v = t["hub_v"] - (zv - shape.TIRE_RADIUS) / t["vertical_scale_m_per_px"]
    uu, vv = np.meshgrid(u, v)
    ok = (uu >= 0) & (uu < im.shape[1] - 1) & (vv >= 0) & (vv < im.shape[0] - 1)
    out = np.full((H, W, 3), 247, np.uint8)
    out[ok] = im[vv[ok].astype(int), uu[ok].astype(int)]
    return Image.fromarray(out)


def side_render(meshes):
    W = int((Y1 - Y0) * PX_PER_M)
    H = int((Z1 - Z0) * PX_PER_M)
    yc = (Y0 + Y1) / 2
    zc = (Z0 + Z1) / 2
    return render.render(meshes, eye=(-20.0, yc, zc), target=(0.0, yc, zc), width=W, height=H, ortho_height=(Z1 - Z0), material_color=PREVIEW_COLORS)


def main(quick=False):
    t0 = time.time()
    if quick:
        P = skin.build(n_cap=20, n_mid=150, n_theta=160, passes=1)
    else:
        P = skin.build()
    print("skin grid", P.shape, "%.1fs" % (time.time() - t0))
    m = skin_mesh(P)
    lo, hi = m.bbox()
    print("bbox min", lo.round(3), "max", hi.round(3))
    print("length %.3f (4.797)  width %.3f (1.920)  height %.3f (1.624)" % (hi[1] - lo[1], hi[0] - lo[0], hi[2]))

    side = side_render([m])
    ref = rectified_reference()
    blend = Image.blend(ref.convert("RGB"), side.convert("RGB"), 0.5)
    render.stacked([ref, side, blend], os.path.join(OUT, "body_side_overlay.png"))

    # front / rear against Storm's references, with cameras fitted by camfit.py
    cams = json.load(open(os.path.join(REF, "cameras.json")))
    for ref_name, out_name in (("storm_1_front.png", "body_front_overlay.png"), ("storm_4_rear.png", "body_rear_overlay.png")):
        c = cams[ref_name]
        refim = Image.open(os.path.join(REF, ref_name)).convert("RGB")
        Wr, Hr = refim.size
        sgn = -1.0 if c["rear"] else 1.0
        pitch = np.radians(c["pitch_deg"])
        eye = np.array([0.0, -sgn * c["distance"], c["height"]])
        target = eye + np.array([0.0, sgn * np.cos(pitch), -np.sin(pitch)])
        fov = np.degrees(2 * np.arctan((Hr / 2) / c["focal_px"]))
        ours = render.render([m], eye=eye, target=target, width=Wr, height=Hr, fov_deg=fov, material_color=PREVIEW_COLORS)
        render.side_by_side([refim, ours, Image.blend(refim, ours, 0.5)], os.path.join(OUT, out_name))

    views = []
    for eye in ((-5.2, -5.6, 2.0), (-5.2, 5.8, 2.2), (0.0, -9.0, 0.9), (0.0, 9.0, 1.0)):
        views.append(render.render([m], eye=eye, target=(0, 0, 0.75), width=700, height=420, fov_deg=26, material_color=PREVIEW_COLORS))
    top = render.side_by_side(views[:2], os.path.join(OUT, "_a.png"))
    bot = render.side_by_side(views[2:], os.path.join(OUT, "_b.png"))
    render.stacked([Image.open(top), Image.open(bot)], os.path.join(OUT, "body_views.png"))
    os.remove(top)
    os.remove(bot)
    print("done %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main(quick="quick" in sys.argv)
