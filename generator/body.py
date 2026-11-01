"""Build the exterior body meshes: skin -> seam cuts -> labels -> panel meshes.

    py body.py [quick]      builds and renders out/body_parts.png

Returns {mesh_name: Mesh} from build(); the same names are used by the jbeam
flexbodies, so the split between panels is the split between soft-body parts.
"""
import os
import sys
import time

import numpy as np
from PIL import Image

import details
import interior
import layout
import meshkit
import render
import skin
import surfcut

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")

PREVIEW = {
    "rtx_leather": (0.05, 0.05, 0.055), "rtx_leather_stitch": (0.07, 0.07, 0.075), "rtx_fabric": (0.30, 0.30, 0.32),
    "rtx_screen_off": (0.02, 0.02, 0.025), "rtx_metal_trim": (0.62, 0.63, 0.65), "rtx_speaker": (0.07, 0.07, 0.075),
    "rtx_mat": (0.08, 0.08, 0.085), "rtx_drl": (0.85, 0.86, 0.88),
    "rtx_seat": (0.80, 0.79, 0.76), "rtx_seat_stitch": (0.50, 0.49, 0.47), "rtx_interior_dark": (0.07, 0.07, 0.08),
    "rtx_interior_trim": (0.56, 0.56, 0.58), "rtx_carpet": (0.10, 0.10, 0.11), "rtx_ambient": (0.10, 0.95, 0.85),
    "rtx_screen_dash": (0.20, 0.45, 0.85), "rtx_screen_rear": (0.20, 0.45, 0.85),
    "rtx_rim": (0.30, 0.31, 0.33), "rtx_rim_dark": (0.03, 0.03, 0.035), "rtx_tire": (0.035, 0.035, 0.035), "rtx_brake": (0.25, 0.25, 0.25),
    "rtx_lens": (0.01, 0.01, 0.015),
    "rtx_halo": (0.10, 0.90, 0.82),
    "rtx_headlight": (0.90, 0.90, 0.88),
    "rtx_inner": (0.05, 0.05, 0.05),
    "rtx_underbody": (0.04, 0.04, 0.04),
    "rtx_paint": (0.86, 0.87, 0.88),
    "rtx_black_gloss": (0.02, 0.02, 0.025),
    "rtx_black_satin": (0.035, 0.035, 0.04),
    "rtx_glass": (0.05, 0.06, 0.07),
    "rtx_glass_roof": (0.03, 0.035, 0.04),
    "rtx_drl": (0.95, 0.95, 0.92),
    "rtx_signal_fl": (0.90, 0.85, 0.70), "rtx_signal_fr": (0.90, 0.85, 0.70),
    "rtx_tailbar": (0.55, 0.04, 0.04),
    "rtx_tail_l": (0.55, 0.04, 0.04), "rtx_tail_r": (0.55, 0.04, 0.04),
}


def build(quick=False, verbose=True):
    t0 = time.time()
    # the skin grid, scaled by meshkit.BODY_DETAIL (even counts; 1.0 gives the old grids)
    def ev(n, lo):
        return max(lo, 2 * int(round(n * meshkit.BODY_DETAIL / 2)))
    if quick:
        P = skin.build(n_cap=ev(22, 10), n_mid=ev(170, 60), n_theta=ev(180, 64), passes=1)
    else:
        P = skin.build(n_cap=ev(40, 12), n_mid=ev(340, 80), n_theta=ev(320, 80), passes=2)
    sm = surfcut.SurfMesh.from_grid(P)
    if verbose:
        print("skin %d verts %d tris (%.1fs)" % (len(sm.V), len(sm.F), time.time() - t0))
    for name, field, mask in layout.cuts():
        n = sm.cut(field, mask)
    if verbose:
        print("after cuts %d verts %d tris (%.1fs)" % (len(sm.V), len(sm.F), time.time() - t0))
    c, n = sm.centroid_fields()
    labels = layout.classify(c, n)
    specs = layout.panel_specs()
    meshes = {}
    unknown = set()
    for lab in sorted(set(labels.tolist())):
        if lab == "hole":
            continue
        if lab not in specs:
            unknown.add(lab)
            continue
        mesh_name, material, gap, flange, offset = specs[lab]
        tri = np.nonzero(labels == lab)[0]
        pm = surfcut.panel_mesh(sm, tri, mesh_name, material, gap=gap, flange=flange, offset=offset)
        if mesh_name in meshes:
            meshes[mesh_name].add_mesh(pm)
        else:
            meshes[mesh_name] = pm
    if unknown:
        raise RuntimeError("labels without a panel spec: %s" % sorted(unknown))
    add_details(meshes)
    if verbose:
        for k in sorted(meshes):
            mm = meshes[k]
            print("  %-26s %7d tris  %s" % (k, mm.triangle_count(), ",".join(sorted(mm.groups))))
        print("body built in %.1fs" % (time.time() - t0))
    return meshes


# flush handles: rear of the front doors, front of the rear-hinged rear doors
HANDLES = {"rtx_door_fl": (0.090, 0.250, 0.925), "rtx_door_rl": (0.395, 0.555, 0.945)}


def add_details(meshes):
    for side, s in ((1.0, "l"), (-1.0, "r")):
        for door, (y0, y1, zc) in HANDLES.items():
            name = door[:-1] + s
            meshes[name].add_mesh(details.door_handle(name, side, y0, y1, zc))
        meshes[f"rtx_campod_{s}"] = details.camera_pod(f"rtx_campod_{s}", side)
        meshes["rtx_bumper_f"].add_mesh(details.corner_sensor("rtx_bumper_f", side, -2.020, 0.640))
        meshes["rtx_body"].add_mesh(details.corner_sensor("rtx_body", side, 2.250, 0.985))
        meshes[f"rtx_liner_f{s}"] = details.wheel_liner(f"rtx_liner_f{s}", shape_y_axle("F"), side)
        meshes[f"rtx_liner_r{s}"] = details.wheel_liner(f"rtx_liner_r{s}", shape_y_axle("R"), side)
    meshes["rtx_lidar"] = details.lidar_crown("rtx_lidar")
    for name, m in meshes.items():
        details.back_faces(m)                                        # paint: dark inner skin
        # pillars, roof frame and glass are single-sided too; without inner skins the cabin had
        # no windshield, pillars or roof from the rear seat (seen in game, 2026-10-31)
        details.back_faces(m, source_material="rtx_black_gloss")
        details.back_faces(m, source_material="rtx_black_satin")
        details.back_faces(m, material="rtx_glass_inner", depth=0.003, source_material="rtx_glass")
    interior.cabin(meshes)


def shape_y_axle(which):
    import shape
    return shape.Y_AXLE_F if which == "F" else shape.Y_AXLE_R


def wheel_set():
    """Wheels and tyres placed at the four corners (preview only; in game the
    flexbody pos/rot does this)."""
    import shape
    import wheels
    w, t = wheels.wheel_mesh(), wheels.tire_mesh()
    out = []
    x = shape.TRACK / 2
    for y in (shape.Y_AXLE_F, shape.Y_AXLE_R):
        for side in (1.0, -1.0):
            R = np.eye(3) if side > 0 else np.diag([-1.0, -1.0, 1.0])   # 180 deg about z
            out.append(w.transformed(R, (side * x, y, shape.WHEEL_Z)))
            out.append(t.transformed(R, (side * x, y, shape.WHEEL_Z)))
    return out


def preview(meshes, out_name="body_parts.png"):
    ms = list(meshes.values()) + wheel_set()
    views = []
    for eye, tgt in (((-5.6, -4.6, 1.35), (0, -0.2, 0.62)), ((-5.0, 5.4, 1.75), (0, 0.2, 0.65)),
                     ((-1.8, -6.6, 1.25), (0, 0, 0.65)), ((2.6, -2.2, 4.6), (0, 0.1, 0.6))):
        views.append(render.render(ms, eye=eye, target=tgt, width=800, height=480, fov_deg=30, material_color=PREVIEW))
    side = render.render(ms, eye=(-20, 0.05, 0.80), target=(0, 0.05, 0.80), width=1600, height=520, ortho_height=1.85, material_color=PREVIEW)
    a = render.side_by_side(views[:2], os.path.join(OUT, "_a.png"))
    b = render.side_by_side(views[2:], os.path.join(OUT, "_b.png"))
    render.stacked([Image.open(a), Image.open(b), side], os.path.join(OUT, out_name))
    os.remove(a)
    os.remove(b)


def preview_cabin(meshes, out_name="cabin.png"):
    glassy = ("rtx_canopy", "rtx_glass_", "rtx_tailgate", "rtx_lidar")
    inside = [m for k, m in meshes.items() if not k.startswith(glassy)]
    cut = [m for k, m in meshes.items() if not k.startswith(glassy) and not k.startswith(("rtx_door", "rtx_campod"))]
    a = render.render(cut, eye=(-3.4, 2.4, 4.6), target=(0, -0.05, 0.50), width=900, height=560, fov_deg=44, material_color=PREVIEW)
    b = render.render(inside, eye=(-0.36, 0.80, 1.16), target=(0.05, -1.2, 0.86), width=900, height=560, fov_deg=74, material_color=PREVIEW)
    render.side_by_side([a, b], os.path.join(OUT, out_name))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    m = build(quick="quick" in sys.argv)
    preview(m)
    preview_cabin(m)
