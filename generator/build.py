"""Build the Lumen robotaxi mod from scratch.

    py build.py [quick] [--install-test]

Writes mod/ (the unpacked mod: vehicles/lumen, lua, ui, scripts) and, with
--install-test, mirrors it into the isolated test profile only
(%LOCALAPPDATA%\\BeamNGTest\\robotaxi). Storm's real profile is never touched here.
"""
import json
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image

import body
import dae
import details
import jbeam_gen
import materials_gen
import render
import shape
import wheels

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
MOD = os.path.join(ROOT, "mod")
VEH = os.path.join(MOD, "vehicles", "lumen")
SRC_LUA = os.path.join(ROOT, "src")          # hand-written Lua/UI sources copied into the mod
TEST_PROFILE = os.path.join(os.environ["LOCALAPPDATA"], "BeamNGTest", "robotaxi")

PAINTS = {
    "Pearl White": {"baseColor": [0.92, 0.93, 0.94, 1.2], "metallic": 0.15, "roughness": 0.35, "clearcoat": 1, "clearcoatRoughness": 0.04},
    "Graphite": {"baseColor": [0.10, 0.105, 0.115, 1.2], "metallic": 0.85, "roughness": 0.38, "clearcoat": 1, "clearcoatRoughness": 0.04},
    "Lunar Silver": {"baseColor": [0.62, 0.64, 0.66, 1.2], "metallic": 0.95, "roughness": 0.32, "clearcoat": 1, "clearcoatRoughness": 0.03},
    "Deep Teal": {"baseColor": [0.03, 0.22, 0.24, 1.2], "metallic": 0.85, "roughness": 0.34, "clearcoat": 1, "clearcoatRoughness": 0.03},
    "Ember Orange": {"baseColor": [0.80, 0.25, 0.06, 1.2], "metallic": 0.6, "roughness": 0.36, "clearcoat": 1, "clearcoatRoughness": 0.03},
}


def decals(meshes):
    for side, s in ((1.0, "l"), (-1.0, "r")):
        meshes[f"rtx_decal_{s}"] = details.side_patch(f"rtx_decal_{s}", "rtx_decal", -0.96, -0.08, 0.48, 0.70, side, nu=64, nv=14, lift=0.0016)  # 4:1 like the 2048x512 texture


def write_info():
    info = {
        "Name": "Lumen", "Brand": "Frontier", "Body Style": "Sedan", "Type": "Car", "Country": "United States",
        "Description": "Lumen is a fully autonomous electric robotaxi: a black-glass canopy, a turquoise autonomy halo that lights up while it drives itself, and a lounge cabin with no steering wheel. Call one to your location, pick a destination, and ride.",
        "Author": "Frontier Studios", "Years": {"min": 2027, "max": 2027}, "default_pc": "robotaxi",
        "defaultPaintName1": "Pearl White", "paints": PAINTS,
    }
    with open(os.path.join(VEH, "info.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=1)


def pc(paint, parts):
    p = PAINTS[paint]
    paint_struct = {"baseColor": p["baseColor"], "metallic": p["metallic"], "roughness": p["roughness"],
                    "clearcoat": p["clearcoat"], "clearcoatRoughness": p["clearcoatRoughness"]}
    return {"format": 2, "model": "lumen", "mainPartName": "lumen", "parts": parts, "paints": [paint_struct, paint_struct, paint_struct]}


def write_configs():
    configs = {
        "robotaxi": (pc("Pearl White", {"paint_design": "lumen_livery_robotaxi", "lumen_sensors": "lumen_sensors"}),
                     {"Configuration": "Robotaxi", "Description": "Fleet robotaxi: livery, sensor crown, autonomy halo.", "Value": 72000, "Population": 0,
                      "Drivetrain": "AWD", "Propulsion": "Electric", "Fuel Type": "Battery", "Power": 410, "Torque": 700, "Weight": 2050, "Config Type": "Factory", "defaultPaintName1": "Pearl White"}),
        "private": (pc("Graphite", {"paint_design": "", "lumen_sensors": ""}),
                    {"Configuration": "Private", "Description": "Private Lumen without the fleet livery or sensor crown.", "Value": 64000, "Population": 0,
                     "Drivetrain": "AWD", "Propulsion": "Electric", "Fuel Type": "Battery", "Power": 410, "Torque": 700, "Weight": 2030, "Config Type": "Factory", "defaultPaintName1": "Graphite"}),
    }
    for name, (cfg, info) in configs.items():
        with open(os.path.join(VEH, f"{name}.pc"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(cfg, f, indent=1)
        with open(os.path.join(VEH, f"info_{name}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(info, f, indent=1)
    return list(configs)


def thumbnails(meshes, wheel_set, names):
    cols = dict(body.PREVIEW)
    for name in names:
        ms = [m for k, m in meshes.items() if not (name == "private" and (k.startswith("rtx_decal") or k == "rtx_lidar"))]
        c = dict(cols)
        if name == "private":
            c["rtx_paint"] = (0.11, 0.115, 0.125)
        img = render.render(ms + wheel_set, eye=(-5.4, -4.4, 1.5), target=(0, 0.1, 0.62), width=500, height=281, fov_deg=30,
                            material_color=c, bg=(0.93, 0.94, 0.95))
        img.save(os.path.join(VEH, f"{name}.jpg"), quality=92)
    shutil.copyfile(os.path.join(VEH, f"{names[0]}.jpg"), os.path.join(VEH, "default.jpg"))


def copy_sources():
    """Copy src/ into mod/ and remove files that src/ no longer has from the trees it owns
    (art, lua, scripts, ui); mod/vehicles is generated and left alone. 2026-10-31: renaming the
    AR textures left the old ar_*.color.png in mod/ and in the release zip."""
    if not os.path.isdir(SRC_LUA):
        return 0
    n = 0
    wanted = set()
    for dirpath, _, files in os.walk(SRC_LUA):
        for fn in files:
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(src, SRC_LUA)
            dst = os.path.join(MOD, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            wanted.add(os.path.normcase(dst))
            n += 1
    for top in os.listdir(SRC_LUA):
        tree = os.path.join(MOD, top)
        if top == "vehicles" or not os.path.isdir(tree):
            continue
        for dirpath, _, files in os.walk(tree):
            for fn in files:
                p = os.path.join(dirpath, fn)
                if os.path.normcase(p) not in wanted:
                    os.remove(p)
                    print("removed stale", os.path.relpath(p, MOD))
    return n


def release():
    """Zip mod/ into out/release/lumen_robotaxi.zip (forward-slash paths, deflate), written to a
    temporary name first so a half-written zip is never left where the installer looks."""
    import zipfile
    out_dir = os.path.join(ROOT, "out", "release")
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, "lumen_robotaxi.zip")
    tmp = final + ".part"
    count = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for dirpath, dirs, files in os.walk(MOD):
            dirs.sort()
            for fn in sorted(files):
                p = os.path.join(dirpath, fn)
                z.write(p, os.path.relpath(p, MOD).replace(os.sep, "/"))
                count += 1
    os.replace(tmp, final)
    return final, count


def install_test():
    """Mirror mod/ into the test profile's unpacked mods (both candidate layouts)."""
    for root in (TEST_PROFILE, os.path.join(TEST_PROFILE, "current")):
        dst = os.path.join(root, "mods", "unpacked", "lumen_robotaxi")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(MOD, dst)
    return True


def main():
    if "scripts" in sys.argv:
        # Lua, UI and input files only; the vehicle in mod/ stays as last built
        print("copied %d source files" % copy_sources())
        if "--install-test" in sys.argv:
            install_test()
            print("installed into the test profile")
        if "--release" in sys.argv:
            print("release: %s (%d files)" % release())
        return
    quick = "quick" in sys.argv
    t0 = time.time()
    if os.path.isdir(VEH):
        shutil.rmtree(VEH)
    os.makedirs(VEH)
    meshes = body.build(quick=quick, verbose=False)
    decals(meshes)
    meshes["rtx_wheel"] = wheels.wheel_mesh("rtx_wheel")
    meshes["rtx_tire"] = wheels.tire_mesh("rtx_tire")
    ml = [m.compact() for m in meshes.values() if m.triangle_count() > 0]
    dae.write_dae(os.path.join(VEH, "lumen.dae"), ml)
    tris = sum(m.triangle_count() for m in ml)
    print("meshes %d, triangles %d (%.1fs)" % (len(ml), tris, time.time() - t0))
    stats = jbeam_gen.build(VEH)
    print("jbeam:", stats)
    if stats["missing"]:
        raise SystemExit("jbeam references missing nodes: %s" % stats["missing"])
    materials_gen.write(VEH)
    write_info()
    names = write_configs()
    thumbnails(meshes, body.wheel_set(), names)
    n = copy_sources()
    print("copied %d source files; mod at %s (%.1fs)" % (n, MOD, time.time() - t0))
    if "--install-test" in sys.argv:
        install_test()
        print("installed into the test profile")
    if "--release" in sys.argv:
        print("release: %s (%d files)" % release())


if __name__ == "__main__":
    main()
