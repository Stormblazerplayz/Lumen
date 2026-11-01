"""Materials (BeamNG PBR, version 1.5) and the small textures they use, all
generated here. Paint follows the vehicle's paint slot 1 through a red palette
mask; glass is translucent with premultiplied alpha; lamps, the autonomy halo,
ambient strips and screens switch to emissive variants through the jbeam glowMap.
"""
import json
import os

import numpy as np
from PIL import Image

import textures

VDIR = "/vehicles/lumen/"


def _solid_png(path, rgba, size=4):
    Image.new("RGBA", (size, size), tuple(int(c) for c in rgba)).save(path)


def write_textures(tex_dir):
    os.makedirs(tex_dir, exist_ok=True)
    _solid_png(os.path.join(tex_dir, "lumen_paintmask_R.color.png"), (255, 0, 0, 255), size=16)
    _wordmark_maps(tex_dir)
    textures.screen_robotaxi(os.path.join(tex_dir, "lumen_screen.color.png"), 1024, 512)
    _dash_strip(os.path.join(tex_dir, "lumen_screen_dash.color.png"))
    _leather_texture(os.path.join(tex_dir, "lumen_leather.color.png"))
    # uniform opacity map for the cabin-side glass (opacity comes from opacityMap x opacityFactor)
    _solid_png(os.path.join(tex_dir, "lumen_glass_o.data.png"), (255, 255, 255, 255), size=16)


def _wordmark_maps(tex_dir):
    """BeamNG takes decal transparency from a separate opacity map (as the vanilla
    alpha-tested materials do), so split the RGBA wordmark into a colour map (RGB,
    transparent areas filled with the ink colour to avoid dark fringes) and a
    grayscale opacity map."""
    tmp = os.path.join(tex_dir, "_wordmark_rgba.png")
    textures.lumen_wordmark(tmp, 2048, 512)
    im = np.asarray(Image.open(tmp).convert("RGBA")).astype(np.float32)
    os.remove(tmp)
    a = im[..., 3:4] / 255.0
    ink = np.array(textures.GRAPHITE, np.float32)
    rgb = im[..., :3] * a + ink * (1 - a)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").save(os.path.join(tex_dir, "lumen_wordmark.color.png"))
    op = np.clip(im[..., 3], 0, 255).astype(np.uint8)
    Image.fromarray(np.dstack([op, op, op]), "RGB").save(os.path.join(tex_dir, "lumen_wordmark_o.data.png"))


def _dash_strip(path, w=1024, h=328):
    """Ride-display texture for the small floating centre display (332 x 106 mm, about 3.1 : 1)."""
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", (w, h), (12, 14, 18))
    d = ImageDraw.Draw(img)
    try:
        big = ImageFont.truetype("segoeuib.ttf", 58)
        mid = ImageFont.truetype("segoeui.ttf", 38)
        small = ImageFont.truetype("segoeui.ttf", 27)
    except OSError:
        big = mid = small = ImageFont.load_default()
    d.text((44, 30), "robotaxi", font=big, fill=(236, 238, 240))
    d.ellipse((268, 52, 288, 72), fill=textures.TURQ)
    d.text((44, 120), "Autonomous ride", font=mid, fill=(200, 204, 210))
    d.text((44, 172), "Sit back. We'll take it from here.", font=small, fill=(130, 136, 146))
    # route ribbon
    d.rounded_rectangle((44, 246, 980, 278), radius=16, fill=(30, 34, 42))
    d.rounded_rectangle((44, 246, 620, 278), radius=16, fill=(24, 214, 196))
    d.text((44, 288), "Live route", font=small, fill=(130, 136, 146))
    img.save(path)


def _leather_texture(path, size=512):
    """Dark grained leather with a grooved double-stitch seam along the inner (v = 0.75) and outer
    (v = 0.25) edge of the wheel rim (cockpit._rim_tube: u along the rim, v around the section).
    The seams are symmetric in v, so they land correctly whichever way the importer flips v."""
    from PIL import ImageFilter
    rng = np.random.default_rng(7)
    noise = rng.standard_normal((size, size)).astype(np.float32)
    g = np.asarray(Image.fromarray(((noise * 0.25 + 0.5).clip(0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.3)), np.float32) / 255.0 - 0.5
    g = g / max(float(g.std()), 1e-6)
    base = np.array([26, 26, 29], np.float32)
    img = base[None, None, :] * (1.0 + 0.16 * g[..., None])
    period, dash, th = 16, 10, 3
    for row in (int(0.75 * size), int(0.25 * size)):
        img[row - 2:row + 2, :, :] *= 0.55                       # the groove the thread sits beside
        for k, off in ((-1, 0), (1, period // 2)):
            r0 = row + k * 14
            for x0 in range(off, size, period):
                img[r0 - th // 2:r0 + th // 2 + 1, x0:x0 + dash, :] = (156, 156, 152)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").save(path)


def mat(name, stage0, **extra):
    m = {"name": name, "mapTo": name, "class": "Material",
         "Stages": [stage0, {}, {}, {}],
         "materialTag0": "beamng", "materialTag1": "vehicle", "version": 1.5, "dynamicCubemap": True}
    m.update(extra)
    return m


def emissive(name, rgb, nits, base=(0.9, 0.9, 0.9), translucent=False):
    st = {"baseColorFactor": [*base, 1], "emissiveFactor": list(rgb), "emissiveIntensityNits": nits,
          "roughnessFactor": 0.25, "metallicFactor": 0.0}
    extra = {"translucent": True, "translucentBlendOp": "None"} if translucent else {}
    return mat(name, st, **extra)


def lens(name, rgb, rough=0.12):
    return mat(name, {"baseColorFactor": [*rgb, 1], "roughnessFactor": rough, "metallicFactor": 0.0, "pixelSpecular": True})


def materials():
    M = {}
    add = lambda m: M.__setitem__(m["name"], m)
    # paint: slot 1 colour through the palette mask, with clear coat
    add(mat("rtx_paint", {"colorPaletteMap": VDIR + "lumen_paintmask_R.color.png", "colorPaletteMapUseUV": 0,
                          "clearCoatFactor": 1, "metallicFactor": 1, "roughnessFactor": 0.4}))
    add(mat("rtx_black_gloss", {"baseColorFactor": [0.012, 0.012, 0.014, 1], "roughnessFactor": 0.08, "metallicFactor": 0.0, "clearCoatFactor": 1}))
    add(mat("rtx_black_satin", {"baseColorFactor": [0.025, 0.025, 0.028, 1], "roughnessFactor": 0.55, "metallicFactor": 0.0}))
    add(mat("rtx_underbody", {"baseColorFactor": [0.02, 0.02, 0.02, 1], "roughnessFactor": 0.9, "metallicFactor": 0.0}))
    add(mat("rtx_inner", {"baseColorFactor": [0.03, 0.03, 0.03, 1], "roughnessFactor": 0.85, "metallicFactor": 0.0}))
    add(lens("rtx_lens", (0.005, 0.005, 0.008), rough=0.03))
    # tinted canopy glass: dark from outside, see-through from inside
    add(mat("rtx_glass", {"baseColorFactor": [0.015, 0.017, 0.02, 0.78], "roughnessFactor": 0.0, "metallicFactor": 0.0, "pixelSpecular": True},
            translucent=True, translucentBlendOp="PreMulAlpha", castShadows=False, translucentRecvShadows=True, alphaRef=0))
    # The same glass seen from the cabin (the inner skin from details.back_faces), built like the
    # stock cabin glass (bastion_glass_int, covet_glass_int): opacity from opacityMap x
    # opacityFactor (PBR v1.5 ignores the base colour's alpha, so rtx_glass above has always
    # been fully opaque, which is the black-glass look from outside), metallic, and a black sky
    # cubemap so the windshield does not mirror the sky back at the rider. 2026-10-31: sharing
    # rtx_glass made the cabin a dark smeared wall.
    add(mat("rtx_glass_inner", {"baseColorFactor": [0.03, 0.035, 0.04, 1], "opacityMap": VDIR + "lumen_glass_o.data.png",
                                "opacityFactor": 0.15, "roughnessFactor": 0.0, "metallicFactor": 1.0, "pixelSpecular": True},
            translucent=True, translucentBlendOp="PreMulAlpha", castShadows=False, translucentRecvShadows=True, alphaRef=0,
            dynamicCubemap=False, cubemap="BlackSkyCubemap"))
    # lamps: off and on variants (glowMap switches them)
    add(lens("rtx_drl", (0.75, 0.76, 0.78)))
    add(emissive("rtx_drl_on", (1.0, 1.0, 1.0), 6000))
    add(emissive("rtx_drl_on_intense", (1.0, 1.0, 1.0), 16000))
    add(lens("rtx_headlight", (0.25, 0.26, 0.28)))
    add(emissive("rtx_headlight_on", (1.0, 0.98, 0.94), 18000))
    add(emissive("rtx_headlight_on_intense", (1.0, 0.98, 0.94), 30000))
    add(lens("rtx_signal_fl", (0.70, 0.66, 0.55)))
    add(lens("rtx_signal_fr", (0.70, 0.66, 0.55)))
    add(emissive("rtx_signal_on", (1.0, 0.55, 0.05), 12000, base=(1.0, 0.6, 0.1)))
    for n in ("rtx_tail_l", "rtx_tail_r", "rtx_tailbar"):
        add(lens(n, (0.30, 0.02, 0.02)))
    add(emissive("rtx_tail_on", (1.0, 0.04, 0.03), 2500, base=(0.6, 0.03, 0.03)))
    add(emissive("rtx_tail_on_intense", (1.0, 0.04, 0.03), 9000, base=(0.8, 0.04, 0.04)))
    # autonomy halo (turquoise, the proposed marker colour for automated driving)
    add(lens("rtx_halo", (0.08, 0.30, 0.28)))
    add(emissive("rtx_halo_on", (0.09, 0.85, 0.78), 4000, base=(0.1, 0.8, 0.75)))
    # wheels
    add(mat("rtx_rim", {"baseColorFactor": [0.62, 0.63, 0.65, 1], "roughnessFactor": 0.28, "metallicFactor": 1.0, "clearCoatFactor": 1}))
    add(mat("rtx_rim_dark", {"baseColorFactor": [0.02, 0.02, 0.025, 1], "roughnessFactor": 0.4, "metallicFactor": 0.0}))
    add(mat("rtx_tire", {"baseColorFactor": [0.03, 0.03, 0.03, 1], "roughnessFactor": 0.85, "metallicFactor": 0.0}))
    add(mat("rtx_brake", {"baseColorFactor": [0.35, 0.35, 0.36, 1], "roughnessFactor": 0.45, "metallicFactor": 1.0}))
    # livery
    add(mat("rtx_decal", {"baseColorMap": VDIR + "lumen_wordmark.color.png", "opacityMap": VDIR + "lumen_wordmark_o.data.png",
                          "roughnessFactor": 0.35, "metallicFactor": 0.0, "clearCoatFactor": 1},
            alphaTest=True, alphaRef=100, castShadows=False, translucentBlendOp="None"))
    # cabin
    add(mat("rtx_seat", {"baseColorFactor": [0.80, 0.79, 0.76, 1], "roughnessFactor": 0.55, "metallicFactor": 0.0}))
    add(mat("rtx_seat_stitch", {"baseColorFactor": [0.48, 0.47, 0.45, 1], "roughnessFactor": 0.6, "metallicFactor": 0.0}))
    add(mat("rtx_interior_dark", {"baseColorFactor": [0.06, 0.06, 0.07, 1], "roughnessFactor": 0.5, "metallicFactor": 0.0}))
    add(mat("rtx_interior_trim", {"baseColorFactor": [0.55, 0.55, 0.57, 1], "roughnessFactor": 0.8, "metallicFactor": 0.0}))
    add(mat("rtx_carpet", {"baseColorFactor": [0.09, 0.09, 0.10, 1], "roughnessFactor": 0.95, "metallicFactor": 0.0}))
    # cockpit (cockpit.py): wheel leather, satin aluminium trim, speaker cloth, floor mats
    add(mat("rtx_leather", {"baseColorFactor": [0.035, 0.035, 0.038, 1], "roughnessFactor": 0.55, "metallicFactor": 0.0}))
    add(mat("rtx_leather_stitch", {"baseColorMap": VDIR + "lumen_leather.color.png", "roughnessFactor": 0.52, "metallicFactor": 0.0}))
    add(mat("rtx_fabric", {"baseColorFactor": [0.20, 0.20, 0.215, 1], "roughnessFactor": 0.96, "metallicFactor": 0.0}))
    add(mat("rtx_metal_trim", {"baseColorFactor": [0.62, 0.63, 0.65, 1], "roughnessFactor": 0.32, "metallicFactor": 1.0}))
    add(mat("rtx_speaker", {"baseColorFactor": [0.05, 0.05, 0.055, 1], "roughnessFactor": 0.95, "metallicFactor": 0.0}))
    add(mat("rtx_mat", {"baseColorFactor": [0.055, 0.055, 0.06, 1], "roughnessFactor": 0.98, "metallicFactor": 0.0}))
    add(lens("rtx_ambient", (0.06, 0.20, 0.19)))
    add(emissive("rtx_ambient_on", (0.09, 0.85, 0.78), 900, base=(0.1, 0.7, 0.65)))
    add(mat("rtx_screen_off", {"baseColorFactor": [0.01, 0.01, 0.012, 1], "roughnessFactor": 0.05, "metallicFactor": 0.0}))
    add(mat("rtx_screen_dash", {"baseColorFactor": [0.02, 0.02, 0.02, 1], "emissiveMap": VDIR + "lumen_screen_dash.color.png",
                                "emissiveFactor": [1, 1, 1], "emissiveIntensityNits": 700, "roughnessFactor": 0.08, "metallicFactor": 0.0}))
    add(mat("rtx_screen_rear", {"baseColorFactor": [0.02, 0.02, 0.02, 1], "emissiveMap": VDIR + "lumen_screen.color.png",
                                "emissiveFactor": [1, 1, 1], "emissiveIntensityNits": 700, "roughnessFactor": 0.08, "metallicFactor": 0.0}))
    return M


def write(vehicle_dir):
    write_textures(vehicle_dir)
    with open(os.path.join(vehicle_dir, "main.materials.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(materials(), f, indent=1)
