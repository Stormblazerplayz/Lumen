"""Textures drawn from scratch with PIL.

robotaxi_decal.png  : our own "Robotaxi" lettering: a slanted, angular brush script
                      with a long underline tail, champagne gold (the look of the
                      real door sticker, drawn as our own strokes, not a copy of it)
screen_robotaxi.png : the centre/rear screen picture (dark ride UI)
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

GOLD = (222, 197, 140)
GOLD_HI = (243, 226, 184)
GOLD_LO = (176, 146, 92)

# Each letter is a list of strokes; each stroke a list of (x, y) in letter units
# (y up, cap height 1.0, x-height ~0.62), placed at an x offset along the word.
LETTERS = [
    # R: stem, bowl, and a leg that sweeps out into the long underline tail
    (0.00, [[(0.00, 0.02), (0.20, 1.02)],
            [(0.16, 1.00), (0.58, 1.01), (0.72, 0.86), (0.64, 0.66), (0.44, 0.57), (0.12, 0.55)],
            [(0.30, 0.56), (0.52, 0.30), (0.70, 0.04), (1.10, -0.07), (2.4, -0.10), (4.2, -0.11), (5.05, -0.08)]]),
    # o
    (0.80, [[(0.17, 0.60), (0.43, 0.63), (0.53, 0.42), (0.45, 0.10), (0.22, 0.02), (0.06, 0.12), (0.02, 0.36), (0.17, 0.60)]]),
    # b: tall ascender and bowl
    (1.40, [[(0.20, 1.05), (0.02, 0.02)],
            [(0.07, 0.28), (0.24, 0.55), (0.44, 0.62), (0.56, 0.46), (0.50, 0.17), (0.30, 0.02), (0.04, 0.05)]]),
    # o
    (2.03, [[(0.17, 0.60), (0.43, 0.63), (0.53, 0.42), (0.45, 0.10), (0.22, 0.02), (0.06, 0.12), (0.02, 0.36), (0.17, 0.60)]]),
    # t: stem and crossbar
    (2.64, [[(0.24, 0.96), (0.10, 0.12), (0.17, 0.02), (0.36, 0.06)],
            [(0.00, 0.62), (0.44, 0.66)]]),
    # a: bowl and stem
    (3.06, [[(0.46, 0.55), (0.22, 0.62), (0.06, 0.48), (0.00, 0.20), (0.12, 0.02), (0.34, 0.08), (0.48, 0.30)],
            [(0.50, 0.63), (0.43, 0.10), (0.48, 0.01), (0.60, 0.04)]]),
    # x
    (3.70, [[(0.02, 0.62), (0.50, 0.00)], [(0.50, 0.63), (0.00, 0.00)]]),
    # i: stem and a slanted dot
    (4.30, [[(0.12, 0.60), (0.03, 0.03)], [(0.15, 0.84), (0.21, 0.90)]]),
]
SLANT = 0.22        # italic shear (x += SLANT * y)


def _stroke_points(stroke, step=0.01):
    pts = np.array(stroke, float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(2, int(np.linalg.norm(b - a) / step))
        for t in np.linspace(0, 1, n)[1:]:
            out.append(a + (b - a) * t)
    return np.array(out)


def robotaxi_decal(path, width=2048, height=540, margin=0.06):
    """Render the lettering. The texture aspect (3.8:1) matches the door patch."""
    sx0, sx1, sy0, sy1 = -0.10, 5.25, -0.24, 1.14       # word extents in letter units (after slant)
    scale = min(width * (1 - 2 * margin) / (sx1 - sx0), height * (1 - 2 * margin) / (sy1 - sy0))
    ox = (width - (sx1 - sx0) * scale) / 2 - sx0 * scale
    oy = (height + (sy1 - sy0) * scale) / 2 + sy0 * scale
    ss = 3  # supersample for smooth edges
    W, H = width * ss, height * ss
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    base_w = 0.085   # stroke thickness in letter units
    for x_off, strokes in LETTERS:
        for st in strokes:
            pts = _stroke_points([(x + x_off, y) for x, y in st])
            n = len(pts)
            # brush: thick in the middle of a stroke, tapering at its ends; the long
            # tail thins out to a hairline
            t = np.linspace(0, 1, n)
            length = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
            taper = np.minimum(1.0, np.minimum(t, 1 - t) * max(length, 0.5) / 0.12)
            width_f = base_w * (0.35 + 0.65 * taper)
            if length > 3.0:   # the underline tail
                width_f = base_w * (0.95 - 0.80 * t) * np.minimum(1.0, t * 30 + 0.4)
            for (px, py), wf in zip(pts, width_f):
                qx = (px + SLANT * py) * scale * ss + ox * ss
                qy = oy * ss - py * scale * ss
                r = wf * scale * ss / 2
                d.ellipse((qx - r, qy - r, qx + r, qy + r), fill=255)
    mask = mask.resize((width, height), Image.LANCZOS)
    # metallic-vinyl look: vertical gradient plus a soft highlight band
    yy = np.linspace(0, 1, height)[:, None]
    grad = np.clip(1.0 - yy, 0, 1)
    band = np.exp(-((yy - 0.35) / 0.12) ** 2)
    col = np.zeros((height, width, 3))
    for k in range(3):
        col[..., k] = GOLD_LO[k] + (GOLD[k] - GOLD_LO[k]) * grad[:, :1] + (GOLD_HI[k] - GOLD[k]) * band[:, :1] * 0.6
    rgba = np.dstack([np.clip(col, 0, 255).astype(np.uint8), np.asarray(mask)])
    Image.fromarray(rgba, "RGBA").save(path)
    return path


def screen_robotaxi(path, width=1024, height=512):
    """Static ride screen: dark map with a route line and a status card."""
    img = Image.new("RGB", (width, height), (16, 18, 22))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(7)
    # map streets
    for _ in range(26):
        x0, y0 = rng.integers(0, width), rng.integers(0, height)
        if rng.random() < 0.5:
            d.line((0, y0, width, y0 + rng.integers(-60, 60)), fill=(34, 38, 46), width=int(rng.integers(3, 9)))
        else:
            d.line((x0, 0, x0 + rng.integers(-60, 60), height), fill=(34, 38, 46), width=int(rng.integers(3, 9)))
    # route
    route = [(120, 520), (180, 430), (330, 420), (420, 300), (610, 280), (700, 150), (860, 120)]
    d.line(route, fill=(62, 132, 255), width=10, joint="curve")
    d.ellipse((108, 508, 132, 532), fill=(255, 255, 255))
    d.ellipse((846, 106, 874, 134), fill=(62, 132, 255), outline=(255, 255, 255), width=4)
    # status card
    d.rounded_rectangle((32, 32, 420, 190), radius=18, fill=(26, 29, 35))
    try:
        big = ImageFont.truetype("segoeuib.ttf", 44)
        small = ImageFont.truetype("segoeui.ttf", 26)
    except OSError:
        big = small = ImageFont.load_default()
    d.text((58, 50), "Robotaxi", font=big, fill=(236, 236, 236))
    d.text((60, 112), "Heading to your destination", font=small, fill=(170, 176, 186))
    d.text((60, 146), "Sit back and relax", font=small, fill=(120, 126, 136))
    img.save(path)
    return path


def _main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
    os.makedirs(out, exist_ok=True)
    lumen_wordmark(os.path.join(out, "lumen_wordmark.png"))
    screen_robotaxi(os.path.join(out, "screen_robotaxi.png"))
    print("ok")



# ---- Lumen livery: "robotaxi" wordmark, geometric strokes (our own letterforms) ----
GRAPHITE = (34, 36, 41)
TURQ = (24, 214, 196)


def _arc(cx, cy, r, a0, a1, n=60):
    t = np.radians(np.linspace(a0, a1, n))
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in t]


def lumen_wordmark(path, width=2048, height=None):
    """Lowercase geometric "robotaxi" (x-height 1.0, stroke 0.17) with a turquoise
    dot on the i and a thin turquoise underline segment."""
    sw = 0.17
    L = []   # (x offset, [strokes]); strokes are point lists in letter units, y up
    x = 0.0
    L.append((x, [[(0, 0), (0, 1.0)], _arc(0.52, 0.48, 0.52, 180, 95)]))        # r
    x += 0.90
    L.append((x, [_arc(0.5, 0.5, 0.5, 0, 361)]))                                 # o
    x += 1.30
    L.append((x, [[(0, 0), (0, 1.48)], _arc(0.5, 0.5, 0.5, 0, 361)]))            # b
    x += 1.30
    L.append((x, [_arc(0.5, 0.5, 0.5, 0, 361)]))                                 # o
    x += 1.28
    L.append((x, [[(0.18, 1.36), (0.18, 0.22)] + _arc(0.43, 0.22, 0.25, 180, 290), [(0.0, 1.0), (0.50, 1.0)]]))   # t
    x += 0.84
    L.append((x, [_arc(0.5, 0.5, 0.5, 0, 361), [(1.0, 1.0), (1.0, 0.0)]]))       # a
    x += 1.42
    L.append((x, [[(0, 1.0), (0.80, 0.0)], [(0.80, 1.0), (0, 0.0)]]))           # x
    x += 1.22
    L.append((x, [[(0, 0), (0, 1.0)]]))                                          # i stem
    total = x + 0.2
    margin_x, margin_y = 40, 50
    if height is None:   # canvas follows the word's aspect so the door patch can match it
        height = int(round(margin_y * 2 + (width - 2 * margin_x) * (1.48 + 0.42 + sw) / (total + sw)))
    scale = min((width - 2 * margin_x) / (total + sw), (height - 2 * margin_y) / (1.48 + 0.42 + sw))
    ss = 3
    W, H = width * ss, height * ss
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    base_y = height - margin_y - (0.42 * scale) - sw * scale / 2

    def to_px(px, py):
        return ((margin_x + sw * scale / 2 + px * scale) * ss, (base_y - py * scale) * ss)

    r = sw * scale * ss / 2
    for xo, strokes in L:
        for st in strokes:
            pts = _stroke_points([(px + xo, py) for px, py in st], step=0.006)
            for px, py in pts:
                cx, cy = to_px(px, py)
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=GRAPHITE + (255,))
    # turquoise dot on the i and a short underline that starts under the "x"
    cx, cy = to_px(x, 1.36)
    rd = sw * scale * ss * 0.62
    d.ellipse((cx - rd, cy - rd, cx + rd, cy + rd), fill=TURQ + (255,))
    ux0, uy = to_px(L[6][0] - 0.05, -0.30)
    ux1, _ = to_px(x + 0.12, -0.30)
    d.rounded_rectangle((ux0, uy - r * 0.45, ux1, uy + r * 0.45), radius=r * 0.45, fill=TURQ + (255,))
    img = img.resize((width, height), Image.LANCZOS)
    img.save(path)
    return path


if __name__ == "__main__":
    _main()
