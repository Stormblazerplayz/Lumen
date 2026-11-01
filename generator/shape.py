"""The Lumen robotaxi body as one smooth implicit function F(x, y, z): F < 0 inside.

Lumen is our own design (Storm, 2026-10-31: "forget a tesla. Just a cool looking
self driving car"): low and wide, 21-inch wheels pushed to the corners, a sharp
forward-leaning nose carrying a razor light bar, a 26-degree windshield that runs
into one continuous black-glass canopy, and a short kammback tail with a crisp lip.

Why implicit: every panel, glass pane, trim strip and decal is cut out of, or
projected onto, the same surface, so neighbouring parts always meet exactly and
normals (from the gradient) are smooth everywhere.

Construction:
  F = smooth_max(F_slab, F_plan, r)
  F_slab  : between the bottom line and the crowned top line of the side view
  F_plan  : inside the plan outline at each height (side section with tuck-under,
            shoulder and canopy tumblehome), closed at the nose and tail by
            superelliptic corners whose tips follow the nose/tail face profiles
  r       : fillet radius between the two

Frame: +x left, +y rearward, +z up, ground z = 0, wheelbase centred on y = 0.
"""
import numpy as np

np.seterr(divide='ignore', invalid='ignore')  # flat Pchip segments divide by zero in an unused branch

# ---- package (our design) -------------------------------------------------------
WHEELBASE = 3.000
OVERHANG_F = 0.800
OVERHANG_R = 0.900
HEIGHT = 1.560
HALF_WIDTH = 0.980          # 1.96 m body
TRACK = 1.700
Y_AXLE_F = -WHEELBASE / 2
Y_AXLE_R = WHEELBASE / 2
Y_FRONT = Y_AXLE_F - OVERHANG_F   # -2.30
Y_REAR = Y_AXLE_R + OVERHANG_R    # +2.40
TIRE_RADIUS = 0.369               # 255/40 R21
WHEEL_Z = TIRE_RADIUS
ARCH_RADIUS = 0.432               # tight arches around the big wheels


# ---- monotone cubic interpolation (Fritsch-Carlson) ----------------------------
class Pchip:
    def __init__(self, xs, ys):
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        order = np.argsort(xs)
        self.x, self.y = xs[order], ys[order]
        h = np.diff(self.x)
        d = np.diff(self.y) / h
        m = np.zeros_like(self.y)
        m[1:-1] = np.where(d[:-1] * d[1:] > 0, 2.0 / (1.0 / np.where(d[:-1] == 0, 1e-12, d[:-1]) + 1.0 / np.where(d[1:] == 0, 1e-12, d[1:])), 0.0)
        m[0], m[-1] = d[0], d[-1]
        self.m, self.h = m, h

    def __call__(self, t):
        t = np.asarray(t, float)
        x, y, m, h = self.x, self.y, self.m, self.h
        tc = np.clip(t, x[0], x[-1])
        i = np.clip(np.searchsorted(x, tc) - 1, 0, len(x) - 2)
        s = (tc - x[i]) / h[i]
        h00 = (1 + 2 * s) * (1 - s) ** 2
        h10 = s * (1 - s) ** 2
        h01 = s * s * (3 - 2 * s)
        h11 = s * s * (s - 1)
        out = h00 * y[i] + h10 * h[i] * m[i] + h01 * y[i + 1] + h11 * h[i] * m[i + 1]
        # linear extrapolation outside the table
        lo, hi = t < x[0], t > x[-1]
        out = np.where(lo, y[0] + m[0] * (t - x[0]), out)
        out = np.where(hi, y[-1] + m[-1] * (t - x[-1]), out)
        return out


def smoothstep(a, b, t):
    s = np.clip((np.asarray(t, float) - a) / (b - a), 0.0, 1.0)
    return s * s * (3 - 2 * s)


def smax(a, b, r):
    """Polynomial smooth maximum (rounded intersection), radius ~r."""
    r = np.maximum(r, 1e-6)
    h = np.clip(0.5 + 0.5 * (a - b) / r, 0.0, 1.0)
    return b + (a - b) * h + r * h * (1 - h)


# ---- side view -------------------------------------------------------------------
# Centre-line top silhouette = smooth max of a hood line and a canopy line, so the
# windshield base gets a small concave fillet instead of a kink.
HOOD = Pchip(
    [-2.55, -2.32, -2.20, -2.00, -1.80, -1.60, -1.40, -1.20, -0.80, 0.0, 2.6],
    [0.600, 0.662, 0.738, 0.815, 0.875, 0.930, 0.978, 1.015, 1.040, 0.95, 0.80])
GREENHOUSE = Pchip(
    [-1.90, -1.60, -1.40, -1.10, -0.80, -0.50, -0.20, 0.10, 0.40, 0.70, 1.00, 1.30, 1.60, 1.90, 2.15, 2.30, 2.40, 2.60],
    [0.620, 0.840, 0.985, 1.150, 1.300, 1.425, 1.510, 1.550, 1.560, 1.548, 1.515, 1.455, 1.370, 1.270, 1.180, 1.135, 1.105, 1.050])
COWL_FILLET = 0.05


def TOP(y):
    a, b = HOOD(y), GREENHOUSE(y)
    return smax(a, b, COWL_FILLET)


# Underbody: flat battery floor at 165 mm, rising to the nose and tail lips.
BOTTOM = Pchip(
    [-2.60, -2.30, -2.10, -1.80, -1.40, -1.00, 1.00, 1.40, 1.90, 2.20, 2.40, 2.60],
    [0.360, 0.290, 0.240, 0.205, 0.180, 0.165, 0.165, 0.180, 0.220, 0.270, 0.310, 0.360])

# Nose: most forward at the light bar (z 0.65), receding below; tail: a near-vertical
# kammback face with the lip at the top.
FRONT_TIP = Pchip(
    [0.05, 0.20, 0.28, 0.38, 0.48, 0.58, 0.65, 0.70, 0.74, 0.78, 0.84, 0.95],
    [-2.020, -2.120, -2.200, -2.255, -2.282, -2.296, -2.300, -2.296, -2.272, -2.215, -2.100, -1.850])
REAR_TIP = Pchip(
    [0.10, 0.25, 0.32, 0.42, 0.55, 0.70, 0.85, 1.00, 1.07, 1.11, 1.14, 1.20, 1.30],
    [2.050, 2.200, 2.280, 2.340, 2.380, 2.398, 2.400, 2.392, 2.385, 2.375, 2.350, 2.250, 2.000])

# Canopy base line (the glass starts here; the autonomy halo runs just below it).
BELT = Pchip(
    [-1.60, -1.40, -1.10, -0.50, 0.30, 1.00, 1.60, 2.00, 2.30, 2.60],
    [1.030, 0.975, 0.968, 0.980, 1.000, 1.020, 1.048, 1.072, 1.085, 1.085])

# ---- plan view ----------------------------------------------------------------------
# Maximum half width: haunches over both axles, a slight waist at the doors.
WMAX = Pchip([-2.60, -2.25, -1.85, -1.50, -1.00, -0.20, 0.60, 1.20, 1.50, 1.90, 2.30, 2.60],
             [0.860, 0.930, 0.970, 0.978, 0.962, 0.950, 0.958, 0.975, 0.980, 0.972, 0.930, 0.860])

CORNER_F = Pchip([0.0, 0.40, 0.60, 0.75, 0.95, 1.20], [0.44, 0.48, 0.50, 0.54, 0.60, 0.60])
CORNER_R = Pchip([0.0, 0.40, 0.70, 1.00, 1.30], [0.38, 0.40, 0.42, 0.46, 0.50])
CORNER_EXP_F = 2.5
CORNER_EXP_R = 2.8

# Body-side section below the canopy, by normalised height s = (z - bottom)/(belt - bottom).
TUCK_S = Pchip([0.0, 0.06, 0.14, 0.24, 0.36, 0.50, 1.0], [0.120, 0.078, 0.040, 0.015, 0.003, 0.0, 0.0])
SHOULDER_S = Pchip([0.0, 0.50, 0.64, 0.78, 0.88, 0.95, 1.0], [0.0, 0.0, 0.006, 0.022, 0.040, 0.058, 0.075])
TUCK_GAIN = Pchip([-2.6, -2.2, -1.8, -1.2, 1.2, 1.9, 2.3, 2.6], [0.70, 0.75, 0.85, 1.0, 1.0, 0.9, 0.8, 0.8])
SHOULDER_GAIN = Pchip([-2.6, -2.0, -1.2, 1.3, 1.8, 2.1, 2.4, 2.6], [1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.8, 1.9])

# Canopy taper in plan: narrow at the windshield base and towards the tail.
GH_TAPER = Pchip([-1.60, -1.40, -1.10, -0.70, -0.40, 1.00, 1.40, 1.80, 2.10, 2.40],
                 [0.270, 0.240, 0.140, 0.050, 0.0, 0.0, 0.040, 0.120, 0.200, 0.260])

# Crown (drop at the full half width): a domed hood and an arched canopy roof.
CROWN = Pchip([-2.6, -1.8, -1.4, -0.9, -0.4, 0.4, 1.2, 1.8, 2.2, 2.6], [0.100, 0.100, 0.105, 0.150, 0.210, 0.225, 0.210, 0.150, 0.090, 0.060])


def belt_z(y):
    """Beltline height used for the greenhouse; lifted above the hood in front of the
    cowl so the hood/fender region has no tumblehome."""
    yb = np.asarray(y, float)
    b = BELT(yb)
    hood_lift = TOP(yb) + 0.06
    w = smoothstep(-1.30, -1.50, yb)   # 1 in front of the cowl
    return b * (1 - w) + np.maximum(b, hood_lift) * w


def inset(y, z):
    """How far the body side sits inboard of WMAX at (y, z)."""
    zb = BOTTOM(y)
    zbelt = belt_z(y)
    s = (z - zb) / np.maximum(zbelt - zb, 0.2)
    sc = np.clip(s, 0.0, 1.0)
    below = TUCK_S(sc) * TUCK_GAIN(y) + SHOULDER_S(sc) * SHOULDER_GAIN(y)
    hp = np.maximum(z - zbelt, 0.0)
    greenhouse = 0.32 * hp + 0.55 * hp * hp
    taper = GH_TAPER(y) * smoothstep(0.0, 0.14, hp)
    return below + greenhouse + taper


BOW_F = 0.18   # plan-view curvature of the nose face (set-back at the full half width)
BOW_R = 0.10


def plan_sdf(ax, y, z):
    """Distance-like function to the plan outline at height z: the body side
    (WMAX - inset) closed by superelliptic corners at the nose and tail tips."""
    yf = FRONT_TIP(z)
    yr = REAR_TIP(z)
    mid = 0.5 * (yf + yr)
    front = y < mid
    t2 = (ax / HALF_WIDTH) ** 2
    yw = np.where(front, y - BOW_F * t2, y + BOW_R * t2)
    hw = WMAX(y) - inset(y, z)
    rc = np.where(front, CORNER_F(z), CORNER_R(z))
    pe = np.where(front, CORNER_EXP_F, CORNER_EXP_R)
    qx = ax - (hw - rc)
    qy = np.where(front, (yf + rc) - yw, yw - (yr - rc))
    mx = np.maximum(qx, 0.0)
    my = np.maximum(qy, 0.0)
    outer = (mx ** pe + my ** pe) ** (1.0 / pe)
    inner = np.minimum(np.maximum(qx, qy), 0.0)
    return outer + inner - rc


def top_z(x, y):
    return TOP(y) - CROWN(y) * (np.asarray(x, float) / HALF_WIDTH) ** 2


def fillet(y, z):
    """Fillet radius between the side-view slab and the plan outline."""
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    r = np.full(np.broadcast(y, z).shape, 0.10)
    r = np.where((y > -1.2) & (y < 1.9), 0.09, r)       # canopy: tight turn, arched roof
    lip = smoothstep(2.15, 2.30, y) * smoothstep(1.02, 1.12, z)
    nose = smoothstep(-2.00, -2.18, y) * smoothstep(0.60, 0.72, z)
    r = r * (1 - 0.80 * lip) * (1 - 0.70 * nose)
    r = np.where(z < 0.45, 0.07, r)                    # underbody edge
    return r


def F(p):
    """Signed-distance-like body function. p: (...,3). Negative inside."""
    p = np.asarray(p, float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    ax = np.abs(x)
    zt = top_z(ax, y)
    zb = BOTTOM(y)
    f_slab = np.maximum(z - zt, zb - z)
    f_plan = plan_sdf(ax, y, z)
    return smax(f_slab, f_plan, fillet(y, z))


def grad(p, e=5e-4):
    p = np.asarray(p, float)
    g = np.zeros_like(p)
    for k in range(3):
        d = np.zeros(3)
        d[k] = e
        g[..., k] = (F(p + d) - F(p - d)) / (2 * e)
    return g


def project(p, iters=4):
    """Pull points onto the surface F=0 along the gradient (Newton steps)."""
    p = np.array(p, float)
    for _ in range(iters):
        f = F(p)
        g = grad(p)
        gg = np.maximum((g * g).sum(-1, keepdims=True), 1e-12)
        p = p - (f[..., None] * g) / gg
    return p
