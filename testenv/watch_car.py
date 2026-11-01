"""Watch one fleet car on Lumen Drive: every path it is given (each new one in a new colour) and
the car's own trajectory (red, a dot every 0.5 s), on one top-down picture.
usage: py watch_car.py [slot] [seconds]"""
import json
import os
import sys
import time

from PIL import Image, ImageDraw

import mcp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "game")


def dump(slot):
    s = mcp.lua(f"return jsonEncode(extensions.lumen_robotaxi.debugDump({slot}))")
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return None


def main():
    slot = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 20
    t0 = time.time()
    while time.time() - t0 < 90 and not dump(slot):
        time.sleep(1)
    paths, traj, log = [], [], []
    t0 = time.time()
    while time.time() - t0 < secs:
        d = dump(slot)
        if d:
            key = (round(d["px"][0], 1), round(d["py"][0], 1), round(d["len"], 0))
            if not paths or paths[-1]["key"] != key:
                paths.append({"key": key, "px": d["px"], "py": d["py"], "t": time.time() - t0, "nx": d["nx"], "ny": d["ny"]})
            traj.append((d["car"][0], d["car"][1], d["car"][2], d["car"][3], time.time() - t0))
            log.append(f"{time.time() - t0:5.1f}s path#{len(paths)} s={d.get('s') or 0:6.1f} lat={d.get('lat') or 0:+.2f} idx={d.get('idx')}")
        time.sleep(0.5)
    print("\n".join(log))
    if not traj:
        return
    cx = sum(p[0] for p in traj) / len(traj)
    cy = sum(p[1] for p in traj) / len(traj)
    span = max(40.0, max(max(abs(p[0] - cx), abs(p[1] - cy)) for p in traj) * 1.4)
    S = 1000

    def P(x, y):
        return ((x - cx) / span * S / 2 + S / 2, S / 2 - (y - cy) / span * S / 2)
    im = Image.new("RGB", (S, S), (18, 20, 24))
    g = ImageDraw.Draw(im)
    cols = [(60, 200, 120), (80, 160, 255), (240, 200, 60), (200, 100, 255), (60, 220, 220), (255, 140, 60)]
    for i, p in enumerate(paths):
        pts = [P(x, y) for x, y in zip(p["px"], p["py"])]
        if len(pts) > 1:
            g.line(pts, fill=cols[i % len(cols)], width=2)
            g.text((pts[0][0] + 4, pts[0][1] + 4), f"#{i + 1} {p['t']:.1f}s", fill=cols[i % len(cols)])
        for x, y in zip(p["nx"], p["ny"]):
            a, b = P(x, y)
            g.ellipse([a - 3, b - 3, a + 3, b + 3], fill=(150, 150, 150))
    tp = [P(p[0], p[1]) for p in traj]
    g.line(tp, fill=(255, 70, 60), width=2)
    for i, p in enumerate(traj):
        a, b = tp[i]
        g.ellipse([a - 3, b - 3, a + 3, b + 3], fill=(255, 70, 60))
        if i % 4 == 0:
            e = P(p[0] + p[2] * 4, p[1] + p[3] * 4)
            g.line([(a, b), e], fill=(255, 160, 150), width=1)
            g.text((a + 5, b - 12), f"{p[4]:.0f}", fill=(255, 180, 170))
    g.text((10, 10), f"car {slot}: {len(paths)} paths, {len(traj)} samples, view {2 * span:.0f} m", fill=(230, 230, 230))
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"watch_car{slot}.png")
    im.save(out)
    print(out)


if __name__ == "__main__":
    main()
