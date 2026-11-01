"""Top-down picture of a Lumen Drive path dump (drive_test.py writes them when a car strays):
the route's nodes (grey dots), the lane segments (blue), the smooth path (green), the slice last
sent to the car (thick green), the car (red arrow). usage: py plot_dump.py dump.json [out.png]"""
import json
import sys

from PIL import Image, ImageDraw

d = json.load(open(sys.argv[1]))
out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(".json", ".png")
cx, cy = d["car"][0], d["car"][1]
R = 70.0                                  # metres shown around the car
S = 900
def P(x, y):
    return ((x - cx) / R * S / 2 + S / 2, S / 2 - (y - cy) / R * S / 2)
im = Image.new("RGB", (S, S), (18, 20, 24))
g = ImageDraw.Draw(im)
for k in range(-7, 8):
    o = k * 10 / R * S / 2
    g.line([(S / 2 + o, 0), (S / 2 + o, S)], fill=(32, 34, 40))
    g.line([(0, S / 2 + o), (S, S / 2 + o)], fill=(32, 34, 40))
for sg in d.get("segs", []):
    g.line([P(sg[0], sg[1]), P(sg[2], sg[3])], fill=(70, 110, 220), width=2)
pts = [P(x, y) for x, y in zip(d["px"], d["py"])]
if len(pts) > 1:
    g.line(pts, fill=(40, 160, 90), width=2)
i0, i1 = d.get("i0") or 1, d.get("i1") or 1
sl = pts[max(0, i0 - 1):i1]
if len(sl) > 1:
    g.line(sl, fill=(80, 255, 140), width=5)
for x, y in zip(d["nx"], d["ny"]):
    a, b = P(x, y)
    g.ellipse([a - 4, b - 4, a + 4, b + 4], fill=(170, 170, 170))
x, y, dx, dy = d["car"]
a, b = P(x, y)
e = P(x + dx * 5, y + dy * 5)
g.ellipse([a - 6, b - 6, a + 6, b + 6], fill=(255, 70, 60))
g.line([(a, b), e], fill=(255, 70, 60), width=4)
g.text((10, 10), f"lat {d.get('lat') or 0:+.2f} m  s {d.get('s') or 0:.1f}/{d.get('len') or 0:.0f}  idx {d.get('idx')}  slice {i0}-{i1}  cruising {d.get('cruising')}  (grid 10 m)", fill=(230, 230, 230))
im.save(out)
print(out)
