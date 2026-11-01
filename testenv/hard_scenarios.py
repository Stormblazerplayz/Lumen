"""Hard scenarios on the test instance, for Lumen Drive and Lumen NN (Storm, 2026-11-01: "Have you tested it through
hard scenarios? Like a car pulling out and a car swerving and a car changing lanes right next to Lumen?").

    py -u hard_scenarios.py [trials_per_kind_and_brain=4] [--kinds pull_out,swerve,side_merge] [--brains drive,nn] [--roads]

Each trial: a fleet Lumen is placed at the start of a long straight served road and sent down it on the brain under
test; a scripted car (BeamNG script AI, testenv/ge/lumen_scenarios.lua) is set up relative to the Lumen's own lane
line and does its maneuver at the trigger moment; the game measures the closest gap between the two bodies, contact,
the Lumen's deceleration, its reaction time and how far it moved sideways. --roads only lists the candidate roads.
Needs the test instance running with the Lumen mod and a level loaded (game.py start, then drive_test.py or similar).
"""
import json
import math
import os
import random
import shutil
import sys
import time

import mcp

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE = os.path.join(os.environ["LOCALAPPDATA"], "BeamNGTest", "robotaxi", "current")
OUT = os.path.normpath(os.path.join(HERE, "..", "out", "game"))
SLOT = 1                                         # the fleet car used for the trials


def lua_json(code, timeout=60):
    return json.loads(mcp.lua(code, timeout=timeout))


def install_extension():
    dst = os.path.join(PROFILE, "lua", "ge", "extensions")
    os.makedirs(dst, exist_ok=True)
    shutil.copyfile(os.path.join(HERE, "ge", "lumen_scenarios.lua"), os.path.join(dst, "lumen_scenarios.lua"))
    return mcp.lua("if extensions.lumen_scenarios then extensions.unload('lumen_scenarios') end "
                   "package.loaded['lumen_scenarios'] = nil extensions.load('lumen_scenarios') "
                   "return tostring(extensions.lumen_scenarios ~= nil)")


FIND_ROADS = r"""
-- straight, wide, public roads in the service area: chains of links that turn less than 4 degrees per node
local mn = map.getMap().nodes
local roads = require('lumen/roads')
local gf = (function() local ok, g = pcall(require, 'lumen/geofence') return ok and g or nil end)()
local out, used = {}, {}
local function nb(id) local n = 0 for _ in pairs(mn[id].links or {}) do n = n + 1 end return n end
for a, na in pairs(mn) do
  for b, l in pairs(na.links or {}) do
    local key = a < b and (a .. '|' .. b) or (b .. '|' .. a)
    if not used[key] and mn[b] and (l.type == nil or l.type == 'public') and (na.radius or 0) >= %(minrad)f and (mn[b].radius or 0) >= %(minrad)f then
      used[key] = true
      local chain, len = {a, b}, na.pos:distance(mn[b].pos)
      local prev, cur = a, b
      local d0 = (mn[b].pos - na.pos):normalized()
      while nb(cur) == 2 and len < 900 do
        local nxt
        for c, lc in pairs(mn[cur].links) do if c ~= prev then nxt = c end end
        if not nxt or not mn[nxt] or (mn[nxt].radius or 0) < %(minrad)f then break end
        local d1 = (mn[nxt].pos - mn[cur].pos):normalized()
        if d1:dot(d0) < 0.9976 then break end                 -- 4 degrees from the chain's first direction
        len = len + mn[cur].pos:distance(mn[nxt].pos)
        chain[#chain + 1] = nxt
        prev, cur = cur, nxt
      end
      local p0, p1 = na.pos, mn[cur].pos
      local served = (not gf) or (not gf.hasZones()) or (gf.served(p0.x, p0.y) and gf.served(p1.x, p1.y))
      if len >= %(minlen)f and served then
        out[#out + 1] = {x0 = p0.x, y0 = p0.y, z0 = p0.z, x1 = p1.x, y1 = p1.y, len = len, rad = math.min(na.radius, mn[cur].radius), n = #chain}
      end
    end
  end
end
table.sort(out, function(p, q) return p.len > q.len end)
local top = {}
for i = 1, math.min(#out, 12) do top[i] = out[i] end
return jsonEncode(top)
"""


def find_roads(minrad=5.5, minlen=260.0):
    return lua_json(FIND_ROADS % {"minrad": minrad, "minlen": minlen}, timeout=120)


def lane_line():
    """the Lumen's lane line right where it is: a point, the direction, the height, and the lane's offset from the road
    centre (from its path)"""
    d = lua_json(f"return jsonEncode(extensions.lumen_robotaxi.debugDump({SLOT}))")
    px, py = d["px"], d["py"]
    i = max(0, min(len(px) - 3, int(d.get("idx") or 1) - 1))
    j = min(len(px) - 1, i + 6)
    fx, fy = px[j] - px[i], py[j] - py[i]
    L = math.hypot(fx, fy) or 1.0
    off = d["segs"][0][4] if d.get("segs") else 0.0
    return {"lineX": px[i], "lineY": py[i], "fx": fx / L, "fy": fy / L, "off": off, "car": d["car"]}


def lumen_id():
    for c in lua_json("return jsonEncode(extensions.lumen_robotaxi.debugFleet())"):
        if c.get("slot") == SLOT:
            return c.get("id"), c.get("speed") or 0
    return None, 0


def trial(kind, brain, road, rng):
    fx, fy = road["x1"] - road["x0"], road["y1"] - road["y0"]
    L = math.hypot(fx, fy)
    fx, fy = fx / L, fy / L
    sx, sy = road["x0"] + fx * 15, road["y0"] + fy * 15
    ok = mcp.lua(f"return tostring(extensions.lumen_robotaxi.debugPlace({SLOT}, {sx:.2f}, {sy:.2f}, {fx:.4f}, {fy:.4f}))")
    if ok != "true":
        return {"error": "place failed"}
    time.sleep(2.0)
    rem = mcp.lua(f"return tostring(extensions.lumen_robotaxi.debugSendTo({SLOT}, {road['x1'] - fx * 10:.2f}, {road['y1'] - fy * 10:.2f}, '{brain}'))")
    if rem in ("false", "nil"):
        return {"error": "no route down the road"}
    # let it get going
    t0 = time.time()
    lid, v = lumen_id()
    while time.time() - t0 < 25 and v < 7.5:
        time.sleep(0.5)
        lid, v = lumen_id()
    line = lane_line()
    same_dir_left = line["off"] - 3.5 >= 0.8                     # a same-direction lane to the left of ours
    side = -1 if same_dir_left else 1
    spec = {"kind": kind, "lumenId": lid, "lineX": line["lineX"], "lineY": line["lineY"], "fx": line["fx"], "fy": line["fy"],
            "z": road["z0"], "advModel": "pickup", "maxT": 9}
    if kind == "pull_out":
        spec.update(s0=rng.uniform(80, 110), lat0=2.8, trigger=rng.uniform(12, 30), accel=rng.uniform(2.0, 3.0), dur=rng.uniform(2.0, 3.0), v=6.0)
    elif kind == "swerve":
        spec.update(s0=rng.uniform(35, 50), lat0=side * 3.5, v=max(3.0, 0.6 * v), trigger=rng.uniform(0.0, 4.0), amp=rng.uniform(1.0, 1.8), dur=rng.uniform(1.0, 1.8))
    else:
        spec.update(s0=rng.uniform(25, 35), lat0=side * 3.5, v=max(3.0, 0.75 * v), trigger=rng.uniform(-3.0, 1.0), dur=rng.uniform(2.0, 3.0))
    r = mcp.lua("return tostring(extensions.lumen_scenarios.start(" + lua_table(spec) + "))")
    if not r.startswith("ok"):
        return {"error": "start: " + r, "spec": spec}
    t0 = time.time()
    st = {}
    while time.time() - t0 < 70:
        time.sleep(0.5)
        st = lua_json("return jsonEncode(extensions.lumen_scenarios.status())")
        if st.get("phase") in ("done", "nofire", "lost"):
            break
    fin = lua_json("return jsonEncode(extensions.lumen_scenarios.finish())")
    mcp.lua(f"return tostring(extensions.lumen_robotaxi.debugRelease({SLOT}))")
    fin.update({"kind": kind, "brain": brain, "spec": spec, "road": road, "lane_off": line["off"], "speed_before": v})
    return fin


def lua_table(d):
    parts = []
    for k, v in d.items():
        if isinstance(v, str):
            parts.append(f"{k} = {json.dumps(v)}")
        elif isinstance(v, bool):
            parts.append(f"{k} = {'true' if v else 'false'}")
        else:
            parts.append(f"{k} = {float(v):.6f}")
    return "{" + ", ".join(parts) + "}"


def summarize(results):
    rows = {}
    for r in results:
        if r.get("error") or r.get("phase") != "done":
            continue
        k = (r["kind"], r["brain"])
        rows.setdefault(k, []).append(r)
    lines = []
    for (kind, brain), rs in sorted(rows.items()):
        hits = sum(1 for r in rs if r.get("hit"))
        gaps = sorted(r["minGap"] for r in rs if r.get("minGap") is not None)
        reacts = sorted(r["reactT"] for r in rs if r.get("reactT") is not None)
        dec = [r.get("maxDecel") or 0 for r in rs]
        lines.append(f"{kind:10} {brain:5}  trials {len(rs):2}  contact {hits}  closest gap min {gaps[0] if gaps else float('nan'):.2f} m, "
                     f"median {gaps[len(gaps) // 2] if gaps else float('nan'):.2f} m  reaction median "
                     f"{reacts[len(reacts) // 2] if reacts else float('nan'):.2f} s ({len(reacts)}/{len(rs)} reacted)  max decel {max(dec) if dec else 0:.1f} m/s2")
    return lines


def main():
    args = sys.argv[1:]
    n = int(args[0]) if args and args[0].isdigit() else 4
    kinds = (args[args.index("--kinds") + 1] if "--kinds" in args else "pull_out,swerve,side_merge").split(",")
    brains = (args[args.index("--brains") + 1] if "--brains" in args else "drive,nn").split(",")
    if mcp.lua("return tostring(extensions.lumen_robotaxi ~= nil and extensions.lumen_robotaxi.debugPlace ~= nil)") != "true":
        raise SystemExit("the Lumen mod (with the test hooks) is not loaded in the test instance")
    print("scenario extension:", install_extension())
    roads = find_roads()
    print(f"{len(roads)} candidate roads")
    for r in roads[:6]:
        print(f"  ({r['x0']:.0f}, {r['y0']:.0f}) -> ({r['x1']:.0f}, {r['y1']:.0f}), {r['len']:.0f} m, half-width {r['rad']:.1f} m")
    if "--roads" in args or not roads:
        return
    rng = random.Random(20261101)
    results = []
    for kind in kinds:
        for brain in brains:
            for i in range(n):
                road = roads[(i + len(results)) % len(roads)]
                r = trial(kind, brain, road, rng)
                results.append(r)
                tag = r.get("error") or f"{r.get('phase')}: contact={r.get('hit')} gap={r.get('minGap')} react={r.get('reactT')} decel={r.get('maxDecel')}"
                print(f"{kind:10} {brain:5} #{i + 1}: {tag}")
    print("\n".join(["", "SUMMARY"] + summarize(results)))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"hard_scenarios_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
