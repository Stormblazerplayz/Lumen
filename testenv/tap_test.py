"""Tap-to-go on the test instance (Storm, 2026-11-01: "BRO I AM CLICKING EVERYWHERE AND I CANT GO ANYWHERE"):
call a Lumen, board it, then tap pins all over the map the way the app's map does (setDestinationXY) and count how
many are accepted. Pins are road nodes drawn with a fixed seed, so runs are comparable.
usage: py -u tap_test.py [pins] [seed]"""
import json
import os
import re
import sys
import time

import mcp

LOG = os.path.join(os.environ["LOCALAPPDATA"], "BeamNGTest", "robotaxi", "current", "beamng.log")


def state():
    try:
        return json.loads(mcp.lua("return jsonEncode(extensions.lumen_robotaxi.getState())"))
    except (ValueError, mcp.McpError):
        return {}


def wait_for(states, timeout):
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        s = state()
        if s.get("state") != last:
            print("  t=%4.0fs state=%s" % (time.time() - t0, s.get("state")))
            last = s.get("state")
        if s.get("state") in states:
            return s
        time.sleep(2)
    return None


def log_since(pos):
    with open(LOG, "rb") as f:
        f.seek(pos)
        return f.read().decode("utf-8", "replace")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    s = state()
    if s.get("state") not in ("boarded", "riding", "arrived", "pulledOver"):
        if s.get("state") in (None, "idle", "?"):
            print("call:", mcp.lua("return tostring(extensions.lumen_robotaxi.callTaxi())"))
        if not wait_for(("waiting",), 300):
            print("FAIL: the Lumen never arrived")
            return 1
        print("board:", mcp.lua("return tostring(extensions.lumen_robotaxi.board())"))
        if not wait_for(("boarded", "riding"), 30):
            print("FAIL: could not board")
            return 1
    # road nodes spread over the map, drawn in the game with a fixed seed
    pins = json.loads(mcp.lua(f"""
local nodes = map.getMap().nodes
local ids = {{}}
for id in pairs(nodes) do ids[#ids + 1] = id end
table.sort(ids)
math.randomseed({seed})
local out = {{}}
for i = 1, {n} do
  local id = ids[math.random(#ids)]
  local p = nodes[id].pos
  out[#out + 1] = {{id = id, x = p.x, y = p.y}}
end
return jsonEncode(out)"""))
    ok, refused = 0, []
    for i, p in enumerate(pins, 1):
        pos = os.path.getsize(LOG)
        t0 = time.time()
        r = mcp.lua(f"return tostring(extensions.lumen_robotaxi.setDestinationXY({p['x']:.1f}, {p['y']:.1f}))")
        dt = time.time() - t0
        time.sleep(0.6)
        text = log_since(pos)
        where = re.findall(r"geofence: destination '[^']*' \([^)]*\) at \([^)]*\), ([^;]*);", text)
        closed = re.findall(r"route needed (\d+) closed link", text)
        if r == "true":
            ok += 1
            tag = "OK      "
        else:
            refused.append((p, where[:1]))
            tag = "REFUSED "
        print(f"{i:3d} {tag} pin ({p['x']:7.0f}, {p['y']:7.0f}) node {p['id']:<14} {dt*1000:5.0f} ms"
              f"{'  closed links used: ' + closed[-1] if closed else ''}  {where[0] if where else ''}")
    print(f"\n{ok}/{len(pins)} pins accepted")
    for p, w in refused:
        print("  refused:", p, w)
    return 0 if ok == len(pins) else 2


if __name__ == "__main__":
    sys.exit(main())
