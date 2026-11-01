"""Service-area checks on the test instance (geofence.lua + the gate in roads.lua).
usage: py geo_test.py [level]
  1. zones and tiers as exported to the app
  2. membership of known points
  3. how many road links each open tier admits (by road tier)
  4. routes: with a tier closed, no route uses a closed road; opened, the route appears
  5. the fleet stays inside the open zones while it roams (sampled for a while)"""
import json
import sys
import time

import mcp
import drive_test

LEVEL = sys.argv[1] if len(sys.argv) > 1 else "west_coast_usa"

POINTS = {
    "west_coast_usa": {
        "downtown grid": (-700, 500), "port (SW)": (-1400, -700), "east hills": (500, 0), "us110 highway": (200, 330),
        "north island": (-200, 2000), "far west sea": (-1500, 1500),
    },
    "east_coast_usa": {
        "harbor town": (650, -250), "mountain road N": (-300, 700), "mountain road S": (200, -800), "off-map corner": (1500, 1500),
    },
}


def lj(code):
    s = mcp.lua(code)
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


def main():
    print("loading", LEVEL, drive_test.load_level(LEVEL))
    time.sleep(3)
    g = lj('return jsonEncode(extensions.lumen_robotaxi.getGeofence())')
    print("open tier:", g["openTier"], "| tiers:", [(t["id"], t["name"], t["open"]) for t in g["tiers"]])
    for z in g["zones"]:
        print("  zone %-28s tier %d %-6s polys %d" % (z["name"], z["tier"], "open" if z["open"] else "closed", len(z["polys"])))

    print("\nmembership (tier 2 open):")
    for name, (x, y) in POINTS[LEVEL].items():
        r = lj(f'local g = require("lumen/geofence") return jsonEncode(g.status({x}, {y}))')
        print("  %-18s (%5d,%5d) -> %s" % (name, x, y, r))

    print("\nlinks admitted per open tier (all links of the graph):")
    for tier in (0, 1, 2, 3):
        r = lj(f'''
local g = require("lumen/geofence")
g.setOpenTier({tier})
local mn = map.getMap().nodes
local n, ok, byTier = 0, 0, {{0, 0, 0}}
local cls = {{0, 0, 0}}
for a, na in pairs(mn) do
  for b, l in pairs(na.links or {{}}) do
    local nb = mn[b]
    if nb and a < b then
      n = n + 1
      local rad = math.min(na.radius or 3, nb.radius or 3)
      local t = g.linkTier(l, rad)
      cls[t] = cls[t] + 1
      if g.linkOK(na.pos, nb.pos, l, rad) then ok = ok + 1 byTier[1][t] = byTier[1][t] + 1 end
    end
  end
end
return jsonEncode({{links = n, admitted = ok, admittedByRoadTier = byTier[1], graphByRoadTier = cls}})''')
        print("  tier %d open: %s" % (tier, r))

    print("\nroute gating (random node pairs, 40 tries):")
    for tier in (1, 2, 3):
        r = lj(f'''
local g = require("lumen/geofence")
local roads = require("lumen/roads")
g.setOpenTier({tier})
roads.gate = {{link = g.linkOK, version = g.version}}
local mn = map.getMap().nodes
local ids = {{}}
for k, n in pairs(mn) do ids[#ids + 1] = k end
math.randomseed(11)
local found, closedUsed, tried = 0, 0, 0
for i = 1, 400 do
  local a, b = ids[math.random(#ids)], ids[math.random(#ids)]
  if g.served(mn[a].pos.x, mn[a].pos.y) and g.served(mn[b].pos.x, mn[b].pos.y) and mn[a].pos:distance(mn[b].pos) > 300 then
    tried = tried + 1
    local p = roads.path(a, b, {{}})
    if p then
      found = found + 1
      for q = 2, #p do
        local na, nb = mn[p[q - 1]], mn[p[q]]
        local l = na.links and na.links[p[q]] or (nb.links and nb.links[p[q - 1]])
        -- a hop that starts more than 80 m from the route start must be allowed
        if na.pos:distance(mn[a].pos) > 80 and not g.linkOK(na.pos, nb.pos, l, math.min(na.radius or 3, nb.radius or 3)) then closedUsed = closedUsed + 1 end
      end
    end
    if tried >= 40 then break end
  end
end
return jsonEncode({{tried = tried, routes = found, closedHops = closedUsed}})''')
        print("  tier %d: %s" % (tier, r))

    # leave the map at its configured tier
    lj('local g = require("lumen/geofence") g.setOpenTier(%d) return 1' % (2,))

    print("\nfleet roaming (60 s): every car position vs the open zones")
    inside = outside = 0
    bad = []
    t0 = time.time()
    while time.time() - t0 < 60:
        cars = lj('return jsonEncode(extensions.lumen_robotaxi.debugFleet())')
        if isinstance(cars, list):
            for c in cars:
                if c.get("x") is None or c.get("status") not in ("cruising", "assigned"):
                    continue
                st = lj(f'local g = require("lumen/geofence") return jsonEncode(g.status({c["x"]}, {c["y"]}))')
                if isinstance(st, dict) and st.get("inside"):
                    inside += 1
                else:
                    outside += 1
                    bad.append((round(c["x"]), round(c["y"]), c.get("mode")))
        time.sleep(4)
    print("  samples inside open zones: %d, outside: %d %s" % (inside, outside, bad[:6]))


if __name__ == "__main__":
    main()
