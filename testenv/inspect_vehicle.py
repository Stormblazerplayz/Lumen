"""Inspect the physics data BeamNG built for the player vehicle (after its own
processing: pressureWheel generation, nodeOffsets, ...). Spawns paused."""
import sys
import time

import mcp

INSPECT = r"""
local function len3(a, b) local dx, dy, dz = a.x-b.x, a.y-b.y, a.z-b.z return math.sqrt(dx*dx+dy*dy+dz*dz) end
local nodes = v.data.nodes
local nanN, lightN = {}, {}
local mass = 0
for _, n in pairs(nodes) do
  local p = n.pos
  if p.x ~= p.x or p.y ~= p.y or p.z ~= p.z then nanN[#nanN+1] = tostring(n.name or n.cid) end
  local w = n.nodeWeight or 0
  mass = mass + w
  if w <= 0.05 then lightN[#lightN+1] = (n.name or tostring(n.cid)) .. ":" .. tostring(w) end
end
local short, cnt = {}, 0
for _, b in pairs(v.data.beams) do
  cnt = cnt + 1
  local a, c = nodes[b.id1], nodes[b.id2]
  if a and c then
    local l = len3(a.pos, c.pos)
    if l < 0.01 then short[#short+1] = string.format("%s-%s(%.4f,%s)", tostring(a.name or a.cid), tostring(c.name or c.cid), l, tostring(b.beamType)) end
  end
end
local wheels = {}
for _, w in pairs(v.data.wheels or {}) do wheels[#wheels+1] = string.format("%s r=%.3f n1=%s n2=%s", tostring(w.name), w.radius or -1, tostring(w.node1), tostring(w.node2)) end
return string.format("nodes=%d beams=%d mass=%.1f nan=[%s] light=[%s] shortBeams(%d)=[%s] wheels=[%s]",
  tableSize(nodes), cnt, mass, table.concat(nanN, ","), table.concat(lightN, ","), #short, table.concat(short, " ", 1, math.min(#short, 25)), table.concat(wheels, " | "))
"""


def vlua(code, vid):
    mcp.call("run_lua_vehicle", {"code": code, "id": vid})
    time.sleep(0.8)
    return mcp.call("run_lua_vehicle", {"code": code, "id": vid})


def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else "debugmin"
    mcp.call("pause_physics", {})
    print(mcp.call("spawn", {"model": "lumen", "config": cfg}, timeout=180))
    time.sleep(4)
    vid = int(mcp.call("get_player_vehicle_id", {}).strip())
    print(vlua(INSPECT, vid)[:4000])


if __name__ == "__main__":
    main()
