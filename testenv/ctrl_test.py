"""The Lumen's own controller on its own: a test Lumen (not a fleet car) on a long straight road
is sent hand-made plans, the same way Lumen Drive sends them: a straight line at 8 m/s, a 2 m step
to the right, then a 15 m radius left curve at 5 m/s. Prints the lateral error from the line it
was told to follow, ten times a second. usage: py ctrl_test.py"""
import json
import math
import time

import mcp

SETUP = r"""
local roads = require("lumen/roads")
local drive = require("lumen/drive")
local p = getPlayerVehicle(0):getPosition()
local best, bestLen
for _, c in ipairs(roads.closestLinks(p, {public = true, maxDist = 600}, 60)) do
  local L = c.L
  if L.len > 140 and (not bestLen or L.len > bestLen) then best, bestLen = L, L.len end
end
if not best then return "no road" end
local u = (best.pb - best.pa):normalized()
local off = drive._laneFor(best.a, best.b)
local r = vec3(u.y, -u.x, 0)
local start = best.pa + u * 12 + r * off
local old = scenetree.findObject("lumen_ctrl_test")
if old then old:delete() end
local veh = core_vehicles.spawnNewVehicle("lumen", {config = "vehicles/lumen/robotaxi.pc", pos = start + vec3(0, 0, 0.6),
  rot = quatFromDir(u, vec3(0, 0, 1)), autoEnterVehicle = false, vehicleName = "lumen_ctrl_test"})
_G.__ctrl = {id = veh:getID(), sx = start.x, sy = start.y, sz = start.z, ux = u.x, uy = u.y, len = best.len - 20}
return jsonEncode(_G.__ctrl)
"""

# a plan around the car: the line it should follow from 3 m behind to 80 m ahead, at a speed
PLAN = r"""
local t = _G.__ctrl
local veh = getObjectByID(t.id)
if not veh then return "gone" end
local pos, dir = veh:getPosition(), veh:getDirectionVector()
local ux, uy = t.ux, t.uy
local rx, ry = uy, -ux
local mode, shift, speed = "%s", %f, %f
local along = (pos.x - t.sx) * ux + (pos.y - t.sy) * uy
local xs, ys, vs = {}, {}, {}
local ref = {}
local function pt(s)
  if mode == "curve" then
    -- a left curve of radius 15 m starting at s0
    local s0, R = t.curveAt, 15
    if s <= s0 then return t.sx + ux * s + rx * shift, t.sy + uy * s + ry * shift end
    local a = (s - s0) / R
    local cx, cy = t.sx + ux * s0 - rx * R + rx * shift, t.sy + uy * s0 - ry * R + ry * shift
    return cx + rx * R * math.cos(a) + ux * R * math.sin(a), cy + ry * R * math.cos(a) + uy * R * math.sin(a)
  end
  return t.sx + ux * s + rx * shift, t.sy + uy * s + ry * shift
end
for k = 0, 55 do
  local s = along - 3 + k * 1.5
  local x, y = pt(s)
  xs[#xs + 1] = string.format("%%.2f", x - pos.x)
  ys[#ys + 1] = string.format("%%.2f", y - pos.y)
  vs[#vs + 1] = string.format("%%.2f", speed)
end
veh:queueLuaCommand(string.format("local c = controller.getController('lumenRobotaxi') if c and c.lumenDrive then c.lumenDrive({ox = %%.3f, oy = %%.3f, x = {%%s}, y = {%%s}, v = {%%s}}) end",
  pos.x, pos.y, table.concat(xs, ","), table.concat(ys, ","), table.concat(vs, ",")))
-- error from the line: distance to the nearest of the plan's points' polyline
local best = 1e9
local px, py = pos.x, pos.y
local prevx, prevy = pt(along - 3)
for k = 1, 55 do
  local x, y = pt(along - 3 + k * 1.5)
  local dx, dy = x - prevx, y - prevy
  local L2 = dx * dx + dy * dy
  local f = L2 > 0 and math.max(0, math.min(1, ((px - prevx) * dx + (py - prevy) * dy) / L2)) or 0
  local qx, qy = prevx + dx * f, prevy + dy * f
  local dd = math.sqrt((px - qx) ^ 2 + (py - qy) ^ 2)
  local side = (px - qx) * (dy) - (py - qy) * (dx)   -- > 0: right of travel
  if dd < math.abs(best) then best = side >= 0 and dd or -dd end
  prevx, prevy = x, y
end
local e = 0
local hx, hy = dir.x, dir.y
return string.format("%%.2f %%.2f %%.2f %%.3f", along, best, veh:getVelocity():dot(dir), math.atan2(hx * uy - hy * ux, hx * ux + hy * uy))
"""


def main():
    print(mcp.lua(SETUP))
    time.sleep(3)
    mcp.lua("_G.__ctrl.curveAt = 1e9 return 1")
    import os
    sp = float(os.environ.get("SPEED", "8"))
    if os.environ.get("SPEED"):
        phases = [("line", 0.0, sp, 14.0), ("line", 3.0, sp, 10.0), ("line", 0.0, sp, 8.0)]
    else:
        phases = [("line", 0.0, 8.0, 9.0), ("line", 2.0, 8.0, 7.0), ("curve", 2.0, 5.0, 12.0)]
    for name, shift, speed, dur in phases:
        if name == "curve":
            mcp.lua("local t = _G.__ctrl local v = getObjectByID(t.id) local p = v:getPosition() "
                    "_G.__ctrl.curveAt = (p.x - t.sx) * t.ux + (p.y - t.sy) * t.uy + 12 return 1")
        print(f"--- {name} shift {shift} m at {speed} m/s")
        t0 = time.time()
        rows = []
        while time.time() - t0 < dur:
            r = mcp.lua(PLAN % (name, shift, speed))
            rows.append(r)
            time.sleep(0.1)
        for r in rows[::int(os.environ.get('STRIDE', '4'))]:
            print("   along/err/fwd/heading:", r)
    mcp.lua("local t = _G.__ctrl local v = getObjectByID(t.id) if v then v:queueLuaCommand(\"local c = controller.getController('lumenRobotaxi') if c then c.lumenDrive({hold = true}) end\") end return 1")


if __name__ == "__main__":
    main()
