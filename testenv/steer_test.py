"""Spawn a test Lumen on a road near the player and measure, for several steering inputs,
the front wheel steer angles and whether 30 % throttle moves the car.
Usage: py steer_test.py [config]"""
import json
import sys
import time

import mcp

CFG = sys.argv[1] if len(sys.argv) > 1 else "vehicles/lumen/private.pc"

spawn = mcp.lua("""
local pp = getPlayerVehicle(0):getPosition()
local n1, n2 = map.findClosestRoad(pp + vec3(0, 40, 0))
local m = map.getMap().nodes
local a, b = m[n1].pos, m[n2].pos
local dir = (b - a):normalized()
local pos = a + (b - a) * 0.5
pos.z = be:getSurfaceHeightBelow(pos + vec3(0, 0, 3)) + 0.3
local v = core_vehicles.spawnNewVehicle('lumen', {config = '%s', pos = pos, rot = quatFromDir(dir, vec3(0, 0, 1)), autoEnterVehicle = false, vehicleName = 'lumen_steertest'})
return v and (v:getID() .. ' ' .. string.format('%%.0f %%.0f', pos.x, pos.y)) or 'fail'
""" % CFG)
print("spawned", spawn)
vid = int(spawn.split()[0])
time.sleep(3)

ANGLE = """
local function npos(name)
  for _, n in pairs(v.data.nodes) do if n.name == name then return obj:getNodePosition(n.cid) end end
end
local fwd = -obj:getDirectionVector()   -- unused, nodes are in vehicle space
local function ang(o, i)
  local a, b = npos(o), npos(i)
  if not a or not b then return -999 end
  local ax = a - b
  -- vehicle space: x left, y back; straight wheels have their axle along x
  return math.deg(math.atan2(ax.y, math.abs(ax.x)))
end
local s = electrics.values
obj:queueGameEngineLua(string.format("extensions.lumen_steerprobe = {fl = %.1f, fr = %.1f, ws = %.2f, thr = %.2f, st = %.2f}", ang('fw1ll', 'fw1l'), ang('fw1rr', 'fw1r'), s.wheelspeed or 0, s.throttle or 0, s.steering_input or 0))
"""

def probe():
    mcp.call("run_lua_vehicle", {"id": vid, "code": ANGLE})
    time.sleep(0.4)
    return json.loads(mcp.lua("return jsonEncode(extensions.lumen_steerprobe or {})"))

def speed():
    return float(mcp.lua(f"return string.format('%.2f', getObjectByID({vid}):getVelocity():length())"))

results = []
for st in (0.0, 0.5, 0.8, 1.0, -1.0):
    mcp.call("run_lua_vehicle", {"id": vid, "code": f"input.event('parkingbrake', 0, 2) input.event('brake', 0, 2) input.event('steering', {st}, 2) input.event('throttle', 0, 2)"})
    time.sleep(1.5)
    a = probe()
    mcp.call("run_lua_vehicle", {"id": vid, "code": "input.event('throttle', 0.3, 2)"})
    time.sleep(2.5)
    b = probe()
    sp = speed()
    mcp.call("run_lua_vehicle", {"id": vid, "code": "input.event('throttle', 0, 2) input.event('brake', 1, 2)"})
    time.sleep(2.0)
    print(f"steer {st:+.1f}: FL {a.get('fl')}  FR {a.get('fr')} deg | throttle 0.3 -> wheelspeed {b.get('ws')} speed {sp}")
    results.append((st, a, b, sp))
print(mcp.lua(f"getObjectByID({vid}):delete() return 'deleted test car'"))
