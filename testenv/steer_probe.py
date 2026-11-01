"""Steering check on a parked Lumen: each front wheel's heading in the car's frame at several
steering inputs (both wheels must turn the same way), plus the tie rod lengths.
Usage: py steer_probe.py [config]"""
import sys
import time

import mcp

CFG = sys.argv[1] if len(sys.argv) > 1 else "vehicles/lumen/robotaxi.pc"
vid = int(mcp.lua("""
local pp = getPlayerVehicle(0):getPosition()
local n1, n2 = map.findClosestRoad(pp + vec3(0, 40, 0))
local m = map.getMap().nodes
local a, b = m[n1].pos, m[n2].pos
local dir = (b - a):normalized()
local pos = a + (b - a) * 0.5
pos.z = be:getSurfaceHeightBelow(pos + vec3(0, 0, 3)) + 0.3
local v = core_vehicles.spawnNewVehicle('lumen', {config = '%s', pos = pos, rot = quatFromDir(dir, vec3(0, 0, 1)), autoEnterVehicle = false, vehicleName = 'lumen_probe'})
return v:getID()
""" % CFG))
time.sleep(4)
CODE = r"""
local function npos(name) for _, n in pairs(v.data.nodes) do if n.name == name then return obj:getNodePosition(n.cid) end end end
local up = obj:getDirectionVectorUp():normalized()
local fwd = obj:getDirectionVector():normalized()
local function flat(x) return (x - up * x:dot(up)):normalized() end
local L = flat(npos('fw1ll') - npos('fw1l'))    -- left axle, pointing out (left)
local R = flat(npos('fw1rr') - npos('fw1r'))    -- right axle, pointing out (right)
local hl, hr = L:cross(up), up:cross(R)        -- rolling directions (forward for straight wheels)
local function signed(h) return math.deg(math.atan2(fwd:cross(h):dot(up), fwd:dot(h))) end
local rl = npos('fs1l'):distance(npos('fh3l'))
local rr = npos('fs1r'):distance(npos('fh3r'))
obj:queueGameEngineLua('extensions.lumen_steerprobe3 = ' .. string.format('%q', string.format('left wheel %+6.1f  right wheel %+6.1f deg (+ = left)  rods %.3f / %.3f', signed(hl), signed(hr), rl, rr)))
"""
for st in (0, 0.5, 1, -0.5, -1):
    mcp.call("run_lua_vehicle", {"id": vid, "code": f"input.event('parkingbrake', 1, 2) input.event('steering', {st}, 2)"})
    time.sleep(1.6)
    mcp.call("run_lua_vehicle", {"id": vid, "code": CODE})
    time.sleep(0.5)
    print(f"steer {st:+.1f}:", mcp.lua("return extensions.lumen_steerprobe3 or 'none'"))
mcp.lua(f"getObjectByID({vid}):delete() return 'ok'")
