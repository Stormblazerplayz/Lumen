"""Toe and camber of all four wheels, measured in the vehicle's own frame, plus ride height.
Usage: py wheel_probe.py <vehId>"""
import json
import sys
import time

import mcp

vid = int(sys.argv[1])
CODE = """
local function npos(name)
  for _, n in pairs(v.data.nodes) do if n.name == name then return obj:getNodePosition(n.cid) end end
end
local fwd = obj:getDirectionVector():normalized()
local up = obj:getDirectionVectorUp():normalized()
local right = fwd:cross(up):normalized()
local out = {}
for _, w in ipairs({{'FL', 'fw1ll', 'fw1l', -1}, {'FR', 'fw1rr', 'fw1r', 1}, {'RL', 'rw1ll', 'rw1l', -1}, {'RR', 'rw1rr', 'rw1r', 1}}) do
  local a, b = npos(w[2]), npos(w[3])
  if a and b then
    local ax = a - b
    local lat = ax:dot(right) * w[4]          -- outward
    local toe = math.deg(math.atan2(ax:dot(fwd), lat))   -- + = toe-in (outer end ahead)
    local camber = math.deg(math.atan2(ax:dot(up), lat)) -- + = top of wheel leans out
    out[#out + 1] = string.format('%s toe %+.2f camber %+.2f hubZ %.3f', w[1], toe, -camber, (a + b):dot(up) * 0.5)
  end
end
local e = electrics.values
obj:queueGameEngineLua('extensions.lumen_wheelprobe = ' .. string.format('%q', table.concat(out, ' | ') .. string.format(' || steer %.2f speed %.2f', e.steering_input or 0, e.wheelspeed or 0)))
"""
mcp.call("run_lua_vehicle", {"id": vid, "code": CODE})
time.sleep(0.5)
print(mcp.lua("return extensions.lumen_wheelprobe or 'none'"))
