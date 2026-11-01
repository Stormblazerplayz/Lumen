"""Find which nodes blow up first: pause physics, spawn, step tick by tick and
rank nodes by speed. Usage: py debug_instability.py [config]"""
import json
import sys
import time

import mcp

VEL_LUA = r"""
local out = {}
for _, n in pairs(v.data.nodes) do
  local vel = obj:getNodeVelocityVector(n.cid)
  out[#out + 1] = {n.name or tostring(n.cid), vel:length()}
end
table.sort(out, function(a, b) return a[2] > b[2] end)
local top = {}
for i = 1, math.min(12, #out) do top[i] = string.format("%s=%.2f", out[i][1], out[i][2]) end
return table.concat(top, " ")
"""


def vlua(code, vid=None):
    args = {"code": code}
    if vid is not None:
        args["id"] = vid
    mcp.call("run_lua_vehicle", args)
    time.sleep(0.6)
    return mcp.call("run_lua_vehicle", args)


def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else "robotaxi"
    mcp.call("get_instability", {})
    mcp.call("pause_physics", {})
    print(mcp.call("spawn", {"model": "lumen", "config": cfg}, timeout=180))
    time.sleep(6)
    vid = int(mcp.call("get_player_vehicle_id", {}).strip())
    print("vid", vid)
    done = 0
    for target in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        mcp.call("step_physics", {"steps": target - done})
        done = target
        time.sleep(0.4)
        res = vlua(VEL_LUA, vid)
        print("after %4d ticks: %s" % (done, res[:600]))
    print("instability:", mcp.call("get_instability", {}))
    mcp.call("resume_physics", {})


if __name__ == "__main__":
    main()
