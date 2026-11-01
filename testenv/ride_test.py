"""End-to-end robotaxi ride in the test instance: call -> arrive -> board -> destination
-> ride -> arrive -> exit -> depart. Logs each state change with timings and takes
screenshots at the key moments. Usage: py ride_test.py [destination_id]"""
import json
import os
import sys
import time

import mcp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "game")


def state():
    s = mcp.lua("local s = extensions.lumen_robotaxi.getState() return jsonEncode(s)")
    try:
        return json.loads(s)
    except ValueError:
        return {"state": "?", "raw": s}


def shot(name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"ride_{name}.png")
    try:
        mcp.screenshot(p)
        print("   shot", p)
    except Exception as e:
        print("   shot failed", e)


def chase(s):
    """Free camera behind and above the ride's car, so screenshots show it driving."""
    tid = s.get("taxiId")
    if not tid:
        return
    try:
        mcp.lua(f"""
local v = getObjectByID({tid})
if not v then return 'none' end
local p, d = v:getPosition(), v:getDirectionVector()
local c = p - d * 10 + vec3(0, 0, 4.5)
core_camera.setByName(0, 'free')
core_camera.globalCameraFunction('free', 'setPosition', c)
core_camera.globalCameraFunction('free', 'setRotation', quatFromDir((p + d * 8 - c):normalized(), vec3(0, 0, 1)))
return 'ok'""")
    except Exception:
        pass


def wait_for(states, timeout, every=3.0, follow=None):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        s = state()
        if s.get("state") != last or int(time.time() - t0) % 15 == 0:
            print("  t=%5.0fs state=%-11s remaining=%s eta=%s fare=%s" % (time.time() - t0, s.get("state"), s.get("remaining") and round(s["remaining"]), s.get("eta") and round(s["eta"]), s.get("fare") and round(s["fare"], 2)))
            last = s.get("state")
        if s.get("state") in states:
            return s, time.time() - t0
        if follow:
            follow(s)
        time.sleep(every)
    return state(), None


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else "__surprise"
    mcp.call("toggle_ui", {"show": True})
    print("call:", mcp.lua("return tostring(extensions.lumen_robotaxi.callTaxi())"))
    s = state()
    print("after call:", s)
    if not s.get("taxiId"):
        return 1
    tid = s["taxiId"]
    s, t = wait_for(("waiting",), 360, follow=chase)
    print("ARRIVED at pickup after", t)
    if t is None:
        shot("stuck_arriving")
        return 2
    chase(s)
    time.sleep(1)
    shot("pickup")
    print("board:", mcp.lua("return tostring(extensions.lumen_robotaxi.board())"))
    time.sleep(3)
    shot("aboard")
    opts = mcp.lua("return jsonEncode(extensions.lumen_robotaxi.getState())")
    print("destination:", dest, mcp.lua(f"return tostring(extensions.lumen_robotaxi.setDestination({json.dumps(dest)}))"))
    time.sleep(10)
    shot("riding_inside")
    chase(state())
    time.sleep(1)
    shot("riding")
    s, t = wait_for(("arrived",), 900, every=5, follow=chase)
    print("ARRIVED at destination after", t, s)
    shot("destination")
    print("exit:", mcp.lua("return tostring(extensions.lumen_robotaxi.exitTaxi())"))
    time.sleep(4)
    shot("dropoff")
    s, t = wait_for(("idle",), 90)
    print("despawned after", t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
