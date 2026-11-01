"""One test cycle against the isolated test instance:
flush BeamNG's jbeam file cache (GE Lua reload), spawn Lumen on the ground at a
fixed spot, wait for it to settle, report instability, take screenshots.

    py cycle.py [config] [--shots name1,name2,...]

Why the reload: BeamNG's jbeam cache is only invalidated by its file watcher, and
files copied in from outside never triggered it (2026-10-31: every 'fix' kept
loading the first, unstable jbeam until the GE VM was reloaded).
"""
import json
import os
import sys
import time

import mcp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "game")
SPOT = (-122.0, -188.0)
VIEWS = {
    "3q_front": {"yaw": 215, "pitch": -10, "distance": 7.0},
    "3q_rear": {"yaw": 140, "pitch": -10, "distance": 7.0},
    "side": {"yaw": 270, "pitch": -4, "distance": 7.5},
    "front": {"yaw": 180, "pitch": -6, "distance": 6.5},
    "top": {"yaw": 215, "pitch": -55, "distance": 8.0},
}


def flush_cache():
    try:
        mcp.call("reload_lua", {}, timeout=60)
    except Exception:
        pass
    time.sleep(6)
    mcp.wait_ready(60)


def spawn(config="robotaxi", yaw_deg=30.0):
    for v in json.loads(mcp.call("get_vehicles", {}) or "[]") or []:
        mcp.call("remove_vehicle", {"id": v["id"]})
    mcp.call("get_instability", {})
    code = f"""
local pos = vec3({SPOT[0]}, {SPOT[1]}, 150)
local h = be:getSurfaceHeightBelow(pos)
local v = core_vehicles.spawnNewVehicle('lumen', {{config = 'vehicles/lumen/{config}.pc', pos = vec3(pos.x, pos.y, h + 0.25), rot = quatFromEuler(0, 0, math.rad({yaw_deg})), autoEnterVehicle = true}})
return tostring(v and v:getID())
"""
    vid = mcp.lua(code).strip()
    time.sleep(8)
    inst = json.loads(mcp.call("get_instability", {}))
    info = mcp.call("get_vehicle", {})
    return vid, inst, info


def shots(names, prefix):
    os.makedirs(OUT, exist_ok=True)
    mcp.call("toggle_ui", {"show": False})
    paths = []
    for n in names:
        mcp.call("orbit_camera", VIEWS[n])
        time.sleep(2.2)
        p = os.path.join(OUT, f"{prefix}_{n}.png")
        mcp.screenshot(p)
        paths.append(p)
    return paths


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    config = args[0] if args else "robotaxi"
    names = ["3q_front"]
    for a in sys.argv[1:]:
        if a.startswith("--shots"):
            names = a.split("=", 1)[1].split(",")
    flush_cache()
    vid, inst, info = spawn(config)
    print("vid", vid, "| instability count", inst.get("count"), "|", info[:220])
    if inst.get("count"):
        print("UNSTABLE:", inst)
        return 1
    for p in shots(names, f"lumen_{config}"):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
