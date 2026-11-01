"""Point the free camera at a spot (or a vehicle) and save a screenshot.
Usage: py lookat.py <vehId|x,y,z> <name> [offset_x offset_y offset_z]"""
import os
import sys
import time

import mcp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "game")


def look(target, name, off=(9.0, 11.0, 6.0)):
    if "," in target:
        tx, ty, tz = (float(v) for v in target.split(","))
        tgt = f"vec3({tx}, {ty}, {tz})"
    else:
        tgt = f"getObjectByID({int(target)}):getPosition()"
    mcp.lua(f"""
local t = {tgt}
local c = t + vec3({off[0]}, {off[1]}, {off[2]})
core_camera.setByName(0, 'free')
core_camera.globalCameraFunction('free', 'setPosition', c)
core_camera.globalCameraFunction('free', 'setRotation', quatFromDir((t + vec3(0, 0, 0.6) - c):normalized(), vec3(0, 0, 1)))
return 'ok'""")
    time.sleep(1.2)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.abspath(os.path.join(OUT, name + ".png"))
    mcp.screenshot(p)
    print(p)


if __name__ == "__main__":
    off = tuple(float(v) for v in sys.argv[3:6]) if len(sys.argv) >= 6 else (9.0, 11.0, 6.0)
    look(sys.argv[1], sys.argv[2], off)
