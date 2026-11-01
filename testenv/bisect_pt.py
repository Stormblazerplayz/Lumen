"""Bisect: strip the powertrain (and controllers) from the installed test jbeam,
then spawn the minimal config in mid-air. Usage: py bisect_pt.py <variant>
variants: nopt (no powertrain/energy storage/vehicleController), noctrl (no
controllers only), restore."""
import glob
import json
import os
import shutil
import sys
import time

import mcp

VEH = os.path.join(os.environ["LOCALAPPDATA"], "BeamNGTest", "robotaxi", "current", "mods", "unpacked", "lumen_robotaxi", "vehicles", "lumen")


def backup_all():
    for f in glob.glob(os.path.join(VEH, "*.jbeam")):
        if not os.path.exists(f + ".orig"):
            shutil.copyfile(f, f + ".orig")


def restore_all():
    for f in glob.glob(os.path.join(VEH, "*.jbeam.orig")):
        shutil.copyfile(f, f[:-5])


def strip(variant):
    backup_all()
    restore_all()
    for f in glob.glob(os.path.join(VEH, "*.jbeam")):
        d = json.load(open(f))
        for name, part in d.items():
            if variant in ("nopt",):
                for k in ("powertrain", "energyStorage", "vehicleController", "rearMotor", "frontMotor", "mainBattery"):
                    part.pop(k, None)
            if variant in ("nopt", "noctrl"):
                if "controller" in part:
                    part["controller"] = [["fileName"]]
        json.dump(d, open(f, "w"), indent=1)


def spawn_air():
    code = """
local pos = vec3(-122, -188, 103.5)
local h = be:getSurfaceHeightBelow(pos + vec3(0,0,50))
local v = core_vehicles.spawnNewVehicle('lumen', {config = 'vehicles/lumen/debugmin.pc', pos = vec3(pos.x, pos.y, h + 2.0), rot = quatFromEuler(0, 0, 0), autoEnterVehicle = true})
return tostring(v and v:getID())
"""
    return mcp.lua(code)


def main():
    v = sys.argv[1]
    if v == "restore":
        restore_all()
        print("restored")
        return
    strip(v)
    mcp.call("resume_physics", {})
    for vv in json.loads(mcp.call("get_vehicles", {}) or "{}") or []:
        pass
    mcp.call("get_instability", {})
    print("spawned", spawn_air())
    time.sleep(5)
    print(v, "-> instability:", mcp.call("get_instability", {}), "| vehicles:", mcp.call("get_vehicles", {})[:160])


if __name__ == "__main__":
    main()
