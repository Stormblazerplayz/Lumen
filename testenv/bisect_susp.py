"""Bisect the spawn explosion by stripping suspension features from the installed
(test-profile) jbeam and respawning. Usage: py bisect_susp.py <variant>
variants: nowheels, nohydro, nobump, nopreload, all (= all four removed), restore
"""
import json
import os
import shutil
import sys
import time

import mcp

VEH = os.path.join(os.environ["LOCALAPPDATA"], "BeamNGTest", "robotaxi", "current", "mods", "unpacked", "lumen_robotaxi", "vehicles", "lumen")
SRC = os.path.join(VEH, "lumen_suspension.jbeam")
BAK = SRC + ".orig"


def strip(variant):
    if not os.path.exists(BAK):
        shutil.copyfile(SRC, BAK)
    d = json.load(open(BAK))
    for part in d.values():
        if variant in ("nowheels", "all"):
            part.pop("pressureWheels", None)
            part["flexbodies"] = [r for r in part.get("flexbodies", []) if not (isinstance(r, list) and r[0] in ("rtx_wheel", "rtx_tire"))]
            part.pop("powertrain", None)
        if variant in ("nohydro", "all"):
            part.pop("hydros", None)
        rows = []
        for r in part.get("beams", []):
            if isinstance(r, list) and len(r) > 2 and isinstance(r[2], dict):
                o = r[2]
                if variant in ("nobump", "all") and o.get("beamType") == "|BOUNDED":
                    continue
                if variant in ("nopreload", "all") and "beamPreload" in o:
                    o = dict(o)
                    o.pop("beamPreload")
                    r = [r[0], r[1], o]
            rows.append(r)
        if "beams" in part:
            part["beams"] = rows
    json.dump(d, open(SRC, "w"), indent=1)


def main():
    variant = sys.argv[1]
    if variant == "restore":
        if os.path.exists(BAK):
            shutil.copyfile(BAK, SRC)
        print("restored")
        return
    strip(variant)
    mcp.call("resume_physics", {})
    mcp.call("get_instability", {})
    print(mcp.call("spawn", {"model": "lumen", "config": "debugmin"}, timeout=180))
    time.sleep(8)
    print(variant, "-> instability:", mcp.call("get_instability", {}), "| vehicles:", mcp.call("get_vehicles", {})[:120])


if __name__ == "__main__":
    main()
