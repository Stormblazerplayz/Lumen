"""Lumen Drive on the test instance: loads a freeroam level, lets the fleet roam on Lumen Drive,
and reports how each car drives: speed, lateral error from its lane line, state and reason, plus
any Lua errors in the log. Screenshots from a chase camera behind each car.
usage: py drive_test.py [level] [seconds]"""
import json
import os
import re
import sys
import time

import mcp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "game")
LOG = os.path.join(os.environ["LOCALAPPDATA"], "BeamNGTest", "robotaxi", "current", "beamng.log")


def lua_json(code):
    s = mcp.lua(code)
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return None


def fleet():
    return lua_json("return jsonEncode(extensions.lumen_robotaxi.debugFleet())") or []


def load_level(level):
    cur = mcp.lua("local l = getCurrentLevelIdentifier() return tostring(l)")
    if cur == level:
        return True
    mcp.lua(f"freeroam_freeroam.startFreeroamByName('{level}') return 1")
    t0 = time.time()
    while time.time() - t0 < 300:
        time.sleep(5)
        try:
            cur = mcp.lua("local l = getCurrentLevelIdentifier() return tostring(l) .. '|' .. tostring(getPlayerVehicle(0) ~= nil)")
        except Exception:
            continue
        if cur.startswith(level + "|true"):
            # the "Input Changes" popup that 0.40 shows on a fresh profile
            try:
                mcp.call("ui_eval", {"js": "(function(){var b=[...document.querySelectorAll('button')].find(x=>/continue/i.test(x.innerText));if(b){b.click();return 'clicked'}return 'none'})()"})
            except Exception:
                pass
            return True
    return False


def chase_shot(car, name):
    if not car.get("id"):
        return
    mcp.lua(f"""
local v = getObjectByID({car['id']})
if not v then return 'none' end
local p, d = v:getPosition(), v:getDirectionVector()
local cam = p - d * 9 + vec3(0, 0, 4.2)
core_camera.setByName(0, 'free')
core_camera.setPosition(0, cam)
local look = (p + d * 6 - cam):normalized()
core_camera.setRotation(0, quatFromDir(look, vec3(0, 0, 1)))
return 'ok'""")
    time.sleep(0.6)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"drive_{name}.png")
    try:
        mcp.screenshot(path)
        print("   shot", path)
    except Exception as e:
        print("   shot failed:", e)


def log_errors(since_size):
    try:
        with open(LOG, "rb") as f:
            f.seek(since_size)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    return [l for l in text.splitlines() if re.search(r"\|E\||Exception|lumen_drive", l)][-20:]


def main():
    level = sys.argv[1] if len(sys.argv) > 1 else "west_coast_usa"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 90
    size0 = os.path.getsize(LOG) if os.path.exists(LOG) else 0
    print("loading", level, load_level(level))
    mcp.lua("extensions.lumen_robotaxi.setDriveDebug(true) return 1")
    # wait for the fleet to spawn and start driving
    t0 = time.time()
    while time.time() - t0 < 120:
        f = fleet()
        if f and sum(1 for c in f if c.get("mode") == "follow") >= 2:
            break
        time.sleep(3)
    stats = {}
    t0 = time.time()
    shots = 0
    while time.time() - t0 < seconds:
        f = fleet()
        line = []
        for c in f:
            k = c["slot"]
            st = stats.setdefault(k, {"maxlat": 0, "n": 0, "sumlat2": 0, "maxv": 0, "start": None, "last": None, "reasons": set()})
            if c.get("lat") is not None and c.get("mode") == "follow" and (c.get("speed") or 0) > 2:
                st["maxlat"] = max(st["maxlat"], abs(c["lat"]))
                st["sumlat2"] += c["lat"] ** 2
                st["n"] += 1
            st["maxv"] = max(st["maxv"], c.get("speed") or 0)
            if c.get("x") is not None:
                if st["start"] is None:
                    st["start"] = (c["x"], c["y"])
                st["last"] = (c["x"], c["y"])
            if c.get("reason"):
                st["reasons"].add(c["reason"])
            line.append(f"{k}:{c.get('status','?')[:5]}/{c.get('mode','-')} fwd={c.get('fwd') or 0:+5.1f} g={c.get('gear')} t={c.get('thr') or 0:.2f} b={c.get('brk') or 0:.2f} lat={c.get('lat') or 0:+.2f} {c.get('state') or ''} {c.get('stopKind') or ''}")
        print(f"{time.time() - t0:5.1f}s  " + " | ".join(line))
        for c in f:
            st = stats[c["slot"]]
            if c.get("service") and not st.get("svc"):
                st["svc"] = c["service"]
                print(f"   car {c['slot']} went to service: {c['service']} (recoveries {c.get('recoveries')})")
            if abs(c.get("lat") or 0) > 1.5 and c.get("mode") == "follow" and time.time() - st.get("spikeShot", -99) > 20:
                st["spikeShot"] = time.time()
                print(f"   car {c['slot']} is {c['lat']:+.2f} m off its line at {c.get('speed') or 0:.1f} m/s")
                dump = lua_json(f"return jsonEncode(extensions.lumen_robotaxi.debugDump({c['slot']}))")
                if dump:
                    os.makedirs(OUT, exist_ok=True)
                    with open(os.path.join(OUT, f"dump_car{c['slot']}_{int(time.time() - t0)}.json"), "w") as fh:
                        json.dump(dump, fh)
                chase_shot(c, f"spike_car{c['slot']}_{int(time.time() - t0)}")
        if shots < 4 and time.time() - t0 > 10 + shots * 18 and f:
            car = f[shots % len(f)]
            chase_shot(car, f"car{car['slot']}_{shots}")
            shots += 1
        time.sleep(2)
    print("\nsummary")
    for k, st in sorted(stats.items()):
        rms = (st["sumlat2"] / st["n"]) ** 0.5 if st["n"] else float("nan")
        moved = 0
        if st["start"] and st["last"]:
            moved = ((st["last"][0] - st["start"][0]) ** 2 + (st["last"][1] - st["start"][1]) ** 2) ** 0.5
        print(f" car {k}: lateral rms {rms:.2f} m, max {st['maxlat']:.2f} m over {st['n']} samples; top speed {st['maxv']:.1f} m/s; moved {moved:.0f} m (straight line)")
        for r in sorted(st["reasons"]):
            print("    reason:", r)
    errs = log_errors(size0)
    print("\nlog errors/warnings:", len(errs))
    for e in errs:
        print("  ", e[:240])


if __name__ == "__main__":
    main()
