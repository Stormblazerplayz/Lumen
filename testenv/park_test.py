"""Parking test on the test instance (lumen/parking.lua). The test copy must be running (py game.py start --owner NAME).

    py park_test.py selftest                   the maths of parking.lua run in the game's own Lua (Dubins ends, rectangles, backing sign)
    py park_test.py probe LEVEL X Y [R]        are the level's parking spaces loaded in freeroam? counts around a point
    py park_test.py direct LEVEL CX CY [N]     N times: put a fleet Lumen in an aisle of the lot at (CX, CY), park it, measure
                                               where it ended up, back it out, measure again; counts successes and failures
    py park_test.py ride LEVEL X Y [DEST]      a scripted ride whose pickup is at (X, Y): call, arrive (parks), board, set a
                                               destination, leave the space, drive, arrive (parks if there is room), exit
    py park_test.py log                        the "parking:" lines of the game log

Screenshots go to out/game/park_*.png. Nothing here touches Storm's game: it only talks to the test copy's MCP port."""
import json
import math
import os
import re
import sys
import time

import mcp
from drive_test import load_level

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out", "game")
LOG = os.path.join(os.environ["LOCALAPPDATA"], "BeamNGTest", "robotaxi", "current", "beamng.log")
PRE = """
local robo = extensions.lumen_robotaxi
local parking = require('lumen/parking')
local drive = require('lumen/drive')
local roads = require('lumen/roads')
"""


def lj(code, timeout=60):
    s = mcp.lua(PRE + code, timeout=timeout)
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return {"raw": s}


def shot(name, top=None):
    """top = (x, y, z): a free camera 16 m straight above that point looking down, else whatever the camera shows"""
    os.makedirs(OUT, exist_ok=True)
    if top:
        mcp.lua("""
core_camera.setByName(0, 'free')
core_camera.setPosition(0, vec3(%f, %f - 2, %f + 16))
core_camera.setRotation(0, quatFromDir(vec3(0, 2, -16):normalized(), vec3(0, 0, 1)))
return 1""" % top)
        time.sleep(0.8)
    p = os.path.join(OUT, f"park_{name}.png")
    try:
        mcp.screenshot(p)
        print("   shot", os.path.normpath(p))
    except Exception as e:
        print("   shot failed:", e)
    return p


def chase(vid, back=10, up=4.5):
    mcp.lua(f"""
local v = getObjectByID({vid})
if not v then return 'none' end
local p, d = v:getPosition(), v:getDirectionVector()
local c = p - d * {back} + vec3(0, 0, {up})
core_camera.setByName(0, 'free')
core_camera.setPosition(0, c)
core_camera.setRotation(0, quatFromDir((p + d * 3 - c):normalized(), vec3(0, 0, 1)))
return 'ok'""")
    time.sleep(0.5)


def log_lines(since=0, pat=r"parking:"):
    try:
        with open(LOG, "rb") as f:
            f.seek(since)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    return [l.split("|", 4)[-1].strip() if "|" in l else l for l in text.splitlines() if re.search(pat, l)]


def log_size():
    return os.path.getsize(LOG) if os.path.exists(LOG) else 0


# ---------------------------------------------------------------------------------
def probe(level, x, y, r=40):
    print("level:", load_level(level))
    out = lj(f"return jsonEncode(parking.probe(vec3({x}, {y}, 0), {r}))")
    print(json.dumps(out, indent=1))
    return out


def lanes(cx, cy, z, r=60):
    return lj(f"""
local out = {{}}
for _, e in ipairs(roads.closestLinks(vec3({cx}, {cy}, {z}), {{maxDist = {r}}}, 80)) do
  local L = e.L
  out[#out + 1] = {{ax = L.pa.x, ay = L.pa.y, az = L.pa.z, bx = L.pb.x, by = L.pb.y, bz = L.pb.z, len = L.len, rad = L.radius, pub = L.public, d = e.d}}
end
return jsonEncode(out)""")


def pick_car():
    st = mcp.lua("return jsonEncode(extensions.lumen_robotaxi.getState())")
    cars = json.loads(st).get("fleet", [])
    for c in cars:
        if c.get("id") and c.get("status") == "cruising":
            return c["slot"], c["id"]
    return None, None


def put_car(slot, x, y, z, hx, hy):
    return lj(f"""
local car = robo._car({slot})
local veh = car and car.id and getObjectByID(car.id)
if not veh then return jsonEncode({{err = 'no car'}}) end
parking.forget(car)
drive.release(car)
car.status = 'assigned' car.needsCruise = false car.pendingTrip = nil car.recovering = nil car.evading = nil car.readyAt = 0
local dir = vec3({hx}, {hy}, 0):normalized()
spawn.safeTeleport(veh, vec3({x}, {y}, {z} + 0.3), quatFromDir(dir, vec3(0, 0, 1)), true, nil, false)
return jsonEncode({{ok = true, id = car.id}})""")


def settle_hold(slot):
    return lj(f"""
local car = robo._car({slot})
local veh = getObjectByID(car.id)
drive.stop(car)
local p, d = veh:getPosition(), veh:getDirectionVector()
return jsonEncode({{x = p.x, y = p.y, z = p.z, hx = d.x, hy = d.y, speed = veh:getVelocity():length()}})""")


def car_pose(slot):
    return lj(f"""
local car = robo._car({slot})
local veh = car and car.id and getObjectByID(car.id)
if not veh then return jsonEncode({{err = 'no car'}}) end
local p, d = veh:getPosition(), veh:getDirectionVector()
return jsonEncode({{x = p.x, y = p.y, z = p.z, hx = d.x, hy = d.y, speed = veh:getVelocity():length(), park = car.park and car.park.phase or nil}})""")


def game_check(slot, key):
    """BeamNG's own checkParking on the space, strict (every corner) and the usual 0.8"""
    return lj(f"""
local car = robo._car({slot})
local sites = extensions.gameplay_city.getSites()
local ps = sites.parkingSpots.objects[{key}]
if not ps then return jsonEncode({{err = 'no space {key}'}}) end
local v1, r1 = ps:checkParking(car.id, 1.0)
local v8, r8 = ps:checkParking(car.id, 0.8)
return jsonEncode({{strict = v1, corners = r1, usual = v8}})""")


def wait_park(slot, timeout=120):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        st = lj(f"return jsonEncode(robo.parkStatus({slot}) or {{none = true}})")
        ph = st.get("phase")
        if ph != last:
            print(f"   t={time.time() - t0:5.1f}s phase={ph} {('space ' + str(st.get('space'))) if st.get('space') else ''}")
            last = ph
        if ph == "parked":
            return st
        if st.get("none"):
            return None
        time.sleep(0.5)
    return lj(f"return jsonEncode(robo.parkStatus({slot}) or {{none = true}})")


def wait_left(slot, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = lj(f"return jsonEncode(robo.parkStatus({slot}) or {{none = true}})")
        if st.get("none"):
            return time.time() - t0
        time.sleep(0.5)
    return None


def direct(level, cx, cy, n=6, arrive_speed=0):
    size0 = log_size()
    print("level:", load_level(level))
    # the fleet needs a while to spawn
    t0 = time.time()
    slot = vid = None
    while time.time() - t0 < 150:
        slot, vid = pick_car()
        if slot:
            break
        time.sleep(3)
    if not slot:
        print("no fleet car after 150 s")
        return 1
    print("using fleet car", slot, "id", vid)
    pr = lj(f"return jsonEncode(parking.probe(vec3({cx}, {cy}, 0), 45))")
    print("probe:", json.dumps(pr))
    z = pr.get("z") or 0
    ls = [l for l in lanes(cx, cy, z) if not l.get("pub") and l.get("len", 0) >= 8 and l.get("rad", 0) >= 1.6]
    print(len(ls), "private lanes near the lot")
    poses = []
    for l in ls:
        for t in (0.5,):
            px, py, pz = l["ax"] + (l["bx"] - l["ax"]) * t, l["ay"] + (l["by"] - l["ay"]) * t, l["az"] + (l["bz"] - l["az"]) * t
            for sgn in (1, -1):
                poses.append((px, py, pz, sgn * (l["bx"] - l["ax"]), sgn * (l["by"] - l["ay"]), l))
    poses = poses[:n]
    results = []
    for k, (px, py, pz, hx, hy, l) in enumerate(poses):
        print(f"\n[{k + 1}/{len(poses)}] aisle pose ({px:.1f}, {py:.1f}) heading {math.degrees(math.atan2(hy, hx)):.0f} deg")
        print("   put:", put_car(slot, px, py, pz, hx, hy))
        time.sleep(3.0)
        base = settle_hold(slot)
        if base.get("speed", 9) > 0.5:
            print("   car still moving:", base)
            time.sleep(3.0)
            base = settle_hold(slot)
        shot(f"direct{k + 1}_0_before", top=(base["x"], base["y"], base["z"]))
        ok = lj(f"""
local car = robo._car({slot})
local veh = getObjectByID(car.id)
local ok, info = parking.start(car, veh, veh:getPosition(), {{what = 'test', nn = false}})
return jsonEncode({{ok = ok, info = info}})""")
        print("   start:", ok)
        r = {"pose": (px, py), "started": bool(ok.get("ok"))}
        if ok.get("ok"):
            time.sleep(6)
            chase(vid, 9, 7)
            shot(f"direct{k + 1}_1_midway")
            st = wait_park(slot, 150)
            if st and st.get("phase") == "parked":
                fin = st.get("final") or {}
                time.sleep(1.0)
                chk = game_check(slot, st["key"])
                pose = car_pose(slot)
                d = math.hypot(pose["x"] - base["x"], pose["y"] - base["y"])
                good = fin.get("inside") and abs(fin.get("side", 9)) <= 0.3 and abs(fin.get("ang", 99)) <= 8
                print(f"   PARKED in {st.get('space')}: along {fin.get('along', 0):+.2f} m, side {fin.get('side', 0):+.2f} m, {fin.get('ang', 0):+.1f} deg, inside {fin.get('inside')}, "
                      f"{d:.1f} m from the stop point, {st.get('t', 0):.0f} s; game checkParking strict={chk.get('strict')} usual={chk.get('usual')}  -> {'GOOD' if good else 'OUT OF TOLERANCE'}")
                r.update(parked=True, good=bool(good), fin=fin, dist=d, game=chk, t=st.get("t"))
                shot(f"direct{k + 1}_2_parked", top=(pose["x"], pose["y"], pose["z"]))
                # leave
                lv = lj(f"""
local car = robo._car({slot})
local veh = getObjectByID(car.id)
return jsonEncode({{ok = parking.leave(car, veh)}})""")
                print("   leave:", lv)
                time.sleep(4)
                chase(vid, 9, 7)
                shot(f"direct{k + 1}_3_backing")
                took = wait_left(slot, 90)
                pose2 = car_pose(slot)
                if took is None:
                    print("   did not finish leaving in 90 s")
                    r["left"] = False
                else:
                    ang = math.degrees(math.atan2(pose2["hy"], pose2["hx"]) - math.atan2(hy, hx))
                    ang = (ang + 180) % 360 - 180
                    d2 = math.hypot(pose2["x"] - base["x"], pose2["y"] - base["y"])
                    print(f"   OUT after {took:.0f} s: {d2:.1f} m from where it started, {ang:+.0f} deg from the aisle heading, speed {pose2['speed']:.2f}")
                    r.update(left=True, leftT=took, leftAng=ang, leftDist=d2)
                    shot(f"direct{k + 1}_4_out", top=(pose2["x"], pose2["y"], pose2["z"]))
            else:
                print("   did not park:", st)
                r["parked"] = False
                lj(f"local car = robo._car({slot}) parking.forget(car) drive.stop(car) return '{{}}'")
        results.append(r)
    print("\n==== summary")
    started = [r for r in results if r["started"]]
    parked = [r for r in started if r.get("parked")]
    good = [r for r in parked if r.get("good")]
    left = [r for r in parked if r.get("left")]
    print(f"poses {len(results)}; a space found and path started {len(started)}; parked {len(parked)}; within 0.3 m / 8 deg / inside the lines {len(good)}; backed out {len(left)}")
    for r in results:
        if not r["started"]:
            print("   no space for the pose", tuple(round(v, 1) for v in r["pose"]))
    for line in log_lines(size0):
        print("  ", line[:300])
    return 0


def ride(level, x, y, dest="__surprise"):
    size0 = log_size()
    print("level:", load_level(level))
    pr = lj(f"return jsonEncode(parking.probe(vec3({x}, {y}, 0), 45))")
    print("probe:", json.dumps(pr))
    z = pr.get("z") or 0
    # the player goes to the lot (the car they are in is moved with them)
    print("player to the pickup:", mcp.lua(f"""
local pv = getPlayerVehicle(0)
if not pv then return 'no player vehicle' end
spawn.safeTeleport(pv, vec3({x}, {y}, {z} + 0.5), quatFromDir(vec3(1, 0, 0), vec3(0, 0, 1)), true, nil, false)
return 'ok'"""))
    time.sleep(4)
    print("call:", mcp.lua("return tostring(extensions.lumen_robotaxi.callTaxi())"))
    t0 = time.time()
    last = None
    state = {}
    while time.time() - t0 < 600:
        s = lj("return jsonEncode(robo.getState())")
        slot = None
        for c in s.get("fleet", []):
            if c.get("id") == s.get("taxiId"):
                slot = c["slot"]
        ps = lj(f"return jsonEncode(robo.parkStatus({slot}) or {{none = true}})") if slot else {}
        key = (s.get("state"), ps.get("phase"))
        if key != last:
            print(f"  t={time.time() - t0:5.0f}s ride={s.get('state')} park={ps.get('phase')} {ps.get('space') or ''}")
            last = key
        if s.get("state") in ("waiting", "idle") and not ps.get("phase") in ("into", "settle"):
            state = s
            break
        time.sleep(2)
    if state.get("state") != "waiting":
        print("never reached waiting:", state)
        for line in log_lines(size0):
            print("  ", line[:300])
        return 2
    slot = [c["slot"] for c in state["fleet"] if c.get("id") == state["taxiId"]][0]
    ps = lj(f"return jsonEncode(robo.parkStatus({slot}) or {{none = true}})")
    pose = car_pose(slot)
    print("at the pickup:", json.dumps(ps))
    chase(state["taxiId"], 9, 6)
    shot("ride_1_waiting_chase")
    shot("ride_1_waiting_top", top=(pose["x"], pose["y"], pose["z"]))
    if ps.get("key"):
        print("game checkParking:", game_check(slot, ps["key"]))
    print("board:", mcp.lua("return tostring(extensions.lumen_robotaxi.board())"))
    time.sleep(3)
    print("destination:", dest, mcp.lua(f"return tostring(extensions.lumen_robotaxi.setDestination({json.dumps(dest)}))"))
    t0 = time.time()
    last = None
    while time.time() - t0 < 900:
        s = lj("return jsonEncode(robo.getState())")
        ps = lj(f"return jsonEncode(robo.parkStatus({slot}) or {{none = true}})")
        key = (s.get("state"), ps.get("phase"), ps.get("stage"))
        if key != last:
            print(f"  t={time.time() - t0:5.0f}s ride={s.get('state')} park={ps.get('phase')}/{ps.get('stage')} remaining={s.get('remaining') and round(s['remaining'])}")
            last = key
            if ps.get("phase") == "leaving" and ps.get("stage") == "back":
                time.sleep(3)
                chase(state["taxiId"], 9, 7)
                shot("ride_2_backing_out")
        if s.get("state") == "arrived":
            break
        if s.get("state") in ("idle", "dispatching"):
            break
        time.sleep(3)
    s = lj("return jsonEncode(robo.getState())")
    print("end state:", s.get("state"))
    if s.get("state") == "arrived":
        ps = lj(f"return jsonEncode(robo.parkStatus({slot}) or {{none = true}})")
        pose = car_pose(slot)
        print("at the destination:", json.dumps(ps))
        chase(s["taxiId"], 9, 6)
        shot("ride_3_arrived_chase")
        shot("ride_3_arrived_top", top=(pose["x"], pose["y"], pose["z"]))
        print("exit:", mcp.lua("return tostring(extensions.lumen_robotaxi.exitTaxi())"))
        time.sleep(4)
    print("\nparking log lines:")
    for line in log_lines(size0):
        print("  ", line[:320])
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "selftest":
        out = lj("return jsonEncode(parking._selfTest())")
        print(json.dumps(out, indent=1))
        return 0 if out.get("ok") else 1
    if cmd == "probe":
        probe(argv[2], float(argv[3]), float(argv[4]), float(argv[5]) if len(argv) > 5 else 40)
    elif cmd == "direct":
        return direct(argv[2], float(argv[3]), float(argv[4]), int(argv[5]) if len(argv) > 5 else 6)
    elif cmd == "ride":
        return ride(argv[2], float(argv[3]), float(argv[4]), argv[5] if len(argv) > 5 else "__surprise")
    elif cmd == "log":
        for line in log_lines(0):
            print(line[:320])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
