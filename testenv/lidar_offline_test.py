"""Offline checks of lumen/lidar.lua: the real module under LuaJIT (the Lua BeamNG uses), with the game mocked:
castRay against a small world (a floor and oriented boxes, hit by its own slab test, not the module's), vehicles,
the player, the robotaxi state, debugDrawer, and guihooks, which encodes to JSON at once with BeamNG's own rules
(common/utils.lua jsonEncode: a Lua sequence becomes an array, an empty table becomes {}).

usage: py -3.12 testenv/lidar_offline_test.py [--file candidate.lua] [--reference older/lidar.lua]
       exit code 1 when anything fails. --file tests another lidar.lua instead of the source tree's (a candidate, or a
       deliberately broken copy to prove a check can fail). Needs lupa with LuaJIT, found the way lua_harness.py finds
       it (set LUPA_PATH if it is not installed).

Why it exists (2026-11-01): the app's Lidar tab (Storm: "Actually just a live lidar view") receives the scan in
decimetres in the car's own frame. A sign or axis slip there puts the world on the wrong side of the car without any
error, so these checks pin the frame down for several headings, and also the 5 Hz rate, the "no sensor car" notes,
that stale points are not sent, and that nothing is scanned while nobody is looking.
--reference runs the world-view scenes through an older lidar.lua too and requires identical dots (the world view
must not change), and prints what each version allocates per frame.
"""
import json
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lua_harness import EXT, lupa  # noqa: E402  (same lupa lookup as the NN tests)

LIDAR = EXT + "/lumen/lidar.lua"
MAGENTA = (1.0, 0.25, 0.85, 1.0)
IDLE_REASON = "Lidar starts when your Lumen is on its way."

PRELUDE = r'''
math.randomseed(7)
-- vec3: as much of BeamNG's vec3 as lidar.lua uses
local V = {}
V.__index = V
local function new(x, y, z) return setmetatable({x = x, y = y, z = z}, V) end
function vec3(x, y, z)
  if type(x) == "table" then return new(x.x, x.y, x.z) end
  return new(x or 0, y or 0, z or 0)
end
V.__add = function(a, b) return new(a.x + b.x, a.y + b.y, a.z + b.z) end
V.__sub = function(a, b) return new(a.x - b.x, a.y - b.y, a.z - b.z) end
V.__mul = function(a, b)
  if type(a) == "number" then return new(b.x * a, b.y * a, b.z * a) end
  return new(a.x * b, a.y * b, a.z * b)
end
V.__unm = function(a) return new(-a.x, -a.y, -a.z) end
function V:dot(b) return self.x * b.x + self.y * b.y + self.z * b.z end
function V:cross(b) return new(self.y * b.z - self.z * b.y, self.z * b.x - self.x * b.z, self.x * b.y - self.y * b.x) end
function V:length() return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z) end
function V:normalized()
  local L = self:length()
  if L < 1e-12 then return new(0, 0, 0) end
  return new(self.x / L, self.y / L, self.z / L)
end
function V:distance(b)
  local dx, dy, dz = self.x - b.x, self.y - b.y, self.z - b.z
  return math.sqrt(dx * dx + dy * dy + dz * dz)
end

-- the world castRay sees: an endless floor at FLOOR_Z (nil: none) and oriented boxes
FLOOR_Z, WALLS, CASTS = nil, {}, 0
function addBox(cx, cy, cz, rx, ry, fx, fy, hr, hf, hz)
  WALLS[#WALLS + 1] = {c = vec3(cx, cy, cz), ax = vec3(rx, ry, 0), ay = vec3(fx, fy, 0), az = vec3(0, 0, 1), h = vec3(hr, hf, hz)}
end
local function boxEntry(o, d, b)
  local tmin, tmax, nmin = -math.huge, math.huge, nil
  local rel = o - b.c
  for _, k in ipairs({{b.ax, b.h.x}, {b.ay, b.h.y}, {b.az, b.h.z}}) do
    local a, e = k[1], k[2]
    local p, q = rel:dot(a), d:dot(a)
    if math.abs(q) < 1e-12 then
      if math.abs(p) > e then return nil end
    else
      local t1, t2, n1 = (-e - p) / q, (e - p) / q, a * -1
      if t1 > t2 then t1, t2, n1 = t2, t1, a end
      if t1 > tmin then tmin, nmin = t1, n1 end
      if t2 < tmax then tmax = t2 end
      if tmin > tmax then return nil end
    end
  end
  if tmin < 0 then return nil end
  return tmin, nmin
end
function castRay(origin, target, includeTerrain, renderGeometry)
  CASTS = CASTS + 1
  local d = target - origin
  local len = d:length()
  d = d * (1 / len)
  local best, bn = nil, nil
  if FLOOR_Z and d.z < -1e-12 then
    local t = (FLOOR_Z - origin.z) / d.z
    if t >= 0 and t <= len then best, bn = t, vec3(0, 0, 1) end
  end
  for _, b in ipairs(WALLS) do
    local t, n = boxEntry(origin, d, b)
    if t and t <= len and (not best or t < best) then best, bn = t, n end
  end
  if not best then return nil end
  return {pt = origin + d * best, norm = bn, dist = best}
end

-- vehicles: level, heading (fx, fy); box = bounding box centre + half extents (width, length, height)
VEHS, ORDER, PLAYER = {}, {}, nil
local Veh = {}
Veh.__index = Veh
function Veh:getID() return self.id end
function Veh:getJBeamFilename() return self.jbeam end
function Veh:getActive() return self.active end
function Veh:getPosition() return vec3(self.pos) end
function Veh:getDirectionVector() return vec3(self.fwd) end
function Veh:getDirectionVectorUp() return vec3(self.up) end
function Veh:getVelocity() return vec3(self.vel) end
function Veh:getSpawnWorldOOBB()
  local b = self.box
  return {getCenter = function() return vec3(b.c) end, getAxis = function(_, i) return vec3(b.axes[i + 1]) end,
          getHalfExtents = function() return vec3(b.h) end}
end
function addVeh(id, jbeam, active, px, py, pz, fx, fy, cx, cy, cz, hw, hl, hh, speed)
  local fwd = vec3(fx, fy, 0):normalized()
  local up = vec3(0, 0, 1)
  local v = setmetatable({id = id, jbeam = jbeam, active = active, pos = vec3(px, py, pz), fwd = fwd, up = up,
                          vel = fwd * (speed or 0), box = {c = vec3(cx, cy, cz), axes = {fwd:cross(up), fwd, up}, h = vec3(hw, hl, hh)}}, Veh)
  if not VEHS[id] then ORDER[#ORDER + 1] = id end
  VEHS[id] = v
end
function removeVeh(id) VEHS[id] = nil end
function vehiclesIterator()
  local i = 0
  return function()
    while true do
      i = i + 1
      local id = ORDER[i]
      if id == nil then return nil end
      if VEHS[id] then return id, VEHS[id] end
    end
  end
end
function getObjectByID(id) return VEHS[id] end
function getPlayerVehicle(n) return PLAYER and VEHS[PLAYER] or nil end
MOVING = nil                       -- id of a vehicle that drives along its heading at its speed
local function moveAll(dt)
  local v = MOVING and VEHS[MOVING]
  if v then
    local step = v.vel * dt
    v.pos = v.pos + step
    v.box.c = v.box.c + step
  end
end

-- BeamNG's JSON rules (common/utils.lua jsonEncode_rec), so the test sees exactly what the app gets
local function enc(v, out)
  local t = type(v)
  if t == "string" then
    out[#out + 1] = '"' .. (v:gsub('[%c"\\]', function(ch) return string.format("\\u%04x", ch:byte()) end)) .. '"'
  elseif t == "number" then
    out[#out + 1] = (v * 0 == 0) and tostring(v) or (v > 0 and "9e999" or "-9e999")
  elseif t == "boolean" then
    out[#out + 1] = tostring(v)
  elseif t == "table" then
    local k1 = next(v)
    if k1 == 1 and next(v, #v) == nil then
      out[#out + 1] = "["
      for i = 1, #v do
        if i > 1 then out[#out + 1] = "," end
        enc(v[i], out)
      end
      out[#out + 1] = "]"
    elseif k1 == nil then
      out[#out + 1] = "{}"
    else
      out[#out + 1] = "{"
      local first = true
      for k, x in pairs(v) do
        if not first then out[#out + 1] = "," end
        first = false
        out[#out + 1] = '"' .. tostring(k) .. '":'
        enc(x, out)
      end
      out[#out + 1] = "}"
    end
  else
    out[#out + 1] = "null"
  end
end
function jsonEncode(v) local out = {} enc(v, out) return table.concat(out) end

NOW, FRAME, SENT, DOTS, LOGS, MESSAGES, COLORS = 0, 0, {}, {}, {}, {}, 0
guihooks = {trigger = function(name, data) SENT[#SENT + 1] = name .. "\t" .. string.format("%.6f", NOW) .. "\t" .. jsonEncode(data) end}
function ColorF(r, g, b, a) COLORS = COLORS + 1 return {r = r, g = g, b = b, a = a} end
debugDrawer = {drawDot = function(self, p, size, c)
  DOTS[#DOTS + 1] = string.format("%d %.9g %.9g %.9g %g %.6g %.6g %.6g %.6g", FRAME, p.x, p.y, p.z, size, c.r, c.g, c.b, c.a)
end}
function ui_message(msg) MESSAGES[#MESSAGES + 1] = msg end
function log(level, tag, msg) LOGS[#LOGS + 1] = level .. " " .. tag .. ": " .. msg end
STATE = {state = "idle"}
ROBOTAXI = {getState = function() return STATE end}
extensions = {lumen_robotaxi = ROBOTAXI}

-- advance the module by `seconds` of frames of `dt` (jitter: dt varies +-40 %); while the app's request is on, it
-- is renewed every `renew` seconds the way the open tab does it; returns the frame count
function run(M, seconds, dt, renew, jitter)
  local t, nextRenew, frames = 0, renew or math.huge, 0
  while t < seconds - 1e-9 do
    local d = jitter and dt * (0.6 + 0.8 * math.random()) or dt
    FRAME, NOW, t = FRAME + 1, NOW + d, t + d
    moveAll(d)
    M.onUpdate(d)
    frames = frames + 1
    if renew and t >= nextRenew then
      if M.isUiEnabled and M.isUiEnabled() then M.setUiEnabled(true) end
      nextRenew = nextRenew + renew
    end
  end
  return frames
end

-- kilobytes allocated per frame over `frames` frames of 1/60 s (the collector stopped meanwhile). This mock's JSON
-- encoder is swapped out for the measurement: its garbage is the test's, not the module's. The JIT is off while
-- measuring: its allocation sinking removes different tables depending on the code around them, which made the
-- numbers move between runs; interpreted, they are the logical allocations and repeat exactly.
function allocPerFrame(M, frames)
  local trigger = guihooks.trigger
  guihooks.trigger = function(name, data) LAST = data end
  jit.off()
  jit.flush()
  collectgarbage("collect")
  collectgarbage("stop")
  local before = collectgarbage("count")
  for _ = 1, frames do
    FRAME, NOW = FRAME + 1, NOW + 1 / 60
    moveAll(1 / 60)
    M.onUpdate(1 / 60)
  end
  local kb = (collectgarbage("count") - before) / frames
  collectgarbage("restart")
  jit.on()
  guihooks.trigger = trigger
  return kb
end
'''


class Lidar:
    """one LuaJIT runtime with the mocks and a lidar.lua loaded"""

    def __init__(self, file=None):
        file = file or LIDAR                  # read at call time: --file replaces LIDAR
        self.lua = lupa.LuaRuntime(unpack_returned_tuples=True)
        self.lua.execute(PRELUDE)
        self.g = self.lua.globals()
        chunk = self.lua.eval("loadfile")(file.replace("\\", "/"))
        assert chunk is not None, "cannot load " + file
        self.M = chunk()
        self.cars = {}

    # -- scene ---------------------------------------------------------------------------------------------
    def car(self, id, heading_deg, pos=(120.0, -40.0, 10.0), jbeam="lumen", active=True, box_fwd=0.62, box_right=0.0,
            half=(0.98, 2.45, 0.75), box_up=0.55, speed=0.0):
        """a level vehicle; heading 0 = north (+y), 90 = east. Returns its frame: centre C (box centre), fwd, right."""
        a = math.radians(heading_deg)
        fx, fy = math.sin(a), math.cos(a)
        rx, ry = fy, -fx
        px, py, pz = pos
        cx, cy, cz = px + fx * box_fwd + rx * box_right, py + fy * box_fwd + ry * box_right, pz + box_up
        self.g.addVeh(id, jbeam, active, px, py, pz, fx, fy, cx, cy, cz, half[0], half[1], half[2], speed)
        f = {"C": (cx, cy, cz), "P": pos, "f": (fx, fy), "r": (rx, ry), "half": half}
        self.cars[id] = f
        return f

    def at(self, frame, fwd_m, right_m, up_m=0.0):
        """world point fwd_m ahead of and right_m to the right of a car frame's box centre"""
        cx, cy, cz = frame["C"]
        return (cx + frame["f"][0] * fwd_m + frame["r"][0] * right_m, cy + frame["f"][1] * fwd_m + frame["r"][1] * right_m, cz + up_m)

    def box(self, frame, fwd_m, right_m, z, half_right, half_fwd, half_up):
        """a box in a car frame's orientation, centred fwd_m / right_m from the box centre, at height z"""
        x, y, _ = self.at(frame, fwd_m, right_m)
        self.g.addBox(x, y, z, frame["r"][0], frame["r"][1], frame["f"][0], frame["f"][1], half_right, half_fwd, half_up)

    def walker(self, id, frame, fwd_m, right_m, active=True):
        """a walking character (BeamNG: a vehicle with the unicycle jbeam) standing on the floor"""
        x, y, _ = self.at(frame, fwd_m, right_m)
        floor = self.g.FLOOR_Z if self.g.FLOOR_Z is not None else frame["P"][2] - 0.2
        self.g.addVeh(id, "unicycle", active, x, y, floor, 0.0, 1.0, x, y, floor + 0.85, 0.35, 0.35, 0.85, 0.0)
        return x, y

    def ride(self, taxi_id, state="arriving"):
        st = self.lua.table()
        st.state = state
        st.taxiId = taxi_id
        self.g.STATE = st

    def state(self, name):
        st = self.lua.table()
        st.state = name
        self.g.STATE = st

    # -- running -------------------------------------------------------------------------------------------
    def run(self, seconds, dt=1 / 60, renew=2.0, jitter=False):
        return self.g.run(self.M, seconds, dt, renew, jitter)

    def sent(self, since=0):
        out = []
        for i in range(since + 1, len(self.g.SENT) + 1):
            name, t, js = self.g.SENT[i].split("\t", 2)
            out.append((name, float(t), js, json.loads(js)))
        return out

    def snaps(self, since=0):
        return [(t, d) for name, t, js, d in self.sent(since) if name == "LumenRobotaxiLidar" and not d.get("none")]

    def nones(self, since=0):
        return [(t, d) for name, t, js, d in self.sent(since) if name == "LumenRobotaxiLidar" and d.get("none")]

    def dots(self):
        return [self.g.DOTS[i] for i in range(1, len(self.g.DOTS) + 1)]


def triples(snap):
    p = snap["pts"]
    if isinstance(p, dict):                 # an empty Lua table arrives as {}
        assert not p, "pts is an object with keys"
        return []
    assert len(p) % 3 == 0, f"pts has {len(p)} numbers, not whole triples"
    return [(p[i], p[i + 1], p[i + 2]) for i in range(0, len(p), 3)]


def check_shape(snap):
    """the wire format the app relies on"""
    assert snap["range"] == 60, snap.get("range")
    tr = triples(snap)
    assert len(tr) <= 800, f"{len(tr)} points sent, the cap is 800"
    for x, y, k in tr:
        assert isinstance(x, int) and isinstance(y, int) and k in (1, 2, 3, 4), f"bad point {(x, y, k)}"
        assert abs(x) <= 660 and abs(y) <= 660, f"point {(x, y)} dm is beyond the sensor's 60 m"
    assert snap["car"]["hl"] > snap["car"]["hw"] > 0, snap["car"]
    assert isinstance(snap["speed"], (int, float)) and snap["speed"] >= 0
    return tr


def floor_under(h, frame):
    h.g.FLOOR_Z = frame["P"][2] - 0.2           # the reference node sits 0.2 m above the road


TESTS = []


def test(f):
    TESTS.append(f)
    return f


@test
def frame_mapping_for_five_headings():
    """a wall face 10 m straight ahead arrives at (0, +100) dm, one 5 m to the right at (+50, 0), whatever the heading"""
    for heading in (0, 90, 37, 211, -123):
        h = Lidar()
        f = h.car(10, heading, speed=7.26)
        floor_under(h, f)
        h.ride(10)
        floor = h.g.FLOOR_Z
        h.box(f, 10.1, 0.0, floor + 1.5, 0.5, 0.1, 1.5)      # front wall, near face exactly 10 m ahead of the box centre
        h.box(f, 0.0, 5.1, floor + 1.5, 0.1, 0.5, 1.5)       # right wall, near face exactly 5 m to the right
        h.M.setUiEnabled(True)
        h.run(0.7)
        snaps = h.snaps()
        assert snaps, f"heading {heading}: no picture sent"
        tr = check_shape(snaps[-1][1])
        statics = {(x, y) for x, y, k in tr if k == 2}
        assert (0, 100) in statics, f"heading {heading}: nothing at (0, 100); statics near the front: {sorted(p for p in statics if p[1] > 80)}"
        assert (50, 0) in statics, f"heading {heading}: nothing at (50, 0); statics near the right: {sorted(p for p in statics if p[0] > 40)}"
        front = [p for p in statics if p[1] > 80]
        right = [p for p in statics if p[0] > 40]
        assert all(y == 100 and abs(x) <= 5 for x, y in front), f"heading {heading}: front wall points {front}"
        assert all(x == 50 and abs(y) <= 5 for x, y in right), f"heading {heading}: right wall points {right}"
        assert not [p for p in statics if p[1] < -20 or p[0] < -20], f"heading {heading}: static points behind or left of the car"
        ground = [(x, y) for x, y, k in tr if k == 1]
        assert len(ground) > 100, f"heading {heading}: only {len(ground)} ground points"
        snap = snaps[-1][1]
        assert snap["car"] == {"hl": 2.45, "hw": 0.98}, snap["car"]
        assert snap["speed"] == 7.3, snap["speed"]           # 7.26 m/s, rounded to 0.1
        assert "you" not in snap, "no player, yet a 'you' was sent"


@test
def the_origin_is_the_box_centre_not_the_sensor_or_the_reference_node():
    """box centre 0.2 m ahead of the reference node and 0.3 m right of the sensor's line: walls still read 100 and 50"""
    for heading in (15, 250):
        h = Lidar()
        f = h.car(10, heading, box_fwd=0.2, box_right=0.3)
        floor_under(h, f)
        h.ride(10)
        floor = h.g.FLOOR_Z
        h.box(f, 10.1, 0.0, floor + 1.5, 1.0, 0.1, 1.5)
        h.box(f, 0.0, 5.1, floor + 1.5, 0.1, 1.0, 1.5)
        h.M.setUiEnabled(True)
        h.run(0.7)
        tr = check_shape(h.snaps()[-1][1])
        front = [(x, y) for x, y, k in tr if k == 2 and y > 80]
        right = [(x, y) for x, y, k in tr if k == 2 and x > 40]
        assert front and all(y == 100 for x, y in front), f"heading {heading}: front wall at {front}"
        assert right and all(x == 50 for x, y in right), f"heading {heading}: right wall at {right}"
        # the sensor is 0.3 m left of the box centre, so the straight-ahead ray meets the wall at x = -3 dm
        assert (-3, 100) in front, f"heading {heading}: straight-ahead hit not at (-3, 100): {sorted(front)}"


@test
def vehicles_people_and_no_phantom_walker():
    h = Lidar()
    f = h.car(10, 30, speed=3.0)
    floor_under(h, f)
    h.ride(10)
    floor = h.g.FLOOR_Z
    # a parked pickup behind-left (box 4.6 m x 1.9 m), the rider walking 8 m ahead-left, and an inactive unicycle
    # 6 m ahead-right (walking mode leaves one like that where the rider got in)
    px, py, _ = h.at(f, -15, -4)
    h.g.addVeh(20, "pickup", True, px, py, floor, f["f"][0], f["f"][1], px, py, floor + 0.75, 0.95, 2.3, 0.75, 0.0)
    h.walker(30, f, 8, -3)
    h.g.PLAYER = 30
    h.walker(31, f, 6, 3, active=False)
    h.M.setUiEnabled(True)
    h.run(0.7)
    snap = h.snaps()[-1][1]
    tr = check_shape(snap)
    cars = [(x, y) for x, y, k in tr if k == 3]
    people = [(x, y) for x, y, k in tr if k == 4]
    assert len(cars) >= 4, f"only {len(cars)} vehicle points"
    assert all(abs(x + 40) <= 12 and abs(y + 150) <= 26 for x, y in cars), f"vehicle points outside the pickup's footprint: {cars}"
    assert people, "the walking rider produced no person points"
    assert all(abs(x + 30) <= 6 and abs(y - 80) <= 6 for x, y in people), f"person points away from the rider: {people}"
    phantom = [(x, y, k) for x, y, k in tr if k in (3, 4) and abs(x - 30) <= 8 and abs(y - 60) <= 8]
    assert not phantom, f"the inactive unicycle shows up: {phantom}"
    assert snap["you"] == {"x": -30, "y": 80, "d": 8.5}, snap.get("you")
    # the world view still paints people in the vehicle magenta
    h.M.setEnabled(True)
    h.run(0.2)
    wx, wy = h.at(f, 8, -3)[:2]
    near = []
    for d in h.dots():
        parts = d.split()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        if math.hypot(x - wx, y - wy) < 0.6 and z > floor + 0.3:
            near.append(tuple(round(float(c), 6) for c in parts[5:9]))
    assert near and all(c == MAGENTA for c in near), f"world-view dots on the walker: {set(near)}"


@test
def you_is_left_out_while_riding():
    """the player sits in the Lumen: it is the sensor car, and no 'you' marker is sent"""
    h = Lidar()
    f = h.car(10, 120)
    floor_under(h, f)
    h.g.PLAYER = 10
    h.ride(77, "riding")                 # robotaxi.lua's ride car id is irrelevant: the player's own Lumen comes first
    h.M.setUiEnabled(True)
    h.run(1.0)
    snaps = h.snaps()
    assert len(snaps) >= 3 and all("you" not in d for _, d in snaps), "a 'you' was sent while the player rides"
    assert not h.nones(), "a 'none' note was sent while the player sits in a Lumen"


@test
def pictures_at_five_hertz():
    # steady 60 fps for 10 s, the request renewed every 2 s like the app does
    h = Lidar()
    f = h.car(10, 0)
    floor_under(h, f)
    h.ride(10)
    h.M.setUiEnabled(True)
    h.run(10.0)
    ts = [t for t, _ in h.snaps()]
    assert 49 <= len(ts) <= 51, f"{len(ts)} pictures in 10 s at 60 fps"
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert all(0.18 <= g <= 0.235 for g in gaps), f"gaps from {min(gaps):.3f} to {max(gaps):.3f} s"
    assert 0.19 <= ts[0] <= 0.235, f"first picture after {ts[0]:.3f} s (it waits one sweep, 0.2 s)"
    # uneven frames around 40 fps
    h = Lidar()
    f = h.car(10, 45)
    floor_under(h, f)
    h.ride(10)
    h.M.setUiEnabled(True)
    h.run(10.0, dt=1 / 40, jitter=True)
    n = len(h.snaps())
    assert 48 <= n <= 52, f"{n} pictures in 10 s with uneven frames"
    # a one-second hitch: the pictures resume at 5 Hz, without a burst of catch-up pictures in the next frames
    h = Lidar()
    f = h.car(10, 45)
    floor_under(h, f)
    h.ride(10)
    h.M.setUiEnabled(True)
    h.run(1.0)
    k = len(h.sent())
    h.run(1.0, dt=1.0)
    h.run(1.0)
    ts = [t for t, _ in h.snaps(k)]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert 5 <= len(ts) <= 7 and all(g >= 0.18 for g in gaps), f"after a hitch: {len(ts)} pictures, gaps {[round(g, 3) for g in gaps]}"
    # 4 fps (a frame is longer than the picture period): one picture per frame
    h = Lidar()
    f = h.car(10, 45)
    floor_under(h, f)
    h.ride(10)
    h.M.setUiEnabled(True)
    frames = h.run(5.0, dt=0.25)
    ts = [t for t, _ in h.snaps()]
    assert len(ts) <= frames and len(set(ts)) == len(ts), "more than one picture in a frame"
    assert len(ts) >= frames - 2, f"{len(ts)} pictures in {frames} slow frames"


@test
def none_is_sent_without_a_sensor_car():
    h = Lidar()
    h.car(5, 0, jbeam="pickup")          # the player drives an ordinary car, no ride
    h.g.PLAYER = 5
    h.state("idle")
    h.M.setUiEnabled(True)
    h.run(10.05)
    sent = h.sent()
    assert sent and all(d.get("none") for _, _, _, d in sent), "a picture was sent without a sensor car"
    ts = [t for _, t, _, _ in sent]
    assert 10 <= len(ts) <= 11, f"{len(ts)} notes in 10 s, expected one a second"
    assert ts[0] < 0.02, f"the first note came after {ts[0]:.3f} s"
    assert all(abs((b - a) - 1.0) < 0.02 for a, b in zip(ts, ts[1:])), "notes are not one a second"
    assert all(d["reason"] == IDLE_REASON for _, _, _, d in sent), {d["reason"] for _, _, _, d in sent}
    assert h.g.CASTS == 0, f"{h.g.CASTS} rays cast without a sensor car"
    # the ride is being dispatched: the new reason shows within a frame
    k = len(h.sent())
    h.state("dispatching")
    h.run(1 / 60)
    new = h.sent(k)
    assert len(new) == 1 and new[0][3]["reason"].startswith("Finding you a Lumen"), new
    # a car is assigned: pictures start one sweep later
    f = h.car(10, 200)
    floor_under(h, f)
    h.ride(10)
    t0 = float(h.g.NOW)
    k = len(h.sent())
    h.run(1.0)
    snaps = h.snaps(k)
    assert 4 <= len(snaps) <= 6 and not h.nones(k), f"{len(snaps)} pictures in the first second after the car came"
    assert 0.19 <= snaps[0][0] - t0 <= 0.235, f"first picture {snaps[0][0] - t0:.3f} s after the car came"
    # the car object vanishes while the ride still names it: the note says so at once
    h.g.removeVeh(10)
    k = len(h.sent())
    h.run(1 / 60)
    new = h.sent(k)
    assert len(new) == 1 and new[0][3].get("none") and new[0][3]["reason"] == "Waiting for your Lumen's lidar.", new
    # no robotaxi extension at all
    h.g.extensions.lumen_robotaxi = None
    k = len(h.sent())
    h.run(1.5)
    assert [d["reason"] for _, d in h.nones(k)][:1] == [IDLE_REASON], h.nones(k)


@test
def nothing_is_scanned_while_nobody_looks():
    h = Lidar()
    f = h.car(10, 0)
    floor_under(h, f)
    h.g.PLAYER = 10                       # a sensor car is there the whole time
    h.run(5.0)
    assert h.g.CASTS == 0 and len(h.g.SENT) == 0 and len(h.g.DOTS) == 0, "work done with both views off"
    # the world view alone: its full 140 rays a frame
    h.M.setEnabled(True)
    c0 = h.g.CASTS
    frames = h.run(1.0)
    assert h.g.CASTS - c0 == 140 * frames, f"{h.g.CASTS - c0} rays in {frames} frames, expected 140 a frame"
    assert len(h.g.DOTS) > 0 and len(h.g.SENT) == 0
    h.M.setEnabled(False)
    c0 = h.g.CASTS
    h.run(1.0)
    assert h.g.CASTS == c0, "still scanning after the world view went off"
    # the app alone: one full sweep (768 rays) per 0.2 s picture, 64 or 65 rays a frame at 60 fps
    h.M.setUiEnabled(True)
    c0 = h.g.CASTS
    frames = h.run(1.0)
    per = (h.g.CASTS - c0) / frames
    assert 64 <= per <= 65, f"{per:.1f} rays a frame with only the app looking"
    # both: the world view's rate
    h.M.setEnabled(True)
    c0 = h.g.CASTS
    frames = h.run(0.5)
    assert h.g.CASTS - c0 == 140 * frames
    h.M.setEnabled(False)
    h.M.setUiEnabled(False)
    assert not h.M.isUiEnabled() and not h.M.isEnabled()
    c0, k = h.g.CASTS, len(h.g.SENT)
    h.run(3.0)
    assert h.g.CASTS == c0 and len(h.g.SENT) == k, "still scanning or sending with both views off"


@test
def the_apps_request_expires_unless_renewed():
    h = Lidar()
    f = h.car(10, 0)
    floor_under(h, f)
    h.g.PLAYER = 10
    h.M.setUiEnabled(True)
    h.run(5.5, renew=None)                # the app went away without saying "off"
    assert h.M.isUiEnabled(), "the request ran out before its 6 s"
    h.run(1.0, renew=None)
    assert not h.M.isUiEnabled(), "the request did not run out after 6 s without a renewal"
    c0, k = h.g.CASTS, len(h.g.SENT)
    h.run(3.0, renew=None)
    assert h.g.CASTS == c0 and len(h.g.SENT) == k, "scanning or sending after the request ran out"
    assert any("stopped renewing" in h.g.LOGS[i] for i in range(1, len(h.g.LOGS) + 1)), "no log line for the lapse"
    # renewed, it stays on; a console test can ask for longer
    h.M.setUiEnabled(True)
    h.run(30.0)
    assert h.M.isUiEnabled(), "lapsed although renewed every 2 s"
    h.M.setUiEnabled(True, 3600)
    h.run(60.0, renew=None)
    assert h.M.isUiEnabled(), "setUiEnabled(true, 3600) lapsed within a minute"


@test
def stale_points_are_not_sent():
    """POINT_LIFE (0.45 s): a wall that disappears is gone from the picture within half a second"""
    h = Lidar()
    f = h.car(10, 70)
    h.g.PLAYER = 10                       # no floor: the wall is the only thing to hit
    h.box(f, 10.1, 0.0, f["P"][2] + 1.0, 3.0, 0.1, 2.5)
    h.M.setUiEnabled(True)
    h.run(0.6)
    assert [p for p in triples(h.snaps()[-1][1]) if p[1] == 100], "the wall is not in the picture"
    h.g.WALLS = h.lua.table()
    t0 = float(h.g.NOW)
    k = len(h.sent())
    h.run(1.0)
    late = [(t, d) for t, d in h.snaps(k) if t - t0 >= 0.46]
    assert late, "no pictures after the wall went"
    for t, d in late:
        assert triples(d) == [], f"{len(triples(d))} points still sent {t - t0:.2f} s after the wall went"
    assert late[-1][1]["pts"] == {}, "an empty picture should arrive as {} (BeamNG's encoder); the app must cope"


@test
def world_view_rules_hold_and_the_app_does_not_disturb_them():
    def scene(ui):
        h = Lidar()
        f = h.car(10, 33, speed=9.0)
        floor_under(h, f)
        h.g.PLAYER = 10
        h.g.MOVING = 10
        floor = h.g.FLOOR_Z
        h.box(f, 14, -6, floor + 3.0, 2.0, 2.0, 3.0)                     # a building corner ahead-left
        px, py, _ = h.at(f, 20, 3)
        h.g.addVeh(20, "pickup", True, px, py, floor, f["f"][0], f["f"][1], px, py, floor + 0.75, 0.95, 2.3, 0.75, 0.0)
        h.M.setEnabled(True)
        if ui:
            h.M.setUiEnabled(True)
        h.run(1.5)
        return h, floor

    h1, floor = scene(False)
    h2, _ = scene(True)
    d1, d2 = h1.dots(), h2.dots()
    assert d1 == d2, f"the world view changes while the app also looks ({len(d1)} vs {len(d2)} dots)"
    assert len(h2.snaps()) >= 6
    kinds = set()
    for d in d1:
        parts = d.split()
        size, r, g, b, a = float(parts[4]), *[float(c) for c in parts[5:9]]
        assert size == 7 and a == 1, d
        if (round(r, 6), round(g, 6), round(b, 6)) == MAGENTA[:3]:
            kinds.add("vehicle")
        elif abs(r - 0.10) < 1e-6:
            assert 0.40 - 1e-6 <= g <= 0.85 + 1e-6 and 0.80 - 1e-6 <= b <= 0.95 + 1e-6, d       # ground: teal to blue
            kinds.add("ground")
        else:
            assert r == 1.0 and abs(b - 0.15) < 1e-6 and 0.15 - 1e-6 <= g <= 0.75 + 1e-6, d     # static: amber to red
            kinds.add("static")
    assert kinds == {"vehicle", "ground", "static"}, kinds


def reference_check(old_path):
    """identical world-view dots from the old and the new lidar.lua, and allocations per frame of each"""
    def scene(file, ui):
        h = Lidar(file)
        f = h.car(10, 33, speed=9.0)
        floor_under(h, f)
        h.g.PLAYER = 10
        h.g.MOVING = 10
        floor = h.g.FLOOR_Z
        h.box(f, 14, -6, floor + 3.0, 2.0, 2.0, 3.0)
        h.box(f, -9, 7, floor + 2.0, 1.0, 4.0, 2.0)
        for i, (fw, rt) in enumerate(((20, 3), (-12, -3.5), (35, -3.2))):
            px, py, _ = h.at(f, fw, rt)
            h.g.addVeh(20 + i, "pickup", True, px, py, floor, f["f"][0], f["f"][1], px, py, floor + 0.75, 0.95, 2.3, 0.75, 0.0)
        h.walker(30, f, 9, 4)
        h.M.setEnabled(True)
        if ui:
            h.M.setUiEnabled(True)
        return h

    a, b = scene(old_path, False), scene(LIDAR, False)
    a.run(3.0)
    b.run(3.0)
    da, db = a.dots(), b.dots()
    same = da == db
    print(f"      world-view dots, old vs new: {len(da)} vs {len(db)}, identical: {same}")
    rows = []
    for label, file, ui, world in (("old, world view", old_path, False, True), ("new, world view", LIDAR, False, True),
                                   ("new, world view + app", LIDAR, True, True), ("new, app only", LIDAR, True, False)):
        h = scene(file, ui)
        if not world:
            h.M.setEnabled(False)
        h.run(0.5)
        rows.append((label, float(h.g.allocPerFrame(h.M, 300))))
    for label, kb in rows:
        print(f"      allocated per frame, {label:24s} {kb:8.1f} KB")
    print("      (the mock vec3 is a Lua table, so every vector op allocates here; BeamNG's FFI vec3 does not cost the same)")
    return same


def main():
    global LIDAR
    args = sys.argv[1:]
    ref = args[args.index("--reference") + 1] if "--reference" in args else None
    if "--file" in args:
        LIDAR = os.path.abspath(args[args.index("--file") + 1]).replace("\\", "/")
        print("testing " + LIDAR)
    failed = 0
    for f in TESTS:
        try:
            f()
            print("PASS  " + f.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL  " + f.__name__ + ": " + str(e))
        except Exception:                                    # noqa: BLE001
            failed += 1
            print("ERROR " + f.__name__ + "\n" + traceback.format_exc())
    total = len(TESTS)
    if ref:
        total += 1
        try:
            if reference_check(ref):
                print("PASS  world_view_identical_to_reference")
            else:
                failed += 1
                print("FAIL  world_view_identical_to_reference")
        except Exception:                                    # noqa: BLE001
            failed += 1
            print("ERROR world_view_identical_to_reference\n" + traceback.format_exc())
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
