"""Offline harness for the Lumen Lua modules: the REAL lumen/nn.lua and lumen/guard.lua with the real weights
under LuaJIT (the same Lua BeamNG uses), driven through a mocked `drive` and mocked game objects. No game, no GPU.
Written 2026-11-01 after the in-game NN stall (Storm: "Only the NN sat there"): it reproduced the fault, which the
simulator could not, and measured the fix.

Needs lupa:  py -3.12 -m pip install lupa   (or put it somewhere and set LUPA_PATH to that folder)

A scene is a plain dict. The car stands at (0, 0) heading +y (north) on a straight path that starts 200 m behind it:
  v        own speed, m/s                       vl      the path's speed limit, m/s (the game's log: limit 21.1 = vl / 0.95)
  signs    [(distance_ahead_m, action), ...]    action: "briefStop" (stop sign), "stop" (red), "alert" (yellow), "slow"
  agents   [{rs, rd, speed, face, hl, hw, acc}, ...]
             rs: metres ahead along the road (negative: behind), rd: metres to the right (+x),
             speed: m/s along `face`, face: "along" | "across" | "against", hl/hw: half length / half width
  len      path length in metres (default 600; the car is at arc length 200)
"""
import json
import math
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXT = os.path.join(ROOT, "src", "lua", "ge", "extensions").replace("\\", "/")
BACK = 200.0                      # the mock path starts this far behind the car

for extra in (os.environ.get("LUPA_PATH"),):
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
try:
    from lupa import luajit21 as lupa
except ImportError as e:         # pragma: no cover
    raise SystemExit("lua_harness needs lupa with LuaJIT: py -3.12 -m pip install lupa  (or set LUPA_PATH). " + str(e))

PRELUDE = r'''
LOGS = {}
function log(level, tag, msg) LOGS[#LOGS + 1] = level .. " " .. tag .. ": " .. msg end
function vec3(x, y, z) return {x = x, y = y, z = z} end
VEHS = {}
function getObjectByID(id) return VEHS[id] end
function jsonReadFile(path) return WEIGHTS end

local function V(x, y) return {x = x, y = y, z = 0, dot = function(self, o) return self.x * o.x + self.y * o.y end,
                               length = function(self) return math.sqrt(self.x * self.x + self.y * self.y) end} end

function mkveh(v)
  return {getPosition = function() return V(0, 0) end,
          getDirectionVector = function() return V(0, 1) end,
          getVelocity = function() return V(0, v) end}
end

BACK = %(back)f
function mkpath(vl, len)
  local P = {x = {}, y = {}, z = {}, s = {}, vl = {}, k = {}, seg = {}, n = 0, len = len, segs = {{rad = 3.6, off = 0}}}
  local n = math.floor(len / 2) + 1
  for i = 1, n do
    local s = (i - 1) * 2
    P.x[i], P.y[i], P.z[i], P.s[i], P.vl[i], P.k[i], P.seg[i] = 0, s - BACK, 0, s, vl, 0, 1
  end
  P.n = n
  return P
end

SIGS, AGENTS = {}, {}
drive = {
  _project = function(P, px, py, idx, win)
    local s = math.max(0, math.min(P.len, py + BACK))
    return s, math.max(1, math.min(P.n - 1, math.floor(s / 2) + 1)), px
  end,
  _signalsNear = function(pos, r) return SIGS end,
  _sense = function(id, pos, now) return AGENTS end,
  _sendPlan = function() end,
  _say = function(car, d, key, text, st) SAID = {key = key, text = text, state = st} end,
}
''' % {"back": BACK}


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


class Harness:
    def __init__(self, file=None, module="lumen/nn", weights=None, preload=None):
        """file: load a different nn.lua (a backup, a candidate) under `module` instead of the source tree's.
        weights: path to a weights json other than the shipped one.
        preload: {module name: file} for modules that are not in the source tree yet (a staged lumen/shield)."""
        self.lua = lupa.LuaRuntime(unpack_returned_tuples=True)
        lua = self.lua
        lua.execute('package.path = "%s/?.lua;" .. package.path' % EXT)
        wpath = weights or os.path.join(ROOT, "src", "lua", "ge", "extensions", "lumen", "nn_weights.json")
        w = json.load(open(wpath, encoding="utf-8"))

        def to_lua(o):
            if isinstance(o, list):
                return lua.table_from([to_lua(x) for x in o])
            if isinstance(o, dict):
                t = lua.table()
                for k, v in o.items():
                    t[k] = to_lua(v)
                return t
            return o

        lua.globals().WEIGHTS = to_lua(w)
        lua.execute(PRELUDE)
        self.g = lua.globals()
        for name, path in (preload or {}).items():
            fn = lua.eval("loadfile")(path.replace("\\", "/"))
            assert fn is not None, "cannot load " + path
            self.g.package.preload[name] = fn
        if file:
            fn = lua.eval("loadfile")(file.replace("\\", "/"))
            assert fn is not None, "cannot load " + file
            self.g.package.preload[module] = fn
        self.nn = lua.eval('require("%s")' % module)
        self.P = None

    # -- scenes --------------------------------------------------------------------------------------------
    def set_scene(self, sc):
        lua = self.lua
        self.P = lua.eval("mkpath")(sc.get("vl", 20.0), sc.get("len", 600.0))
        mk_pos = lua.eval("function(x, y) return {x = x, y = y, z = 0} end")
        mk_dir = lua.eval("function(x, y) return {x = x, y = y} end")
        sigs = lua.table()
        for i, (dist, action) in enumerate(sc.get("signs", []), start=1):
            t = lua.table()
            t.pos = mk_pos(0.0, dist + 2.6)            # nn.lua: dist = ss - (s + FRONT), FRONT = 2.6
            t.dir = mk_dir(0.0, 1.0)
            t.action = action
            t.sig = "sig%d" % i
            sigs[i] = t
        self.g.SIGS = sigs
        ags = lua.table()
        for i, a in enumerate(sc.get("agents", []), start=1):
            fx, fy = {"along": (0.0, 1.0), "across": (1.0, 0.0), "against": (0.0, -1.0)}[a.get("face", "along")]
            sp = float(a.get("speed", 0.0))           # negative: moving against the way it faces (reversing)
            t = lua.table()
            t.id = 1000 + i
            t.x, t.y = float(a["rd"]), float(a["rs"])
            t.fx, t.fy = fx, fy
            t.vx, t.vy = fx * sp, fy * sp
            t.speed = abs(sp)                           # the game's speed is a magnitude (drive.lua sense())
            t.hl, t.hw = a.get("hl", 2.3), a.get("hw", 0.95)
            t.acc = a.get("acc", 0.0)
            t.yaw = 0.0
            ags[i] = t
        self.g.AGENTS = ags
        self.g.VEHS[1] = lua.eval("mkveh")(float(sc.get("v", 0.0)))

    def new_car(self, sc, stop_at_end=False):
        """a fresh car and drive state for a scene"""
        self.set_scene(sc)
        car = self.lua.table()
        car.id = 1
        d = self.lua.table()
        d.path, d.mode, d.idx, d.stopAtEnd = self.P, "follow", int(BACK / 2) + 1, stop_at_end
        car.drv = d
        return car, d

    def debug(self, sc, car=None):
        """raw network outputs, the observation and the guard cap for a scene"""
        if car is None:
            car, _ = self.new_car(sc)
        else:
            self.set_scene(sc)
        return self.nn.debug(self.g.drive, car)

    def request(self, sc, car=None):
        """(speed the network asks for, the rules' cap, the observation as a list)"""
        r = self.debug(sc, car)
        vnet = min(sigmoid(r.out[1]) * 1.1 * r.vlim, 20.0)
        return vnet, r.vcap, [r.obs[i] for i in range(1, 92)], r

    def settle(self, sc, steps=30):
        """iterate the network with its own previous-target memory on a car that stays put (the game's loop for a
        standing car); returns (settled request, the rules' cap, first-step request)"""
        car, d = self.new_car(sc)
        first = None
        vn = cap = 0.0
        for _ in range(steps):
            r = self.nn.debug(self.g.drive, car)
            vn = min(sigmoid(r.out[1]) * 1.1 * r.vlim, 20.0)
            cap = r.vcap
            if first is None:
                first = vn
            d.nnPrevT = min(vn, cap)
            d.nnPrevD = math.tanh(r.out[2]) * 4
        return vn, cap, first

    def plan_steps(self, sc, seconds, dt=0.2, stop_at_end=False):
        """call nn.plan() every dt on a car that stays put; returns [(t, d)] snapshots of the fields that matter"""
        car, d = self.new_car(sc, stop_at_end)
        rows, t = [], 0.0
        while t < seconds - 1e-9:
            ok = self.nn.plan(self.g.drive, car, self.g.VEHS[1], d, t)
            assert ok, "nn.plan returned false"
            rows.append({"t": round(t, 3), "target": d.nnPrevT, "held": d.nnHeld or 0.0,
                         "relief": t < (d.nnReliefUntil or 0.0), "reliefs": int(d.nnReliefs or 0)})
            t += dt
        return rows, d

    def logs(self, contains=""):
        return [v for v in self.g.LOGS.values() if contains in v]


def break_policy(h):
    """test hook: output 1 of the loaded network becomes the constant -6 (it asks for ~0.05 m/s whatever it sees)"""
    W = h.nn.load()
    W.b3[1] = -6.0
    for i in range(1, 65):
        W.w3[(i - 1) * 2 + 1] = 0.0
