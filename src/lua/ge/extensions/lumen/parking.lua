-- Lumen parking (Frontier Studios).
-- Storm, 2026-11-02: "Also make the NN park in the parking spaces if there are ones. That is a requirement."
-- (the screenshot: a Lumen waiting for its rider in the aisle of a car park, across the painted spaces).
--
-- A Lumen that reaches a stop (a pickup, or the destination) looks for a free marked space close to it and
-- parks there instead of standing in the aisle. The spaces are BeamNG's own parking data: the level's
-- city.sites.json, loaded by gameplay_city (position, rotation, size; a car faces the space's +y by default).
-- The maneuver is the same whichever driver is selected, Lumen Drive or Lumen NN: a rule-based plan of two
-- circular arcs and straights (a Dubins path for the rear axle, which is the point the controller steers),
-- driven at walking pace by Lumen Drive's follower (drive.followCustom) and stopped by proximity in the middle of
-- the space. The NN plans lanes and a space is not a lane, so it is not asked.
--
-- Leaving is the entry path run backwards: the car backs along it until it is clear of the row and points the way
-- it came, then the route goes on from there (a plain straight reverse would leave the nose pointing at the row
-- of cars). The back-up is steered here, with the controller's reverse speed loop (drive.reverseCmd).
--
-- Every pose of the swept body is checked before a space is used: against the boxes of the vehicles standing
-- near (BeamNG's own parking system does not track cars parked in car parks, so map.objects would miss them),
-- and with rays for walls, posts, steps and drops.
--
-- Limits (2026-11-02): a space is used only where a clear path exists, so a parallel street space inside a
-- continuous row of parked cars is skipped; the car goes in nose first (it does not back in); spaces the level's
-- data does not have cannot be used; one maneuver at a time per car, no second attempt after a failed one.

local M = {}

local drive = require("lumen/drive")
local logTag = "lumen_parking"

local sqrt, abs, min, max, floor, ceil = math.sqrt, math.abs, math.min, math.max, math.floor, math.ceil
local sin, cos, atan2, atan, pi, deg = math.sin, math.cos, math.atan2, math.atan, math.pi, math.deg

M.enabled = true                       -- off switch for tests: M.setEnabled(false)

-- the search
local SEARCH_RADIUS = 38               -- m around the stop point
local MAX_PATH = 90                    -- m of driving to a space, at most
local MAX_CHECKS = 500                 -- paths checked against the standing vehicles per search (cheap, only maths)
local MAX_RAY_TESTS = 6                -- paths that also get the ray checks for walls and kerbs (the expensive part)
local SPACE_FAILS = 8                  -- checks that failed before a space (one direction of it) is dropped
local TEST_BUDGET = 0.09               -- s of CPU for all those checks together
local WALK_WEIGHT = 1.5                -- a metre the rider walks counts this much against a metre driven
local MAX_ARC = 4.7                    -- rad: no loop-the-loop
local RADII = {5.8, 7.5, 10.5}         -- m, rear axle (the Lumen's tightest turn is 5.2)
local FINALS = {5.0, 3.5, 2.2}         -- m of straight into the space after the last arc
local LEADS = {0, 3, 6}                -- m driven straight before the first arc (to swing wide)
local MIN_WID, MIN_LEN = 2.2, 5.2      -- smallest space used (the body is 1.9 by 4.8)

-- the Lumen (lumen.jbeam / drive.lua): the reference node is the car's position; the rear axle is 1.18 m behind
-- it, the front bumper 2.6 m ahead of it, the rear bumper 2.2 m behind it
local REAR_AXLE, FRONT, REAR, HALF_W = 1.18, 2.6, 2.2, 0.95
local BODY_AHEAD = (FRONT - REAR) / 2      -- 0.2: the middle of the body is this far ahead of the reference node
local BODY_HALF = (FRONT + REAR) / 2       -- 2.4
local WHEELBASE, MAX_WHEEL = 3.0, math.rad(30)

-- driving
local STEP = 0.8                       -- m between path points
local SPEED, SPEED_SLOW = 2.6, 1.4     -- m/s on the way, and on the last straight
local TAIL = 3.5                       -- the path runs this far past the goal; the car is stopped by proximity
local CHECK_STEP = 1.2                 -- m between the poses checked along a path
local CAR_MARGIN, WALL_MARGIN = 0.2, 0.25
local START_SKIP = 2.5                 -- m at the start of a path where standing neighbours are not checked
local STOP_AHEAD = 0.05                -- m: stop when the reference node is this close to the goal (plus the speed)
local STILL_LIMIT = 12                 -- s standing still on the way in before giving up (40 while waiting for a person)
local INTO_LIMIT = 150                 -- s for the whole way in

-- leaving
local EXIT_BACK = 7.5                  -- m backed at least (the space is 6 m deep)
local EXIT_ANGLE = math.rad(25)        -- and until the car points within this of the way it came in
local REV_SPEED, REV_LOOK = 1.5, 2.6   -- m/s backing, m of look-ahead
local LEAVE_WAIT = 45                  -- s waiting for something behind to move before giving up
local RELEASE = "input.event('parkingbrake', 0, 2, nil, nil, nil, 'ai') input.event('brake', 0, 2, nil, nil, nil, 'ai') electrics.set_warn_signal(false) "

local reserved = {}                    -- [space key] = the car going to or sitting in it
local stats = {tries = 0, parked = 0, noSpace = 0, aborted = 0, left = 0, leaveGaveUp = 0, spotsSeen = 0, checked = 0, blocked = 0}
local noDataLevel = nil

-- ---------------------------------------------------------------------------------
-- geometry
-- ---------------------------------------------------------------------------------
local function mod2pi(a)
  local r = a - 2 * pi * floor(a / (2 * pi))
  if r > 2 * pi - 1e-9 then r = 0 end
  return r
end

local function angDiff(a, b)               -- a - b wrapped to (-pi, pi]
  local d = (a - b) % (2 * pi)
  if d > pi then d = d - 2 * pi end
  return d
end

-- do two rectangles (centre, forward unit vector, half length, half width) overlap?
local function rectsOverlap(ax, ay, afx, afy, ahl, ahw, bx, by, bfx, bfy, bhl, bhw)
  local dx, dy = bx - ax, by - ay
  local anx, any = -afy, afx
  local bnx, bny = -bfy, bfx
  local function apart(nx, ny)
    local ra = ahl * abs(afx * nx + afy * ny) + ahw * abs(anx * nx + any * ny)
    local rb = bhl * abs(bfx * nx + bfy * ny) + bhw * abs(bnx * nx + bny * ny)
    return abs(dx * nx + dy * ny) > ra + rb
  end
  return not (apart(afx, afy) or apart(anx, any) or apart(bfx, bfy) or apart(bnx, bny))
end

-- The shortest of the four curve-straight-curve paths (LSL, RSR, LSR, RSL) between two poses for a turning
-- radius R. t, p and q are the first arc, the straight and the second arc in units of R. nil when none exists.
-- (checked offline against 20000 random poses: the sampled end is within 1 mm and 0.001 degrees)
local function dubins(x0, y0, th0, x1, y1, th1, R)
  local dx, dy = x1 - x0, y1 - y0
  local d = sqrt(dx * dx + dy * dy) / R
  local theta = atan2(dy, dx)
  local alpha, beta = mod2pi(th0 - theta), mod2pi(th1 - theta)
  local sa, sb, ca, cb = sin(alpha), sin(beta), cos(alpha), cos(beta)
  local cab = cos(alpha - beta)
  local bl, bk, bt, bp, bq
  local p2 = 2 + d * d - 2 * cab + 2 * d * (sa - sb)
  if p2 >= 0 then
    local tmp = atan2(cb - ca, d + sa - sb)
    local t, p, q = mod2pi(-alpha + tmp), sqrt(p2), mod2pi(beta - tmp)
    bl, bk, bt, bp, bq = t + p + q, "LSL", t, p, q
  end
  p2 = 2 + d * d - 2 * cab + 2 * d * (sb - sa)
  if p2 >= 0 then
    local tmp = atan2(ca - cb, d - sa + sb)
    local t, p, q = mod2pi(alpha - tmp), sqrt(p2), mod2pi(-beta + tmp)
    if not bl or t + p + q < bl then bl, bk, bt, bp, bq = t + p + q, "RSR", t, p, q end
  end
  p2 = -2 + d * d + 2 * cab + 2 * d * (sa + sb)
  if p2 >= 0 then
    local p = sqrt(p2)
    local tmp = atan2(-ca - cb, d + sa + sb) - atan2(-2.0, p)
    local t, q = mod2pi(-alpha + tmp), mod2pi(-mod2pi(beta) + tmp)
    if not bl or t + p + q < bl then bl, bk, bt, bp, bq = t + p + q, "LSR", t, p, q end
  end
  p2 = -2 + d * d + 2 * cab - 2 * d * (sa + sb)
  if p2 >= 0 then
    local p = sqrt(p2)
    local tmp = atan2(ca + cb, d - sa - sb) - atan2(2.0, p)
    local t, q = mod2pi(alpha - tmp), mod2pi(beta - tmp)
    if not bl or t + p + q < bl then bl, bk, bt, bp, bq = t + p + q, "RSL", t, p, q end
  end
  if not bl then return nil end
  return {kind = bk, t = bt, p = bp, q = bq, length = bl * R, R = R}
end

-- points along an arc of `ang` radians from (x, y, th), turning "L" or "R"; returns the end pose
local function arcPoints(out, x, y, th, turn, ang, R, step)
  local n = max(1, ceil(ang * R / step))
  local sgn = turn == "L" and 1 or -1
  local cx, cy = x - sgn * R * sin(th), y + sgn * R * cos(th)
  local px, py = x, y
  for i = 1, n do
    local t2 = th + sgn * ang * i / n
    px, py = cx + sgn * R * sin(t2), cy - sgn * R * cos(t2)
    out[#out + 1] = {px, py}
  end
  return px, py, th + sgn * ang
end

-- the rear-axle line of a Dubins path from (x0, y0, th0), appended to out (the first point is not added)
local function dubinsPoints(out, x0, y0, th0, D, step)
  local R = D.R
  local px, py, th = arcPoints(out, x0, y0, th0, D.kind:sub(1, 1), D.t, R, step)
  local len = D.p * R
  local n = max(1, ceil(len / step))
  local sx, sy = px, py
  for i = 1, n do out[#out + 1] = {sx + cos(th) * len * i / n, sy + sin(th) * len * i / n} end
  px, py = sx + cos(th) * len, sy + sin(th) * len
  return arcPoints(out, px, py, th, D.kind:sub(3, 3), D.q, R, step)
end

-- position, direction and segment of a polyline {x, y, s, n, len} at arc length s
local function polyAt(rv, s, hint)
  local n = rv.n
  s = clamp(s, 0, rv.len)
  local i = clamp(hint or 1, 1, n - 1)
  while i < n - 1 and rv.s[i + 1] < s do i = i + 1 end
  while i > 1 and rv.s[i] > s do i = i - 1 end
  local ds = rv.s[i + 1] - rv.s[i]
  local t = ds > 1e-6 and clamp((s - rv.s[i]) / ds, 0, 1) or 0
  local dx, dy = rv.x[i + 1] - rv.x[i], rv.y[i + 1] - rv.y[i]
  local l = sqrt(dx * dx + dy * dy)
  if l < 1e-6 then dx, dy, l = 1, 0, 1 end
  return rv.x[i] + dx * t, rv.y[i] + dy * t, dx / l, dy / l, i
end

-- nearest point of the polyline to (px, py), searching forward from segment `hint`: segment, fraction, distance
local function projectPoly(rv, px, py, hint)
  local best, bd, bt = hint, 1e18, 0
  for i = max(1, hint - 2), min(rv.n - 1, hint + 30) do
    local ax, ay = rv.x[i], rv.y[i]
    local dx, dy = rv.x[i + 1] - ax, rv.y[i + 1] - ay
    local l2 = dx * dx + dy * dy
    local t = 0
    if l2 > 1e-8 then t = clamp(((px - ax) * dx + (py - ay) * dy) / l2, 0, 1) end
    local d2 = (px - (ax + dx * t)) ^ 2 + (py - (ay + dy * t)) ^ 2
    if d2 < bd then best, bd, bt = i, d2, t end
  end
  return best, bt, sqrt(bd)
end

local function polyFromPoints(pts, first)
  local x, y, s = {}, {}, {}
  local acc, lx, ly = 0, nil, nil
  for i = first or 1, #pts do
    local px, py = pts[i][1], pts[i][2]
    if lx then acc = acc + sqrt((px - lx) ^ 2 + (py - ly) ^ 2) end
    x[#x + 1], y[#y + 1], s[#s + 1] = px, py, acc
    lx, ly = px, py
  end
  return {x = x, y = y, s = s, n = #x, len = acc}
end

-- ---------------------------------------------------------------------------------
-- what is around: the level's parking data, the vehicles standing near, rays for the static world
-- ---------------------------------------------------------------------------------
local function getSites()
  local ext = extensions
  local gc = ext and ext.gameplay_city
  if not gc and ext and ext.load then
    pcall(ext.load, "gameplay_city")
    gc = ext.gameplay_city
  end
  if not gc then return nil end
  local s = gc.getSites and gc.getSites()
  if not s and gc.loadSites then
    pcall(gc.loadSites)
    s = gc.getSites and gc.getSites()
  end
  return s
end

-- one space as plain numbers, or nil and why it is not used
local function spotShape(ps, carZ)
  local tags = ps.customFields and ps.customFields.tags
  if tags and tags.banned then return nil, "banned" end
  local scl = ps.scl
  if not scl or scl.x < MIN_WID or scl.y < MIN_LEN or scl.x > 5.5 or scl.y > 9.5 then return nil, "size" end
  if carZ and abs(ps.pos.z - carZ) > 2.5 then return nil, "level" end
  local y = ps.rot * vec3(0, 1, 0)
  local l = sqrt(y.x * y.x + y.y * y.y)
  if l < 0.5 then return nil, "tilted" end
  return {key = ps.id or ps.name, name = tostring(ps.name or ps.id), cx = ps.pos.x, cy = ps.pos.y, z = ps.pos.z,
          ux = y.x / l, uy = y.y / l, wid = scl.x, len = scl.y, street = (tags and tags.street) and true or false}
end

-- footprints of the vehicles standing within `radius` of (cx, cy), other than the car itself: centre, forward,
-- half length, half width. Moving vehicles and people on foot are not obstacles to plan around.
local function vehicleBoxes(selfId, cx, cy, radius)
  local out = {}
  for _, v in ipairs(getAllVehicles()) do
    local id = v:getID()
    if id ~= selfId and v:getActive() then
      local x, y = be:getObjectOOBBCenterXYZ(id)
      if x and (x - cx) ^ 2 + (y - cy) ^ 2 < radius * radius and v:getVelocity():length() < 2.5 and v:getJBeamFilename() ~= "unicycle" then
        local a0x, a0y = be:getObjectOOBBHalfAxisXYZ(id, 0)
        local a1x, a1y = be:getObjectOOBBHalfAxisXYZ(id, 1)
        local a2x, a2y = be:getObjectOOBBHalfAxisXYZ(id, 2)
        local axes = {{a0x, a0y}, {a1x, a1y}, {a2x, a2y}}      -- the third of these is the height: it lies flat least
        table.sort(axes, function(p, q) return p[1] * p[1] + p[2] * p[2] > q[1] * q[1] + q[2] * q[2] end)
        local l1 = sqrt(axes[1][1] ^ 2 + axes[1][2] ^ 2)
        local l2 = sqrt(axes[2][1] ^ 2 + axes[2][2] ^ 2)
        if l1 > 0.05 then
          local hw = max(l2, 0.1)
          out[#out + 1] = {x = x, y = y, fx = axes[1][1] / l1, fy = axes[1][2] / l1, hl = l1, hw = hw, r = sqrt(l1 * l1 + hw * hw), id = id}
        end
      end
    end
  end
  return out
end

local function spotFree(spot, boxes)
  for _, b in ipairs(boxes) do
    if rectsOverlap(spot.cx, spot.cy, spot.ux, spot.uy, spot.len / 2 - 0.3, spot.wid / 2 - 0.2, b.x, b.y, b.fx, b.fy, b.hl, b.hw) then return false end
  end
  return true
end

local function groundAt(x, y, zhint)
  local h = castRayStatic(vec3(x, y, zhint + 1.5), vec3(0, 0, -1), 3.5)
  if h >= 3.5 then return nil end
  return zhint + 1.5 - h
end

local RAY_ALONG = {-1, -0.5, 0, 0.5, 1}
-- is the body at (bx, by) heading (hx, hy) on level ground with nothing solid within WALL_MARGIN? ok, ground height, why not
local function staticClear(bx, by, hx, hy, zhint, rays)
  local gz = groundAt(bx, by, zhint)
  if not gz then return false, nil, "drop" end
  local nx, ny = -hy, hx
  for _, a in ipairs({-1, 1}) do
    for _, b in ipairs({-1, 1}) do
      local g = groundAt(bx + hx * a * BODY_HALF + nx * b * HALF_W, by + hy * a * BODY_HALF + ny * b * HALF_W, gz)
      if not g or abs(g - gz) > 0.30 then return false, gz, "step" end
    end
  end
  if rays then
    local hl, hw = BODY_HALF + WALL_MARGIN, HALF_W + WALL_MARGIN
    local o = vec3(bx, by, gz + 0.55)
    for _, b in ipairs({-1, 1}) do
      for _, a in ipairs(RAY_ALONG) do
        local tx, ty = hx * a * hl + nx * b * hw, hy * a * hl + ny * b * hw
        local len = sqrt(tx * tx + ty * ty)
        if castRayStatic(o, vec3(tx / len, ty / len, 0), len) < len then return false, gz, "wall" end
      end
    end
    for _, a in ipairs({-1, 1}) do
      if castRayStatic(o, vec3(hx * a, hy * a, 0), hl) < hl then return false, gz, "wall" end
    end
  end
  return true, gz
end

-- The poses of a rear-axle path every CHECK_STEP m: distance along the path, direction, middle of the body.
local function poseList(pts)
  local out = {}
  local n = #pts
  local s, nextS = 0, 0
  local lx, ly = pts[1][1], pts[1][2]
  for i = 1, n do
    local x, y = pts[i][1], pts[i][2]
    if i > 1 then s = s + sqrt((x - lx) ^ 2 + (y - ly) ^ 2) end
    lx, ly = x, y
    if s >= nextS or i == n then
      nextS = s + CHECK_STEP
      local a, b = pts[min(i + 1, n)], pts[max(i - 1, 1)]
      local hx, hy
      if i < n then hx, hy = a[1] - x, a[2] - y else hx, hy = x - b[1], y - b[2] end
      local l = sqrt(hx * hx + hy * hy)
      if l < 1e-6 then hx, hy, l = 1, 0, 1 end
      hx, hy = hx / l, hy / l
      out[#out + 1] = {s = s, hx = hx, hy = hy, i = i, bx = x + hx * (REAR_AXLE + BODY_AHEAD), by = y + hy * (REAR_AXLE + BODY_AHEAD)}
    end
  end
  return out, n
end

local CAR_REACH = sqrt((BODY_HALF + CAR_MARGIN) ^ 2 + (HALF_W + CAR_MARGIN) ^ 2)

-- Vehicles only: is the swept body clear of the standing vehicles' boxes? Cheap, so hundreds of paths can be
-- tried. Returns true, or false, "car" and the distance along the path of the pose that hits.
local function carsClear(poses, boxes)
  for _, p in ipairs(poses) do
    if p.s >= START_SKIP then
      for _, bb in ipairs(boxes) do
        local reach = CAR_REACH + bb.r
        if (bb.x - p.bx) ^ 2 + (bb.y - p.by) ^ 2 < reach * reach
           and rectsOverlap(p.bx, p.by, p.hx, p.hy, BODY_HALF + CAR_MARGIN, HALF_W + CAR_MARGIN, bb.x, bb.y, bb.fx, bb.fy, bb.hl, bb.hw) then
          return false, "car", p.s
        end
      end
    end
  end
  return true
end

-- The static world: level ground, no step or drop, rays for walls and posts. The rays are the expensive part.
-- Returns true, or false, why ("wall", "step", "drop") and the distance along the path of the pose.
local function groundClear(poses, n, z0, z1)
  local gzPrev
  for _, p in ipairs(poses) do
    local ok, gz, why = staticClear(p.bx, p.by, p.hx, p.hy, gzPrev or (z0 + (z1 - z0) * (p.i - 1) / max(1, n - 1)), p.s >= START_SKIP)
    if not ok then return false, why, p.s end
    if gzPrev and abs(gz - gzPrev) > 0.12 then return false, "step", p.s end
    gzPrev = gz
  end
  return true
end

-- ---------------------------------------------------------------------------------
-- the search
-- ---------------------------------------------------------------------------------
local function logLine(text) log("I", logTag, "parking: " .. text) end

local function sayTo(st, text, kind)
  if st and st.say then pcall(st.say, text, kind or "go") end
end

local function release(car)
  local st = car.park
  if st and st.key and reserved[st.key] == car then reserved[st.key] = nil end
end

function M.forget(car)
  release(car)
  car.park = nil
end

-- the fleet is gone (a new level, the service stopped): no space is held for anyone
function M.reset()
  reserved = {}
end

-- the rear-axle path of one candidate: lead-in, Dubins path, straight into the space, tail past the goal
local function entryPoints(c, sx, sy, th0)
  local pts = {{sx, sy}}
  local x0, y0 = sx, sy
  if c.lead > 0 then
    local n = max(1, ceil(c.lead / STEP))
    for i = 1, n do pts[#pts + 1] = {sx + cos(th0) * c.lead * i / n, sy + sin(th0) * c.lead * i / n} end
    x0, y0 = sx + cos(th0) * c.lead, sy + sin(th0) * c.lead
  end
  local px, py = dubinsPoints(pts, x0, y0, th0, c.D, STEP)
  local n = max(1, ceil(c.sf / STEP))
  for i = 1, n do pts[#pts + 1] = {px + c.ux * c.sf * i / n, py + c.uy * c.sf * i / n} end
  local iG = #pts
  local gx, gy = px + c.ux * c.sf, py + c.uy * c.sf
  n = max(1, ceil(TAIL / STEP))
  for i = 1, n do pts[#pts + 1] = {gx + c.ux * TAIL * i / n, gy + c.uy * TAIL * i / n} end
  return pts, iG
end

local function startImpl(car, veh, aim, opts)
  opts = opts or {}
  if not M.enabled then return false, "parking is switched off" end
  if car.park then return false, "already parking" end
  local sites = getSites()
  local level = getCurrentLevelIdentifier and getCurrentLevelIdentifier() or "?"
  if not sites or not sites.getRadialParkingSpots then
    if noDataLevel ~= level then
      noDataLevel = level
      logLine(string.format("level %s has no parking data (no city.sites.json), so cars stop where they are", tostring(level)))
    end
    return false, "no parking data on this level"
  end
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  local hl = sqrt(dir.x * dir.x + dir.y * dir.y)
  if hl < 0.3 then return false, "the car is not level" end
  local hx, hy = dir.x / hl, dir.y / hl
  local th0 = atan2(hy, hx)
  local sx, sy = pos.x - hx * REAR_AXLE, pos.y - hy * REAR_AXLE          -- the rear axle now
  aim = aim or pos
  local what = opts.what or "stop"
  local t0 = os.clock()
  stats.tries = stats.tries + 1

  local near = sites:getRadialParkingSpots(aim, 0, SEARCH_RADIUS)
  stats.spotsSeen = stats.spotsSeen + #near
  local boxes = vehicleBoxes(veh:getID(), aim.x, aim.y, SEARCH_RADIUS + 40)
  local skipped = {}
  local usable = 0
  local cands = {}
  for _, e in ipairs(near) do
    local spot, why = spotShape(e.ps, pos.z)
    if spot and reserved[spot.key] and reserved[spot.key] ~= car and reserved[spot.key].park and reserved[spot.key].park.key == spot.key then
      spot, why = nil, "reserved"
    end
    if spot and not spotFree(spot, boxes) then spot, why = nil, "occupied" end
    if not spot then
      skipped[why] = (skipped[why] or 0) + 1
    else
      usable = usable + 1
      local walk = sqrt((spot.cx - aim.x) ^ 2 + (spot.cy - aim.y) ^ 2)
      for _, sigma in ipairs({1, -1}) do
        local ux, uy = spot.ux * sigma, spot.uy * sigma
        local thu = atan2(uy, ux)
        local gx, gy = spot.cx - ux * BODY_AHEAD, spot.cy - uy * BODY_AHEAD          -- the reference node at the goal
        local grx, gry = gx - ux * REAR_AXLE, gy - uy * REAR_AXLE                    -- and the rear axle
        for _, sf in ipairs(FINALS) do
          local erx, ery = grx - ux * sf, gry - uy * sf
          for _, lead in ipairs(LEADS) do
            local x0, y0 = sx + hx * lead, sy + hy * lead
            for _, R in ipairs(RADII) do
              local D = dubins(x0, y0, th0, erx, ery, thu, R)
              if D and D.t <= MAX_ARC and D.q <= MAX_ARC then
                local total = lead + D.length + sf
                if total <= MAX_PATH then
                  cands[#cands + 1] = {spot = spot, sigma = sigma, ux = ux, uy = uy, gx = gx, gy = gy, sf = sf, lead = lead, D = D,
                                       total = total, walk = walk, score = total + WALK_WEIGHT * walk}
                end
              end
            end
          end
        end
      end
    end
  end
  table.sort(cands, function(a, b) return a.score < b.score end)

  -- Best score first. Every path gets the cheap check against the standing vehicles; only those that pass it get
  -- the rays (walls, kerbs). A space that fails SPACE_FAILS times, or once in the last stretch (the straight into
  -- the space, which every path to it shares), is dropped.
  local tested, rayTests, blockedBy = 0, 0, {}
  local tTest = os.clock()
  local chosen, chosenPts, chosenG, rank
  local failed = {}
  for k, c in ipairs(cands) do
    if tested >= MAX_CHECKS or rayTests >= MAX_RAY_TESTS or os.clock() - tTest > TEST_BUDGET then break end
    local fk = tostring(c.spot.key) .. ":" .. c.sigma
    if (failed[fk] or 0) < SPACE_FAILS then
      tested = tested + 1
      local pts, iG = entryPoints(c, sx, sy, th0)
      local poses, np = poseList(pts)
      local ok, why, failS = carsClear(poses, boxes)
      if ok then
        rayTests = rayTests + 1
        ok, why, failS = groundClear(poses, np, pos.z, c.spot.z)
      end
      if ok then
        chosen, chosenPts, chosenG, rank = c, pts, iG, k
        break
      end
      blockedBy[why] = (blockedBy[why] or 0) + 1
      failed[fk] = (failS and failS >= c.total - c.sf - 1.0) and SPACE_FAILS or (failed[fk] or 0) + 1
    end
  end
  stats.checked = stats.checked + tested
  local function list(t)
    local out = {}
    for k, v in pairs(t) do out[#out + 1] = k .. " " .. v end
    table.sort(out)
    return #out > 0 and table.concat(out, ", ") or "none"
  end
  if not chosen then
    stats.noSpace = stats.noSpace + 1
    stats.blocked = stats.blocked + tested
    local msg = string.format("car %s (%s): no space used: %d marked spaces within %d m of the stop, %d free (not used: %s), %d paths planned, %d checked and blocked (%s; %d with rays), %.0f ms",
      tostring(car.slot), what, #near, SEARCH_RADIUS, usable, list(skipped), #cands, tested, list(blockedBy), rayTests, (os.clock() - t0) * 1000)
    logLine(msg)
    return false, "no free space with a clear path"
  end

  local z0, z1 = pos.z, chosen.spot.z
  local custom = {}
  for i, p in ipairs(chosenPts) do custom[i] = {p[1], p[2], z0 + (z1 - z0) * (i - 1) / max(1, #chosenPts - 1)} end
  local okf, P = drive.followCustom(car, custom, {speed = SPEED, slowSpeed = SPEED_SLOW, slowLen = chosen.sf + TAIL + 0.5})
  if not okf then return false, "could not start the path" end
  if car.drv then car.drv.s, car.drv.lat = 0, 0 end          -- (the old path's progress, until the first plan)

  local sp = chosen.spot
  local st = {phase = "into", key = sp.key, name = sp.name, spot = sp, ux = chosen.ux, uy = chosen.uy, gx = chosen.gx, gy = chosen.gy,
              sf = chosen.sf, pts = chosenPts, iG = chosenG, th0 = th0, P = P, t = 0, stillT = 0, what = what, say = opts.say,
              startX = pos.x, startY = pos.y, kind = chosen.D.kind, R = chosen.D.R, lead = chosen.lead, total = chosen.total,
              walk = chosen.walk, nn = opts.nn}
  car.park = st
  if sp.key then reserved[sp.key] = car end
  logLine(string.format("car %s (%s): space \"%s\" is %.1f m from the stop, going in nose first; path %.1f m (%s, radius %.1f m, lead-in %.0f m, straight %.1f m); candidate %d of %d, %d checked, %.0f ms",
    tostring(car.slot), what, sp.name, chosen.walk, chosen.total, chosen.D.kind, chosen.D.R, chosen.lead, chosen.sf, rank, #cands, tested, (os.clock() - t0) * 1000))
  sayTo(st, string.format("A free space %d m from the %s. Pulling in nose first at walking pace.%s", floor(chosen.walk + 0.5), what,
    opts.nn and " Lumen NN does not drive this part: parking is a rule-based maneuver, so Lumen Drive does it." or ""), "go")
  return true, string.format("space %s, %.0f m", sp.name, chosen.total)
end

-- ---------------------------------------------------------------------------------
-- going in, and the measurement of where the car ended up
-- ---------------------------------------------------------------------------------
local function measure(veh, st)
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  local l = sqrt(dir.x * dir.x + dir.y * dir.y)
  if l < 1e-3 then l = 1 end
  local hx, hy = dir.x / l, dir.y / l
  local sp = st.spot
  local bx, by = pos.x + hx * BODY_AHEAD, pos.y + hy * BODY_AHEAD
  local along = (bx - sp.cx) * st.ux + (by - sp.cy) * st.uy
  local side = -(bx - sp.cx) * st.uy + (by - sp.cy) * st.ux
  local ang = deg(atan2(st.ux * hy - st.uy * hx, st.ux * hx + st.uy * hy))
  -- all four corners of the body inside the marked rectangle?
  local inside = true
  local nx, ny = -hy, hx
  for _, a in ipairs({-1, 1}) do
    for _, b in ipairs({-1, 1}) do
      local cx, cy = bx + hx * a * BODY_HALF + nx * b * HALF_W, by + hy * a * BODY_HALF + ny * b * HALF_W
      local ca = (cx - sp.cx) * st.ux + (cy - sp.cy) * st.uy
      local cs = -(cx - sp.cx) * st.uy + (cy - sp.cy) * st.ux
      if abs(ca) > sp.len / 2 or abs(cs) > sp.wid / 2 then inside = false end
    end
  end
  return {along = along, side = side, ang = ang, inside = inside, fromStop = sqrt((pos.x - st.startX) ^ 2 + (pos.y - st.startY) ^ 2)}
end

local function giveUp(car, veh, st, why)
  local gd = -1
  if veh then
    local p = veh:getPosition()
    gd = sqrt((p.x - st.gx) ^ 2 + (p.y - st.gy) ^ 2)
  end
  logLine(string.format("car %s (%s): gave up on space \"%s\" (%s) after %.0f s, %.1f m from the goal; the car stays where it is",
    tostring(car.slot), st.what, st.name, why, st.t, gd))
  sayTo(st, "I could not get into the space (" .. why .. "). Stopping here instead.", "warn")
  drive.stop(car)
  M.forget(car)
  stats.aborted = stats.aborted + 1
  return "failed"
end

local function stepInto(car, veh, st, dt, now)
  local d = car.drv
  if not d or not d.custom or d.path ~= st.P or d.mode ~= "follow" then
    return giveUp(car, veh, st, "another plan took over the car")
  end
  d.noSolids = true                            -- the path was checked; walls and neighbours within centimetres are expected
  local pos = veh:getPosition()
  local speed = veh:getVelocity():length()
  local remain = (st.gx - pos.x) * st.ux + (st.gy - pos.y) * st.uy
  local lateral = abs(-(pos.x - st.gx) * st.uy + (pos.y - st.gy) * st.ux)
  if speed < 0.25 then st.stillT = st.stillT + dt else st.stillT = 0 end
  if st.stillT > ((d.state == "wait") and 40 or STILL_LIMIT) then return giveUp(car, veh, st, "standing still") end
  if st.t > INTO_LIMIT then return giveUp(car, veh, st, "too slow") end
  -- on the last straight (not a loop that merely passes the goal on the way) and at the goal, a little early
  -- because the car still rolls: hold
  local onLast = (d.s or 0) >= st.P.len - TAIL - st.sf - 2.0
  if onLast and remain <= st.sf + 1.5 and lateral < 1.0 and remain <= STOP_AHEAD + 0.3 * speed then
    st.phase, st.settleT = "settle", 0
    drive.stop(car)
  end
  return nil
end

local function stepSettle(car, veh, st, dt, now)
  st.settleT = st.settleT + dt
  if (veh:getVelocity():length() < 0.12 and st.settleT > 0.4) or st.settleT > 3 then
    local m = measure(veh, st)
    st.phase, st.final = "parked", m
    stats.parked = stats.parked + 1
    logLine(string.format("car %s (%s): parked in \"%s\": middle of the body %+.2f m along and %+.2f m across the middle of the space, %+.1f deg off its axis, body inside the marked rectangle: %s; %.1f m from the stop point, %.0f s after starting",
      tostring(car.slot), st.what, st.name, m.along, m.side, m.ang, m.inside and "yes" or "NO", m.fromStop, st.t))
    sayTo(st, string.format("Parked in the space, %s.", m.inside and "inside the lines" or "close to the lines"), "go")
    return "parked"
  end
  return nil
end

-- ---------------------------------------------------------------------------------
-- leaving: back along the way in
-- ---------------------------------------------------------------------------------
local function buildReverse(st, veh)
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  local l = sqrt(dir.x * dir.x + dir.y * dir.y)
  if l < 0.3 then return nil, "the car is not level" end
  local hx, hy = dir.x / l, dir.y / l
  local rx, ry = pos.x - hx * REAR_AXLE, pos.y - hy * REAR_AXLE
  local pts = st.pts
  local iNow, bd = st.iG, 1e18
  for i = max(2, st.iG - 10), #pts do
    local d2 = (pts[i][1] - rx) ^ 2 + (pts[i][2] - ry) ^ 2
    if d2 < bd then iNow, bd = i, d2 end
  end
  if bd > 9 then return nil, string.format("the car is %.1f m from where it parked", sqrt(bd)) end
  local list = {{rx, ry}}
  local back, lastX, lastY = 0, rx, ry
  for i = iNow - 1, 1, -1 do
    local px, py = pts[i][1], pts[i][2]
    back = back + sqrt((px - lastX) ^ 2 + (py - lastY) ^ 2)
    lastX, lastY = px, py
    list[#list + 1] = {px, py}
    -- the direction the car had here on the way in
    local fx, fy = pts[i + 1][1] - px, pts[i + 1][2] - py
    if back >= EXIT_BACK and abs(angDiff(atan2(fy, fx), st.th0)) < EXIT_ANGLE then break end
  end
  if #list < 3 then return nil, "no way back" end
  return polyFromPoints(list)
end

-- is anything in the way of the next few metres of backing? other vehicles' boxes, and people on foot
local function reverseBlocked(car, veh, st, sNow, now)
  if not st.boxes or now - st.boxesT > 0.5 then
    local pos = veh:getPosition()
    st.boxes, st.boxesT = vehicleBoxes(veh:getID(), pos.x, pos.y, 25), now
  end
  local rv = st.rv
  local walkers = drive._walkers and drive._walkers(now) or {}
  for k = 0, 3 do
    local x, y, tx, ty = polyAt(rv, sNow + k * 1.5, st.rvI)
    local hx, hy = -tx, -ty                                  -- backing, so the car faces the way the path came from
    local bx, by = x + hx * (REAR_AXLE + BODY_AHEAD), y + hy * (REAR_AXLE + BODY_AHEAD)
    if k > 0 then
      for _, bb in ipairs(st.boxes) do
        if rectsOverlap(bx, by, hx, hy, BODY_HALF + 0.12, HALF_W + 0.12, bb.x, bb.y, bb.fx, bb.fy, bb.hl, bb.hw) then return true, "a vehicle" end
      end
    end
    for _, p in ipairs(walkers) do
      if (p.x - bx) ^ 2 + (p.y - by) ^ 2 < 3.2 * 3.2 then return true, "a person" end
    end
  end
  return false
end

local function leaveImpl(car, veh)
  local st = car.park
  if not st then return false end
  if st.phase == "leaving" then return true end
  if st.phase ~= "parked" then return false end
  local rv, why = buildReverse(st, veh)
  if not rv then
    logLine(string.format("car %s: cannot back out of \"%s\": %s", tostring(car.slot), st.name, tostring(why)))
    M.forget(car)
    return false
  end
  st.phase, st.rv, st.rvI, st.stage = "leaving", rv, 1, "wait"
  st.leaveT, st.waitT, st.checkT, st.clearN, st.backT, st.stuckT, st.vcmd, st.cmdT, st.backDist = 0, 0, 0, 0, 0, 0, 0, 0, 0
  st.boxes = nil
  logLine(string.format("car %s: leaving \"%s\": backing %.1f m along the way it came in", tostring(car.slot), st.name, rv.len))
  sayTo(st, "Backing out of the space along the way I came in.", "go")
  return true
end

-- Back out of the space: true when the car is leaving (or already was), false when it is not parked in one or
-- cannot back out (the state is dropped, and the car drives on as it would have).
function M.leave(car, veh)
  local ok, a = pcall(leaveImpl, car, veh)
  if ok then return a end
  log("E", logTag, "parking: leave failed: " .. tostring(a))
  M.forget(car)
  return false
end

local function endLeave(car, veh, st, why)
  local hdg = ""
  if veh then
    local dir = veh:getDirectionVector()
    hdg = string.format(", %.0f deg from the way it came in", abs(deg(angDiff(atan2(dir.y, dir.x), st.th0))))
  end
  logLine(string.format("car %s: out of \"%s\" after %.0f s and %.1f m%s%s", tostring(car.slot), st.name, st.leaveT, st.backDist, hdg,
    why and (" (" .. why .. ")") or ""))
  if why then stats.leaveGaveUp = stats.leaveGaveUp + 1 end
  sayTo(st, why and ("Out of the space (" .. why .. "). Carrying on.") or "Out of the space. Carrying on.", "go")
  M.forget(car)
  stats.left = stats.left + 1
  return "left"
end

-- Pure pursuit for a car backing along a rear-axle line (rear axle at rx, ry, the car facing hx, hy): the steering
-- for the controller, -1..1, right positive. The direction of travel is the opposite of the way the car faces,
-- and backing turns the wheels the other way round from driving forward (the rear goes where the wheel points).
local function reverseSteer(rv, sNow, seg, rx, ry, hx, hy)
  local tx, ty = polyAt(rv, sNow + REV_LOOK, seg)
  local lx, ly = tx - rx, ty - ry
  local dx, dy = -hx, -hy
  local lat = lx * dy - ly * dx                             -- right of the direction of travel positive
  local lon = lx * dx + ly * dy
  local ld = max(1.0, sqrt(lx * lx + ly * ly))
  local wheel = -atan(2 * sin(atan2(lat, lon)) * WHEELBASE / ld)
  return clamp(wheel / MAX_WHEEL, -1, 1)
end

local function stepLeaving(car, veh, st, dt, now)
  local d = car.drv
  st.leaveT = st.leaveT + dt
  if st.stage ~= "wait" and d and d.mode ~= "off" then
    return endLeave(car, veh, st, "another plan took over the car")
  end

  if st.stage == "wait" then
    st.waitT, st.checkT = st.waitT + dt, st.checkT - dt
    if st.checkT <= 0 then
      st.checkT = 0.25
      local blocked, what = reverseBlocked(car, veh, st, 0, now)
      if blocked then
        if st.clearN >= 0 then logLine(string.format("car %s: waits to back out of \"%s\": %s is behind", tostring(car.slot), st.name, what)) end
        st.clearN = -1
      elseif st.clearN < 0 then
        st.clearN = 1
      else
        st.clearN = st.clearN + 1
      end
      if st.clearN >= 2 then
        drive.release(car)
        veh:queueLuaCommand(RELEASE)
        st.stage = "back"
        st.lastPos = veh:getPosition()
      end
    end
    if st.stage == "wait" and st.waitT > LEAVE_WAIT then return endLeave(car, veh, st, "something stayed behind the car") end
    return nil
  end

  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  local l = sqrt(dir.x * dir.x + dir.y * dir.y)
  if l < 0.3 then return endLeave(car, veh, st, "the car is tilted") end
  local hx, hy = dir.x / l, dir.y / l
  local speed = veh:getVelocity():length()

  if st.stage == "stop" then
    st.stopT = st.stopT + dt
    if speed < 0.15 or st.stopT > 1.8 then
      drive.reverseCmd(car, 0, 0)
      drive.stop(car)
      return endLeave(car, veh, st, st.why)
    end
    return nil
  end

  -- stage "back": pure pursuit for a reversing car on the rear-axle line
  local rv = st.rv
  local rx, ry = pos.x - hx * REAR_AXLE, pos.y - hy * REAR_AXLE
  local seg, frac, off = projectPoly(rv, rx, ry, st.rvI)
  st.rvI = seg
  local sNow = rv.s[seg] + (rv.s[seg + 1] - rv.s[seg]) * frac
  local remaining = rv.len - sNow
  st.backT = st.backT + dt
  if pos:distance(st.lastPos) > 0.12 then
    st.backDist = st.backDist + pos:distance(st.lastPos)
    st.lastPos, st.stuckT = pos, 0
  elseif st.vcmd > 0.4 then
    st.stuckT = st.stuckT + dt
  end

  local finish
  if remaining < 0.6 then finish = nil
  elseif off > 3.5 then finish = "left the line it came in on"
  elseif st.stuckT > 4 then finish = "stuck"
  elseif st.leaveT > 90 then finish = "too slow" end
  if remaining < 0.6 or finish then
    st.stage, st.stopT, st.why = "stop", 0, finish
    drive.reverseCmd(car, 0.05, 0)
    return nil
  end

  st.cmdT = st.cmdT - dt
  st.blockT = st.blockT or 0
  if st.cmdT <= 0 then
    st.cmdT = 0.05
    st.chkT = (st.chkT or 0) - 0.05
    if st.chkT <= 0 then
      st.chkT = 0.15
      st.blocked = reverseBlocked(car, veh, st, sNow, now)
    end
    local blocked = st.blocked
    local steer = reverseSteer(rv, sNow, seg, rx, ry, hx, hy)
    local v = min(REV_SPEED, 0.5 + 0.8 * st.backT)
    if remaining < 2.5 then v = max(0.5, v * remaining / 2.5) end
    if blocked then
      v = 0.05
      st.blockT = st.blockT + 0.05
      if st.blockT > LEAVE_WAIT then
        st.stage, st.stopT, st.why = "stop", 0, "something stayed in the way"
      end
    else
      st.blockT = max(0, st.blockT - 0.1)
    end
    st.vcmd = v
    drive.reverseCmd(car, v, steer)
  end
  return nil
end

-- ---------------------------------------------------------------------------------
-- public
-- ---------------------------------------------------------------------------------
-- A Lumen that has arrived at `aim` (a vec3: the pickup or the destination stop) and is standing: park it in a
-- free marked space near there. opts = {what = "pickup" | "drop-off", say = function(text, kind), nn = true when
-- Lumen NN is the selected driver}. Returns true when the maneuver has started, else false and why.
function M.start(car, veh, aim, opts)
  local ok, a, b = pcall(startImpl, car, veh, aim, opts)
  if ok then return a, b end
  log("E", logTag, "parking: start failed: " .. tostring(a))
  M.forget(car)
  return false, "error"
end

-- every frame for a fleet car; returns an event or nil: "parked" (stopped in the space), "failed" (gave up, the
-- car holds where it is), "left" (out of the space, or the back-out was given up)
function M.update(car, veh, dt, now)
  local st = car.park
  if not st then return nil end
  st.t = st.t + dt
  local function step()
    if st.phase == "into" then return stepInto(car, veh, st, dt, now) end
    if st.phase == "settle" then return stepSettle(car, veh, st, dt, now) end
    if st.phase == "leaving" then return stepLeaving(car, veh, st, dt, now) end
    return nil
  end
  local ok, ev = pcall(step)
  if ok then return ev end
  log("E", logTag, string.format("parking: car %s: update failed in phase %s: %s", tostring(car.slot), tostring(st.phase), tostring(ev)))
  pcall(function()
    drive.reverseCmd(car, 0, 0)
    drive.stop(car)
  end)
  local was = st.phase
  M.forget(car)
  stats.aborted = stats.aborted + 1
  return (was == "leaving") and "left" or "failed"
end

-- under the maneuver's control: going in, settling, or backing out
function M.busy(car)
  local st = car.park
  return st ~= nil and (st.phase == "into" or st.phase == "settle" or st.phase == "leaving")
end

function M.parked(car)
  local st = car.park
  return st ~= nil and st.phase == "parked"
end

-- parked, or backing out: the car must finish leaving before anything else drives it
function M.holds(car)
  local st = car.park
  return st ~= nil and (st.phase == "parked" or st.phase == "leaving")
end

function M.setEnabled(on)
  M.enabled = on and true or false
  logLine("parking is " .. (M.enabled and "on" or "off"))
  return M.enabled
end

function M.stats()
  local out = {}
  for k, v in pairs(stats) do out[k] = v end
  return out
end

-- for the test scripts: what the level's parking data holds near a point (are the spaces loaded in freeroam?)
function M.probe(pos, radius)
  local sites = getSites()
  if not sites or not sites.getRadialParkingSpots then return {loaded = false} end
  radius = radius or 40
  local near = sites:getRadialParkingSpots(pos, 0, radius)
  local boxes = vehicleBoxes(-1, pos.x, pos.y, radius + 40)
  local free, sample, byWhy, zRef = 0, {}, {}, nil
  for _, e in ipairs(near) do
    local spot, why = spotShape(e.ps, pos.z ~= 0 and pos.z or nil)         -- z = 0: any height
    if spot and not spotFree(spot, boxes) then spot, why = nil, "occupied" end
    if spot then
      free = free + 1
      zRef = zRef or spot.z
      if #sample < 5 then sample[#sample + 1] = string.format("%s (%.1f,%.1f)", spot.name, spot.cx, spot.cy) end
    else
      byWhy[why] = (byWhy[why] or 0) + 1
    end
  end
  return {loaded = true, total = sites.parkingSpots and #sites.parkingSpots.sorted or 0, near = #near, free = free, z = zRef, sample = sample, notUsed = byWhy}
end

-- In-game check of the maths (the Lua copy of what the offline Python mirrors verified): random Dubins paths must
-- end on their goal pose, the rectangle test must tell touching from apart, an entry path must end on its goal,
-- backing must steer the right way, and the cost of one cheap check is measured. For the test scripts.
function M._selfTest()
  local out = {}
  math.randomseed(12345)
  local worstPos, worstHdg, n = 0, 0, 0
  for _ = 1, 2000 do
    local x0, y0, th0 = (math.random() - 0.5) * 60, (math.random() - 0.5) * 60, (math.random() - 0.5) * 2 * pi
    local x1, y1, th1 = (math.random() - 0.5) * 60, (math.random() - 0.5) * 60, (math.random() - 0.5) * 2 * pi
    local D = dubins(x0, y0, th0, x1, y1, th1, 6.0)
    if D then
      local pts = {{x0, y0}}
      local px, py, th = dubinsPoints(pts, x0, y0, th0, D, STEP)
      worstPos = max(worstPos, sqrt((px - x1) ^ 2 + (py - y1) ^ 2))
      worstHdg = max(worstHdg, abs(angDiff(th, th1)))
      n = n + 1
    end
  end
  out.dubins = {paths = n, worstPositionError = worstPos, worstHeadingErrorDeg = deg(worstHdg)}
  out.rects = {
    apartBy10cm = not rectsOverlap(0, 0, 1, 0, 2.4, 0.95, 0, 2.0, 1, 0, 2.4, 0.95),
    overlapBy10cm = rectsOverlap(0, 0, 1, 0, 2.4, 0.95, 0, 1.8, 1, 0, 2.4, 0.95),
    crossed = rectsOverlap(0, 0, 1, 0, 2.4, 0.95, 0, 0, 0, 1, 2.4, 0.95),
    farApart = not rectsOverlap(0, 0, 1, 0, 2.4, 0.95, 20, 0, 1, 0, 2.4, 0.95),
  }
  -- an entry path to a space 9 m to the right and 8 m ahead, nose pointing right (-y): it ends on the goal
  local ux, uy, sf = 0, -1, 3.5
  local grx, gry = 8, -9
  local D = dubins(0, 0, 0, grx - ux * sf, gry - uy * sf, atan2(uy, ux), 5.8)
  if D then
    local c = {lead = 0, sf = sf, D = D, ux = ux, uy = uy}
    local pts, iG = entryPoints(c, 0, 0, 0)
    local g, last = pts[iG], pts[#pts]
    out.entry = {points = #pts, goalError = sqrt((g[1] - grx) ^ 2 + (g[2] - gry) ^ 2), tail = sqrt((last[1] - g[1]) ^ 2 + (last[2] - g[2]) ^ 2), length = D.length + sf, kind = D.kind}
    -- the cost of one cheap check against 30 standing vehicles
    local boxes = {}
    for i = 1, 30 do
      boxes[i] = {x = (i % 6) * 5 + 3, y = floor(i / 6) * 3 - 6, fx = 1, fy = 0, hl = 2.3, hw = 0.9, r = sqrt(2.3 ^ 2 + 0.9 ^ 2)}
    end
    local t0 = os.clock()
    for _ = 1, 100 do
      local p = entryPoints(c, 0, 0, 0)
      carsClear(poseList(p), boxes)
    end
    out.cheapCheckMs = (os.clock() - t0) * 10
  else
    out.entry = {err = "no dubins path"}
  end
  -- backing along a line that curves to the car's left must turn the wheel left (negative), to the right: right
  local left = polyFromPoints({{-1.18, 0}, {-3, 0.8}, {-6, 3}})
  local right = polyFromPoints({{-1.18, 0}, {-3, -0.8}, {-6, -3}})
  out.reverse = {curvesLeft = reverseSteer(left, 0, 1, -1.18, 0, 1, 0), curvesRight = reverseSteer(right, 0, 1, -1.18, 0, 1, 0)}
  out.ok = (out.dubins.worstPositionError < 0.01 and out.dubins.worstHeadingErrorDeg < 0.1 and out.rects.apartBy10cm and out.rects.overlapBy10cm
    and out.rects.crossed and out.rects.farApart and out.entry.goalError ~= nil and out.entry.goalError < 0.01
    and out.reverse.curvesLeft < 0 and out.reverse.curvesRight > 0) and true or false
  return out
end

-- for the test scripts: where a car is in the maneuver
function M.status(car)
  local st = car.park
  if not st then return nil end
  return {phase = st.phase, stage = st.stage, space = st.name, key = st.key, final = st.final, t = st.t, what = st.what,
          kind = st.kind, radius = st.R, path = st.total, rev = st.rv and st.rv.len or nil}
end

return M
