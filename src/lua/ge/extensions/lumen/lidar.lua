-- Lumen colored lidar view (Frontier Studios).
-- Rays leave the roof sensor crown of the Lumen you are in (or your robotaxi) and are
-- drawn as a point cloud colored by what they hit:
--   ground / road  : teal, fading to blue with distance
--   static objects : amber to red with height (buildings, walls, poles, trees)
--   vehicles       : magenta (ray vs each nearby car's bounding box; castRay ignores cars)
-- The same scan is what a future driving network can learn from (Storm: "colored lidar").
--
-- One scan, two viewers, and it runs only while at least one of them is on:
--   the world view   setEnabled (robotaxi.lua's toggleLidar): the dots drawn into the 3D world
--   the app's tab    setUiEnabled: the same points top-down around the car, sent to the UI app
--                    as "LumenRobotaxiLidar" five times a second. Storm, 2026-11-01: "Make the app
--                    able to view the front camera so you can see your robotaxi coming", then
--                    "Actually just a live lidar view" (a HUD app cannot show a camera feed).

local M = {}

local logTag = "lumen_lidar"
local AZ_STEPS = 96                -- rays around per elevation ring
local ELEVATIONS = {-24, -16, -10, -6, -3, -1, 1, 4}   -- degrees
local MAX_RANGE = 60
local RAYS_PER_FRAME = 140
local POINT_LIFE = 0.45            -- seconds a point stays visible
-- roof crown (0, -0.30, 1.56) measured from the reference node b0 (0, 0.32, 0.20), which
-- is where getPosition() is; vehicle space, front is -y, left is +x
local SENSOR_LOCAL = vec3(0, -0.62, 1.36)

local enabled = false
local cursor = 0
local points = {}                  -- ring buffer {pos, color (false while only the app looks), t, kind code}
local head = 0
local MAX_POINTS = AZ_STEPS * #ELEVATIONS     -- also the number of rays in one full sweep
local clock = 0

-- the app's Lidar tab. The kind codes are what the app reads: 1 ground, 2 static, 3 vehicle, 4 person.
local KIND = {ground = 1, static = 2, vehicle = 3, person = 4}
local UI_RATE = 0.2                -- s between pictures (5 Hz)
local UI_IDLE_RATE = 1.0           -- s between "no sensor car" notes
local UI_LEASE = 6                 -- s one request from the app lasts; the app renews it every 2 s
local UI_MAX_POINTS = 800
local ui = {on = false, lease = 0, nextT = 0, mode = nil, reason = nil, n = 0, pts = {},
            snap = {range = MAX_RANGE, car = {hl = 0, hw = 0}}, you = {x = 0, y = 0, d = 0}, none = {none = true}}

-- The Lumen whose sensor we read: the one you are in, else your ride's car. Without one, the
-- second value says why (the app's Lidar tab shows it to the rider).
local function sensorVehicle()
  local v = getPlayerVehicle(0)
  if v and v:getJBeamFilename() == "lumen" then return v end
  if extensions.lumen_robotaxi then
    local s = extensions.lumen_robotaxi.getState()
    if s and s.taxiId then
      local t = getObjectByID(s.taxiId)
      if t then return t end
      return nil, "Waiting for your Lumen's lidar."
    end
    if s and s.state == "dispatching" then return nil, "Finding you a Lumen. Its lidar shows here once it is on the way." end
    if s and s.state == "departing" then return nil, "Ride complete. Call a Lumen to watch the next one arrive." end
  end
  return nil, "Lidar starts when your Lumen is on its way."
end

-- one slab of the box test: the ray's (tmin, tmax) narrowed by the two faces across axis a with
-- half extent ext, or nil when the ray misses the box
local function slab(o, dir, a, ext, tmin, tmax)
  local e = o:dot(a)
  local f = dir:dot(a)
  if math.abs(f) > 1e-6 then
    local t1 = (-ext - e) / f
    local t2 = (ext - e) / f
    if t1 > t2 then t1, t2 = t2, t1 end
    if t1 > tmin then tmin = t1 end
    if t2 < tmax then tmax = t2 end
    if tmin > tmax or tmax < 0 then return nil end
  elseif math.abs(e) > ext then
    return nil
  end
  return tmin, tmax
end

local function rayBox(origin, dir, center, ax, ay, az, h)
  -- slab test in the box frame; returns the entry distance or nil. 2026-11-01: the three axes
  -- are unrolled (same arithmetic in the same order, so the same hits) instead of two new tables
  -- per ray per nearby car, which was most of the scan's garbage once the app tab kept it running
  local o = origin - center
  local tmin, tmax = slab(o, dir, ax, h.x, -math.huge, math.huge)
  if tmin then tmin, tmax = slab(o, dir, ay, h.y, tmin, tmax) end
  if tmin then tmin = slab(o, dir, az, h.z, tmin, tmax) end
  return tmin and tmin > 0 and tmin or nil
end

local function nearbyBoxes(origin, selfId)
  local boxes = {}
  for vid, v in vehiclesIterator() do
    -- 2026-11-01: inactive vehicles are skipped. Walking mode leaves the rider's unicycle inactive
    -- where they got in (gameplay/walk.lua setUnicycleInactive), and its box stayed in the scan as
    -- a phantom at the pickup spot.
    if vid ~= selfId and v:getActive() then
      local p = v:getPosition()
      if p:distance(origin) < MAX_RANGE + 5 then
        local ok, box = pcall(function() return v:getSpawnWorldOOBB() end)
        if ok and box then
          -- the walking character is a vehicle with the "unicycle" jbeam: a person for the app
          boxes[#boxes + 1] = {c = box:getCenter(), x = box:getAxis(0), y = box:getAxis(1), z = box:getAxis(2), h = box:getHalfExtents(),
                               person = v:getJBeamFilename() == "unicycle"}
        end
      end
    end
  end
  return boxes
end

local function colorFor(kind, dist, height)
  -- a person keeps the vehicle color in the world view, as before the app learned to tell them apart
  if kind == "vehicle" or kind == "person" then return ColorF(1.0, 0.25, 0.85, 1) end
  if kind == "ground" then
    local t = math.min(1, dist / MAX_RANGE)
    return ColorF(0.10, 0.85 - 0.45 * t, 0.80 + 0.15 * t, 1)
  end
  local t = math.min(1, math.max(0, height / 6))
  return ColorF(1.0, 0.75 - 0.6 * t, 0.15, 1)
end

local function scan(veh, rays)
  local pos = veh:getPosition()
  local fwd = veh:getDirectionVector():normalized()
  local up = veh:getDirectionVectorUp():normalized()
  local right = fwd:cross(up):normalized()
  -- vehicle space: front is -y, left is +x
  local origin = pos + right * (-SENSOR_LOCAL.x) + fwd * (-SENSOR_LOCAL.y) + up * SENSOR_LOCAL.z
  local boxes = nearbyBoxes(origin, veh:getID())
  local groundZ = pos.z
  local total = AZ_STEPS * #ELEVATIONS
  for _ = 1, rays do
    cursor = (cursor % total) + 1
    local ring = math.floor((cursor - 1) / AZ_STEPS) + 1
    local azi = ((cursor - 1) % AZ_STEPS) / AZ_STEPS * 2 * math.pi
    local el = math.rad(ELEVATIONS[ring])
    local horiz = fwd * math.cos(azi) + right * math.sin(azi)
    local dir = (horiz * math.cos(el) + up * math.sin(el)):normalized()
    local target = origin + dir * MAX_RANGE
    local hitDist, kind = nil, "static"
    local res = castRay(origin, target, true, false)
    if res and res.pt then
      hitDist = origin:distance(res.pt)
      if res.norm and res.norm.z > 0.82 and (res.pt.z - groundZ) < 1.2 then kind = "ground" end
    end
    for _, b in ipairs(boxes) do
      local t = rayBox(origin, dir, b.c, b.x, b.y, b.z, b.h)
      if t and (not hitDist or t < hitDist) then hitDist, kind = t, b.person and "person" or "vehicle" end
    end
    if hitDist and hitDist > 1.2 then
      local p = origin + dir * hitDist
      head = (head % MAX_POINTS) + 1
      -- the world view's color is made only while it draws (the app colors by kind itself), and
      -- a slot's table is reused rather than replaced on every hit
      local color = enabled and colorFor(kind, hitDist, p.z - groundZ) or false
      local pt = points[head]
      if pt then
        pt[1], pt[2], pt[3], pt[4] = p, color, clock, KIND[kind]
      else
        points[head] = {p, color, clock, KIND[kind]}
      end
    end
  end
end

-- nobody looks any more: drop the points, they would be stale when the scan starts again
local function stopIfIdle()
  if not enabled and not ui.on then points, head = {}, 0 end
end

local function dm(v) return math.floor(v * 10 + 0.5) end

-- One picture for the app, in the car's own top-down frame: origin at the middle of its bounding
-- box (where the app draws the car), x to its right, y along its nose, in whole decimetres. The
-- frame is level (pitch and roll ignored) so a slope does not squeeze the picture.
local function publish(veh)
  local f = veh:getDirectionVector()
  local fx, fy = f.x, f.y
  local L = math.sqrt(fx * fx + fy * fy)
  if L > 1e-3 then fx, fy = fx / L, fy / L else fx, fy = 0, 1 end
  local c, hl, hw = veh:getPosition(), 2.3, 0.95      -- drive.lua sizeOf's defaults
  local ok, box = pcall(function() return veh:getSpawnWorldOOBB() end)
  if ok and box then
    c = box:getCenter()
    local he = box:getHalfExtents()
    if he and he.y > 0.3 then hl, hw = math.max(he.x, he.y), math.min(he.x, he.y) end
  end
  local cx, cy = c.x, c.y
  local pts, n = ui.pts, 0
  for i = 0, MAX_POINTS - 1 do
    -- newest first; slots are written in time order, so the first empty or stale one ends the walk
    local pt = points[(head - 1 - i) % MAX_POINTS + 1]
    if not pt or clock - pt[3] >= POINT_LIFE or n >= UI_MAX_POINTS * 3 then break end
    local dx, dy = pt[1].x - cx, pt[1].y - cy
    pts[n + 1], pts[n + 2], pts[n + 3] = dm(dx * fy - dy * fx), dm(dx * fx + dy * fy), pt[4]
    n = n + 3
  end
  for i = n + 1, ui.n do pts[i] = nil end
  ui.n = n
  local s = ui.snap
  s.pts, s.speed = pts, math.floor(veh:getVelocity():length() * 10 + 0.5) / 10
  s.car.hl, s.car.hw = math.floor(hl * 100 + 0.5) / 100, math.floor(hw * 100 + 0.5) / 100
  s.you = nil
  local pv = getPlayerVehicle(0)
  if pv and pv:getID() ~= veh:getID() then
    -- the rider outside the car (on foot or in another vehicle): where they are, seen from the car
    local p = pv:getPosition()
    local dx, dy = p.x - cx, p.y - cy
    local x, y = dx * fy - dy * fx, dx * fx + dy * fy
    local you = ui.you
    you.x, you.y, you.d = dm(x), dm(y), math.floor(math.sqrt(x * x + y * y) * 10 + 0.5) / 10
    s.you = you
  end
  -- guihooks.trigger encodes at once (common/guihooks.lua), so these tables are safe to reuse
  guihooks.trigger("LumenRobotaxiLidar", s)
end

-- What the app's Lidar tab is sent: a picture five times a second while there is a sensor car,
-- otherwise once a second why not.
local function uiTick(veh, reason)
  local mode = veh and "car" or "none"
  if mode ~= ui.mode then
    -- a reason goes out at once; a car's first picture waits one sweep, so the app never shows a
    -- half-scanned circle (after "none" the buffer holds nothing fresh)
    if mode == "none" then ui.nextT = clock elseif ui.mode == "none" then ui.nextT = clock + UI_RATE end
    ui.mode = mode
  elseif mode == "none" and reason ~= ui.reason then
    ui.nextT = clock                                   -- a new reason shows at once too
  end
  if clock < ui.nextT then return end
  if veh then
    publish(veh)
    local nt = ui.nextT + UI_RATE                      -- 5 Hz on average whatever the frame rate,
    ui.nextT = nt > clock and nt or clock + UI_RATE    -- and no burst of catch-up pictures after a hitch
  else
    ui.reason = reason
    ui.none.reason = reason
    guihooks.trigger("LumenRobotaxiLidar", ui.none)
    ui.nextT = clock + UI_IDLE_RATE
  end
end

local function onUpdate(dtReal)
  if not enabled and not ui.on then return end
  clock = clock + dtReal
  if ui.on and clock > ui.lease then
    -- the app renews its request every 2 s while the tab is open; silence means the app went away
    -- without saying so (a UI reload), and the scan must not keep running for nobody
    ui.on = false
    log("I", logTag, "app lidar feed off: the Lidar tab stopped renewing its request")
    stopIfIdle()
    if not enabled then return end
  end
  local veh, reason = sensorVehicle()
  if veh then
    -- the world view keeps its full rate; the app alone needs one full sweep per picture, which
    -- is about half the rays at 60 fps
    scan(veh, enabled and RAYS_PER_FRAME or math.max(1, math.min(RAYS_PER_FRAME, math.ceil(MAX_POINTS * dtReal / UI_RATE))))
  end
  if enabled then
    for i = 1, MAX_POINTS do
      local pt = points[i]
      if pt and pt[2] and clock - pt[3] < POINT_LIFE then
        debugDrawer:drawDot(pt[1], 7, pt[2])
      end
    end
  end
  if ui.on then uiTick(veh, reason) end
end

function M.setEnabled(on)
  enabled = on and true or false
  if not enabled then stopIfIdle() end
  ui_message(enabled and "Lidar view on: teal is road, amber is the world, magenta is traffic." or "Lidar view off.", 4, "lumen_lidar", "visibility")
  return enabled
end

function M.isEnabled() return enabled end

-- The app's Lidar tab: on while the tab is open. A request lasts UI_LEASE seconds (or `seconds`,
-- for a console test) and the app renews it every 2 s, so an app that disappears without saying
-- "off" (a UI reload) cannot leave the scan running, and a Lua reload under an open tab heals at
-- the next renewal.
function M.setUiEnabled(on, seconds)
  if on then
    if not ui.on then
      ui.mode, ui.reason = nil, nil
      ui.nextT = clock + UI_RATE                       -- the first picture waits one sweep (uiTick)
    end
    ui.on = true
    ui.lease = clock + (tonumber(seconds) or UI_LEASE)
  else
    ui.on = false
    stopIfIdle()
  end
  return ui.on
end

function M.isUiEnabled() return ui.on end

M.onUpdate = onUpdate
M.onExtensionLoaded = function() log("I", logTag, "Lumen lidar view ready") end

return M
