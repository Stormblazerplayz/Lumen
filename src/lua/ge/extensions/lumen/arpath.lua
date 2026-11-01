-- Lumen AR path (Frontier Studios).
-- Shows the robotaxi's path planner on the road itself (Storm, 2026-10-31: "Make it show the
-- AR path planner attached to the car in the real world"): a turquoise ribbon that leaves
-- your Lumen's front bumper and follows its planned route in its lane, chevrons flowing along
-- it, and a pulsing ring with a light beam at the stop point. The colour shows what the
-- planner is doing: turquoise driving, amber waiting for traffic or a signal (with a stop bar
-- in front of the car), red while blocked and finding another way.
-- Drawn with the same road-projected dynamic decals as the game's navigation markers, so it
-- hugs the real road surface.

local M = {}

local roads = require("lumen/roads")

local logTag = "lumen_arpath"
-- Plain PNGs, like the game's own ground arrows (art/shapes/arrows/arrow_groundmarkers_1.png).
-- 2026-10-31: as *.color.png the texture cooker converted them on first use, and the decal
-- manager kept the "IMPORTING TEXTURE" placeholder it was handed meanwhile for the whole session.
local BAND_TEX = "/art/lumen_robotaxi/lumen_ar_band.png"
local RING_TEX = "/art/lumen_robotaxi/lumen_ar_ring.png"
local ARROW_TEX = "art/shapes/arrows/arrow_groundmarkers_1.png"   -- the game's own chevron

local RANGE = 120              -- m of path drawn ahead of the car
local FADE_FROM = 80           -- m where the path starts to fade out
local SAMPLE = 1.5             -- m between smoothed path samples
local BAND_STEP = 2.6          -- m between ribbon decals (each a little longer, so they overlap)
local ARROW_STEP = 7.0         -- m between chevrons
local ARROW_SPEED = 6.0        -- m/s the chevrons flow forward
local FRONT = 2.6              -- m from the reference node to the front bumper
local REAR = 2.1               -- m from the reference node to the rear bumper
local BLEND = 8.0              -- m over which the ribbon bends from the car into its lane

local COLORS = {
  go = {0.09, 0.84, 0.77},
  wait = {1.00, 0.68, 0.16},
  blocked = {1.00, 0.27, 0.24},
  reverse = {1.00, 0.68, 0.16},
}
local yaw = {}                 -- [vehId] = {dir, rate}: yaw rate from the change in heading

local enabled = true
local pool = {}
local clock = 0
local UP = vec3(0, 0, 1)

local function decal(i)
  local d = pool[i]
  if not d then
    d = {texture = BAND_TEX, position = vec3(), forwardVec = vec3(0, 1, 0), color = ColorF(1, 1, 1, 0),
         scale = vec3(1, 1, 2), fadeStart = RANGE + 60, fadeEnd = RANGE + 90}
    pool[i] = d
  end
  return d
end

local function setDecal(i, tex, pos, fwd, rgb, a, sx, sy)
  local d = decal(i)
  d.texture = tex
  d.position:set(pos.x, pos.y, pos.z)
  d.forwardVec:set(fwd.x, fwd.y, fwd.z)
  d.color = ColorF(rgb[1], rgb[2], rgb[3], a)
  d.scale:set(sx, sy, 2.5)
end

-- the planned route from the car's current segment, offset into its lane
local function lanePoints(trip, carPos)
  local pts, offs = trip.points, roads.tripLaneOffsets(trip)
  local idx = math.max(1, math.min(trip.idx, #pts - 1))
  local a, b = pts[idx], pts[idx + 1]
  local raw = {a + (b - a) * clamp(carPos:xnormOnLine(a, b), 0, 1)}
  local rawOff = {offs[idx] or 0}
  for i = idx + 1, #pts do
    raw[#raw + 1] = pts[i]
    rawOff[#rawOff + 1] = offs[math.min(i, #offs)] or 0
  end
  local out = {}
  for i = 1, #raw do
    local p = raw[i]
    local dNext = raw[i + 1] and (raw[i + 1] - p) or nil
    local dPrev = raw[i - 1] and (p - raw[i - 1]) or nil
    local d = (dPrev and dPrev:normalized() or vec3()) + (dNext and dNext:normalized() or vec3())
    if d:length() < 1e-3 then d = dNext or dPrev or vec3(0, 1, 0) end
    d = d:normalized()
    out[i] = p + d:cross(UP):normalized() * rawOff[i]
  end
  return out
end

-- Catmull-Rom through the lane points, sampled about every SAMPLE metres
local function smooth(points)
  local out = {}
  local n = #points
  for i = 1, n - 1 do
    local p0, p1, p2, p3 = points[math.max(1, i - 1)], points[i], points[i + 1], points[math.min(n, i + 2)]
    local steps = math.max(1, math.ceil(p1:distance(p2) / SAMPLE))
    for s = 0, steps - 1 do
      local t = s / steps
      local t2, t3 = t * t, t * t * t
      out[#out + 1] = (p1 * 2 + (p2 - p0) * t + (p0 * 2 - p1 * 5 + p2 * 4 - p3) * t2 + (p1 * 3 - p0 - p2 * 3 + p3) * t3) * 0.5
    end
  end
  out[#out + 1] = points[n]
  return out
end

-- bend the start of the path out of the car's nose with its current heading
local function blendFromCar(path, front, dir)
  local acc, j = 0, nil
  for i = 2, #path do
    acc = acc + path[i]:distance(path[i - 1])
    if path[i]:distance(front) >= BLEND and (path[i] - front):dot(dir) > 0 then
      j = i
      break
    end
  end
  if not j then return path end
  local target = path[j]
  local tdir = (path[math.min(#path, j + 1)] - path[math.max(1, j - 1)]):normalized()
  local L = front:distance(target)
  local out = {}
  local steps = math.max(2, math.ceil(L / SAMPLE))
  for s = 0, steps - 1 do
    local t = s / steps
    local h00, h10, h01, h11 = 2 * t ^ 3 - 3 * t ^ 2 + 1, t ^ 3 - 2 * t ^ 2 + t, -2 * t ^ 3 + 3 * t ^ 2, t ^ 3 - t ^ 2
    out[#out + 1] = front * h00 + dir * (L * h10) + target * h01 + tdir * (L * h11)
  end
  for i = j, #path do out[#out + 1] = path[i] end
  return out
end

-- Reversing: where the car is backing to, from the rear bumper, curving with its current yaw
-- rate like a parking camera's guide lines (Storm: "It's on the back too right? Cause it
-- reverses."). About 2.5 s of travel, 3 to 9 m.
local function reversePath(veh, carPos, dir, speed, dt)
  local id = veh:getID()
  local y = yaw[id]
  if y and dt and dt > 0 then
    local cross = y.dir:cross(dir).z
    local rate = math.asin(clamp(cross, -1, 1)) / dt
    y.rate = y.rate + (rate - y.rate) * math.min(1, dt * 4)
    y.dir = vec3(dir)
  else
    y = {dir = vec3(dir), rate = 0}
    yaw[id] = y
  end
  local heading = -dir
  local p = carPos - dir * REAR
  local out = {vec3(p)}
  local total = clamp(speed * 2.5, 3, 9)
  local step = 0.5
  local turn = (speed > 0.3) and (y.rate * step / speed) or 0
  turn = clamp(turn, -0.12, 0.12)
  local acc = 0
  while acc < total do
    -- the direction of travel is fixed to the body, so it turns with the body's yaw rate
    local c, s = math.cos(turn), math.sin(turn)
    heading = vec3(heading.x * c - heading.y * s, heading.x * s + heading.y * c, 0)
    p = p + heading * step
    out[#out + 1] = vec3(p)
    acc = acc + step
  end
  return out
end

local function alphaAt(d)
  local a = math.min(1, math.max(0, (d - 0.5) / 3))
  if d > FADE_FROM then a = a * math.max(0, 1 - (d - FADE_FROM) / (RANGE - FADE_FROM)) end
  return a
end

local function onPreRender(dt)
  if not enabled then return end
  local rt = extensions.lumen_robotaxi
  local data = rt and rt.getArPath and rt.getArPath()
  if not data then return end
  local veh, trip = data.veh, data.trip
  if not veh or not trip or not trip.points or #trip.points < 2 then return end
  local carPos, dir = veh:getPosition(), veh:getDirectionVector()
  if core_camera.getPosition():distance(carPos) > 300 then return end
  dir = vec3(dir.x, dir.y, 0):normalized()
  clock = clock + (dt or 0)

  local ok, path
  local plan = data.plan
  local atEnd = true                     -- the drawn path ends at the trip's stop point
  if data.reversing then
    ok, path = pcall(reversePath, veh, carPos, dir, data.speed or 1, dt)
  elseif plan and plan.points and #plan.points >= 2 then
    -- Lumen Drive's own plan: exactly the line the car is following, from its front bumper on
    ok, path = true, {}
    local front = carPos + dir * FRONT
    for _, p in ipairs(plan.points) do
      if #path > 0 or (p - front):dot(dir) > 0 then
        if #path == 0 then path[1] = vec3(front.x, front.y, p.z) end
        path[#path + 1] = p
      end
    end
    local total = 0
    for i = 2, #plan.points do total = total + plan.points[i]:distance(plan.points[i - 1]) end
    atEnd = plan.endS ~= nil and plan.startS + total >= plan.endS - 1
  else
    yaw[veh:getID()] = nil
    ok, path = pcall(function()
      local front = carPos + dir * FRONT
      front.z = carPos.z
      return blendFromCar(smooth(lanePoints(trip, carPos)), front, dir)
    end)
  end
  if not ok or not path or #path < 2 then return end

  local rgb = data.reversing and COLORS.reverse or (COLORS[data.state] or COLORS.go)
  local count = 0
  -- ribbon and chevrons by arc length along the path
  local phase = (clock * ARROW_SPEED) % ARROW_STEP
  local nextBand, nextArrow = BAND_STEP * 0.5, phase
  local acc = 0
  for i = 2, #path do
    local a, b = path[i - 1], path[i]
    local seg = a:distance(b)
    if seg > 1e-4 then
      local fwd = (b - a) * (1 / seg)
      while nextBand <= acc + seg and nextBand <= RANGE do
        local p = a + fwd * (nextBand - acc)
        count = count + 1
        setDecal(count, BAND_TEX, p, fwd, rgb, 0.85 * alphaAt(nextBand), 1.25, BAND_STEP * 1.18)
        nextBand = nextBand + BAND_STEP
      end
      while nextArrow <= acc + seg and nextArrow <= RANGE do
        if data.state ~= "blocked" or data.reversing then
          local p = a + fwd * (nextArrow - acc)
          count = count + 1
          setDecal(count, ARROW_TEX, p, fwd, rgb, 0.9 * alphaAt(nextArrow), 0.95, 1.6)
        end
        nextArrow = nextArrow + ARROW_STEP
      end
    end
    acc = acc + seg
    if acc > RANGE then break end
  end
  -- planned stop: a bar across the lane where the car will stop (a light, a stopped car, a yield)
  if plan and plan.stopS and plan.stopKind ~= "end" and not data.reversing and #plan.points >= 2 then
    local want = plan.stopS + FRONT + 0.6 - plan.startS
    local run = 0
    for i = 2, #plan.points do
      local a, b = plan.points[i - 1], plan.points[i]
      local seg = a:distance(b)
      if run + seg >= want and seg > 1e-4 then
        local p = a + (b - a) * ((want - run) / seg)
        local f = (b - a) * (1 / seg)
        if p:distance(carPos) < RANGE then
          count = count + 1
          setDecal(count, BAND_TEX, p, f:cross(UP):normalized(), COLORS.blocked, 0.9, 0.6, 3.4)
        end
        break
      end
      run = run + seg
    end
  elseif not plan and data.state == "wait" and not data.reversing then
    count = count + 1
    local right = dir:cross(UP):normalized()
    setDecal(count, BAND_TEX, carPos + dir * (FRONT + 1.4), right, COLORS.blocked, 0.9, 0.55, 3.2)
  end
  -- the car it is yielding to: red chevrons along that car's predicted path
  if plan and plan.threat and #plan.threat >= 2 then
    for i = 2, #plan.threat do
      local a, b = plan.threat[i - 1], plan.threat[i]
      local seg = a:distance(b)
      if seg > 0.5 then
        count = count + 1
        setDecal(count, ARROW_TEX, (a + b) * 0.5, (b - a) * (1 / seg), COLORS.blocked, 0.8, 0.95, 1.6)
      end
    end
  end
  -- the stop point: a pulsing ring on the road and a thin beam of light
  local stop = path[#path]
  if atEnd and acc <= RANGE + 40 and not data.reversing then
    local pulse = 0.5 + 0.5 * math.sin(clock * 3.2)
    count = count + 1
    setDecal(count, RING_TEX, stop, dir, rgb, 0.75 + 0.2 * pulse, 4.2 + 0.5 * pulse, 4.2 + 0.5 * pulse)
    debugDrawer:drawCylinder(stop, stop + UP * 3.2, 0.05, ColorF(rgb[1], rgb[2], rgb[3], 0.35 + 0.2 * pulse))
  end
  for i = count + 1, #pool do pool[i].color = ColorF(0, 0, 0, 0) end
  if count > 0 then Engine.Render.DynamicDecalMgr.addDecals(pool, count) end
end

function M.setEnabled(on)
  enabled = on and true or false
  return enabled
end

function M.isEnabled() return enabled end

M.onPreRender = onPreRender
M.onExtensionLoaded = function() log("I", logTag, "Lumen AR path ready") end

return M
