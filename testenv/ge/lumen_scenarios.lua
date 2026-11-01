-- Hard scenarios for the Lumen test instance (testenv only, never shipped). Storm, 2026-11-01: "Have you tested it
-- through hard scenarios? Like a car pulling out and a car swerving and a car changing lanes right next to Lumen?"
--
-- One scripted car per trial, driven by BeamNG's script AI (a timed trajectory: no steering sign to guess), next to
-- a Lumen driving its own route on the brain under test. The maneuver is issued at the moment the Lumen reaches the
-- trigger geometry, from where the scripted car really is, so the timing does not depend on how fast the Lumen
-- accelerated. Everything is measured here, every frame: the gap between the two bodies (separating-axis distance
-- between their boxes), the Lumen's speed and deceleration, when it first reacted, and whether either car was hit.
--
--   kinds   pull_out    parked at the kerb ahead, pulls into the Lumen's lane when the Lumen is `trigger` m behind it
--           swerve      ahead in the next lane, slower; swerves `amp` m into the Lumen's lane as the Lumen comes level
--           side_merge  in the next lane right beside the Lumen, changes into its lane over `dur` s
--
-- Driven from Python (testenv/hard_scenarios.py) through MCP: start(spec), status(), finish().

local M = {}
local logTag = "lumen_scenarios"
local sqrt, abs, max, min = math.sqrt, math.abs, math.max, math.min

local S = nil                         -- the running trial

local function right2(x, y) return y, -x end

-- the box of a vehicle now: centre, forward, half length, half width (half extents from its spawn box)
local function boxOf(v, ext)
  local p, d = v:getPosition(), v:getDirectionVector()
  local L = sqrt(d.x * d.x + d.y * d.y)
  local fx, fy = d.x / max(L, 1e-6), d.y / max(L, 1e-6)
  local rx, ry = right2(fx, fy)
  return {x = p.x + fx * ext.oy + rx * ext.ox, y = p.y + fy * ext.oy + ry * ext.ox, fx = fx, fy = fy, hl = ext.hl, hw = ext.hw}
end

local function extOf(v)
  local e = {hl = 2.3, hw = 0.95, ox = 0, oy = 0}
  local ok, bb = pcall(function() return v:getSpawnWorldOOBB() end)
  if ok and bb then
    local he = bb:getHalfExtents()
    e.hl, e.hw = max(he.x, he.y), min(he.x, he.y)
    local c, p, d = bb:getCenter(), v:getPosition(), v:getDirectionVector()
    local L = sqrt(d.x * d.x + d.y * d.y)
    local fx, fy = d.x / max(L, 1e-6), d.y / max(L, 1e-6)
    local rx, ry = right2(fx, fy)
    e.oy = (c.x - p.x) * fx + (c.y - p.y) * fy
    e.ox = (c.x - p.x) * rx + (c.y - p.y) * ry
  end
  return e
end

-- separating-axis gap between two boxes (0 when they touch or overlap): a lower bound of the true distance that
-- equals it whenever the closest features are faces, which is the case for cars side by side or nose to tail
local function gap(a, b)
  local best = -1e9
  local axes = {{a.fx, a.fy}, {right2(a.fx, a.fy)}, {b.fx, b.fy}, {right2(b.fx, b.fy)}}
  local dx, dy = b.x - a.x, b.y - a.y
  for _, ax in ipairs(axes) do
    local ux, uy = ax[1], ax[2]
    local arx, ary = right2(a.fx, a.fy)
    local brx, bry = right2(b.fx, b.fy)
    local ra = a.hl * abs(a.fx * ux + a.fy * uy) + a.hw * abs(arx * ux + ary * uy)
    local rb = b.hl * abs(b.fx * ux + b.fy * uy) + b.hw * abs(brx * ux + bry * uy)
    best = max(best, abs(dx * ux + dy * uy) - ra - rb)
  end
  return max(0, best)
end

-- the script AI's trajectory: points {x, y, z, t} from the car's pose; lateral(t) is the offset from the lane line
-- (right of travel positive) and speed(t) the speed, both functions of time from now
local function script(origin, fx, fy, latNow, lateral, speed, dur, z)
  local rx, ry = right2(fx, fy)
  local pts, s, t = {}, 0, 0
  local step = 0.1
  while t <= dur + 1e-6 do
    local d = lateral(t)
    pts[#pts + 1] = {x = origin.x + fx * s + rx * (d - latNow), y = origin.y + fy * s + ry * (d - latNow), z = z, t = t}
    s = s + speed(t) * step
    t = t + step
  end
  return pts
end

local function smooth(u) u = max(0, min(1, u)) return u * u * (3 - 2 * u) end

local function follow(v, pts)
  -- ai.startFollowing is the script AI that BeamNGpy's execute_script uses
  local parts = {}
  for i, p in ipairs(pts) do parts[i] = string.format("{x=%.3f,y=%.3f,z=%.3f,t=%.2f}", p.x, p.y, p.z, p.t) end
  v:queueLuaCommand("ai.startFollowing({" .. table.concat(parts, ",") .. "}, 0, 0, 'never')")
end

-- spec: {kind, lumenId, lineX, lineY, fx, fy (the Lumen's lane line: a point on it and its direction), z,
--        advModel, advConfig, s0 (where along the line the scripted car starts, from the line point),
--        lat0 (its lateral offset there), v (its speed before the maneuver), trigger, amp, dur, accel, maxT}
function M.start(spec)
  if S then M.finish() end
  local lumen = getObjectByID(spec.lumenId)
  if not lumen then return "no Lumen " .. tostring(spec.lumenId) end
  local fx, fy = spec.fx, spec.fy
  local rx, ry = right2(fx, fy)
  local px = spec.lineX + fx * spec.s0 + rx * spec.lat0
  local py = spec.lineY + fy * spec.s0 + ry * spec.lat0
  local pos = vec3(px, py, spec.z + 0.5)
  local rot = quatFromDir(vec3(fx, fy, 0), vec3(0, 0, 1))
  local adv = core_vehicles.spawnNewVehicle(spec.advModel or "pickup", {pos = pos, rot = rot, config = spec.advConfig, autoEnterVehicle = false})
  if not adv then return "spawn failed" end
  local lumenDmg = map.objects[spec.lumenId] and map.objects[spec.lumenId].damage or 0
  S = {spec = spec, adv = adv, advId = adv:getID(), lumen = lumen, lumenExt = extOf(lumen), advExt = nil, t = 0, phase = "setup",
       minGap = 1e9, minGapT = nil, vMin = 1e9, vAtTrigger = nil, maxDecel = 0, reactT = nil, triggerT = nil, hit = false,
       lumenDmg0 = lumenDmg, advDmg0 = nil, lastV = nil, decelF = 0, latAtTrigger = nil, maxLatShift = 0, events = {}}
  return "ok " .. tostring(S.advId)
end

local function lumenLat(sp, p)
  local rx, ry = right2(sp.fx, sp.fy)
  return (p.x - sp.lineX) * rx + (p.y - sp.lineY) * ry, (p.x - sp.lineX) * sp.fx + (p.y - sp.lineY) * sp.fy
end

local function onUpdate(dtReal, dtSim)
  if not S then return end
  local dt = dtSim or dtReal
  if not dt or dt <= 0 then return end
  local sp = S.spec
  local adv, lumen = getObjectByID(S.advId), getObjectByID(sp.lumenId)
  if not adv or not lumen then S.phase = "lost" return end
  S.t = S.t + dt
  if not S.advExt then
    S.advExt = extOf(adv)
    S.advDmg0 = map.objects[S.advId] and map.objects[S.advId].damage or 0
    -- parked until its maneuver: the parking brake holds it; the others are driven by the script at once
    adv:queueLuaCommand("ai.setMode('disabled') input.event('parkingbrake', " .. (sp.kind == "pull_out" and "1" or "0") .. ", 2)")
  end
  local lb, ab = boxOf(lumen, S.lumenExt), boxOf(adv, S.advExt)
  local g = gap(lb, ab)
  local vel = lumen:getVelocity()
  local v = vel:length()
  if S.lastV then
    local a = (S.lastV - v) / dt
    S.decelF = S.decelF + (a - S.decelF) * min(1, dt * 6)           -- smoothed deceleration
  end
  S.lastV = v
  local lat, along = lumenLat(sp, lumen:getPosition())
  local advLat, advAlong = lumenLat(sp, adv:getPosition())
  -- where the scripted car is relative to the Lumen, along the lane: its rear minus the Lumen's front
  local ahead = (advAlong - S.advExt.hl) - (along + S.lumenExt.hl)
  local damL = map.objects[sp.lumenId] and map.objects[sp.lumenId].damage or 0
  local damA = map.objects[S.advId] and map.objects[S.advId].damage or 0
  if damL - S.lumenDmg0 > 50 or (S.advDmg0 and damA - S.advDmg0 > 50) or g <= 0.02 then
    if not S.hit then S.hit = true S.events[#S.events + 1] = string.format("contact at %.2f s (gap %.2f, damage +%.0f / +%.0f)", S.t, g, damL - S.lumenDmg0, damA - (S.advDmg0 or 0)) end
  end

  if S.phase == "setup" then
    -- the pre-maneuver part: pull_out stands parked; the others drive their lane at sp.v
    if sp.kind ~= "pull_out" and not S.cruiseIssued then
      S.cruiseIssued = true
      local o = adv:getPosition()
      follow(adv, script(o, sp.fx, sp.fy, advLat, function() return advLat end, function() return sp.v end, 40, o.z))
    end
    local fire
    if sp.kind == "pull_out" then
      fire = ahead < sp.trigger and ahead > -2
    elseif sp.kind == "swerve" then
      fire = ahead < sp.trigger                                      -- the Lumen is catching up: swerve as it comes level
    else
      fire = ahead < sp.trigger and ahead > -S.advExt.hl * 2          -- side by side
    end
    if fire then
      S.phase, S.triggerT, S.vAtTrigger, S.latAtTrigger, S.aheadAtTrigger = "maneuver", S.t, v, lat, ahead
      local o = adv:getPosition()
      local lat0 = advLat
      local pts
      if sp.kind == "pull_out" then
        -- from standstill into the Lumen's lane centre over sp.dur s, accelerating at sp.accel
        pts = script(o, sp.fx, sp.fy, lat0, function(t) return lat0 * (1 - smooth(t / sp.dur)) end,
          function(t) return min(sp.accel * t, sp.v) end, sp.dur + 6, o.z)
      elseif sp.kind == "swerve" then
        -- sp.amp m toward the Lumen's lane (lat goes up toward 0) over 1 s, held sp.dur s, back over 1 s
        local toward = lat0 < 0 and 1 or -1
        pts = script(o, sp.fx, sp.fy, lat0, function(t)
          local u = t < 1 and smooth(t) or (t < 1 + sp.dur and 1 or 1 - smooth(t - 1 - sp.dur))
          return lat0 + toward * sp.amp * u
        end, function() return sp.v end, sp.dur + 5, o.z)
      else
        -- a full lane change into the Lumen's lane over sp.dur s, at its current speed
        local vv = adv:getVelocity():length()
        pts = script(o, sp.fx, sp.fy, lat0, function(t) return lat0 * (1 - smooth(t / sp.dur)) end, function() return max(vv, 3) end, sp.dur + 6, o.z)
      end
      if sp.kind == "pull_out" then adv:queueLuaCommand("input.event('parkingbrake', 0, 2)") end
      follow(adv, pts)
      S.events[#S.events + 1] = string.format("maneuver at %.2f s: Lumen %.1f m/s, scripted car %.1f m ahead and %.1f m aside", S.t, v, ahead, advLat - lat)
    elseif S.t > (sp.maxSetup or 40) then
      S.phase = "nofire"
    end
  elseif S.phase == "maneuver" then
    local tm = S.t - S.triggerT
    S.minGap = min(S.minGap, g)
    if g == S.minGap then S.minGapT = tm end
    S.vMin = min(S.vMin, v)
    S.maxDecel = max(S.maxDecel, S.decelF)
    S.maxLatShift = max(S.maxLatShift, abs(lat - (S.latAtTrigger or lat)))
    if not S.reactT and (S.decelF > 1.5 or abs(lat - (S.latAtTrigger or lat)) > 0.3) then S.reactT = tm end
    if tm > (sp.maxT or 9) then S.phase = "done" end
  end
end

function M.status()
  if not S then return {phase = "none"} end
  return {phase = S.phase, t = S.t, hit = S.hit, minGap = S.minGap < 1e8 and S.minGap or nil, minGapT = S.minGapT,
          vAtTrigger = S.vAtTrigger, vMin = S.vMin < 1e8 and S.vMin or nil, maxDecel = S.maxDecel, reactT = S.reactT,
          maxLatShift = S.maxLatShift, aheadAtTrigger = S.aheadAtTrigger, events = S.events, advId = S.advId}
end

function M.finish()
  if not S then return "none" end
  local st = M.status()
  local adv = getObjectByID(S.advId)
  if adv then adv:delete() end
  S = nil
  return st
end

M._gap, M._script = gap, script            -- for testenv/scenarios_offline_test.py
M.onUpdate = onUpdate
M.onExtensionLoaded = function() log("I", logTag, "hard scenarios ready (test instance only)") end
return M
