-- Lumen robotaxi vehicle controller (Frontier Studios).
-- Drives the autonomy halo: electrics.values.rtx_autonomy is 1 while the vehicle AI
-- is driving (any mode other than "disabled") or while the robotaxi service asks for
-- it (setAutonomy). The jbeam glowMap reads rtx_autonomy to switch rtx_halo on.
-- Fleet cars also report a little telemetry to the service twice a second (setTelemetry):
-- throttle, brake, gear and wheel speed. Throttle with no wheel speed means "pushing
-- against something", which is how the service tells a stuck car from one waiting at a
-- red light.
-- Lumen Drive's hands and feet (lumenDrive): the service sends a plan (a line to drive and a
-- speed for each point of it) five times a second, and this follows it every frame. Steering is
-- pure pursuit from the rear axle; the pedals are a speed loop. It never holds the brake pedal at
-- a standstill (BeamNG's arcade gearbox reads that as reverse), it holds with the parking brake.
-- No plan for PLAN_TIMEOUT means the service stalled: stop and hold.

local M = {}
M.type = "auxiliary"
M.relevantDevice = nil

local TELEMETRY_PERIOD = 0.5

-- the Lumen's geometry (generator/shape.py, lumen.jbeam): reference node b0, axles at y = -1.5 and +1.5
local WHEELBASE = 3.0
local REAR_AXLE = 1.18            -- m behind the reference node
local MAX_WHEEL = math.rad(30)    -- effective bicycle-model lock (inner wheel 32 deg, outer less)
local PLAN_TIMEOUT = 1.5          -- s
local KP, KI = 0.42, 0.10         -- speed loop
local plan = nil                  -- {ox, oy, x = {}, y = {}, v = {}, hold, halt, brake}
local planAt = -1e9
local pilotOn = false
local clock = 0
local pIdx = 1
local integ = 0
local thrOut, brkOut = 0, 0

local forced = 0          -- set by the robotaxi service (GE) through setAutonomy
local fade = 0            -- smoothed value so the halo does not flicker on AI mode changes
local telemetry = false
local telemetryTimer = 0
local hornSeq = nil       -- remaining {on, seconds} steps of a honk pattern
local hornT = 0
local evade = nil         -- target speed (m/s) backwards while backing away from a reversing car
local evadeSteer = 0      -- steering while backing, -1..1 right positive; 0 unless the service steers the back-up (parking)

local function init()
  electrics.values.rtx_autonomy = 0
  forced = 0
  fade = 0
  telemetry = false
  telemetryTimer = 0
end

local function reset()
  -- keep the service settings across a vehicle reset (recovery, teleport); an old plan is for
  -- the old place, so hold until the service sends a new one
  electrics.values.rtx_autonomy = forced
  fade = forced
  plan, pIdx, integ = nil, 1, 0
end

local function setAutonomy(on)
  forced = on and 1 or 0
end

local function setTelemetry(on)
  telemetry = on and true or false
  telemetryTimer = 0
end

-- "double": a friendly beep-beep at the pickup; "short": one tap at a car that will not move
local function honk(pattern)
  if pattern == "double" then
    hornSeq = {{true, 0.16}, {false, 0.11}, {true, 0.18}, {false, 0}}
  else
    hornSeq = {{true, 0.28}, {false, 0}}
  end
  hornT = 0
  electrics.horn(hornSeq[1][1])
end

-- Back away at a set speed (the service matches the reversing car's speed). In BeamNG's arcade
-- gearbox the brake pedal at a standstill or while backing gives reverse power, and the
-- throttle while backing brakes, so a simple speed loop on those two inputs holds the speed.
local function reverseAt(speed, steer)
  if speed and speed > 0 then
    evade = speed
    evadeSteer = math.max(-1, math.min(1, steer or 0))
  else
    evade = nil
    evadeSteer = 0
    input.event("brake", 0, 2, nil, nil, nil, "ai")
    input.event("throttle", 0, 2, nil, nil, nil, "ai")
  end
end

local function pedals(throttle, brake, parking)
  input.event("throttle", throttle, "FILTER_AI", nil, nil, nil, "ai")
  input.event("brake", brake, "FILTER_AI", nil, nil, nil, "ai")
  input.event("parkingbrake", parking, "FILTER_AI", nil, nil, nil, "ai")
end

-- The service's plan, or nil to let go (backing up runs on reverseAt; teleports need nothing).
-- The first plan takes the car from BeamNG's AI for good and selects the arcade gearbox.
local function lumenDrive(p)
  if p == nil then
    pilotOn, plan = false, nil
    return
  end
  if not pilotOn then
    pilotOn = true
    integ = 0
    if ai and ai.mode ~= "disabled" then ai.setMode("disabled") end
    local mc = controller.mainController
    if mc and mc.setGearboxMode then pcall(mc.setGearboxMode, "arcade") end
  end
  if p.x then
    p.n = #p.x
    for i = 1, p.n do
      p.x[i] = p.x[i] + p.ox
      p.y[i] = p.y[i] + p.oy
    end
    pIdx = 1
  end
  plan = p
  planAt = clock
end

-- distance-ahead lookups on the plan from segment i, fraction t
local function walkPlan(i, t, dist)
  local n = plan.n
  local x, y = plan.x[i] + (plan.x[i + 1] - plan.x[i]) * t, plan.y[i] + (plan.y[i + 1] - plan.y[i]) * t
  local left = dist
  local j = i
  while j < n do
    local nx, ny = plan.x[j + 1], plan.y[j + 1]
    local seg = math.sqrt((nx - x) ^ 2 + (ny - y) ^ 2)
    if seg >= left then
      local f = seg > 1e-6 and left / seg or 0
      return x + (nx - x) * f, y + (ny - y) * f, j
    end
    left = left - seg
    x, y = nx, ny
    j = j + 1
  end
  -- past the end: carry on along the last segment
  local dx, dy = plan.x[n] - plan.x[n - 1], plan.y[n] - plan.y[n - 1]
  local L = math.sqrt(dx * dx + dy * dy)
  if L < 1e-6 then return plan.x[n], plan.y[n], n end
  return plan.x[n] + dx / L * left, plan.y[n] + dy / L * left, n
end

local function speedAhead(i, t, dist)
  local _, _, j = walkPlan(i, t, dist)
  j = math.min(j, plan.n)
  return math.min(plan.v[math.max(1, j)] or 0, plan.v[math.min(plan.n, j + 1)] or 0)
end

local function pilot(dt)
  local dir = obj:getDirectionVector()
  local hx, hy = dir.x, dir.y
  local hl = math.sqrt(hx * hx + hy * hy)
  if hl < 1e-3 then return end
  hx, hy = hx / hl, hy / hl
  local fwd = obj:getVelocity():dot(dir)
  local stale = clock - planAt > PLAN_TIMEOUT
  if not plan or plan.hold or plan.halt or stale or not plan.x or (plan.n or 0) < 2 then
    -- stop and hold: brake while rolling, then the parking brake (never the pedal at a standstill)
    local speed = math.abs(fwd)
    if speed > 0.35 then
      local b = (plan and plan.halt) and 0.85 or math.min(0.8, 0.22 + 0.12 * speed)
      pedals(0, b, 0)
    else
      pedals(0, 0, 1)
    end
    integ = 0
    thrOut, brkOut = 0, 0
    return
  end
  local pos = obj:getPosition()
  local rx, ry = pos.x - hx * REAR_AXLE, pos.y - hy * REAR_AXLE
  -- closest segment, searching forward from the last one
  local n = plan.n
  local best, bestD2, bestT = pIdx, math.huge, 0
  for i = math.max(1, pIdx - 3), math.min(n - 1, pIdx + 40) do
    local ax, ay = plan.x[i], plan.y[i]
    local dx, dy = plan.x[i + 1] - ax, plan.y[i + 1] - ay
    local L2 = dx * dx + dy * dy
    local t = 0
    if L2 > 1e-8 then t = math.max(0, math.min(1, ((rx - ax) * dx + (ry - ay) * dy) / L2)) end
    local qx, qy = ax + dx * t, ay + dy * t
    local d2 = (rx - qx) ^ 2 + (ry - qy) ^ 2
    if d2 < bestD2 then best, bestD2, bestT = i, d2, t end
  end
  pIdx = best

  -- steering: pure pursuit on a point a speed-dependent distance ahead
  local speed = math.max(0, fwd)
  -- the look-ahead grows with speed (about 1.3 s at highway speed): 2026-11-01, a 20 m cap at
  -- 29 m/s made pure pursuit overshoot until the car spun on US Route 110
  local look = math.max(3.5, math.min(40, 2.5 + 0.5 * speed + 0.025 * speed * speed))
  local tx, ty = walkPlan(best, bestT, look)
  local lx, ly = tx - rx, ty - ry
  local lat = lx * hy - ly * hx              -- right of the car is (hy, -hx)
  local lon = lx * hx + ly * hy
  local ld = math.max(1, math.sqrt(lx * lx + ly * ly))
  local alpha = math.atan2(lat, lon)
  local wheel = math.atan(2 * math.sin(alpha) * WHEELBASE / ld)
  -- never more steering than the speed allows (about 6 m/s2 of lateral acceleration)
  local lim = math.atan(WHEELBASE * 6.0 / math.max(1, speed * speed)) / MAX_WHEEL
  local steer = math.max(-math.min(1, lim), math.min(math.min(1, lim), wheel / MAX_WHEEL))
  input.event("steering", steer, "FILTER_AI", nil, nil, nil, "ai")

  -- speed: the plan's speed a little ahead (so braking starts on time), a PI loop on the pedals
  local target = speedAhead(best, bestT, 1.0 + 0.6 * speed)
  if plan.brake then
    pedals(0, 1, 0)
    integ = 0
    return
  end
  if target < 0.05 and math.abs(fwd) < 0.35 then
    pedals(0, 0, 1)
    integ = 0
    thrOut, brkOut = 0, 0
    return
  end
  local err = target - fwd
  integ = math.max(-3, math.min(3, integ + err * dt))
  local u = KP * err + KI * integ
  local thr, brk = 0, 0
  if u >= 0 then thr = math.min(1, u) else brk = math.min(1, -u * 0.9) end
  -- creeping to a stop: the parking brake, not the pedal (arcade reverse)
  if target < 0.3 and fwd < 0.6 and brk > 0 then brk, thr = 0, 0 end
  -- One pedal at a time, and a released pedal is exactly zero: BeamNG's electric arcade logic
  -- (shiftLogic/electricMotor.lua, updateInGearArcade) only leaves neutral for drive when the
  -- smoothed brake input is <= 0. 2026-11-01: a brake output easing toward zero (never reaching
  -- it) left cars in neutral at full throttle after every stop and back-up.
  if thr > 0 then brkOut = 0 end
  if brk > 0 then thrOut = 0 end
  thrOut = thrOut + (thr - thrOut) * math.min(1, dt * 8)
  brkOut = brkOut + (brk - brkOut) * math.min(1, dt * 10)
  if thrOut < 0.02 then thrOut = 0 end
  if brkOut < 0.02 then brkOut = 0 end
  pedals(thrOut, brkOut, (target < 0.3 and fwd < 0.6) and 1 or 0)
end

local function sendTelemetry()
  local e = electrics.values
  local aiMode = (ai ~= nil and ai.mode) or "disabled"
  obj:queueGameEngineLua(string.format(
    "if extensions.lumen_robotaxi then extensions.lumen_robotaxi.onCarTelemetry(%d, %.3f, %.3f, %d, %.3f, %q) end",
    obj:getId(), e.throttle or 0, e.brake or 0, e.gearIndex or 0, math.abs(e.wheelspeed or 0), aiMode))
end

local function updateGFX(dt)
  clock = clock + dt
  local aiDriving = (ai ~= nil and ai.mode ~= nil and ai.mode ~= "disabled") and 1 or 0
  local target = math.max(forced, aiDriving, pilotOn and 1 or 0)
  -- quick on, slower off
  local rate = target > fade and 6 or 2
  fade = fade + (target - fade) * math.min(1, dt * rate)
  electrics.values.rtx_autonomy = fade > 0.5 and 1 or 0

  if pilotOn and not evade then pilot(dt) end

  if evade then
    local back = -obj:getVelocity():dot(obj:getDirectionVector())
    local err = evade - back
    if err > 0 then
      input.event("throttle", 0, 2, nil, nil, nil, "ai")
      input.event("brake", math.min(0.55, 0.12 + 0.18 * err), 2, nil, nil, nil, "ai")
    else
      input.event("brake", 0, 2, nil, nil, nil, "ai")
      input.event("throttle", math.min(0.6, -0.25 * err), 2, nil, nil, nil, "ai")
    end
    input.event("steering", evadeSteer, 2, nil, nil, nil, "ai")
  end

  if hornSeq then
    hornT = hornT + dt
    if hornT >= hornSeq[1][2] then
      table.remove(hornSeq, 1)
      hornT = 0
      if #hornSeq == 0 then
        hornSeq = nil
        electrics.horn(false)
      else
        electrics.horn(hornSeq[1][1])
      end
    end
  end

  if telemetry then
    telemetryTimer = telemetryTimer + dt
    if telemetryTimer >= TELEMETRY_PERIOD then
      telemetryTimer = 0
      sendTelemetry()
    end
  end
end

M.init = init
M.reset = reset
M.updateGFX = updateGFX
M.setAutonomy = setAutonomy
M.setTelemetry = setTelemetry
M.honk = honk
M.reverseAt = reverseAt
M.lumenDrive = lumenDrive

return M
