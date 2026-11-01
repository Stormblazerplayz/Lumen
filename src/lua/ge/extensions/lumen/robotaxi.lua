-- Lumen robotaxi service (Frontier Studios).
-- A fleet of four Lumens works each map (Storm, 2026-10-31: "I want 4 per map only. Just to
-- start with"). Free cars cruise the roads around you; call one and the nearest free Lumen
-- drives to a stop point next to you. Get in, pick a destination, ride, get dropped off, and
-- the car goes back to cruising.
-- A car that gets stuck fades out and goes in for service (Storm: "make it just disappear and
-- say it's getting serviced"); the next free Lumen is sent instead, and the serviced car
-- rejoins the fleet out of sight a little later. The app shows the ETA and a 2D map of the
-- car, its route and you.
-- Driving is BeamNG's built-in rule-based AI for now (Storm: "rule based for now and NN
-- later"); the turquoise autonomy halo is lit on every fleet car.
--
-- Ride states: idle -> dispatching -> arriving -> waiting -> boarded -> riding -> arrived
--              -> departing -> idle
--   dispatching: waiting for a free Lumen (all busy or in service)
--   arriving:    a Lumen is driving to the stop point nearest you
--   waiting:     it is stopped there with hazards on; walk up and get in, or choose Board
--   boarded:     you are inside; choose a destination
--   riding:      it drives you there
--   pulling:     you asked it to pull over; it signals, moves to the kerb and stops
--   pulledOver:  stopped at the kerb on hazards, fare paused; get out or continue the trip
--   arrived:     stopped at the destination, fare shown; get out or choose Exit
--   departing:   a few seconds of "thanks", then idle (the car is cruising again)
-- Car statuses: offline (about to appear) -> cruising <-> assigned; any -> fading -> service
--               -> offline
--
-- Controls: the "Lumen Robotaxi" UI app, the radial menu (Robotaxi), and input actions
-- (Options > Controls > Bindings > Gameplay > Lumen Robotaxi).

local M = {}
M.dependencies = {"core_vehicles"}

local roads = require("lumen/roads")
local drive = require("lumen/drive")   -- Lumen Drive: the fleet's own driver
-- The service area (lumen/geofence.lua). If it fails to load the whole map stays open rather than
-- the service going down.
local okGeo, geofence = pcall(require, "lumen/geofence")
if not okGeo then
  log("E", "lumen_robotaxi", "geofence failed to load, the map is not restricted: " .. tostring(geofence))
  geofence = nil
end
-- Parking in marked spaces at the pickup and the destination (lumen/parking.lua). Storm, 2026-11-02: "make the NN
-- park in the parking spaces if there are ones". If it fails to load the cars simply stop where they are.
local okPark, parking = pcall(require, "lumen/parking")
if not okPark then
  log("E", "lumen_robotaxi", "parking failed to load, cars stop where they are: " .. tostring(parking))
  local no = function() return false end
  parking = {start = no, update = function() end, busy = no, parked = no, holds = no, leave = no, forget = function() end,
             reset = function() end, stats = function() return {} end, status = function() return nil end, setEnabled = no}
end

local logTag = "lumen_robotaxi"

-- fleet
local FLEET_SIZE = 20                  -- Storm 2026-11-01: "Make it 20 lummens" (2026-10-31 it was "4 per map only. Just to start with")
local MODEL = "lumen"
local CONFIG = "vehicles/lumen/robotaxi.pc"
local NAME_PREFIX = "lumen_fleet_"
local SPAWN_MIN, SPAWN_MAX = 250, 900  -- fleet cars appear this far from you, out of sight
local SPAWN_GAP = 2                    -- s between fleet spawns; each spawn is a short hitch (20 cars fill in about 40 s)
local SPAWN_SPACING = 90               -- m kept between fleet cars when one appears (was 160 with 4 cars)
local HIDDEN_BEYOND = 450              -- m; a spot closer than this must be out of sight
local SERVICE_RADIUS = 1100            -- free cars further away are moved closer, out of sight
-- Distance that matters is DRIVING time: on winding maps a car 1 km away in a straight line can
-- be 5 km of road (Storm, 2026-10-31: "Why did it call a lumen all the way over there???").
local FLEET_MAX_ETA = 150              -- s: a free car further than this by road is moved closer, out of sight
local MAX_DISPATCH_ETA = 150           -- s: if every free car is further, a hidden one is brought closer first
local NEAR_ETA = 100                   -- s: how close a repositioned car is placed, by road
-- 2026-11-01: 90 s -> 30 s. Storm's log showed 3 cars retired in 10 minutes (2 damaged, 1 stuck) and
-- only 2 of 4 on the road; every call during the 90 s away failed or waited (Storm: "WHY IS IT FAILING
-- TO GET A TAXI?"). A serviced car is back in 30 s, in a fresh spot near the player.
local SERVICE_TIME = 30                -- s a serviced car is away before it rejoins
local FADE_TIME = 1.2                  -- s a car takes to fade out when it goes to service
local READY_DELAY = 1.0                -- s after a spawn or teleport before AI commands

-- rides
local ARRIVE_STOP = 6                  -- m from the stop point: stop the car
local APPROACH_DIST, APPROACH_DECEL = 70, 1.8
local BOARD_DIST = 60
local WAIT_TIMEOUT = 240
local REPICK_DIST = 120                -- you moved this far from the pickup: move the pickup
local OFF_ROUTE = 35
local AGGRESSION = 0.32

-- Standing still is only a problem when nothing explains it. Each check sorts a stop into
-- traffic (a vehicle close ahead in our lane), a signal (red light, stop sign or yield ahead)
-- or blocked (nothing in front). Traffic and signals are waited out; a blocked car backs up
-- and takes another way, and only goes to service when that fails. (Storm, 2026-10-31: "Does
-- it say the car is stuck if its in traffic? ... Does the robotaxi try to find another way if
-- it is still and not behind a car?")
local PUSH_TIME = 7                    -- s of throttle while the wheels do not turn
local BLOCKED_TIME = 10                -- s stopped with nothing ahead and no signal
local QUEUE_LIMIT = 90                 -- s in a queue that never moves (a wreck ahead) counts as blocked
local TRAFFIC_RANGE, SIGNAL_RANGE = 18, 45
local RECOVER_LIMIT = 2                -- back-up-and-reroute attempts before service
local BACKUP_DIST = 4                  -- m backed up at most when blocked, then another way
local BACKUP_SPEED = 1.3               -- m/s, the controller's reverse speed loop
local BACKUP_MAX_TIME = 5              -- s
-- a queue whose front car has not moved for this long with nothing holding it (no light, stop
-- sign or yield) is a blocked road: a crash, a car parked across the lane
local JAM_TIME = 20
local HONK_WAIT, HONK_GAP = 12, 30     -- honk at a vehicle that sits still ahead on a green
-- a vehicle close ahead backing towards the Lumen: back away at its speed (Storm, 2026-10-31)
local EVADE_RANGE = 14                 -- m centre to centre (about 9 m bumper to bumper)
local EVADE_ROOM = 7                   -- m that must be free behind to back up
local EVADE_MAX_SPEED = 8              -- m/s
local EVADE_MAX_DIST = 30              -- m backed up before giving up
local FLIP_TIME = 2.5
-- trying hard but going nowhere (crumpled bodywork rubbing a wheel, a jammed corner): 15 of the
-- last 20 s on the throttle and under 40 m covered. 2026-10-31: a car with a crushed rear crawled
-- at 0.1 to 3 m/s on 30-50 % throttle and slipped past both rules above.
local HIST_N, EFFORT_MIN, EFFORT_DIST = 40, 30, 40
-- Collisions (BeamNG's beam damage sum). Any real impact takes a Lumen out of service: a jump of
-- HIT_DAMAGE between two checks, or DAMAGE_LIMIT in total from small knocks. With a rider aboard
-- it stops, lets them out, and the next free Lumen finishes the trip. 2026-10-31: the old limits
-- (2500 empty, 9000 riding) let a car that had driven into another one carry on (Storm: "How
-- does it not need service?").
-- 2026-11-01, Storm: "Lumen hit a car then kept going": a scrape added less than 150 in one check, so the car drove on until
-- small knocks summed to DAMAGE_LIMIT. 50 counts a real contact; each one is logged with its size so this can be calibrated.
local HIT_DAMAGE, DAMAGE_LIMIT = 50, 1000
-- a parked (undriven) car standing in our lane this long counts as a blockage, not traffic
local PARKED_BLOCK_TIME = 8

local FARE_BASE, FARE_PER_KM, FARE_PER_MIN = 2.50, 1.20, 0.30
local UI_RATE = 0.25
local SETTINGS_FILE = "settings/lumen_robotaxi.json"

local fleet = {}                       -- [slot] = car, see newCar()
local ride = nil                       -- the player's ride, see newRide()
local enabled = true                   -- the service switch (persisted)
local active = false                   -- the fleet is running on this level
local initAt = nil                     -- real time at which to start the fleet
local simTime, realTime = 0, 0
local lastSpawnAt = -1e9
local uiTimer = 0
local lidarOn = false
local arPathOn = true                   -- the AR path planner view on the road (lumen_arpath)
local destCache, destCacheAt = nil, -1e9
-- Lumen thinking out loud (Storm: "Can you make Lumen have reasoning?"): every decision the ride
-- car makes is written here in plain words, newest last, and shown live in the app.
local thoughts = {}
local THOUGHTS_KEPT = 6
local function think(text, kind)
  local last = thoughts[#thoughts]
  if last and last.text == text then
    last.t = realTime
    return
  end
  thoughts[#thoughts + 1] = {text = text, kind = kind or "info", t = realTime}
  while #thoughts > THOUGHTS_KEPT do table.remove(thoughts, 1) end
  log("I", logTag, "thinking: " .. text)
end
local blockedRoads = {}                -- [roads.linkKey] = sim time until which the fleet avoids it
local BLOCKED_MEMORY = 300             -- s a blocked stretch of road is avoided
local RIDE_RECOVER_LIMIT = 4           -- recovery attempts per ride before the car goes to service

-- the blocked roads still remembered, as an avoid set for roads.planTrip
local function avoidSet()
  local out, any = {}, false
  for k, untilT in pairs(blockedRoads) do
    if simTime < untilT then
      out[k], any = true, true
    else
      blockedRoads[k] = nil
    end
  end
  return any and out or nil
end

-- ---------------------------------------------------------------------------------
-- helpers
-- ---------------------------------------------------------------------------------
local function msg(text, ttl, icon)
  ui_message(text, ttl or 5, "lumen_robotaxi", icon or "local_taxi")
end

local function walkExt() return extensions.gameplay_walk end

local function playerPos()
  local v = getPlayerVehicle(0)
  return v and v:getPosition() or core_camera.getPosition()
end

local function playerDir()
  local v = getPlayerVehicle(0)
  return v and v:getDirectionVector() or core_camera.getForward()
end

local function isFreeroam()
  if career_career and career_career.isActive and career_career.isActive() then return false end
  local gs = core_gamestate and core_gamestate.state and core_gamestate.state.state
  return gs == "freeroam"
end

local function missionActive()
  local mm = gameplay_missions_missionManager
  return mm and mm.getForegroundMissionId and mm.getForegroundMissionId() ~= nil or false
end

local function minutesText(sec)
  if not sec then return "a moment" end
  if sec < 50 then return "under a minute" end
  local m = math.max(1, math.floor(sec / 60 + 0.5))
  return m == 1 and "1 min" or (m .. " min")
end

local function round(v, step)
  step = step or 1
  return math.floor(v / step + 0.5) * step
end

-- ---------------------------------------------------------------------------------
-- fleet cars
-- ---------------------------------------------------------------------------------
local function newCar(slot)
  return {
    slot = slot, name = NAME_PREFIX .. slot, id = nil, status = "offline",
    spawnAt = 0, serviceUntil = 0, readyAt = 0, needsCruise = false,
    tele = {throttle = 0, brake = 0, gear = 0, ws = 0, ai = "disabled", at = -1e9},
    anchor = nil, anchorT = 0, pushT = 0, flipT = 0, fadeT = 0, checkT = 0, relocateT = 0,
    hist = {i = 0, effort = {}, moved = {}}, lastPos = nil,
    blockedT = 0, queueT = 0, aheadStillT = 0, inTraffic = false, signal = false,
    recoveries = 0, recoverPos = nil, recovering = nil, honkAt = -1e9,
    evading = nil, evadeCheckT = 0, wasTraffic = false, wasSignal = false,
    approach = false, approachT = 0, reason = nil,
  }
end

local function carVeh(car)
  if not car or not car.id then return nil end
  return getObjectByID(car.id)
end

local function carCommand(car, code)
  local v = carVeh(car)
  if v then v:queueLuaCommand(code) end
end

local function carBySlot(slot) return slot and fleet[slot] or nil end

local function rideCar() return ride and carBySlot(ride.slot) or nil end

-- the service's own inputs use source "ai" so they still apply while a rider's controls are
-- locked (input.lua drops "local" events for locked inputs)
local CONTROLLER = "local c = controller.getController('lumenRobotaxi') if c then c.setAutonomy(true) c.setTelemetry(true) end "
local RELEASE = "input.event('parkingbrake', 0, 2, nil, nil, nil, 'ai') input.event('brake', 0, 2, nil, nil, nil, 'ai') electrics.set_warn_signal(false) "

local function resetHealth(car)
  car.anchor, car.anchorT, car.pushT, car.flipT = nil, 0, 0, 0
  car.hist, car.lastPos = {i = 0, effort = {}, moved = {}}, nil
  car.blockedT, car.queueT, car.aheadStillT, car.inTraffic, car.signal = 0, 0, 0, false, false
end

local HONK_DOUBLE = "local c = controller.getController('lumenRobotaxi') if c and c.honk then c.honk('double') end"
local HONK_SHORT = "local c = controller.getController('lumenRobotaxi') if c and c.honk then c.honk('short') end"

-- nearest vehicle in our lane within range metres ahead (or behind): distance, speed, id
local function vehicleNear(veh, range, behind)
  local id = veh:getID()
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  if behind then dir = -dir end
  local bestD, bestV, bestId
  for oid, o in pairs(map.objects) do
    if oid ~= id and o.pos then
      local rel = o.pos - pos
      local fwd = rel:dot(dir)
      if fwd > 1.5 and fwd < range and (rel - dir * fwd):length() < 2.8 and (not bestD or fwd < bestD) then
        bestD, bestV, bestId = fwd, o.vel and o.vel:length() or 0, oid
      end
    end
  end
  return bestD, bestV, bestId
end

-- someone is driving it: the player's vehicle or a traffic car. Anything else standing in the
-- lane (a parked car, the car you just got out of) will not move, so waiting or honking is pointless.
local function isDriven(id)
  if not id then return false end
  if id == be:getPlayerVehicleID(0) then return true end
  local tr = gameplay_traffic and gameplay_traffic.getTrafficData and gameplay_traffic.getTrafficData()
  return (tr and tr[id] ~= nil) and true or false
end

-- Free distance behind the rear bumper (m, at most `want`) for anything static: rays from both
-- rear corners and the middle at three heights catch guardrails, walls, poles and bollards, and a
-- ray down 1.5 m behind the car checks the ground is still there (a road edge on a hill). castRayStatic
-- hits terrain and static objects, not vehicles; those are vehicleNear's job. 2026-10-31: backing
-- up looked for vehicles only, and a Lumen reversed into a guardrail (Storm: "It backed into a guardrail").
local REAR_BUMPER, HALF_WIDTH = 2.1, 0.85      -- m behind the reference node; half the body width
local function roomBehind(veh, want)
  local pos, dir, up = veh:getPosition(), veh:getDirectionVector(), veh:getDirectionVectorUp()
  local right = dir:cross(up):normalized()
  local back = -dir
  local free = want
  for _, side in ipairs({-HALF_WIDTH, 0, HALF_WIDTH}) do
    for _, h in ipairs({0, 0.3, 0.6}) do
      local o = pos + back * (REAR_BUMPER - 0.2) + right * side + up * h
      free = math.min(free, math.max(0, castRayStatic(o, back, want + 0.2) - 0.2))
    end
  end
  local g = pos + back * (REAR_BUMPER + 1.5) + up * 0.6
  if castRayStatic(g, -up, 2.0) >= 2.0 then free = math.min(free, 1.0) end
  return free
end

-- a vehicle close ahead in our lane that is moving backwards towards us: its closing speed
local function reversingThreat(veh)
  local id = veh:getID()
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  local best
  for oid, o in pairs(map.objects) do
    if oid ~= id and o.pos and o.vel and o.dirVec then
      local rel = o.pos - pos
      local fwd = rel:dot(dir)
      if fwd > 2 and fwd < EVADE_RANGE and (rel - dir * fwd):length() < 2.8 then
        local closing = -o.vel:dot(dir)
        if closing > 0.6 and o.vel:dot(o.dirVec) < -0.3 and (not best or closing > best) then best = closing end
      end
    end
  end
  return best
end

-- true while the car waits before a red light, a stop sign or a yield
local STOP_ACTIONS = {stop = true, alert = true, briefStop = true, yield = true}
local function signalAt(id, pos, dir)
  local ts = core_trafficSignals
  if not ts or not ts.getBestSignal then return false end
  local ok, sig = pcall(ts.getBestSignal, pos, dir)
  if not ok or not sig then return false end
  local okp, pl = pcall(function() return sig:getVehPlacement(id) end)
  if not okp or not pl or pl.dist > SIGNAL_RANGE or pl.dot < 0.6 or pl.relDist > 0.2 then return false end
  local oks, _, data = pcall(function() return sig:getState() end)
  return (oks and data and STOP_ACTIONS[data.action]) and true or false
end

local function signalAhead(veh)
  return signalAt(veh:getID(), veh:getPosition(), veh:getDirectionVector())
end

-- The queue in front of us, car by car: from us to the car ahead, from it to the one ahead of
-- it, following each car's heading round bends, until a gap of more than 20 m. Returns how many
-- cars, and the front one's id, position and heading.
local function queueHead(veh, maxCars)
  local seen = {[veh:getID()] = true}
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  local n, headId, headPos, headDir = 0, nil, nil, nil
  for _ = 1, maxCars do
    local bestD, bestId, bestO
    for oid, o in pairs(map.objects) do
      if not seen[oid] and o.pos then
        local rel = o.pos - pos
        local fwd = rel:dot(dir)
        if fwd > 1.5 and fwd < 20 and (rel - dir * fwd):length() < 2.8 and (not bestD or fwd < bestD) then
          bestD, bestId, bestO = fwd, oid, o
        end
      end
    end
    if not bestId then break end
    seen[bestId] = true
    n = n + 1
    headId, headPos = bestId, bestO.pos
    headDir = bestO.dirVec or dir
    pos, dir = bestO.pos, headDir
  end
  return n, headId, headPos, headDir
end

-- Free cars roam on their own with Lumen Drive, on roads around the player.
local function cruise(car)
  -- parked in a space, or backing out of one: get out first (the cruise check calls this again)
  if parking.holds(car) then
    local pv = carVeh(car)
    if pv and parking.leave(car, pv) then return end
  end
  carCommand(car, CONTROLLER .. "electrics.set_warn_signal(false)")
  drive.cruise(car, playerPos())
  car.needsCruise = false
  car.cruiseCheckT = 0
  resetHealth(car)
end

-- Lumen Drive follows the trip and stops at its end, pulled in to the curb. 2026-10-31 night: this
-- used to hand the route to BeamNG's built-in AI (Storm: "Beamng's AI. Did I ever say to use that?").
local function driveTrip(car, trip)
  carCommand(car, CONTROLLER .. "electrics.set_warn_signal(false)")
  -- in a space, or backing out of one: leave first and keep the trip until the car is out (updateFleet starts it)
  if parking.holds(car) then
    car.pendingTrip = trip
    local pv = carVeh(car)
    if pv then parking.leave(car, pv) end
    return
  end
  drive.follow(car, trip, {curb = true})
  resetHealth(car)
end

local SIGNALS_OFF = "if electrics.values.signal_left_input == 1 then electrics.toggle_left_signal() end " ..
  "if electrics.values.signal_right_input == 1 then electrics.toggle_right_signal() end "

-- Stop where it is and hold: the controller brakes, then holds on the parking brake with the
-- pedal released. Holding the brake pedal at a standstill selects reverse in BeamNG's arcade
-- gearbox: v1 did that and floored it backwards at every pickup (Storm, 2026-10-31: "When the
-- Lumen reaches the player it just reverses at full speed").
local function stopCar(car, hazards)
  drive.stop(car)
  if hazards then carCommand(car, "electrics.set_warn_signal(true)") end
end

local function setUsable(car, usable)
  local v = carVeh(car)
  if not v then return end
  v.playerUsable = usable and true or false
  local w = walkExt()
  if w then
    if usable then
      if w.removeVehicleFromBlacklist then w.removeVehicleFromBlacklist(car.id) end
    elseif w.addVehicleToBlacklist then
      w.addVehicleToBlacklist(car.id)
    end
  end
end

-- keep the rider's keyboard or pad from fighting the AI while they ride
local CONTROLS = {"steering", "throttle", "brake", "parkingbrake", "clutch"}
local function lockPlayerControls(veh, locked)
  if not veh then return end
  local parts = {}
  for _, c in ipairs(CONTROLS) do
    if locked then
      parts[#parts + 1] = string.format("input.setAllowedInputSource(%q, 'local', false) input.setAllowedInputSource(%q, 'ai', true)", c, c)
    else
      parts[#parts + 1] = string.format("input.setAllowedInputSource(%q, nil)", c)
    end
  end
  veh:queueLuaCommand(table.concat(parts, " "))
end

local function otherCarPositions(except)
  local out = {}
  for _, c in ipairs(fleet) do
    if c ~= except then
      local v = carVeh(c)
      if v then out[#out + 1] = v:getPosition() end
    end
  end
  return out
end

-- Memory guard (2026-11-01, Storm's game: 4000+ "failed to create texture" in a few minutes, the PC down to 0.7 GB of
-- memory while the fleet kept growing; BeamNG's own canSpawnAnotherVehicle only acts on consoles). A Lumen spawns only
-- while minSpawnGB is free, and below shrinkGB the farthest idle Lumen out of sight is retired every 20 s until it recovers.
local memGuard = {minSpawnGB = 1.5, shrinkGB = 0.9, checkT = 0, loggedAt = -1e9}
local function freeMemGB()
  local ok, mi = pcall(function() return Engine.Platform.getMemoryInfo() end)
  if ok and type(mi) == "table" and mi.availableBytes then return mi.availableBytes / 1073741824 end
  return nil
end

local function spawnCar(car)
  local free = freeMemGB()
  if free and free < memGuard.minSpawnGB then
    car.spawnAt = simTime + 30
    if realTime - memGuard.loggedAt > 60 then
      memGuard.loggedAt = realTime
      log("W", logTag, string.format("memory: only %.2f GB free, so fleet car %d waits before spawning (it needs %.1f GB, to keep textures loading)",
        free, car.slot, memGuard.minSpawnGB))
    end
    return false
  end
  local origin = playerPos()
  local avoid = otherCarPositions(car)
  -- on the rider's own piece of the served network: a car in another piece cannot drive to them
  local comp = roads.areaComponent(origin)
  local pos, dir, up = roads.findSpot(origin, SPAWN_MIN, SPAWN_MAX, {avoid = avoid, spacing = SPAWN_SPACING, farEnough = HIDDEN_BEYOND, component = comp})
  if not pos then
    -- small or crowded map: anywhere out of sight on that piece
    pos, dir, up = roads.findSpot(origin, 60, 4000, {avoid = avoid, spacing = 60, farEnough = HIDDEN_BEYOND, component = comp})
  end
  if not pos then
    pos, dir, up = roads.findSpot(origin, 60, 4000, {avoid = avoid, spacing = 60, farEnough = HIDDEN_BEYOND})
  end
  if not pos then
    car.spawnAt = simTime + 5
    return false
  end
  local veh = core_vehicles.spawnNewVehicle(MODEL, {
    config = CONFIG, pos = pos + up * 0.3, rot = quatFromDir(dir, up),
    autoEnterVehicle = false, vehicleName = car.name,
  })
  lastSpawnAt = realTime
  if not veh then
    log("E", logTag, "fleet car " .. car.slot .. " failed to spawn")
    car.spawnAt = simTime + 10
    return false
  end
  car.id = veh:getID()
  car.drv = nil
  parking.forget(car)
  if veh:getDirectionVector():dot(dir) < 0 then
    log("W", logTag, "fleet car " .. car.slot .. " spawned facing backwards; turning it around")
    spawn.safeTeleport(veh, pos, quatFromDir(dir, up), true, nil, false)
  end
  setUsable(car, false)
  car.status = "cruising"
  car.needsCruise = true
  car.readyAt = simTime + READY_DELAY
  resetHealth(car)
  car.lastDamage = nil
  local gz = geofence and geofence.hasZones() and geofence.status(pos.x, pos.y) or nil
  log("I", logTag, string.format("fleet car %d spawned as %d, %.0f m from the player%s", car.slot, car.id, pos:distance(origin),
    gz and (gz.inside and string.format(", in %s", tostring(gz.zone)) or ", OUTSIDE every open zone") or ""))
  return true
end

-- a hidden road spot from which a Lumen reaches (L, stop) within maxEta seconds of driving
-- forward declaration (ride section). It must come before relocate(): declared below it, the name
-- in relocate() was a global that was always nil, so relocating never aimed at the pickup.
local pickupForPlayer

local function spotNear(L, stop, maxEta, avoidPositions)
  for _ = 1, 10 do
    local pos, dir, up = roads.findSpot(stop, 120, 700, {avoid = avoidPositions, spacing = 100, farEnough = HIDDEN_BEYOND, component = roads.component(L)})
    if not pos then return nil end
    local trip = roads.planTrip(pos, dir, L, stop, {avoid = avoidSet()})
    if trip and trip.eta <= maxEta then return pos, dir, up, trip end
  end
  return nil
end

local function relocate(car, veh)
  local pos, dir, up
  local pk = pickupForPlayer and pickupForPlayer(true)
  if pk then pos, dir, up = spotNear(pk.link, pk.pos, FLEET_MAX_ETA * 0.8, otherCarPositions(car)) end
  if not pos then
    pos, dir, up = roads.findSpot(playerPos(), 300, 850, {avoid = otherCarPositions(car), spacing = SPAWN_SPACING, farEnough = HIDDEN_BEYOND,
      component = roads.areaComponent(playerPos())})
  end
  if not pos then return false end
  parking.forget(car)
  drive.release(car)
  spawn.safeTeleport(veh, pos, quatFromDir(dir, up), true, nil, false)
  car.needsCruise = true
  car.readyAt = simTime + 0.6
  resetHealth(car)
  return true
end

-- ---------------------------------------------------------------------------------
-- health: a car that cannot carry on goes to service
-- ---------------------------------------------------------------------------------
local function health(car, veh, dt, driving)
  local pos = veh:getPosition()
  local speed = veh:getVelocity():length()
  if veh:getDirectionVectorUp().z < 0.35 then car.flipT = car.flipT + dt else car.flipT = 0 end
  if car.flipT > FLIP_TIME then return "flipped" end
  local o = map.objects[car.id]
  if o and o.damage then
    local delta = car.lastDamage and (o.damage - car.lastDamage) or 0
    local hit = car.lastDamage ~= nil and delta >= HIT_DAMAGE
    car.lastDamage = o.damage
    if hit then
      drive.abCount(car, "hit")
      log("I", logTag, string.format("fleet car %d: contact, damage +%.0f in one check (total %.0f), brain %s", car.slot, delta, o.damage, tostring(drive.brainOf(car))))
    end
    if hit or o.damage > DAMAGE_LIMIT then return "damaged" end
  end
  if not driving then
    car.lastPos = nil
    car.blockedT, car.queueT, car.aheadStillT, car.inTraffic = 0, 0, 0, false
    return nil
  end
  local t = car.tele
  local fresh = realTime - t.at < 2
  -- why is it standing? a vehicle close ahead or a signal is normal; nothing at all is not
  local aheadD, aheadV, aheadId
  if speed < 1.5 then aheadD, aheadV, aheadId = vehicleNear(veh, TRAFFIC_RANGE) end
  -- Lumen Drive knows when it is waiting on purpose (a light, a stop sign, a stopped car, a yield);
  -- 2026-11-01: without this, cars waiting at red lights were judged blocked, backed up and sent to service
  local kind = drive.stopKind(car)
  car.signal = speed < 1.5 and (kind == "signal" or signalAhead(veh)) or false
  car.inTraffic = aheadD ~= nil or car.signal or (speed < 1.5 and drive.state(car) == "wait")
  -- the driver's emergency layer is holding the car for a wall or a post (people are different: they move
  -- away). 2026-11-01: back up and take another way instead of waiting for ever.
  if kind == "obstacle" and speed < 0.5 then
    car.obstacleT = (car.obstacleT or 0) + dt
    if car.obstacleT > 5 then
      car.obstacleT = 0
      car.blockCause = "obstacle"
      return "blocked"
    end
  else
    car.obstacleT = 0
  end
  -- Lumen Drive stopped for a parked car that leaves no room to pass (drive.lua, 2026-11-01): it will not move,
  -- so take another way after PARKED_BLOCK_TIME instead of the 90 s a queue gets
  if kind == "parked" and speed < 0.5 then
    car.parkStopT = (car.parkStopT or 0) + dt
    if car.parkStopT > PARKED_BLOCK_TIME then
      car.parkStopT = 0
      car.blockCause = "parked"
      return "blocked"
    end
  else
    car.parkStopT = 0
  end
  if aheadD and (aheadV or 0) < 0.3 and not car.signal then car.aheadStillT = car.aheadStillT + dt else car.aheadStillT = 0 end
  car.aheadParked = aheadD ~= nil and not isDriven(aheadId)
  if car.aheadParked and car.aheadStillT > PARKED_BLOCK_TIME then
    car.aheadStillT = 0
    car.blockCause = "parked"
    return "blocked"
  end
  -- A queue that does not move (Storm, 2026-10-31: "Why is it not trying to find a different path
  -- by now?"): walk it to its front car. Nothing holding that car and no movement for JAM_TIME
  -- means the road is blocked, so take another way instead of waiting out QUEUE_LIMIT.
  car.queueLen, car.queueHeld = 0, false
  if aheadD and not car.signal and speed < 0.6 then
    local n, headId, headPos, headDir = queueHead(veh, 15)
    local ho = headId and map.objects[headId]
    car.queueLen = n
    car.queueHeld = headId ~= nil and signalAt(headId, headPos, headDir)
    if ho and (not ho.vel or ho.vel:length() < 0.3) and not car.queueHeld then car.jamT = (car.jamT or 0) + dt else car.jamT = 0 end
    if car.jamT > JAM_TIME then
      car.jamT = 0
      car.blockCause = "jam"
      car.jamCars = n
      return "blocked"
    end
  else
    car.jamT = 0
  end
  if car.recoverPos and pos:distance(car.recoverPos) > 40 then car.recoveries, car.recoverPos = 0, nil end
  -- pushing against something solid: throttle on, wheels not turning
  local driven = drive.mode(car) == "follow"
  local pushing = fresh and driven and t.throttle > 0.12 and t.ws < 0.6 and speed < 0.5
  if pushing then car.pushT = car.pushT + dt else car.pushT = math.max(0, car.pushT - dt) end
  if car.pushT > PUSH_TIME then
    car.pushT = 0
    return "blocked"
  end
  -- standing with nothing ahead and no signal: something is in the way
  if speed < 0.6 and not car.inTraffic then car.blockedT = car.blockedT + dt else car.blockedT = math.max(0, car.blockedT - 2 * dt) end
  if car.blockedT > BLOCKED_TIME then
    car.blockedT = 0
    return "blocked"
  end
  -- a queue that never moves (a wreck ahead, a deadlock) eventually counts as blocked too
  if car.inTraffic and speed < 0.6 then car.queueT = car.queueT + dt else car.queueT = 0 end
  if car.queueT > QUEUE_LIMIT then
    car.queueT = 0
    return "blocked"
  end
  local h = car.hist
  h.i = h.i % HIST_N + 1
  h.effort[h.i] = (fresh and driven and t.throttle > 0.2) and 1 or 0
  h.moved[h.i] = car.lastPos and pos:distance(car.lastPos) or 0
  car.lastPos = pos
  local effort, moved = 0, 0
  for k = 1, HIST_N do
    effort = effort + (h.effort[k] or 0)
    moved = moved + (h.moved[k] or 0)
  end
  if effort >= EFFORT_MIN and moved < EFFORT_DIST then return "stuck" end
  return nil
end

local rideCarLost, newRide, setState   -- forward declarations (ride section)
local recover, updateRecovery          -- forward declarations (recovery, ride section)
local updateEvasion                    -- forward declaration (evasion, ride section)
local ensureAppInLayout                -- forward declaration (settings section)

local function sendToService(car, reason)
  if car.status == "fading" or car.status == "service" then return end
  drive.abCount(car, "svc_" .. tostring(reason))
  parking.forget(car)
  log("I", logTag, string.format("fleet car %d (%s) goes to service: %s", car.slot, tostring(car.id), reason))
  local wasRide = ride and ride.slot == car.slot
  car.reason = reason
  if carVeh(car) then
    drive.halt(car)
    car.status = "fading"
    car.fadeT = 0
  else
    car.status = "service"
    car.serviceUntil = simTime + SERVICE_TIME
    car.id = nil
  end
  if wasRide then
    think("This Lumen cannot carry on (" .. tostring(reason) .. "). Sending it for service and calling the next free one.", "warn")
    rideCarLost(car, reason)
  end
end

local function finishFade(car, dtReal)
  car.fadeT = car.fadeT + dtReal
  local veh = carVeh(car)
  local a = 1 - car.fadeT / FADE_TIME
  if veh and a > 0 then
    pcall(function() veh:setMeshAlpha(a, "", false) end)
    return
  end
  car.id = nil                          -- before delete, so onVehicleDestroyed ignores it
  car.drv = nil
  car.status = "service"
  car.serviceUntil = simTime + SERVICE_TIME
  if veh then veh:delete() end
end

local abOn, abLogT = false, 0
local function updateFleet(dt, dtReal)
  memGuard.checkT = memGuard.checkT + dtReal
  if memGuard.checkT >= 20 then
    memGuard.checkT = 0
    local free = freeMemGB()
    if free and free < memGuard.shrinkGB then
      local pp, pick, pickD = playerPos(), nil, -1
      for _, c in ipairs(fleet) do
        local v = carVeh(c)
        if v and c.status == "cruising" and not (ride and ride.slot == c.slot) and not c.testHold then
          local p = v:getPosition()
          local dd = p:distance(pp)
          if dd > pickD and roads.isHidden(p, HIDDEN_BEYOND) then pick, pickD = c, dd end
        end
      end
      if pick then
        log("W", logTag, string.format("memory: only %.2f GB free; retiring fleet car %d (%.0f m away, out of sight) so the game can keep loading textures",
          free, pick.slot, pickD))
        sendToService(pick, "memory")
      end
    end
  end
  if abOn then
    abLogT = abLogT + dt
    if abLogT >= 60 then
      abLogT = 0
      log("I", logTag, "abtest: " .. jsonEncode(drive.abReport() or {}))
    end
  end
  for _, car in ipairs(fleet) do
    if car.status == "offline" then
      if simTime >= car.spawnAt and realTime - lastSpawnAt >= SPAWN_GAP then spawnCar(car) end
    elseif car.status == "service" then
      if simTime >= car.serviceUntil then
        car.status = "offline"
        car.spawnAt = simTime
      end
    elseif car.status == "fading" then
      finishFade(car, dtReal)
    else
      local veh = carVeh(car)
      if not veh then
        -- removed by someone else (vehicle menu, a mission, a script)
        parking.forget(car)
        car.id = nil
        car.status = "service"
        car.serviceUntil = simTime + 20
        if ride and ride.slot == car.slot then rideCarLost(car, "unavailable") end
      else
        if car.needsCruise and car.status == "cruising" and simTime >= car.readyAt and not car.testHold then cruise(car) end
        if car.status == "cruising" and not car.needsCruise and not car.recovering and not car.evading and not parking.busy(car) and not car.testHold then
          -- keep roaming: the next road before this one runs out, or again after a failed pick
          car.cruiseCheckT = (car.cruiseCheckT or 0) + dt
          if car.cruiseCheckT > 1 then
            car.cruiseCheckT = 0
            local left = drive.remaining(car)
            if drive.mode(car) ~= "follow" then
              cruise(car)
            elseif left and left < 150 and not drive.keepCruising(car, playerPos()) and left < 40 then
              cruise(car)
            end
          end
        end
        -- the parking maneuver (lumen/parking.lua): going into a space, holding there, or backing out of it
        if car.park and not car.recovering and not car.evading then
          -- released 2026-11-01 before any in-game test (Storm: "Forget the tests just release em"): an error in the
          -- maneuver must cost one car its parking, never the fleet's update loop
          local okp, pev = pcall(parking.update, car, veh, dt, simTime)
          if not okp then
            if not car.parkErr then
              car.parkErr = true
              log("E", logTag, string.format("parking failed for car %d, it stops where it is: %s", car.slot, tostring(pev)))
            end
            pcall(parking.forget, car)
            car.park, pev = nil, nil
          end
          if pev then
            local oke, err = pcall(M._parkEvent, car, pev)
            if not oke then log("E", logTag, "parking event failed for car " .. tostring(car.slot) .. ": " .. tostring(err)) end
          end
        end
        if not car.recovering and not car.evading then drive.update(car, veh, dt, simTime) end
        if drive.debug then drive.drawDebug(car, veh) end
        if car.recovering then updateRecovery(car, veh, dt) end
        if car.pendingTrip and simTime >= car.readyAt then
          local trip = car.pendingTrip
          car.pendingTrip = nil
          driveTrip(car, trip)
        end
        car.evadeCheckT = car.evadeCheckT + dt
        if car.evadeCheckT >= 0.1 and not car.recovering then
          local step = car.evadeCheckT
          car.evadeCheckT = 0
          local moving = car.status == "cruising" or (ride and ride.slot == car.slot and (ride.state == "arriving" or ride.state == "riding" or ride.state == "waiting"))
          if moving or car.evading then updateEvasion(car, veh, step) end
        end
        car.checkT = car.checkT + dt
        if car.checkT >= 0.5 and not car.recovering and not car.evading then
          local step = car.checkT
          car.checkT = 0
          local driving = car.status == "cruising" or (ride and ride.slot == car.slot and (ride.state == "arriving" or ride.state == "riding" or ride.state == "pulling"))
          local occupied = getPlayerVehicle(0) and getPlayerVehicle(0):getID() == car.id
          -- (a car going into or backing out of a space is judged by the maneuver itself, not by the stuck tests)
          local reason = (simTime >= car.readyAt) and health(car, veh, step, driving and not parking.busy(car)) or nil
          local rideTries = (ride and ride.slot == car.slot) and (ride.recoveries or 0) or 0
          if reason == "blocked" and car.recoveries < RECOVER_LIMIT and rideTries < RIDE_RECOVER_LIMIT then
            recover(car, veh)
          elseif reason and not (occupied and not driving) then
            sendToService(car, reason == "blocked" and "stuck" or reason)
          end
          -- what the car is doing goes to the thinking feed from Lumen Drive itself (setReasonHook)
          local mine = ride and ride.slot == car.slot and (ride.state == "arriving" or ride.state == "riding")
          car.wasTraffic, car.wasSignal = car.inTraffic, car.signal
          -- a polite honk at a vehicle that sits still ahead when nothing is stopping it
          if car.aheadStillT > HONK_WAIT and not car.aheadParked and car.queueLen <= 1 and not car.queueHeld and simTime - car.honkAt > HONK_GAP then
            car.honkAt = simTime
            carCommand(car, HONK_SHORT)
            if mine then think("The car ahead has not moved for " .. HONK_WAIT .. " s with nothing stopping it. Honking.", "warn") end
          end
          if car.status == "cruising" and not reason then
            car.relocateT = car.relocateT + step
            if car.relocateT >= 6 then
              car.relocateT = 0
              local pos = veh:getPosition()
              local far = pos:distance(playerPos())
              if far > 500 and roads.isHidden(pos, HIDDEN_BEYOND) then
                local tooFar = far > SERVICE_RADIUS
                if not tooFar then
                  -- a car that cannot get to the player at all (stranded on a piece of road
                  -- that does not connect) is moved too
                  -- (the rider's pickup spot is looked up once per 2 s for the whole fleet: with 20 cars
                  -- a lookup per car every 6 s was a scan and an A* each)
                  local rc = M._relo
                  if not rc then
                    rc = {t = -1e9}
                    M._relo = rc
                  end
                  if realTime - rc.t > 2 then rc.t, rc.pk = realTime, pickupForPlayer(true) end
                  local pk = rc.pk
                  if pk then
                    local cp, sp = roads.componentAt(pos, veh:getDirectionVector()), roads.component(pk.link)
                    if cp and sp and cp ~= sp then
                      tooFar = true                  -- on another piece: no route exists, none is planned
                    else
                      local trip = roads.planTrip(pos, veh:getDirectionVector(), pk.link, pk.pos)
                      tooFar = trip == nil or trip.eta > FLEET_MAX_ETA
                    end
                  end
                end
                if tooFar then relocate(car, veh) end
              end
            end
          end
        end
      end
    end
  end
end

local function startFleet()
  if active or not enabled or not isFreeroam() or missionActive() then return false end
  if roads.nodeCount() < 40 then
    log("I", logTag, "no road network on this level; the robotaxi fleet stays off")
    return false
  end
  active = true
  for i = 1, FLEET_SIZE do
    local car = fleet[i] or newCar(i)
    fleet[i] = car
    -- adopt a fleet car that survived a Lua reload
    local obj = scenetree.findObject(car.name)
    if obj and obj.getID and getObjectByID(obj:getID()) then
      car.id = obj:getID()
      car.status = "cruising"
      car.needsCruise = true
      car.readyAt = simTime + 0.2
      setUsable(car, false)
    else
      car.status = "offline"
      car.spawnAt = simTime + (i - 1) * SPAWN_GAP
    end
  end
  -- the player is sitting in a fleet car (a Lua reload mid-ride): pick the ride back up
  local pv = getPlayerVehicle(0)
  for _, car in ipairs(fleet) do
    if pv and car.id and pv:getID() == car.id then
      car.status = "assigned"
      car.needsCruise = false
      stopCar(car, false)
      ride = newRide()
      ride.slot = car.slot
      setState("boarded")
      lockPlayerControls(pv, true)
      msg("Your Lumen is ready. Choose a destination.", 5)
    end
  end
  log("I", logTag, "robotaxi fleet started: " .. FLEET_SIZE .. " Lumens")
  if geofence and geofence.hasZones() then
    local e = geofence.export()
    local names = {}
    for _, z in ipairs(e.zones) do names[#names + 1] = string.format("%s (tier %d, %s)", z.name, z.tier, z.open and "open" or "closed") end
    log("I", logTag, string.format("geofence: level %s, open tier %d (%s); zones: %s", tostring(e.level), e.openTier,
      geofence.tierName(e.openTier), table.concat(names, "; ")))
  elseif geofence then
    log("I", logTag, "geofence: this map has no service area, nothing is restricted")
  end
  ensureAppInLayout()
  return true
end

local function stopFleet(deleteCars)
  parking.reset()
  for _, car in ipairs(fleet) do
    local v = carVeh(car)
    car.id = nil
    if deleteCars and v then v:delete() end
    car.status = "offline"
  end
  fleet = {}
  ride = nil
  active = false
end

-- ---------------------------------------------------------------------------------
-- destinations
-- ---------------------------------------------------------------------------------
local function prettyName(raw)
  local s = tostring(raw or "")
  s = s:gsub("^spawns?_", ""):gsub("_", " ")
  s = s:gsub("(%a)([%w']*)", function(a, b) return a:upper() .. b end)
  return s ~= "" and s or "Destination"
end

local function namedPlaces()
  local out = {}
  local ppos = playerPos()
  local ok, objs = pcall(scenetree.findClassObjects, "SpawnSphere")
  if ok and objs then
    for _, name in ipairs(objs) do
      local o = scenetree.findObject(name)
      if o then
        local pos = o:getPosition()
        local d = pos:distance(ppos)
        if d > 150 then out[#out + 1] = {id = name, name = prettyName(name), pos = pos, dist = d} end
      end
    end
  end
  table.sort(out, function(a, b) return a.dist < b.dist end)
  return out
end

local function destinationOptions()
  if destCache and realTime - destCacheAt < 2 then return destCache end
  local opts = {}
  local tp = core_groundMarkers and core_groundMarkers.getTargetPos and core_groundMarkers.getTargetPos()
  if tp then opts[#opts + 1] = {id = "__waypoint", name = "My map waypoint", kind = "waypoint"} end
  for i, p in ipairs(namedPlaces()) do
    if i > 8 then break end
    opts[#opts + 1] = {id = p.id, name = p.name, kind = "place", km = round(p.dist / 1000, 0.1),
                       served = (not geofence) or geofence.served(p.pos.x, p.pos.y)}
  end
  opts[#opts + 1] = {id = "__surprise", name = "Surprise me", kind = "surprise"}
  destCache, destCacheAt = opts, realTime
  return opts
end

local function resolveDestination(id)
  if id == "__waypoint" then
    local tp = core_groundMarkers and core_groundMarkers.getTargetPos and core_groundMarkers.getTargetPos()
    if tp then return vec3(tp), "your waypoint" end
    return nil
  elseif id == "__surprise" then
    local ppos = playerPos()
    local picks = {}
    for _, n in pairs(roads.nodes()) do
      if n.radius and n.radius > 3 then
        local d = n.pos:distance(ppos)
        if d > 700 and d < 3000 and ((not geofence) or geofence.served(n.pos.x, n.pos.y)) then picks[#picks + 1] = n.pos end
      end
    end
    if #picks > 0 then return vec3(picks[math.random(#picks)]), "a surprise destination" end
    return nil
  end
  local px, py = tostring(id):match("^xy:(-?[%d%.]+),(-?[%d%.]+)$")
  if px then
    -- Storm, 2026-10-31: "Why can I not put in a custom destination?" Any point on the map. The
    -- height comes from the nearest road node, so the road search compares like with like.
    local x, y = tonumber(px), tonumber(py)
    if x and y then
      local best, bestD2 = nil, math.huge
      for _, n in pairs(roads.nodes()) do
        local d2 = (n.pos.x - x) ^ 2 + (n.pos.y - y) ^ 2
        if d2 < bestD2 then best, bestD2 = n, d2 end
      end
      log("I", logTag, string.format("custom destination: pin at (%.1f, %.1f), nearest road node %.0f m away (height %.1f)", x, y,
        math.sqrt(bestD2), best and best.pos.z or 0))
      return vec3(x, y, best and best.pos.z or 0), "your pin"
    end
    return nil
  end
  local o = scenetree.findObject(id)
  if o then return o:getPosition(), prettyName(id) end
  return nil
end

-- ---------------------------------------------------------------------------------
-- rides
-- ---------------------------------------------------------------------------------
newRide = function()
  return {
    state = "dispatching", stateT = 0, slot = nil, pickup = nil, trip = nil, dest = nil,
    eta = nil, resume = false, fareStarted = false, distance = 0, rideTime = 0, fare = 0,
    lastPos = nil, retryT = 0, playerOwnVeh = nil, exiting = false, repickT = 0, replanT = 0,
  }
end

setState = function(s)
  if ride.state ~= s then log("I", logTag, "ride: " .. tostring(ride.state) .. " -> " .. s) end
  ride.state = s
  ride.stateT = 0
end

local function computeFare()
  if not ride or not ride.fareStarted then return 0 end
  return FARE_BASE + FARE_PER_KM * (ride.distance / 1000) + FARE_PER_MIN * (ride.rideTime / 60)
end

-- Standing in a car park or on a driveway: the Lumen comes in, if a lane it can use is right
-- there (Storm, 2026-10-31: "It also just stops when the pick up spot is in a parking lot").
-- If a car park beats it once, the next car waits at the street (ride.publicOnly).
-- Vehicles standing still that a stop point must keep clear of: parked cars, and your own car
-- once you are out of it (Storm, 2026-10-31: "did it hit my car?" -- the pickup had been put on
-- the FCV they had just got out of, and the Lumen crept into it). Fleet cars and people on foot
-- do not count.
local STOP_CLEAR_ALONG, STOP_CLEAR_SIDE = 7.5, 2.6   -- m, centre to centre along / across the road
local function stopBlockers()
  local mine = {}
  for _, c in ipairs(fleet) do
    if c.id then mine[c.id] = true end
  end
  local out = {}
  for id, o in pairs(map.objects) do
    if not mine[id] and o.pos and (not o.vel or o.vel:length() < 1) then
      local v = getObjectByID(id)
      if v and v:getJBeamFilename() ~= "unicycle" then out[#out + 1] = o.pos end
    end
  end
  return out
end

-- the stop point on link L near t, slid along the road (3 m steps, up to 24 m) to the nearest
-- spot no standing vehicle occupies; second result true when it had to move
local function clearStop(L, t, blockers)
  local base = roads.stopPoint(L, t)
  local u = (L.pb - L.pa):normalized()
  local function free(p)
    for _, q in ipairs(blockers) do
      local rel = q - p
      local along = rel:dot(u)
      if math.abs(along) < STOP_CLEAR_ALONG and (rel - u * along):length() < STOP_CLEAR_SIDE then return false end
    end
    return true
  end
  if free(base) then return base, false end
  for k = 1, 8 do
    for _, sgn in ipairs({1, -1}) do
      local p = roads.stopPoint(L, t + sgn * k * 3 / math.max(L.len, 1))
      if free(p) then return p, true end
    end
  end
  return base, false
end

-- Stop-point candidates near pos, nearest first: a parking-lot lane right there (unless
-- publicOnly), then the nearest public roads. Callers take the first one a Lumen can reach,
-- because road networks have islands (Storm, 2026-10-31: "How can it not find a route?").
-- Is a private lane (a car-park aisle, a driveway, a park path) clear of walls and posts for a car?
-- Rays sideways (1.1 m) 0.6 m above the lane every 2.5 m for 30 m either side of the stop point.
-- 2026-11-01, Storm: a Lumen drove into a park and hit the wall at its gate.
local function laneClear(L, t)
  local dx, dy = L.pb.x - L.pa.x, L.pb.y - L.pa.y
  local hl = math.sqrt(dx * dx + dy * dy)
  if hl < 1 then return false end
  local rx, ry = dy / hl, -dx / hl
  local len = math.max(L.len, 1)
  for off = -30, 30, 2.5 do
    local f = clamp(t + off / len, 0, 1)
    local p = L.pa + (L.pb - L.pa) * f
    local o = vec3(p.x, p.y, p.z + 0.6)
    if castRayStatic(o, vec3(rx, ry, 0), 1.1) < 1.1 or castRayStatic(o, vec3(-rx, -ry, 0), 1.1) < 1.1 then return false end
  end
  return true
end

local function stopCandidates(pos, publicOnly, maxDist, n)
  local blockers = stopBlockers()
  local out = {}
  if not publicOnly then
    -- 2026-11-01, Storm: "Lumen navigation doesn't want to go into parking lots." Only the single closest link
    -- was ever considered, so one pole beside the nearest aisle (or a public road a metre closer) ruled the
    -- whole lot out. The four nearest lot aisles within 35 m are tried now; each must still pass every check.
    local lotsOk = (not geofence) or (geofence.openTier() >= 2 and geofence.served(pos.x, pos.y))
    if lotsOk then
      local tried = 0
      for _, c in ipairs(roads.closestLinks(pos, {maxDist = 35}, 16)) do
        local L = c.L
        if not L.public and L.radius >= 2.0 and (L.link.drivability or 0) >= 0.2 then
          tried = tried + 1
          if laneClear(L, c.t) then
            local p, moved = clearStop(L, c.t, blockers)
            out[#out + 1] = {link = L, pos = p, private = true, moved = moved}
            break
          end
          if tried >= 4 then break end
        end
      end
    end
  end
  for _, c in ipairs(roads.closestLinks(pos, {public = true, maxDist = maxDist}, n or 10)) do
    local p, moved = clearStop(c.L, c.t, blockers)
    out[#out + 1] = {link = c.L, pos = p, moved = moved}
  end
  return out
end

-- Nothing within `near` metres is a covered road: widen the search (600, 2000, 6000 m) and use the
-- nearest covered spots instead of refusing, 80 m or more apart so one cut-off road cannot use them
-- all up. remote = the spot is not near the point. Storm, 2026-11-01: "when calling a Lumen just put
-- the pickup point on the nearest available spot and not just say it can't get there".
local wideCache = {x = 0, y = 0, t = -1e9, out = nil}
local function widenedCandidates(pos, publicOnly, near)
  local c = stopCandidates(pos, publicOnly, near)
  if #c > 0 then return c, false end
  -- the wide scans are not cheap and callers ask every few seconds: reuse a result for 2 s
  if wideCache.out and realTime - wideCache.t < 2 and (wideCache.x - pos.x) ^ 2 + (wideCache.y - pos.y) ^ 2 < 400 then
    return wideCache.out, #wideCache.out > 0
  end
  local out = {}
  for _, r in ipairs({600, 2000, 6000}) do
    local wide = stopCandidates(pos, true, r, 40)
    if #wide > 0 then
      for _, k in ipairs(wide) do
        local apart = true
        for _, o in ipairs(out) do
          if o.pos:distance(k.pos) < 80 then apart = false break end
        end
        if apart then
          k.remote = true
          out[#out + 1] = k
          if #out >= 8 then break end
        end
      end
      break
    end
  end
  wideCache = {x = pos.x, y = pos.y, t = realTime, out = out}
  return out, #out > 0
end

-- what the rider is told when a Lumen has been found
local function onTheWayText()
  local t = "A Lumen is on the way, " .. minutesText(ride.eta) .. "."
  local pk = ride.pickup
  if pk and pk.remote then
    t = t .. string.format(" You are outside the service area, so it will meet you at the nearest covered road, %d m away. The spot is marked on your map: walk there.",
      math.floor(playerPos():distance(pk.pos) / 10 + 0.5) * 10)
  end
  return t
end

-- one line about where a point lies against the service area, for the log
local function geoDescribe(x, y)
  if not geofence then return "no service area loaded" end
  if not geofence.hasZones() then return "this map has no service area" end
  local st = geofence.status(x, y)
  if st.inside then
    return string.format("inside %s (zone tier %s, open tier %s)", tostring(st.zone), tostring(st.tier), tostring(st.openTier))
  end
  if st.zone then
    return string.format("in %s, which is closed (opens with %s; open tier %s)", st.zone, tostring(st.opensWith), tostring(st.openTier))
  end
  return string.format("outside every zone (open tier %s)", tostring(st.openTier))
end

-- a refusal: told to the rider and written to the log with the reason
local function geoRefuse(text, ttl)
  log("I", logTag, "geofence: refused: " .. text)
  msg(text, ttl or 7, "warning")
end

-- Why a point is outside the service area, as a sentence for the rider; nil when it is inside (or
-- the map has no service area). you: the point is where the rider stands.
local function outsideText(pos, you)
  if not geofence or not geofence.hasZones() or geofence.served(pos.x, pos.y) then return nil end
  local _, closed, tier = geofence.where(pos.x, pos.y)
  local text
  if closed then
    text = string.format("%s %s, which is outside the Lumen service area for now. It opens with tier %d: %s.",
      you and "You are in" or "That is in", closed.name, tier or 1, geofence.tierName(tier or 1))
  else
    text = (you and "You are" or "That spot is") .. " outside the Lumen service area."
  end
  if you then
    local ok, L, _, d = pcall(roads.closestLink, pos, {public = true, maxDist = 4000})
    if ok and L and d then text = text .. string.format(" The nearest covered road is about %d m away.", math.floor(d / 10 + 0.5) * 10) end
    text = text .. " Driving a Lumen yourself works anywhere."
  end
  return text
end

pickupForPlayer = function(publicOnly)
  return widenedCandidates(playerPos(), publicOnly, 150)[1]
end

local function chooseCar(L, stop, report)
  -- Free cars, nearest to the stop first (straight line). Only the nearest 6 are planned for at first
  -- (each plan is an A*, about 10 ms: 20 cars would be a 200 ms hitch per call); the next 6 only when
  -- none of them can get there. A car on another piece of the service area than the stop has no route
  -- by construction, so it is not planned at all.
  local BATCH = 6
  local free = {}
  for _, car in ipairs(fleet) do
    if car.status == "cruising" and simTime >= car.readyAt then
      local veh = carVeh(car)
      if veh then
        local p = veh:getPosition()
        free[#free + 1] = {car = car, veh = veh, pos = p, d = p:distance(stop)}
      end
    end
  end
  table.sort(free, function(x, y) return x.d < y.d end)
  local stopPiece = roads.component(L)
  local best, bestTrip
  local i = 1
  while i <= #free and not best do
    for j = i, math.min(i + BATCH - 1, #free) do
      local f = free[j]
      local dir = f.veh:getDirectionVector()
      local piece = roads.componentAt(f.pos, dir)
      local other = (stopPiece and piece and piece ~= stopPiece) or false
      local trip = nil
      if not other then
        trip = roads.planTrip(f.pos, dir, L, stop, {avoid = avoidSet(), speed = f.veh:getVelocity():length()})
      end
      if report then
        -- for the dispatch log: where the car is, which piece of the service area it is on, and what the planner said
        report[#report + 1] = {slot = f.car.slot, pos = f.pos, comp = piece, eta = trip and trip.eta or nil, skipped = other}
      end
      if trip and (not bestTrip or trip.eta < bestTrip.eta) then best, bestTrip = f.car, trip end
    end
    i = i + BATCH
  end
  return best, bestTrip
end

local function anyCarComing()
  for _, car in ipairs(fleet) do
    if car.status ~= "assigned" then return true end
  end
  return false
end

-- sends the nearest free Lumen to the player; false when none is free right now
local function dispatch()
  local pp = playerPos()
  local cands, widened = widenedCandidates(pp, ride.publicOnly, 150)
  if geofence and not ride.geoLogged then
    ride.geoLogged = true
    log("I", logTag, string.format("geofence: pickup request at (%.0f, %.0f), %s; %d covered stop(s) found%s", pp.x, pp.y,
      geoDescribe(pp.x, pp.y), #cands, widened and " by the widened search (the rider is outside the service area)" or ""))
  end
  if #cands == 0 then
    log("I", logTag, "geofence: pickup refused: no covered road within 6 km of the rider")
    return false, "noroad"
  end
  local free = 0
  for _, c in ipairs(fleet) do
    if c.status == "cruising" and simTime >= c.readyAt then free = free + 1 end
  end
  if free == 0 then return false, "busy" end
  -- the closest spot that a free Lumen can actually drive to (Storm, 2026-10-31: "Its right
  -- there dude" -- the one spot tried was a lot lane no road connects to, and that read as busy)
  local function pieceText(comp)
    local info = comp and roads.pieceInfo(comp)
    if not info then return "no served piece" end
    return string.format("piece %s (%.1f km around %.0f, %.0f%s)", tostring(comp), info.km, info.x, info.y, info.main and ", the main one" or "")
  end
  local pk, car, trip
  local reports = {}
  for i = 1, math.min(#cands, 6) do
    reports[i] = {}
    car, trip = chooseCar(cands[i].link, cands[i].pos, reports[i])
    if car then
      pk = cands[i]
      break
    end
  end
  if not car then
    -- every candidate stop with every free car: where each is, and why there is no route
    log("I", logTag, string.format("dispatch: none of %d free car(s) can plan a route to any of the %d nearest stop(s) for the rider at (%.0f, %.0f)",
      free, math.min(#cands, 6), pp.x, pp.y))
    for i = 1, math.min(#cands, 6) do
      local c = cands[i]
      local parts = {}
      for _, r in ipairs(reports[i]) do
        parts[#parts + 1] = string.format("car %d at (%.0f, %.0f) on %s: %s", r.slot, r.pos.x, r.pos.y,
          r.comp and ("piece " .. tostring(r.comp)) or "no served road",
          r.eta and string.format("route of %.0f s", r.eta) or (r.skipped and "on another piece, no route (not planned)" or "no route"))
      end
      log("I", logTag, string.format("dispatch: stop %d at (%.0f, %.0f) is on %s%s; %s", i, c.pos.x, c.pos.y, pieceText(roads.component(c.link)),
        c.remote and " (remote)" or "", #parts > 0 and table.concat(parts, "; ") or "no free car was tried"))
    end
  end
  local pending, farEta = nil, trip and trip.eta
  if not car or trip.eta > MAX_DISPATCH_ETA then
    -- every free Lumen is far by road, or none can get here at all (stranded on a piece of road
    -- that does not connect): bring a hidden one to a spot about a minute and a half away
    local spare
    for _, c in ipairs(fleet) do
      local v = carVeh(c)
      if c.status == "cruising" and simTime >= c.readyAt and v and roads.isHidden(v:getPosition(), HIDDEN_BEYOND) and (not spare or c == car) then spare = c end
    end
    if spare then
      local order = pk and {pk} or cands
      for i = 1, math.min(#order, 3) do
        local pos, dir, up, near = spotNear(order[i].link, order[i].pos, NEAR_ETA, otherCarPositions(spare))
        if pos then
          drive.release(spare)
          spawn.safeTeleport(carVeh(spare), pos, quatFromDir(dir, up), true, nil, false)
          car, trip, pending, pk = spare, near, near, order[i]
          car.readyAt = simTime + 0.8
          resetHealth(car)
          thoughts = {}
          think(farEta and string.format("The closest free Lumen was %s away by road, so I brought one closer from out of sight.", minutesText(farEta))
            or "No free Lumen could reach you from where it was, so I brought one closer from out of sight.")
          break
        end
      end
    end
  end
  -- Dispatch never refuses because of the fence or an island (Storm, 2026-11-01: "No Lumen can reach the
  -- roads right here", 3 free): the free car FARTHEST from the rider, hidden or not, is moved to a spot
  -- on the pickup's own piece of road and dispatched from there. Only "no free car" refuses.
  if not car then
    local order = {}
    for _, c in ipairs(fleet) do
      local v = carVeh(c)
      if c.status == "cruising" and simTime >= c.readyAt and v then order[#order + 1] = {car = c, d = v:getPosition():distance(pp)} end
    end
    table.sort(order, function(x, y) return x.d > y.d end)
    local function spotFor(stopC, avoidPos)
      local comp = roads.component(stopC.link)
      for pass = 1, 2 do
        for _ = 1, 8 do
          local sp, sd, su = roads.findSpot(stopC.pos, pass == 1 and 120 or 40, pass == 1 and 900 or 1800,
            {avoid = avoidPos, spacing = pass == 1 and 100 or 30, farEnough = pass == 1 and HIDDEN_BEYOND or 0, component = comp})
          if not sp then break end
          local t = roads.planTrip(sp, sd, stopC.link, stopC.pos, {avoid = avoidSet()})
          if t then return sp, sd, su, t end
        end
      end
      -- last resort: the stop's own road, 45 m before the stop in the direction of travel
      local L = stopC.link
      local u = (L.pb - L.pa):normalized()
      for _, sgn in ipairs({1, -1}) do
        local from, to = (sgn == 1) and L.a or L.b, (sgn == 1) and L.b or L.a
        local sp = stopC.pos - u * (45 * sgn)
        local along = (sp - L.pa):dot(u)
        if roads.legal(L, from, to) and along > 2 and along < L.len - 2 then
          local h = be:getSurfaceHeightBelow(sp + vec3(0, 0, 3))
          if h and h > -1e6 then sp = vec3(sp.x, sp.y, h) end
          local sd = u * sgn
          local t = roads.planTrip(sp, sd, stopC.link, stopC.pos, {avoid = avoidSet()})
          if t then return sp, sd, vec3(0, 0, 1), t end
        end
      end
      return nil
    end
    for _, o in ipairs(order) do
      for i = 1, math.min(#cands, 3) do
        local stopC = cands[i]
        local sp, sd, su, st = spotFor(stopC, otherCarPositions(o.car))
        if sp then
          local v = carVeh(o.car)
          local oldPiece = pieceText(roads.componentAt(v:getPosition(), v:getDirectionVector()))
          drive.release(o.car)
          spawn.safeTeleport(v, sp, quatFromDir(sd, su), true, nil, false)
          car, trip, pending, pk = o.car, st, st, stopC
          car.readyAt = simTime + 0.8
          resetHealth(car)
          thoughts = {}
          think(string.format("No free Lumen could drive to you from where it was, so I moved the farthest one (%d m from you) onto your road network.", math.floor(o.d)), "go")
          log("I", logTag, string.format("dispatch: moved car %d, the FARTHEST free car (%.0f m from the rider, was on %s), to (%.0f, %.0f) on %s so it can reach the stop at (%.0f, %.0f)",
            car.slot, o.d, oldPiece, sp.x, sp.y, pieceText(roads.component(stopC.link)), stopC.pos.x, stopC.pos.y))
          break
        end
      end
      if car then break end
    end
  end
  if not car then
    log("I", logTag, string.format("dispatch: refused: %d free car(s) but none could be placed on a route to any of the %d nearest stop(s)", free, math.min(#cands, 6)))
    return false, "unreachable"
  end
  ride.pickup = pk
  car.status = "assigned"
  car.needsCruise = false
  ride.slot = car.slot
  ride.recoveries = 0
  ride.trip = trip
  ride.eta = trip.eta
  setState("arriving")
  if pending then car.pendingTrip = pending else driveTrip(car, trip) end
  log("I", logTag, string.format("dispatched fleet car %d (%d): %.0f m, about %.0f s", car.slot, car.id, trip.remaining, trip.eta))
  if not pending then thoughts = {} end
  think(string.format("Picked the closest free Lumen by driving time: %.1f km away, %s.", trip.remaining / 1000, minutesText(trip.eta)))
  if trip.committed then think("It is moving too fast for the next turn, so it will carry on to the next intersection and turn there.") end
  if geofence then
    log("I", logTag, string.format("geofence: pickup stop at (%.0f, %.0f), %s, %.0f m from the rider: %s%s", pk.pos.x, pk.pos.y,
      geoDescribe(pk.pos.x, pk.pos.y), pk.pos:distance(playerPos()), pk.remote and "moved to the nearest covered road" or "allowed",
      pk.private and " (a parking-lot lane)" or ""))
  end
  if pk.private then think("You are in a parking lot, so I will drive in to you.") end
  if pk.moved then think("A parked car is on the spot nearest you, so I will stop just clear of it.") end
  if pk.remote then
    local d = playerPos():distance(pk.pos)
    pk.waitExtra = math.min(900, d / 1.5)          -- the rider has to walk there: the Lumen waits that much longer
    think(string.format("You are outside the service area, so I will meet you at the nearest covered road, %d m away. The spot is marked on your map: walk there.",
      math.floor(d / 10 + 0.5) * 10), "go")
    log("I", logTag, string.format("pickup moved to the nearest covered road, %.0f m from the rider", d))
  end
  return true
end

local function releaseCar(car)
  if not car or car.status ~= "assigned" then return end
  setUsable(car, false)
  car.status = "cruising"
  car.needsCruise = true
  car.readyAt = simTime + 1.5          -- give the rider a moment to step away
end

-- Sit in any seat (Storm, 2026-10-31: "I can't sit in any seat I want"). The car has an
-- interior camera per seat; the app and the radial menu switch between them, and the camera
-- key cycles them too. The last seat chosen is remembered.
local SEATS = {
  {id = "rear_right", name = "Rear right"}, {id = "rear_left", name = "Rear left"},
  {id = "front_passenger", name = "Front passenger"}, {id = "driver", name = "Driver's seat"},
}
local seat = "rear_right"

local function setPassengerCamera()
  local ok = pcall(core_camera.setByName, 0, "onboard." .. seat)
  if not ok then pcall(core_camera.setByName, 0, "driver") end
end

function M.setSeat(id)
  for _, s in ipairs(SEATS) do
    if s.id == id then
      seat = id
      local inside = ride and ride.slot and rideCar() and rideCar().id and getPlayerVehicle(0) and getPlayerVehicle(0):getID() == rideCar().id
      if inside then setPassengerCamera() end
      return true
    end
  end
  return false
end

local function startRide(quiet)
  local car = rideCar()
  local veh = carVeh(car)
  if not veh or not ride.dest then return false end
  local trip = roads.planTrip(veh:getPosition(), veh:getDirectionVector(), ride.dest.link, ride.dest.pos, {avoid = avoidSet()})
  if not trip then
    if not quiet then msg("Lumen can't find a road route there. Pick another destination.", 6, "warning") end
    return false
  end
  ride.trip = trip
  ride.eta = trip.eta
  ride.lastPos = veh:getPosition()
  if not ride.fareStarted then
    ride.fareStarted = true
    ride.distance, ride.rideTime = 0, 0
  end
  setState("riding")
  -- Parked in a space: back out first (lumen/parking.lua), then the route is planned again from where the car
  -- ends up (M._parkEvent). This plan only answers "is there a route" and gives the ETA.
  if parking.holds(car) then
    if parking.leave(car, veh) then
      if not ride.leaving then
        ride.leaving = true
        think(string.format("Backing out of the space first, then the route to %s: %.1f km, %s.", ride.dest.name, trip.remaining / 1000, minutesText(trip.eta)), "go")
        msg(string.format("Heading to %s, about %s.", ride.dest.name, minutesText(trip.eta)), 6)
      end
      return true
    end
  end
  ride.leaving = nil
  if ride.parkTried == "riding" then ride.parkTried = nil end      -- a new destination gets its own space
  driveTrip(car, trip)
  think(string.format("Route to %s planned: %.1f km on public roads, %s.", ride.dest.name, trip.remaining / 1000, minutesText(trip.eta)), "go")
  msg(string.format("Heading to %s, about %s.", ride.dest.name, minutesText(trip.eta)), 6)
  return true
end

local function onBoarded()
  if not ride or ride.state ~= "waiting" then return end
  local car = rideCar()
  local veh = carVeh(car)
  if not veh then return end
  lockPlayerControls(veh, true)
  setPassengerCamera()
  carCommand(car, "electrics.set_warn_signal(false)")
  setState("boarded")
  if ride.resume and ride.dest then
    msg("Welcome back. Continuing to " .. ride.dest.name .. ".", 5)
    startRide()
  else
    msg("Welcome aboard. Choose a destination.", 5)
  end
end

local function exitPlayer(car)
  local veh = carVeh(car)
  local pv = getPlayerVehicle(0)
  if not veh or not pv or pv:getID() ~= car.id then return end
  lockPlayerControls(veh, false)
  ride.exiting = true
  local ok = false
  local w = walkExt()
  if w and w.setWalkingMode then ok = w.setWalkingMode(true, nil, nil, true) end
  if not ok and ride.playerOwnVeh and getObjectByID(ride.playerOwnVeh) then
    be:enterVehicle(0, getObjectByID(ride.playerOwnVeh))
  end
  ride.exiting = false
end

local function finishRide(text, why)
  log("I", logTag, "ride finished: " .. tostring(why or "trip complete"))
  local car = rideCar()
  if car then releaseCar(car) end
  ride.slot, ride.trip = nil, nil
  setState("departing")
  msg(text or string.format("Thanks for riding with Lumen. Fare $%.2f.", ride.fare or 0), 6)
end

-- the ride's car went to service (stuck, wrecked, flipped) or vanished
rideCarLost = function(car, reason)
  if not ride then return end
  local inside = car.id and getPlayerVehicle(0) and getPlayerVehicle(0):getID() == car.id
  if inside then exitPlayer(car) end
  local wasAboard = ride.state == "boarded" or ride.state == "riding" or ride.state == "pulling" or ride.state == "pulledOver"
  if ride.state == "arrived" then
    ride.slot, ride.trip = nil, nil
    setState("departing")
    return
  end
  if ride.state == "arriving" and ride.pickup and ride.pickup.private then ride.publicOnly = true end
  ride.slot, ride.trip = nil, nil
  ride.parkTried, ride.leaving = nil, nil
  if wasAboard then
    ride.resume = ride.dest ~= nil
    if ride.dest and ride.dest.private and ride.dest.target then
      local P, _, _, tp = roads.closestLink(ride.dest.target, {public = true, maxDist = 400})
      if P then
        ride.dest = {link = P, pos = roads.stopPoint(P, tp), name = ride.dest.name, target = ride.dest.target}
        think("That parking lot was too tricky, so the next Lumen will drop you at the street instead.", "warn")
      end
    end
    ride.fare = computeFare()
  end
  setState("dispatching")
  local what = reason == "damaged" and "Your Lumen was in a collision" or reason == "flipped" and "Your Lumen rolled over"
    or reason == "unavailable" and "Your Lumen is no longer available" or "Your Lumen got stuck"
  local ok = dispatch()
  local tail
  if ok then
    tail = (ride.resume and " Another Lumen is coming to finish your trip, " or " Another Lumen is on the way, ") .. minutesText(ride.eta) .. "."
  else
    tail = " All Lumens are busy; the next free one is yours."
  end
  msg(what .. (reason == "unavailable" and "." or " and is getting serviced.") .. tail, 9, "build")
end

-- Park in a free marked space near the stop (lumen/parking.lua), whichever driver is selected. true: the
-- maneuver is running (M._parkEvent calls arrive() again once the car is in, or has given up); "wait": the car
-- is still rolling and brakes to a stop first (arrive() comes round again); false: nothing to do (no data, no
-- free space with a clear path, parking off).
function M._parkBegin(car)
  local veh = carVeh(car)
  if not veh or not ride or car.park then return false end
  local aim = (ride.state == "arriving" and ride.pickup and ride.pickup.pos) or (ride.state == "riding" and ride.dest and ride.dest.pos)
  if not aim then return false end
  if veh:getVelocity():length() > 2.0 then
    if drive.mode(car) ~= "hold" then drive.stop(car) end
    return "wait"
  end
  local ok = parking.start(car, veh, aim, {
    what = ride.state == "arriving" and "pickup" or "drop-off", nn = drive.brainOf(car) == "nn",
    say = function(text, kind) think(text, kind) end,
  })
  if ok then
    carCommand(car, "electrics.set_warn_signal(false)")
    resetHealth(car)
    return true
  end
  return false
end

local function arrive()
  local car = rideCar()
  -- a free marked space near the stop first, if there is one
  if (ride.state == "arriving" or ride.state == "riding") and ride.parkTried ~= ride.state then
    local r = M._parkBegin(car)
    if r == "wait" then return end
    ride.parkTried = ride.state
    if r then return end
  end
  if ride.state == "arriving" then
    stopCar(car, true)
    carCommand(car, HONK_DOUBLE)
    think("At the pickup. Holding on the parking brake with hazards on, and a beep so you know I am here.", "go")
    setUsable(car, true)
    setState("waiting")
    msg("Your Lumen is here. Walk up and get in, or choose Board.", 8)
  elseif ride.state == "riding" then
    stopCar(car, true)
    ride.fare = computeFare()
    think("Arrived. Stopped clear of the junction with hazards on.", "go")
    setState("arrived")
    msg(string.format("You've arrived at %s. Fare $%.2f. Get out when you're ready.", ride.dest and ride.dest.name or "your destination", ride.fare), 10)
  end
end

-- lumen/parking.lua reports: "parked" (stopped in the space), "failed" (gave up; the car holds where it is) and
-- "left" (out of the space, or the back-out was given up)
function M._parkEvent(car, ev)
  local mine = ride and ride.slot == car.slot
  if ev == "left" then
    if mine and ride.leaving then
      ride.leaving = nil
      if ride.state == "riding" and not startRide(true) then
        -- no route from where the car ended up: the plan made before backing out is the best there is
        local pv = carVeh(car)
        if pv and ride.trip then driveTrip(car, ride.trip) end
      end
    elseif car.status == "cruising" then
      car.needsCruise = true
    end
    return
  end
  -- parked or failed: the stop that arrive() was waiting for
  if mine and (ride.state == "arriving" or ride.state == "riding") then arrive() end
end

local function updateDrive(car, veh, dt, dtReal)
  local pos = veh:getPosition()
  local trip = ride.trip
  roads.measure(trip, pos)
  -- Traffic-aware ETA (Storm: "Does the eta take traffic into account?"). The route estimate
  -- assumes about 82 % of the limit; when the car has recently been slower than that (queues,
  -- lights), the next few hundred metres stretch by the same factor, and slow vehicles waiting
  -- on the route ahead add time. The shown ETA counts down and eases toward the result.
  local speedNow = veh:getVelocity():length()
  local vexp = trip.segV[trip.idx] or 11
  ride.avgSpeed = ride.avgSpeed and (ride.avgSpeed + (speedNow - ride.avgSpeed) * math.min(1, dtReal / 20)) or vexp
  ride.queueCheckT = (ride.queueCheckT or 0) - dtReal
  if ride.queueCheckT <= 0 then
    ride.queueCheckT = 1
    ride.queueDelay = roads.queueDelay(trip, pos, car.id)
  end
  local flow = clamp(vexp / math.max(ride.avgSpeed, 1), 1, 2.5)
  local extra = (math.min(trip.remaining, 400) / vexp) * (flow - 1) + (ride.queueDelay or 0)
  ride.trafficDelay = (ride.trafficDelay or 0) + (extra - (ride.trafficDelay or 0)) * math.min(1, dtReal * 0.25)
  local wasSlow = ride.traffic
  ride.traffic = ride.trafficDelay > 15
  if ride.traffic and not wasSlow then
    think(string.format("Slow traffic on the route. Adding about %d s to the ETA.", math.floor(ride.trafficDelay + 0.5)), "wait")
  end
  local target = trip.eta + ride.trafficDelay
  ride.eta = ride.eta and math.max(0, ride.eta - dtReal) or target
  ride.eta = ride.eta + (target - ride.eta) * math.min(1, dtReal * 0.5)
  if ride.state == "riding" then
    if ride.lastPos then ride.distance = ride.distance + pos:distance(ride.lastPos) end
    ride.lastPos = pos
    ride.rideTime = ride.rideTime + dt
    ride.fare = computeFare()
  end
  -- the player walked or drove away from the pickup: move the pickup to them
  if ride.state == "arriving" then
    ride.repickT = ride.repickT + dtReal
    local remote = ride.pickup.remote
    if ride.repickT > (remote and 8 or 3) and (remote or playerPos():distance(ride.pickup.pos) > REPICK_DIST) then
      ride.repickT = 0
      local pk = pickupForPlayer(ride.publicOnly)
      -- a remote pickup moves only when the nearest covered spot to the rider has moved
      if pk and remote and pk.pos:distance(ride.pickup.pos) < 60 then pk = nil end
      local trip2 = pk and roads.planTrip(pos, veh:getDirectionVector(), pk.link, pk.pos, {avoid = avoidSet(), speed = veh:getVelocity():length()})
      if trip2 then
        ride.pickup, ride.trip = pk, trip2
        ride.parkTried = nil                 -- a new pickup gets its own space
        driveTrip(car, trip2)
        msg("Your Lumen is heading to your new location.", 4)
        return
      end
    end
  end
  -- off the planned route (avoidance, a missed turn): plan again from here
  ride.replanT = ride.replanT - dtReal
  if trip.offRoute > OFF_ROUTE and ride.replanT <= 0 then
    ride.replanT = 3
    think("I am off the planned route. Planning again from here.", "warn")
    local target = ride.state == "arriving" and ride.pickup or ride.dest
    local trip2 = roads.planTrip(pos, veh:getDirectionVector(), target.link, target.pos, {avoid = avoidSet(), speed = veh:getVelocity():length()})
    if trip2 then
      ride.trip = trip2
      driveTrip(car, trip2)
      return
    end
  end
  local onLast = trip.idx >= #trip.points - 1
  local toStop = pos:distance(trip.stop)
  local speed = veh:getVelocity():length()
  local dir = veh:getDirectionVector()
  -- lined up with the stop road: never "arrive" half way round a corner
  local aligned = math.abs(dir:dot(trip.stopDir)) > 0.85
  local passed = onLast and aligned and toStop < 15 and (trip.stop - pos):dot(dir) < 0
  -- at the stop point, wherever it falls on the route: a route planned again after a detour can
  -- pass the stop early and loop round the block back to it (Storm, 2026-10-31: "it routed over
  -- the pickup point")
  local atStop = aligned and toStop < 8 and not onLast
  if drive.arrived(car) or atStop or passed or (onLast and aligned and toStop < 12 and speed < 0.8) then
    arrive()
    return
  end
  -- stopped close to the stop point (the AI finished its path short of it, or came in from a
  -- side lane in a parking lot): close enough counts. So does standing behind a vehicle that has
  -- not moved for 3 s right by the stop: never creep into a parked car to reach the exact spot.
  local parkedAhead = car.inTraffic and not car.signal and car.aheadStillT >= 3 and toStop < 25
  if trip.idx >= #trip.points - 2 and toStop < 30 and speed < 0.6 and (not car.inTraffic or parkedAhead) then
    ride.stopT = (ride.stopT or 0) + dtReal
    if ride.stopT > 2.5 then
      if parkedAhead then think("A parked car is on my stop, so I stopped right behind it instead of squeezing in.", "go") end
      arrive()
    end
  else
    ride.stopT = 0
  end
end

-- ---------------------------------------------------------------------------------
-- blocked: back up (only with nothing close behind) and take another way
-- ---------------------------------------------------------------------------------
-- The road the car is blocked on is remembered for BLOCKED_MEMORY seconds and every route the
-- fleet plans keeps off it, so a Lumen never circles back into the same blockage. From here it
-- turns around (starts at the node behind it) unless that is far longer than another way.
local function alternativeTrip(veh, target)
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  local CL = roads.closestLink(pos, {maxDist = 60, dir = dir})
  -- Within 60 m of the stop point a blockage is a local problem (a curb, a parked car at the spot),
  -- not a closed road: do not blacklist the street for 5 minutes. 2026-11-01: a curb hit at a pickup
  -- put DR294_5 - DR294_4 on the blocked list for every Lumen, which cuts later pickups off.
  if CL and CL ~= target.link and pos:distance(target.pos) > 60 then
    blockedRoads[roads.linkKey(CL.a, CL.b)] = simTime + BLOCKED_MEMORY
    log("I", logTag, "remembering a blocked road " .. CL.a .. " - " .. CL.b .. " for " .. BLOCKED_MEMORY .. " s")
    think("Remembering this road as blocked for " .. math.floor(BLOCKED_MEMORY / 60) .. " minutes. Every Lumen will route around it.", "warn")
  end
  local avoid = avoidSet()
  local behind = CL and (((CL.pb - CL.pa):dot(dir) >= 0) and CL.a or CL.b)
  local turned = behind and roads.planTrip(pos, dir, target.link, target.pos, {startNode = behind, avoid = avoid})
  local normal = roads.planTrip(pos, dir, target.link, target.pos, {avoid = avoid})
  if turned and (not normal or turned.eta < normal.eta * 2.5) then return turned end
  return normal
end

recover = function(car, veh)
  drive.abCount(car, "blocked")
  car.recoveries = car.recoveries + 1
  if ride and ride.slot == car.slot then ride.recoveries = (ride.recoveries or 0) + 1 end
  local pos = veh:getPosition()
  car.recoverPos = pos
  local carBehind = vehicleNear(veh, 9, true) ~= nil
  local room = roomBehind(veh, BACKUP_DIST + 2)
  local backUp = not carBehind and room >= 3.6
  car.recovering = {t = 0, backUp = backUp, start = pos, lastPos = pos, lastMove = 0, checkT = 0}
  car.readyAt = simTime + BACKUP_MAX_TIME + 3
  if backUp then
    drive.release(car)
    -- the controller's reverse speed loop: slow and steady, and it stops when told
    carCommand(car, "ai.setMode('disabled') input.event('parkingbrake', 0, 2, nil, nil, nil, 'ai') electrics.set_warn_signal(true) " ..
      string.format("local c = controller.getController('lumenRobotaxi') if c and c.reverseAt then c.reverseAt(%.2f) end", BACKUP_SPEED))
  end
  log("I", logTag, string.format("fleet car %d blocked (attempt %d, %s): %s", car.slot, car.recoveries, tostring(car.blockCause),
    backUp and "backing up, then another way" or string.format("taking another way (%.1f m free behind, car behind: %s)", room, tostring(carBehind))))
  if ride and ride.slot == car.slot then
    local why = car.blockCause == "jam" and string.format("The queue ahead (%d cars) has not moved for %d s and nothing is holding its front car, no light and no stop sign. The road is blocked.", car.jamCars or 1, JAM_TIME)
      or car.blockCause == "parked" and "A parked car is standing in my lane. Nobody is driving it, so it will not move."
      or car.blockCause == "obstacle" and "A wall or a post is on my path, so I will not drive into it."
      or "Something is in the way and nothing explains it (no car ahead, no signal)."
    local how = backUp and " Backing up a little, then taking another way."
      or carBehind and " A car is close behind, so I will not back up. Taking another way."
      or string.format(" Only %.1f m is free behind me (a rail, a wall or a drop), so I will not back up. Taking another way.", room)
    think(why .. how, "warn")
  end
  car.blockCause, car.jamCars = nil, nil
  if ride and ride.slot == car.slot and (ride.state == "arriving" or ride.state == "riding") then
    msg(backUp and "Something is blocking the road. Your Lumen is backing up to find another way." or "Your Lumen is finding another way.", 5)
  end
end

-- A vehicle ahead is backing towards us: back away at its speed while there is room, honk once
-- to warn it, and carry on with the route once it stops.
local function stopEvading(car, why)
  carCommand(car, "local c = controller.getController('lumenRobotaxi') if c and c.reverseAt then c.reverseAt(0) end")
  drive.stop(car)
  car.evading = nil
  resetHealth(car)
  car.readyAt = simTime + 1.5
  local mine = ride and ride.slot == car.slot
  local veh = carVeh(car)
  if mine and ride.state == "waiting" then
    carCommand(car, "electrics.set_warn_signal(true)")
  elseif mine and veh and (ride.state == "arriving" or ride.state == "riding") then
    local target = ride.state == "arriving" and ride.pickup or ride.dest
    local trip = roads.planTrip(veh:getPosition(), veh:getDirectionVector(), target.link, target.pos, {avoid = avoidSet()})
    if trip then
      ride.trip = trip
      driveTrip(car, trip)
    end
  else
    car.needsCruise = true
  end
  if mine then think(why, "go") end
end

updateEvasion = function(car, veh, dt)
  local threat = reversingThreat(veh)
  local mine = ride and ride.slot == car.slot
  if not car.evading then
    if not threat then return end
    local room = vehicleNear(veh, EVADE_ROOM, true) == nil and roomBehind(veh, 4) >= 3
    if car.honkAt < simTime - 4 then
      car.honkAt = simTime
      carCommand(car, HONK_SHORT)
    end
    if not room then
      if mine then think("The car ahead is reversing towards us but there is no room behind. Holding still and honking to warn it.", "warn") end
      return
    end
    local speed = math.min(EVADE_MAX_SPEED, threat)
    car.evading = {t = 0, start = veh:getPosition(), speed = speed, calm = 0}
    drive.release(car)
    carCommand(car, "ai.setMode('disabled') input.event('parkingbrake', 0, 2, nil, nil, nil, 'ai') electrics.set_warn_signal(true) " ..
      string.format("local c = controller.getController('lumenRobotaxi') if c and c.reverseAt then c.reverseAt(%.2f) end", speed))
    log("I", logTag, string.format("fleet car %d backing away from a reversing vehicle at %.1f m/s", car.slot, speed))
    if mine then think(string.format("The car ahead is reversing towards us at %.1f m/s. Backing up at the same speed.", speed), "warn") end
    return
  end
  local e = car.evading
  e.t = e.t + dt
  if threat then
    e.calm = 0
    local speed = math.min(EVADE_MAX_SPEED, threat)
    if math.abs(speed - e.speed) > 0.3 then
      e.speed = speed
      carCommand(car, string.format("local c = controller.getController('lumenRobotaxi') if c and c.reverseAt then c.reverseAt(%.2f) end", speed))
    end
  else
    e.calm = e.calm + dt
  end
  if e.calm > 1.0 then
    stopEvading(car, "The car ahead stopped reversing. Carrying on.")
  elseif veh:getPosition():distance(e.start) > EVADE_MAX_DIST then
    stopEvading(car, "Backed up " .. EVADE_MAX_DIST .. " m, the most I will. Stopping here.")
  elseif vehicleNear(veh, 4, true) ~= nil then
    stopEvading(car, "Something is close behind now. Stopping.")
  elseif roomBehind(veh, 2) < 1.2 then
    stopEvading(car, "A rail, a wall or a drop is right behind me now. Stopping.")
  end
end

-- The way to the pickup spot is blocked (Storm, 2026-11-01: "Can the Lumen change the pickup spot if it is
-- blocked?"; the test instance the same morning: a Lumen stood five minutes beside the rider's own parked car).
-- When the blockage is close to the spot, or this ride has been blocked twice, meet the rider at the next spot
-- that the car can reach instead of fighting its way to the same one. Returns the new trip, or nil to keep the spot.
local function movePickup(car, veh)
  local pk = ride.pickup
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  if not pk or (pos:distance(pk.pos) > 150 and (ride.recoveries or 0) < 2) then return nil end
  local pp = playerPos()
  local cands = widenedCandidates(pp, ride.publicOnly, 150)
  for i = 1, math.min(#cands, 8) do
    local c = cands[i]
    if c.pos:distance(pk.pos) > 15 and c.pos:distance(pos) > 10 then
      local trip = roads.planTrip(pos, dir, c.link, c.pos, {avoid = avoidSet()})
      if trip then
        local d = math.floor(c.pos:distance(pp) / 5 + 0.5) * 5
        log("I", logTag, string.format("pickup moved: the way to (%.0f, %.0f) was blocked %.0f m from it; new spot (%.0f, %.0f), %d m from the rider, %.0f m of route",
          pk.pos.x, pk.pos.y, pos:distance(pk.pos), c.pos.x, c.pos.y, d, trip.remaining))
        ride.pickup = c
        ride.parkTried = nil                               -- lumen/parking.lua: the new spot gets its own parking attempt
        think(string.format("The way to your pickup spot is blocked, so I will meet you at another spot, %d m from you.", d), "go")
        msg(string.format("Pickup moved: your Lumen will meet you %d m from you. The new spot is on your map.", d), 8, "info")
        return trip
      end
    end
  end
  return nil
end

updateRecovery = function(car, veh, dt)
  local r = car.recovering
  r.t = r.t + dt
  if r.backUp then
    local pos = veh:getPosition()
    local done = r.t > BACKUP_MAX_TIME or pos:distance(r.start) > BACKUP_DIST
    r.checkT = r.checkT + dt
    if not done and r.checkT >= 0.1 then
      r.checkT = 0
      -- stop with a margin: anything static within 1.2 m, a vehicle within 6 m centre to centre
      if roomBehind(veh, 2) < 1.2 or vehicleNear(veh, 6, true) ~= nil then done = true end
      -- reversing but not moving: something solid behind that the rays missed
      if pos:distance(r.lastPos) > 0.25 then r.lastPos, r.lastMove = pos, r.t end
      if r.t - r.lastMove > 1.5 then done = true end
    end
    if not done then return end
    carCommand(car, "local c = controller.getController('lumenRobotaxi') if c and c.reverseAt then c.reverseAt(0) end")
  end
  car.recovering = nil
  carCommand(car, "input.event('brake', 0, 2, nil, nil, nil, 'ai') electrics.set_warn_signal(false)")
  resetHealth(car)
  if car.status == "assigned" and ride and ride.slot == car.slot and (ride.state == "arriving" or ride.state == "riding") then
    local target = ride.state == "arriving" and ride.pickup or ride.dest
    local trip = ride.state == "arriving" and movePickup(car, veh) or nil
    trip = trip or alternativeTrip(veh, target)
    if trip then
      ride.trip = trip
      ride.eta = trip.eta
      driveTrip(car, trip)
    else
      sendToService(car, "stuck")
    end
  else
    cruise(car)
  end
end

local function updateRide(dt, dtReal)
  ride.stateT = ride.stateT + dtReal
  local s = ride.state
  if s == "dispatching" then
    ride.retryT = ride.retryT - dtReal
    if ride.retryT <= 0 then
      ride.retryT = 2
      local ok, why = dispatch()
      if ok then
        msg(onTheWayText(), 9)
      elseif why == "noroad" or why == "unreachable" then
        geoRefuse(why == "noroad" and (outsideText(playerPos(), true) or "No road nearby. Move closer to a road and call again.")
          or "No Lumen can reach the roads right here. Move to a bigger road and call again.", 7)
        ride = nil
      elseif not anyCarComing() then
        msg("No Lumens are available on this map right now.", 6, "warning")
        ride = nil
      end
    end
    return
  end
  local car = rideCar()
  local veh = carVeh(car)
  if s == "departing" then
    if ride.stateT > 6 then ride = nil end
    return
  end
  if not veh then
    if car then rideCarLost(car, "unavailable") else ride = nil end
    return
  end
  if s == "pulling" then
    M._updatePulling(car, veh, dtReal)
    if ride.lastPos then ride.distance = ride.distance + veh:getPosition():distance(ride.lastPos) end
    ride.lastPos = veh:getPosition()
    ride.rideTime = ride.rideTime + dt
    ride.fare = computeFare()
  elseif s == "arriving" or s == "riding" then
    -- (the parking maneuver drives the car, not the route: lumen/parking.lua)
    if not parking.busy(car) then updateDrive(car, veh, dt, dtReal) end
  elseif s == "waiting" then
    if ride.stateT > WAIT_TIMEOUT + ((ride.pickup and ride.pickup.waitExtra) or 0) then
      releaseCar(car)
      ride = nil
      msg("Your Lumen couldn't wait any longer and has left. Call again any time.", 7)
    end
  end
end

-- ---------------------------------------------------------------------------------
-- public API (UI app, radial menu, input actions)
-- ---------------------------------------------------------------------------------
function M.callTaxi()
  if not isFreeroam() then
    msg("Robotaxis are available in Freeroam.", 5, "info")
    return false
  end
  if not enabled then
    msg("The robotaxi service is paused. Turn it on in the Robotaxi menu.", 5, "info")
    return false
  end
  if not active then startFleet() end
  if not active then
    msg("Robotaxis need a map with roads.", 5, "info")
    return false
  end
  if ride and ride.state ~= "departing" then
    msg(ride.state == "waiting" and "Your Lumen is already here." or "You already have a Lumen.", 4)
    return false
  end
  ride = newRide()
  thoughts = {}
  local ok, why = dispatch()
  if ok then
    msg(onTheWayText(), 9)
    return true
  end
  if why == "noroad" then
    ride = nil
    geoRefuse(outsideText(playerPos(), true) or "No road nearby. Move closer to a road and call again.", 7)
    return false
  end
  if why == "unreachable" then
    ride = nil
    msg("No Lumen can reach the roads right here. Move to a bigger road and call again.", 7, "warning")
    return false
  end
  think("Every Lumen has a rider or is being serviced. The first one free comes to you.", "wait")
  msg("All Lumens are busy right now. The next free one is yours.", 6)
  return true
end

function M.board()
  if not ride or ride.state ~= "waiting" then
    msg("Your Lumen isn't here yet.", 3)
    return false
  end
  local car = rideCar()
  local veh = carVeh(car)
  if not veh then return false end
  if playerPos():distance(veh:getPosition()) > BOARD_DIST then
    msg("Walk up to your Lumen to get in.", 4)
    return false
  end
  local w = walkExt()
  if w and w.isWalking and w.isWalking() then
    w.getInVehicle(veh)
  else
    local own = getPlayerVehicle(0)
    ride.playerOwnVeh = own and own:getID() or nil
    be:enterVehicle(0, veh)
  end
  onBoarded()
  return true
end

function M.setDestination(id)
  local s = ride and ride.state
  if s ~= "boarded" and s ~= "riding" and s ~= "arrived" and s ~= "pulledOver" then return false end
  local pos, name = resolveDestination(id)
  if not pos then
    msg("That destination isn't available. Pick another one.", 5, "warning")
    return false
  end
  -- a destination inside a parking lot: drive in, like pickups (Storm: "Can they go in parking
  -- lots?"); otherwise the nearest public road the car can actually reach
  local cands = stopCandidates(pos, false, 600)
  -- a spot outside the service area is refused unless a covered road is right beside it (150 m)
  local nearD = cands[1] and cands[1].pos:distance(pos) or nil
  local outside = outsideText(pos, false)
  log("I", logTag, string.format("geofence: destination '%s' (%s) at (%.0f, %.0f), %s; nearest covered road %s", tostring(id), tostring(name),
    pos.x, pos.y, geoDescribe(pos.x, pos.y), nearD and string.format("%.0f m away", nearD) or "none within 600 m"))
  -- Outside the service area (or no covered road near it): the nearest covered road is used, with
  -- a note, instead of a refusal (Storm, 2026-11-01).
  local movedDest = false
  if #cands == 0 then
    cands = widenedCandidates(pos, true, 600)
    movedDest = #cands > 0
  elseif outside and nearD > 150 then
    movedDest = true
  end
  if #cands == 0 then
    geoRefuse("Lumen can't find a road near there. Pick another destination.", 6)
    return false
  end
  if s == "pulledOver" then carCommand(rideCar(), SIGNALS_OFF) end
  local oldDest = ride.dest
  ride.resume = false
  for i, c in ipairs(cands) do
    ride.dest = {link = c.link, pos = c.pos, name = name, private = c.private, target = vec3(pos)}
    if startRide(true) then
      log("I", logTag, string.format("geofence: destination %s: stop at (%.0f, %.0f), %.0f m from the requested point, %s%s",
        movedDest and "moved to the nearest covered road" or "allowed", c.pos.x, c.pos.y, c.pos:distance(pos),
        geoDescribe(c.pos.x, c.pos.y), c.private and " (a parking-lot lane)" or ""))
      if movedDest then
        local d = math.floor(c.pos:distance(pos) / 10 + 0.5) * 10
        think(string.format("%s is outside the service area, so I will take you to the nearest covered road, %d m from it.", name, d), "go")
        msg(string.format("%s is outside the service area. Heading to the nearest covered road, %d m from it.", name, d), 9, "info")
        log("I", logTag, string.format("destination moved to the nearest covered road, %d m from the pick", d))
      end
      if c.private then
        think("Your destination is in a parking lot, so I will drive in.")
      elseif i > 1 then
        think(string.format("The road nearest %s does not connect to the rest of the map, so I will stop at the closest one that does (%d m away).",
          name, math.floor(c.pos:distance(pos) + 0.5)))
      end
      return true
    end
  end
  ride.dest = oldDest
  geoRefuse((geofence and geofence.hasZones() and ("No route to " .. name .. " inside the Lumen service area.") or ("No road that connects to " .. name .. "."))
    .. " Pick another destination.", 6)
  return false
end

-- Pull over (Storm, 2026-10-31: "Pullover doesn't have a pathplanner and it doesn't pull off the
-- road"): Lumen Drive eases the car right onto the edge of the road, and onto the shoulder when
-- rays find it flat and open, clear of junctions, and stops there. The AR path shows the maneuver.
local PULL_TIMEOUT = 30

function M.pullOver()
  if not ride or ride.state ~= "riding" then return false end
  local car = rideCar()
  if not carVeh(car) then return false end
  local ok, shift, shoulder = drive.pullOver(car)
  if not ok then
    msg("There is no safe place to pull over just ahead. Try again in a moment.", 4)
    return false
  end
  local signal = map.getRoadRules().rightHandDrive and "left" or "right"
  carCommand(car, "electrics.set_warn_signal(false) " .. SIGNALS_OFF ..
    "if electrics.values.signal_" .. signal .. "_input ~= 1 then electrics.toggle_" .. signal .. "_signal() end")
  setState("pulling")
  think(string.format("Pulling over: signaling and moving %.1f m to the side%s, clear of intersections.", shift,
    (shoulder or 0) > 0.2 and string.format(", %.1f m of it onto the shoulder", shoulder) or ""), "go")
  msg("Pulling over.", 4)
  return true
end

function M.continueTrip()
  if not ride or ride.state ~= "pulledOver" or not ride.dest then return false end
  carCommand(rideCar(), SIGNALS_OFF)
  return startRide()
end

local function pulledOver(car)
  carCommand(car, SIGNALS_OFF .. "electrics.set_warn_signal(true)")
  drive.stop(car)
  ride.fare = computeFare()
  setState("pulledOver")
  msg("Pulled over. Get out here, or continue your trip.", 6)
end

-- called from updateRide while pulling
local function updatePulling(car, veh, dtReal)
  if drive.arrived(car) or ride.stateT > PULL_TIMEOUT then pulledOver(car) end
end
M._updatePulling = updatePulling

function M.exitTaxi()
  if not ride then return false end
  local car = rideCar()
  if ride.state == "riding" or ride.state == "pulling" then
    stopCar(car, true)
    ride.fare = computeFare()
  end
  if car then exitPlayer(car) end
  finishRide(nil, "exit requested")
  return true
end

function M.cancel()
  if not ride then return false end
  if ride.state == "boarded" or ride.state == "riding" or ride.state == "arrived" or ride.state == "pulling" or ride.state == "pulledOver" then
    return M.exitTaxi()
  end
  releaseCar(rideCar())
  ride = nil
  msg("Ride cancelled.", 3)
  return true
end

-- the AR path view: what the planner will do, drawn on the road ahead of your Lumen
function M.getArPath()
  if not arPathOn or not ride or not ride.trip or not (ride.state == "arriving" or ride.state == "riding" or ride.state == "pulling") then return nil end
  local car = rideCar()
  local veh = carVeh(car)
  if not veh then return nil end
  local state = "go"
  if car.recovering or car.blockedT > 3 then
    state = "blocked"
  elseif drive.state(car) == "wait" then
    state = "wait"
  end
  -- moving backwards (a back-up, or the AI reversing on its own): the view shows the reverse path
  local fwdSpeed = veh:getVelocity():dot(veh:getDirectionVector())
  local reversing = fwdSpeed < -0.3 or (car.tele.gear < 0 and realTime - car.tele.at < 2 and fwdSpeed < 0.2)
  return {veh = veh, trip = ride.trip, plan = drive.arPlan(car), state = state, kind = ride.state == "arriving" and "pickup" or "destination",
          reversing = reversing, speed = math.abs(fwdSpeed)}
end

function M.toggleArPath()
  arPathOn = not arPathOn
  if extensions.lumen_arpath then extensions.lumen_arpath.setEnabled(arPathOn) end
  msg(arPathOn and "AR path on: the road shows where your Lumen is going." or "AR path off.", 4, "visibility")
  return arPathOn
end

function M.toggleLidar()
  lidarOn = not lidarOn
  if extensions.lumen_lidar then extensions.lumen_lidar.setEnabled(lidarOn) end
  return lidarOn
end

local appAdded = false                 -- the app was put on the freeroam HUD once (persisted)

local function saveSettings()
  pcall(jsonWriteFile, SETTINGS_FILE, {enabled = enabled, appAdded = appAdded}, true)
end

-- ---------------------------------------------------------------------------------
-- the brain: Lumen Drive (the rules) or Lumen NN (the neural network)
-- ---------------------------------------------------------------------------------
-- Storm, 2026-11-01: "Make the N button switch from Lumen Drive to Lumen NN." The app's Drive | NN
-- toggle, the radial menu and the lumen_robotaxi_brain input action all land here and switch every
-- fleet car live. Lumen NN (lumen/nn.lua) plans through the NN HOOK in drive.lua and falls back to the
-- rules when it cannot (no weights, an error).
local nnMod = nil
local function nnModule()
  if nnMod == nil then
    local ok, m = pcall(require, "lumen/nn")
    nnMod = ok and type(m) == "table" and m or false
  end
  return nnMod or nil
end

local nnFailed = false
local function nnReady()
  if nnFailed then return false end
  local nm = nnModule()
  if not nm then return false end
  local ok, w = pcall(nm.load)                -- runs on every UI publish: an error here must never reach it
  if not ok then
    nnFailed = true
    log("E", logTag, "Lumen NN failed while loading and stays off: " .. tostring(w))
    return false
  end
  return w ~= nil
end

function M.setBrain(name, quiet)
  if name ~= "drive" and name ~= "nn" then return false end
  if name == "nn" and not nnReady() then
    local nm, why = nnModule(), nil
    if nm then local _info; _info, why = nm.info() end
    log("W", logTag, "brain switch to nn refused: " .. tostring(why or "the module did not load"))
    if not quiet then msg("Lumen NN has no trained weights in this build, so Lumen Drive keeps driving.", 6, "warning") end
    return false
  end
  if drive.getBrain() == name then return true end
  local ids = {}
  for _, car in ipairs(fleet) do ids[#ids + 1] = string.format("%d:%s", car.slot, tostring(car.id)) end
  log("I", logTag, string.format("brain switch %s -> %s; fleet cars (slot:id) %s", drive.getBrain(), name, table.concat(ids, " ")))
  drive.setBrain(name)
  for _, car in ipairs(fleet) do
    if car.drv then car.drv.nnAr, car.drv.reasonKey, car.drv.nnErr, car.drv.nnSeen = nil, nil, nil, nil end
  end
  if not quiet then
    think(name == "nn"
      and "Switched to Lumen NN: a neural network trained in simulation now chooses my speed and my place in the lane. The rules still handle the final stop, pulling over and backing away."
      or "Switched to Lumen Drive: the rule-based driver is back in charge.", "go")
    msg(name == "nn" and "Lumen NN is driving." or "Lumen Drive is driving.", 4, "navigation")
  end
  return true
end

function M.toggleBrain()
  M.setBrain(drive.getBrain() == "nn" and "drive" or "nn")
  return drive.getBrain()
end

-- A/B test of the two drivers on the real roads (2026-11-02, Storm's /goal "Make the NN actually better
-- than Drive"): odd slots drive with Lumen NN and even slots with Lumen Drive (flip = true swaps them),
-- and drive.lua counts what each did. Off unless asked; the test instance runs it (testenv/ab_test.py).
function M.abTest(on, flip)
  if not on then
    abOn = false
    drive.ab = nil
    for _, car in ipairs(fleet) do car.brain = nil end
    log("I", logTag, "abtest: off")
    return false
  end
  if not nnReady() then return false end
  drive.abReset()
  abOn, abLogT = true, 0
  for _, car in ipairs(fleet) do
    car.brain = ((car.slot % 2 == 1) ~= (flip and true or false)) and "nn" or "drive"
    if car.drv then car.drv.nnAr, car.drv.reasonKey, car.drv.nnErr, car.drv.nnSeen = nil, nil, nil, nil end
  end
  log("I", logTag, "abtest: on, odd slots on " .. (flip and "Drive" or "NN") .. ", even slots on " .. (flip and "NN" or "Drive"))
  return true
end

function M.abStats() return drive.abReport() end

-- test instance: the NN's observation and decision for one fleet car (training/nn_ingame.py)
function M.debugNN(slot)
  local car = fleet[slot]
  local nm = nnModule()
  if not car or not nm or not nnReady() or not nm.debug then return nil end
  return nm.debug(drive, car)
end

-- first run only: put the Lumen Robotaxi app on the freeroam HUD (players can move or remove it
-- like any other app; it is never re-added after that)
ensureAppInLayout = function()
  if appAdded or not ui_appLayouts then return end
  pcall(function()
    local file = ui_appLayouts.getLayoutForType and ui_appLayouts.getLayoutForType("freeroam")
    if type(file) == "table" then file = file.filename end
    file = file or "/settings/ui_apps/layouts/default/freeroam.uilayout.json"
    local layout
    for _, x in ipairs(ui_appLayouts.getAvailableLayouts() or {}) do
      if x.filename == file then layout = x end
    end
    if not layout then return end
    for _, a in ipairs(layout.apps or {}) do
      if a.appName == "lumenRobotaxi" then return end
    end
    ui_appLayouts.addApp(file, "lumenRobotaxi", {position = "absolute", top = "110px", right = "20px", width = "330px", height = "540px"})
    local cur = ui_appLayouts.getCurrentLayout and ui_appLayouts.getCurrentLayout()
    if cur and cur.filename == file then ui_appLayouts.setCurrentLayout(file) end
    log("I", logTag, "added the Lumen Robotaxi app to " .. file)
  end)
  appAdded = true
  saveSettings()
end

function M.setEnabled(on)
  enabled = on and true or false
  saveSettings()
  if enabled then
    initAt = realTime + 0.5
    msg("Robotaxi service on. Lumens are heading into the area.", 5)
  else
    if ride then M.cancel() end
    stopFleet(true)
    msg("Robotaxi service paused. The fleet has left the map.", 5)
  end
  return enabled
end

function M.onAction(name)
  if name == "call" then return M.callTaxi()
  elseif name == "board" then
    if ride and ride.state == "waiting" then return M.board() end
    if ride and (ride.state == "arrived" or ride.state == "boarded") then return M.exitTaxi() end
  elseif name == "pullover" then
    if ride and ride.state == "riding" then return M.pullOver() end
    if ride and ride.state == "pulledOver" then return M.continueTrip() end
  elseif name == "cancel" then return M.cancel()
  elseif name == "lidar" then return M.toggleLidar()
  elseif name == "brain" then return M.toggleBrain()
  end
end

-- telemetry from each fleet car's vehicle controller (lumenRobotaxi.lua), twice a second
function M.onCarTelemetry(id, throttle, brake, gear, ws, aiMode)
  for _, car in ipairs(fleet) do
    if car.id == id then
      local t = car.tele
      t.throttle, t.brake, t.gear, t.ws, t.ai, t.at = throttle, brake, gear, ws, aiMode, realTime
      return
    end
  end
end

function M.getState()
  local cars = {}
  for _, car in ipairs(fleet) do cars[#cars + 1] = {slot = car.slot, id = car.id, status = car.status, reason = car.reason} end
  return {
    state = ride and ride.state or "idle", taxiId = rideCar() and rideCar().id or nil,
    dest = ride and ride.dest and ride.dest.name or nil, eta = ride and ride.eta or nil,
    remaining = ride and ride.trip and ride.trip.remaining or nil, fare = ride and ride.fare or nil,
    active = active, enabled = enabled, fleet = cars, brain = drive.getBrain(),
  }
end

-- testing and tools: send a fleet car to service as if it were stuck
function M.setDriveDebug(on) drive.debug = on and true or false return drive.debug end
function M.debugDump(slot) local c = fleet[slot] return c and drive.debugDump(c) or nil end
-- testing: the fleet car table itself, and where its parking maneuver is (lumen/parking.lua)
function M._car(slot) return fleet[slot] end
function M.parkStatus(slot) local c = fleet[slot] return c and parking.status(c) or nil end

-- test instance readout: each fleet car and what Lumen Drive is doing with it
function M.debugFleet()
  local out = {}
  for _, car in ipairs(fleet) do
    local v = carVeh(car)
    local d = car.drv or {}
    local p = v and v:getPosition()
    out[#out + 1] = {slot = car.slot, id = car.id, status = car.status, mode = d.mode, s = d.s, lat = d.lat,
      len = d.path and d.path.len, state = d.state, reason = d.reason, stopKind = d.stopKind, stopS = d.stopS,
      speed = v and v:getVelocity():length() or nil, x = p and p.x, y = p and p.y, recovering = car.recovering ~= nil,
      evading = car.evading ~= nil, cruising = d.cruising, service = car.reason, recoveries = car.recoveries, park = car.park and car.park.phase or nil,
      fwd = v and v:getVelocity():dot(v:getDirectionVector()) or nil, gear = car.tele and car.tele.gear, thr = car.tele and car.tele.throttle, brk = car.tele and car.tele.brake}
  end
  return out
end

function M.debugService(slot, reason)
  local car = carBySlot(slot or (ride and ride.slot))
  if car and car.id then sendToService(car, reason or "stuck") return true end
  return false
end

-- Test instance only (testenv/hard_scenarios.py, 2026-11-01; Storm: "Have you tested it through hard scenarios?").
-- Put fleet car `slot` at (x, y) facing (dx, dy), send it to the public road nearest (tx, ty) on the brain given
-- ("drive" or "nn"), and keep it on that trip (no roaming re-plans) until debugRelease.
function M.debugPlace(slot, x, y, dx, dy)
  local car = carBySlot(slot)
  local veh = car and carVeh(car)
  if not veh then return false end
  drive.release(car)
  local dir = vec3(dx, dy, 0):normalized()
  local L = roads.closestLink(vec3(x, y, 0), {maxDist = 30})
  local z = L and (L.pa.z + 0.3) or veh:getPosition().z
  spawn.safeTeleport(veh, vec3(x, y, z), quatFromDir(dir, vec3(0, 0, 1)), true, nil, false)
  car.testHold = true
  car.readyAt = simTime + 0.8
  resetHealth(car)
  return true
end

function M.debugSendTo(slot, tx, ty, brain)
  local car = carBySlot(slot)
  local veh = car and carVeh(car)
  if not veh then return false end
  local L, _, _, t = roads.closestLink(vec3(tx, ty, veh:getPosition().z), {public = true, maxDist = 80})
  if not L then return false end
  local trip = roads.planTrip(veh:getPosition(), veh:getDirectionVector(), L, roads.stopPoint(L, t), {avoid = avoidSet()})
  if not trip then return false end
  car.brain = (brain == "nn" or brain == "drive") and brain or nil
  car.testHold = true
  if car.drv then car.drv.nnAr, car.drv.nnErr, car.drv.nnSeen = nil, nil, nil end
  driveTrip(car, trip)
  return trip.remaining
end

function M.debugRelease(slot)
  local car = carBySlot(slot)
  if not car then return false end
  car.testHold, car.brain = nil, nil
  car.needsCruise = true
  return true
end

-- ---------------------------------------------------------------------------------
-- UI
-- ---------------------------------------------------------------------------------
local function publishUi()
  local p = {state = ride and ride.state or "idle", lidar = lidarOn, arpath = arPathOn, enabled = enabled, active = active, fleetSize = FLEET_SIZE,
             brain = drive.getBrain(), nnReady = nnReady()}
  local pp, pd = playerPos(), playerDir()
  p.player = {x = round(pp.x, 0.5), y = round(pp.y, 0.5), dx = round(pd.x, 0.001), dy = round(pd.y, 0.001)}
  local cars, available, nearest = {}, 0, nil
  for _, car in ipairs(fleet) do
    local v = carVeh(car)
    if v and car.status ~= "fading" then
      local cp, cd = v:getPosition(), v:getDirectionVector()
      cars[#cars + 1] = {x = round(cp.x, 0.5), y = round(cp.y, 0.5), dx = round(cd.x, 0.001), dy = round(cd.y, 0.001),
                         mine = (ride ~= nil and ride.slot == car.slot), status = car.status}
      if car.status == "cruising" then
        available = available + 1
        local d = cp:distance(pp)
        if not nearest or d < nearest then nearest = d end
      end
    end
  end
  p.fleet, p.available = cars, available
  p.geo = geofence and geofence.status(pp.x, pp.y) or nil
  -- straight-line guess for the idle screen; real routes are measured on dispatch
  p.nearestEta = nearest and round(nearest * 1.4 / 10) or nil
  if ride then
    p.eta = ride.eta and round(ride.eta) or nil
    p.remaining = ride.trip and round(ride.trip.remaining) or nil
    p.fare = round(ride.fare or 0, 0.01)
    p.destName = ride.dest and ride.dest.name or nil
    p.rideKm = round(ride.distance / 1000, 0.1)
    p.resume = ride.resume
    local car = rideCar()
    local v = carVeh(car)
    if v and ride.trip then p.route = roads.routeXY(ride.trip, v:getPosition()) end
    if ride.pickup and (ride.state == "dispatching" or ride.state == "arriving" or ride.state == "waiting") then
      p.pickup = {x = round(ride.pickup.pos.x, 0.5), y = round(ride.pickup.pos.y, 0.5), remote = ride.pickup.remote and true or nil}
    end
    if ride.dest and (ride.state == "boarded" or ride.state == "riding" or ride.state == "arrived" or ride.state == "pulling" or ride.state == "pulledOver" or ride.resume) then
      p.dest = {x = round(ride.dest.pos.x, 0.5), y = round(ride.dest.pos.y, 0.5)}
    end
    if ride.state == "boarded" or ride.state == "riding" or ride.state == "arrived" or ride.state == "pulledOver" then
      p.destinations = destinationOptions()
    end
    p.traffic = ride.traffic or false
    if #thoughts > 0 then
      local list = {}
      for i = math.max(1, #thoughts - 3), #thoughts do
        local th = thoughts[i]
        list[#list + 1] = {text = th.text, kind = th.kind, age = math.floor(realTime - th.t)}
      end
      p.thoughts = list
    end
    if ride.state == "boarded" or ride.state == "riding" or ride.state == "arrived" or ride.state == "pulling" or ride.state == "pulledOver" then
      p.seats, p.seat = SEATS, seat
    end
  end
  guihooks.trigger("LumenRobotaxiUpdate", p)
end

function M.requestUi() publishUi() end

-- the road network for the app's 2D map (once per level; the app caches it)
function M.requestRoads()
  if not getCurrentLevelIdentifier() then return end
  guihooks.trigger("LumenRobotaxiRoads", roads.exportRoads())
  if geofence then guihooks.trigger("LumenRobotaxiZones", geofence.export()) end
end

-- a destination picked on the app's 2D map (world metres): any point, snapped to a road the
-- fleet can reach inside the service area
-- test instance: why a spot does or does not get a parking-lot stop. For the nearest links within 35 m of (x, y):
-- public or a lot lane, its half-width, drivability, whether the wall rays pass, and the service-area gate.
function M.debugLot(x, y)
  local pos = vec3(tonumber(x), tonumber(y), 0)
  local L0 = roads.closestLink(pos, {maxDist = 35})
  if L0 then pos.z = L0.pa.z end
  local out = {lotsOk = (not geofence) or (geofence.openTier() >= 2 and geofence.served(pos.x, pos.y)), links = {}}
  for _, c in ipairs(roads.closestLinks(pos, {maxDist = 35}, 12)) do
    local L = c.L
    out.links[#out.links + 1] = {d = c.d, public = L.public and true or false, type = L.link and L.link.type, radius = L.radius,
      drivability = L.link and L.link.drivability, clear = (not L.public) and laneClear(L, c.t) or nil}
  end
  local cands = stopCandidates(pos, false, 150)
  out.first = cands[1] and {x = cands[1].pos.x, y = cands[1].pos.y, private = cands[1].private and true or false} or nil
  return out
end

function M.setDestinationXY(x, y)
  x, y = tonumber(x), tonumber(y)
  if not x or not y then return false end
  return M.setDestination(string.format("xy:%.1f,%.1f", x, y))
end

-- the service area, for tools and the app
function M.getGeofence() return geofence and geofence.export() or nil end

-- test switch: open tiers up to n on the current map (until the game restarts)
function M.setOpenTier(n)
  if not geofence then return false end
  geofence.setOpenTier(n)
  M.requestRoads()
  return geofence.openTier()
end

-- ---------------------------------------------------------------------------------
-- radial menu
-- ---------------------------------------------------------------------------------
local function registerRadial()
  if not core_quickAccess or not core_quickAccess.addEntry then return end
  core_quickAccess.addEntry({level = "/root/sandbox/", title = "Robotaxi", icon = "cars", uniqueID = "lumen_robotaxi", ["goto"] = "/root/sandbox/lumen_robotaxi/", categoryOrder = 15})
  core_quickAccess.addEntry({level = "/root/sandbox/lumen_robotaxi/", uniqueID = "lumen_robotaxi_gen", generator = function(entries)
    local s = ride and ride.state or "idle"
    if s == "idle" and enabled then
      table.insert(entries, {title = "Call a Lumen", icon = "cars", onSelect = function() M.callTaxi() return {"hide"} end})
    end
    if s == "waiting" then
      table.insert(entries, {title = "Board", icon = "carPlus", onSelect = function() M.board() return {"hide"} end})
    end
    if s == "boarded" or s == "riding" then
      for _, d in ipairs(destinationOptions()) do
        table.insert(entries, {title = d.name, icon = "flag", onSelect = function() M.setDestination(d.id) return {"hide"} end})
      end
    end
    if s == "riding" then
      table.insert(entries, {title = "Pull over", icon = "local_parking", onSelect = function() M.pullOver() return {"hide"} end})
    end
    if s == "pulledOver" then
      table.insert(entries, {title = "Continue trip", icon = "navigation", onSelect = function() M.continueTrip() return {"hide"} end})
      table.insert(entries, {title = "Get out", icon = "carPlus", onSelect = function() M.exitTaxi() return {"hide"} end})
    end
    if s == "boarded" or s == "riding" or s == "arrived" or s == "pulling" or s == "pulledOver" then
      for _, st in ipairs(SEATS) do
        if st.id ~= seat then
          table.insert(entries, {title = "Sit: " .. st.name, icon = "airline_seat_recline_normal", onSelect = function() M.setSeat(st.id) return {"hide"} end})
        end
      end
    end
    if s == "arrived" or s == "boarded" then
      table.insert(entries, {title = "Get out", icon = "carPlus", onSelect = function() M.exitTaxi() return {"hide"} end})
    end
    if s ~= "idle" and s ~= "departing" then
      table.insert(entries, {title = "Cancel ride", icon = "close", onSelect = function() M.cancel() return {"hide"} end})
    end
    table.insert(entries, {title = lidarOn and "Hide lidar view" or "Show lidar view", icon = "eye", onSelect = function() M.toggleLidar() return {"hide"} end})
    table.insert(entries, {title = arPathOn and "Hide AR path" or "Show AR path", icon = "navigation", onSelect = function() M.toggleArPath() return {"hide"} end})
    table.insert(entries, {title = drive.getBrain() == "nn" and "Drive with Lumen Drive (rules)" or "Drive with Lumen NN (neural network)", icon = "navigation",
      onSelect = function() M.toggleBrain() return {"hide"} end})
    table.insert(entries, {title = enabled and "Pause robotaxi service" or "Start robotaxi service", icon = "power_settings_new",
      onSelect = function() M.setEnabled(not enabled) return {"hide"} end})
  end})
end

-- ---------------------------------------------------------------------------------
-- hooks
-- ---------------------------------------------------------------------------------
local function onUpdate(dtReal, dtSim)
  realTime = realTime + dtReal
  simTime = simTime + dtSim
  if initAt and realTime >= initAt then
    initAt = nil
    startFleet()
  end
  if active then
    if missionActive() then
      -- missions get a clean world; the fleet comes back when the mission ends
      if ride then M.cancel() end
      stopFleet(true)
      initAt = realTime + 5
    else
      updateFleet(dtSim, dtReal)
      if ride then updateRide(dtSim, dtReal) end
    end
  end
  uiTimer = uiTimer + dtReal
  if uiTimer >= UI_RATE then
    uiTimer = 0
    publishUi()
  end
end

local function onVehicleSwitched(oldId, newId)
  if ride then log("D", logTag, string.format("vehicle switched %s -> %s during %s", tostring(oldId), tostring(newId), ride.state)) end
  if not ride or not ride.slot or ride.exiting then return end
  local car = rideCar()
  if not car or not car.id then return end
  if newId == car.id and ride.state == "waiting" then
    onBoarded()
  elseif oldId == car.id and newId ~= car.id then
    local s = ride.state
    if s == "boarded" or s == "riding" or s == "arrived" or s == "pulling" or s == "pulledOver" then
      -- the rider got out by themselves
      lockPlayerControls(carVeh(car), false)
      if s == "riding" or s == "pulling" then
        stopCar(car, true)
        ride.fare = computeFare()
      end
      finishRide(nil, "the rider left the car (vehicle switch)")
    end
  end
end

local function onVehicleDestroyed(vid)
  for _, car in ipairs(fleet) do
    if car.id == vid then
      parking.forget(car)
      car.id = nil
      car.status = "service"
      car.serviceUntil = simTime + 20
      if ride and ride.slot == car.slot then rideCarLost(car, "unavailable") end
    end
  end
end

local function onExtensionLoaded()
  local s = jsonReadFile(SETTINGS_FILE)
  if s and s.enabled == false then enabled = false end
  if s and s.appAdded then appAdded = true end
  registerRadial()                  -- (the brain always starts as Lumen Drive: an NN switch is never remembered)
  if geofence then roads.gate = {link = geofence.linkOK, version = geofence.version, classOK = geofence.classOK} end
  drive.setReasonHook(function(car, text, kind)
    if ride and ride.slot == car.slot and (ride.state == "arriving" or ride.state == "riding" or ride.state == "pulling") then think(text, kind) end
  end)
  if getCurrentLevelIdentifier() then initAt = realTime + 1 end
  log("I", logTag, "Lumen robotaxi service ready (" .. (enabled and "on" or "paused") .. ")")
end

local function onClientPostStartMission()
  roads.invalidate()
  parking.reset()
  fleet, ride, active = {}, nil, false
  initAt = realTime + 5               -- let the level and the player's vehicle settle first
  registerRadial()
end

local function onClientEndMission()
  parking.reset()
  fleet, ride, active, initAt = {}, nil, false, nil
  roads.invalidate()
end

M.onUpdate = onUpdate
M.onVehicleSwitched = onVehicleSwitched
M.onVehicleDestroyed = onVehicleDestroyed
M.onExtensionLoaded = onExtensionLoaded
M.onClientPostStartMission = onClientPostStartMission
M.onClientEndMission = onClientEndMission
M.reset = function() if ride then M.cancel() end stopFleet(true) initAt = realTime + 1 end

return M
