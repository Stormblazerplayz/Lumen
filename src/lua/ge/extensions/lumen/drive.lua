-- Lumen Drive (Frontier Studios): the Lumen's own driver.
-- Storm, 2026-10-31: "Beamng's AI. Did I ever say to use that?" The fleet used to hand its
-- routes to BeamNG's built-in AI driver. Lumen Drive replaces it completely, rule based for now
-- (a lidar network comes later):
--   lane model   the lane to drive in, from the road graph's lane data (never the wrong carriageway)
--   path         that lane as a smooth line: junction turns through the junction, a gentle blend out
--                of wherever the car is, a stop pulled in to the curb
--   speed        the posted limit, curves (lateral acceleration), comfortable braking into stops
--   rules        traffic lights and stop signs, following distance to the car ahead
--   prediction   every nearby car's path over the next 4 s (along its road, or by its turn rate),
--                tested against the Lumen's own plan; a conflict means yield
--   maneuvers    pull over onto the edge of the road (rays check the shoulder), cruising on its own
-- The car's controller (lua/vehicle/controller/lumenRobotaxi.lua) steers and works the pedals to
-- follow the plan this module sends five times a second. The AR path draws that same plan, and
-- the reasons behind each decision go to the "Lumen is thinking" feed.
-- Plain module: require("lumen/drive").

local M = {}

local roads = require("lumen/roads")
local guard = require("lumen/guard")   -- committed yellow and junction discipline, shared with lumen/nn.lua
local logTag = "lumen_drive"
-- the service area, for the log lines about where roaming cars go (nil if it failed to load)
local okGeo, geofence = pcall(require, "lumen/geofence")
if not okGeo then geofence = nil end

-- ---------------------------------------------------------------------------------
-- tuning (2026-10-31, first version; to be tuned on the test instance)
-- ---------------------------------------------------------------------------------
local SAMPLE = 1.5                   -- m between path samples
local HORIZON_T, HORIZON_MIN, HORIZON_MAX = 9, 45, 200   -- s / m of path sent to the car
local PLAN_RATE = 0.2                -- s between plans (near the camera; see M.update for the far cars)
local roamTargets = setmetatable({}, {__mode = "k"})   -- [car] = {x, y}: where each roaming car is heading, to spread the fleet
local A_LAT = 2.2                    -- m/s2 lateral acceleration allowed in curves
local A_ACC = 1.6                    -- m/s2
local A_DEC = 2.2                    -- m/s2 comfortable braking into stops
local A_HARD = 6.0                   -- m/s2 emergency braking
local V_MAX = 20                     -- m/s (72 km/h). 2026-11-01: a Lumen lost control at 25 m/s on a divided road; fast roads are no longer fenced by speed class (geofence.lua FAST), so this is the highway limit until the high-speed work is done
local LIMIT_SHARE = 0.95             -- of the posted limit
local V_PRIVATE = 3.5                -- m/s in parking lots and driveways (was 5.5: "it sped up and hit one of the walls", 2026-11-01)
local FRONT, HALF_LEN, HALF_W = 2.6, 2.4, 0.95   -- m: reference node to front bumper; half body
local STOP_GAP = 2.5                 -- m left to a stopped vehicle ahead
local LINE_GAP = 1.2                 -- m between the front bumper and a stop line
local IDM_T, IDM_S0, IDM_A, IDM_B = 1.5, 2.5, 1.6, 2.2
local LANE_HALF = 1.9                -- m: a vehicle this close to our path line is in our lane
local PRED_STEPS, PRED_DT = 16, 0.25 -- 4 s of prediction
local SENSE_RANGE = 75               -- m around the car
local CURB_MARGIN = 0.35             -- m between the body and the road edge at a stop
local STOP_SIGN_WAIT = 1.6           -- s at a full stop before going on from a stop sign

-- ---------------------------------------------------------------------------------
-- small helpers
-- ---------------------------------------------------------------------------------
local sqrt, abs, min, max, floor = math.sqrt, math.abs, math.min, math.max, math.floor

local function rhd()
  local r = map.getRoadRules and map.getRoadRules()
  return r and r.rightHandDrive or false
end

local function flipLanes(lanes)
  local out = {}
  for i = #lanes, 1, -1 do
    local c = lanes:sub(i, i)
    out[#out + 1] = c == "+" and "-" or c == "-" and "+" or c
  end
  return table.concat(out)
end

-- lanes for travel a -> b, left to right as seen driving that way ('+' = our direction), and the
-- road's half width; the same rules map.lua and ai.lua use when a road has no lane data
local function laneConfig(a, b)
  local mn = map.getMap().nodes
  local na, nb = mn[a], mn[b]
  if not na or not nb then return nil end
  local l = (na.links and na.links[b]) or (nb.links and nb.links[a])
  if not l then return nil end
  local rad = min(na.radius or 2, nb.radius or 2)
  local lanes = l.lanes
  if not lanes or lanes == "" then
    local n = max(1, floor(rad * 2 / 3.61 + 0.5))
    if l.oneWay then
      lanes = string.rep("+", n)
    else
      local h = max(1, floor(n * 0.5))
      lanes = rhd() and (string.rep("+", h) .. string.rep("-", h)) or (string.rep("-", h) .. string.rep("+", h))
    end
  end
  if l.inNode ~= a then lanes = flipLanes(lanes) end
  return lanes, rad, l
end

-- our lane for travel a -> b: its centre's offset from the road centre line (m, + = right of
-- travel), the road's half width, the lane width, the link's speed limit (m/s), private or not
local function laneFor(a, b)
  local lanes, rad, l = laneConfig(a, b)
  if not lanes then return 0, 2, 3.6, 13.9, false end
  local n = #lanes
  local w = 2 * rad / n
  local pick
  if rhd() then
    for i = 1, n do if lanes:sub(i, i) == "+" then pick = i break end end
  else
    for i = n, 1, -1 do if lanes:sub(i, i) == "+" then pick = i break end end
  end
  if not pick then pick = rhd() and 1 or n end
  local private = l.type == "private" or (l.drivability or 1) < roads.MIN_DRIVABILITY
  local vlim = l.speedLimit and tonumber(l.speedLimit) or 13.9
  if private then vlim = min(vlim, V_PRIVATE) end
  return -rad + (pick - 0.5) * w, rad, w, max(4, min(vlim, 33)), private
end

local function right2(dx, dy) return dy, -dx end   -- right of travel for a 2D direction

local function bezier(out, sx, sy, sz, c1x, c1y, c2x, c2y, ex, ey, ez, meta)
  local L = sqrt((ex - sx) ^ 2 + (ey - sy) ^ 2)
  local steps = max(2, math.ceil(L / SAMPLE))
  for k = 1, steps do
    local t = k / steps
    local u = 1 - t
    local b0, b1, b2, b3 = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
    out[#out + 1] = {b0 * sx + b1 * c1x + b2 * c2x + b3 * ex, b0 * sy + b1 * c1y + b2 * c2y + b3 * ey,
                     sz + (ez - sz) * t, meta}
  end
end

local function line(out, sx, sy, sz, ex, ey, ez, meta)
  local L = sqrt((ex - sx) ^ 2 + (ey - sy) ^ 2)
  local steps = max(1, math.ceil(L / SAMPLE))
  for k = 1, steps do
    local t = k / steps
    out[#out + 1] = {sx + (ex - sx) * t, sy + (ey - sy) * t, sz + (ez - sz) * t, meta}
  end
end

-- ---------------------------------------------------------------------------------
-- the path: a trip from roads.planTrip turned into the line the car drives
-- ---------------------------------------------------------------------------------
-- path = {x, y, z, s, vl, k (curvature), seg, n, len, junctions = {s ...}, nodesAt = {s ...}}
local function finalize(raw, segs)
  local P = {x = {}, y = {}, z = {}, s = {}, vl = {}, k = {}, seg = {}, n = 0, len = 0, junctions = {}}
  local lastX, lastY
  for _, p in ipairs(raw) do
    if not lastX or (p[1] - lastX) ^ 2 + (p[2] - lastY) ^ 2 > 0.04 then
      local n = P.n + 1
      P.n = n
      P.x[n], P.y[n], P.z[n] = p[1], p[2], p[3]
      P.seg[n] = p[4] or 1
      P.s[n] = n == 1 and 0 or P.s[n - 1] + sqrt((p[1] - lastX) ^ 2 + (p[2] - lastY) ^ 2)
      lastX, lastY = p[1], p[2]
    end
  end
  P.len = P.n > 0 and P.s[P.n] or 0
  for i = 1, P.n do
    local sg = segs[P.seg[i]] or segs[#segs]
    P.vl[i] = sg and sg.vl or 13.9
    -- curvature from points about 3 m either side (Menger), steadier than neighbours 1.5 m apart
    local a, c = max(1, i - 2), min(P.n, i + 2)
    if c - a >= 2 then
      local ax, ay, bx, by, cx, cy = P.x[a], P.y[a], P.x[i], P.y[i], P.x[c], P.y[c]
      local area2 = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
      local la = sqrt((bx - ax) ^ 2 + (by - ay) ^ 2)
      local lb = sqrt((cx - bx) ^ 2 + (cy - by) ^ 2)
      local lc = sqrt((cx - ax) ^ 2 + (cy - ay) ^ 2)
      P.k[i] = (la * lb * lc > 1e-6) and (2 * area2 / (la * lb * lc)) or 0
    else
      P.k[i] = 0
    end
  end
  return P
end

-- Is the ground d metres to the right of (x, y) as high as the road there? A curb or a sidewalk
-- shows as a step of more than 8 cm; no ground at all (a drop) is not flat either.
local function flatTo(x, y, z, rx, ry, d)
  local function groundZ(px, py)
    local h = castRayStatic(vec3(px, py, z + 1.5), vec3(0, 0, -1), 3)
    return h < 3 and (z + 1.5 - h) or nil
  end
  local z0, z1 = groundZ(x, y), groundZ(x + rx * d, y + ry * d)
  return z0 ~= nil and z1 ~= nil and abs(z1 - z0) <= 0.08
end

-- opts.curb: pull the end in to the curb (pickups, drop-offs); opts.endExtra: metres to keep
-- following the road past the stop point (cruising, so the car never brakes for it)
local function buildPath(trip, veh, opts)
  opts = opts or {}
  local mn = map.getMap().nodes
  local nodesList, pts = trip.nodes, trip.points
  local carPos, carDir = veh:getPosition(), veh:getDirectionVector()
  local hx, hy = carDir.x, carDir.y
  local hl = sqrt(hx * hx + hy * hy)
  if hl < 1e-3 then hx, hy = 0, 1 else hx, hy = hx / hl, hy / hl end

  -- the car's own road, to know which way its first lane runs
  local first = nodesList[1]
  local CL = roads.closestLink(carPos, {maxDist = 40, dir = carDir})
  local behind
  if CL and (CL.a == first or CL.b == first) then behind = (CL.a == first) and CL.b or CL.a end
  -- not on a road that reaches the first node (near a junction the closest road can be another
  -- branch): take the first node's own road nearest the car. 2026-11-01: without it the first
  -- lane line was drawn from the car's own spot and shifted a lane to the right, into the curb.
  if not behind and first and mn[first] then
    local fp, bestD = mn[first].pos, nil
    for nb in pairs(mn[first].links or {}) do
      local q = mn[nb]
      if q then
        local t = clamp(carPos:xnormOnLine(q.pos, fp), 0, 1)
        local dd = (q.pos + (fp - q.pos) * t):distance(carPos)
        if not bestD or dd < bestD then behind, bestD = nb, dd end
      end
    end
    if bestD and bestD > 15 then behind = nil end
  end

  -- segments along the road centre lines, each with its lane offset
  local segs = {}
  for i = 1, #pts - 1 do
    local A, B = pts[i], pts[i + 1]
    local na, nb
    if i == 1 then
      na, nb = behind, first
      if behind and mn[behind] and mn[first] then
        local pa, pb = mn[behind].pos, mn[first].pos
        local t = clamp(carPos:xnormOnLine(pa, pb), 0, 1)
        A = pa + (pb - pa) * t
      end
    else
      na, nb = nodesList[i - 1], nodesList[i]
    end
    local off, rad, w, vl, private = 0, 3, 3.6, 13.9, false
    if na and nb then off, rad, w, vl, private = laneFor(na, nb)
    elseif nodesList[1] and nodesList[2] then off, rad, w, vl, private = laneFor(nodesList[1], nodesList[2]) end
    local dx, dy = B.x - A.x, B.y - A.y
    local len = sqrt(dx * dx + dy * dy)
    if len > 0.05 then
      dx, dy = dx / len, dy / len
      local rx, ry = right2(dx, dy)
      segs[#segs + 1] = {ti = i, ax = A.x + rx * off, ay = A.y + ry * off, az = A.z, bx = B.x + rx * off, by = B.y + ry * off, bz = B.z,
                         dx = dx, dy = dy, len = len, off = off, rad = rad, w = w, vl = min(V_MAX, vl * LIMIT_SHARE), private = private,
                         node = (i < #pts - 1) and nodesList[i] or nil}
    end
  end
  if #segs == 0 then return nil end
  -- Merge pieces that run straight on in the same lane: near junctions roads come in pieces a few
  -- metres long, and a turn needs the room of the whole straight for its arc.
  do
    local merged = {segs[1]}
    for q = 2, #segs do
      local m1, b = merged[#merged], segs[q]
      if m1.dx * b.dx + m1.dy * b.dy > math.cos(math.rad(4)) and abs(m1.off - b.off) < 0.3 and m1.private == b.private then
        m1.bx, m1.by, m1.bz = b.bx, b.by, b.bz
        local dx, dy = m1.bx - m1.ax, m1.by - m1.ay
        m1.len = sqrt(dx * dx + dy * dy)
        if m1.len > 1e-6 then m1.dx, m1.dy = dx / m1.len, dy / m1.len end
        m1.node, m1.vl = b.node, min(m1.vl, b.vl)
      else
        merged[#merged + 1] = b
      end
    end
    segs = merged
  end

  -- the stop: pulled toward the curb on public roads
  local last = segs[#segs]
  if opts.curb and not last.private then
    -- 2026-11-01, Storm: "It hit this but it reversed after a lil" (a Lumen climbed the sidewalk at a
    -- pickup in Downtown Belasco, got stuck, backed off). The pull trusted the road's node radius,
    -- which can be wider than the paved surface. Now at most 0.6 m, and only as far as the ground
    -- under the car's right side (plus 15 cm) is as high as the road: no curb, no sidewalk.
    local edgeOff = last.rad - HALF_W - CURB_MARGIN
    local want = min(edgeOff - last.off, 0.6)
    local rx, ry = right2(last.dx, last.dy)
    local pull = 0
    for _, cand in ipairs({want, want * 0.6, want * 0.3}) do
      if cand > 0.15 and flatTo(last.bx, last.by, last.bz, rx, ry, cand + HALF_W + 0.15) then
        pull = cand
        break
      end
    end
    if pull > 0.15 then
      last.bx, last.by = last.bx + rx * pull, last.by + ry * pull
      last.curb = pull
    end
  end

  -- Turns are fillets: a circular arc of radius R tangent to the incoming and outgoing lane lines
  -- around the point where those lines cross, skipping any short pieces in between (a junction is
  -- often several nodes a few metres apart). R: toward the curb (right in right-hand traffic) the
  -- tightest comfortable arc, 5 m, because a bigger one cuts INTO the corner; across a junction a
  -- wide one; on an ordinary bend the road's own curve. The Bezier control length is the circular
  -- one for the chord, so the arc's radius really is R. 2026-11-01: arcs capped by half a 3 m
  -- piece came out at about 1 m radius and the car ran wide at every junction.
  local function isJunction(node) return node ~= nil and map.getNodeLinkCount(node) > 2 end
  local joins = {}
  do
    local i = 1
    while i < #segs do
      local j, run = i + 1, 0
      while j < #segs and segs[j].len < 12 and run + segs[j].len <= 30 do run = run + segs[j].len j = j + 1 end
      local a1, a2 = segs[i], segs[j]
      local cross = a1.dx * a2.dy - a1.dy * a2.dx
      local theta = math.acos(clamp(a1.dx * a2.dx + a1.dy * a2.dy, -1, 1))
      local junction, nr = false, 4
      for q = i, j - 1 do
        local nd = segs[q].node
        if isJunction(nd) then junction = true nr = max(nr, mn[nd] and mn[nd].radius or 4) end
      end
      local R
      if junction then
        local curbSide = (cross < 0) ~= rhd()
        R = curbSide and 5.0 or clamp(nr + 4, 7, 14)
      else
        R = max(25, (run + 10) / max(theta, 0.05))
      end
      -- the lane lines' crossing point
      local vx, vy
      if abs(cross) > math.sin(math.rad(5)) then
        local px, py = a2.ax - a1.bx, a2.ay - a1.by
        local u = (px * a2.dy - py * a2.dx) / cross
        vx, vy = a1.bx + a1.dx * u, a1.by + a1.dy * u
      end
      joins[#joins + 1] = {i = i, j = j, theta = theta, R = R, vx = vx, vy = vy, junction = junction}
      i = j
    end
  end

  local raw = {}
  -- out of wherever the car is now, into the first lane, leaving room for the first turn
  local s1 = segs[1]
  local v = veh:getVelocity():length()
  local blend = clamp(6 + 1.2 * v, 8, 30)
  local along = clamp(((carPos.x - s1.ax) * s1.dx + (carPos.y - s1.ay) * s1.dy), 0, s1.len)
  local firstRoom = s1.len
  if joins[1] and joins[1].i == 1 and joins[1].vx then
    firstRoom = (joins[1].vx - s1.ax) * s1.dx + (joins[1].vy - s1.ay) * s1.dy - joins[1].R * math.tan(min(joins[1].theta, math.rad(170)) * 0.5)
  end
  local eAlong = clamp(along + blend, along + 1.5, max(along + 1.5, firstRoom - 0.5))
  local ex, ey = s1.ax + s1.dx * eAlong, s1.ay + s1.dy * eAlong
  local k = 0.42 * sqrt((ex - carPos.x) ^ 2 + (ey - carPos.y) ^ 2)
  raw[1] = {carPos.x, carPos.y, carPos.z, 1}
  bezier(raw, carPos.x, carPos.y, carPos.z, carPos.x + hx * k, carPos.y + hy * k, ex - s1.dx * k, ey - s1.dy * k, ex, ey, s1.az, 1)
  local curX, curY, curZ = ex, ey, s1.az
  local junctionS = {}
  local ji, i = 1, 1
  while i <= #segs do
    local sg = segs[i]
    local jn = joins[ji]
    if not jn or jn.i ~= i then
      if (sg.bx - curX) * sg.dx + (sg.by - curY) * sg.dy > 0.2 then line(raw, curX, curY, curZ, sg.bx, sg.by, sg.bz, i) end
      break
    end
    local nx = segs[jn.j]
    local sx, sy, exx, eyy
    if jn.vx then
      -- fillet around the vertex; shrink it if there is not room on either side
      local t = jn.R * math.tan(min(jn.theta, math.rad(170)) * 0.5)
      local roomIn = (jn.vx - curX) * sg.dx + (jn.vy - curY) * sg.dy - 0.5
      local roomOut = (nx.bx - jn.vx) * nx.dx + (nx.by - jn.vy) * nx.dy
      roomOut = jn.j == #segs and roomOut - 0.5 or roomOut * 0.5
      t = max(0.3, min(t, roomIn, roomOut))
      sx, sy = jn.vx - sg.dx * t, jn.vy - sg.dy * t
      exx, eyy = jn.vx + nx.dx * t, jn.vy + nx.dy * t
    else
      -- (nearly) parallel lines: an S from the end of one to just into the next
      sx, sy = sg.bx, sg.by
      local tout = min(4, nx.len * 0.5)
      exx, eyy = nx.ax + nx.dx * tout, nx.ay + nx.dy * tout
    end
    if (sx - curX) * sg.dx + (sy - curY) * sg.dy > 0.2 then line(raw, curX, curY, curZ, sx, sy, sg.bz, i) end
    local chord = sqrt((exx - sx) ^ 2 + (eyy - sy) ^ 2)
    local th = max(jn.theta, 0.01)
    local kk = chord * (2 / 3) * math.tan(th / 4) / math.sin(th / 2)
    if jn.junction then junctionS[#junctionS + 1] = #raw end
    bezier(raw, sx, sy, sg.bz, sx + sg.dx * kk, sy + sg.dy * kk, exx - nx.dx * kk, eyy - nx.dy * kk, exx, eyy, nx.az, i)
    curX, curY, curZ = exx, eyy, nx.az
    i = jn.j
    ji = ji + 1
  end
  local P = finalize(raw, segs)
  P.segs = segs
  for _, ri in ipairs(junctionS) do
    if P.s[min(P.n, ri)] then P.junctions[#P.junctions + 1] = P.s[min(P.n, ri)] end
  end
  P.curb = last.curb
  P.private = last.private
  P.trip = trip
  return P
end

-- closest point on the path to (px, py), searching near idx: s, index, lateral (+ right)
local function project(P, px, py, idx, ahead)
  local best, bestD2, bestT = idx or 1, math.huge, 0
  local i0 = max(1, (idx or 1) - 4)
  local i1 = min(P.n - 1, (idx or 1) + (ahead or 60))
  for i = i0, i1 do
    local ax, ay = P.x[i], P.y[i]
    local dx, dy = P.x[i + 1] - ax, P.y[i + 1] - ay
    local L2 = dx * dx + dy * dy
    local t = L2 > 1e-8 and clamp(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1) or 0
    local qx, qy = ax + dx * t, ay + dy * t
    local d2 = (px - qx) ^ 2 + (py - qy) ^ 2
    if d2 < bestD2 then best, bestD2, bestT = i, d2, t end
  end
  if P.n < 2 then return 0, 1, 0 end
  -- 2026-11-01: no segment scored (a NaN position, a stale index) left best at idx, and best + 1
  -- can be past the end of the path: clamp it like poseAt does
  best = clamp(best, 1, P.n - 1)
  local dx, dy = P.x[best + 1] - P.x[best], P.y[best + 1] - P.y[best]
  local L = sqrt(dx * dx + dy * dy)
  local s = P.s[best] + L * bestT
  local lat = 0
  if L > 1e-6 then
    local rx, ry = right2(dx / L, dy / L)
    lat = (px - (P.x[best] + dx * bestT)) * rx + (py - (P.y[best] + dy * bestT)) * ry
  end
  return s, best, lat
end

-- position and direction on the path at arc length s
local function poseAt(P, s, hint)
  -- 2026-11-01 01:44: Storm's log had "drive.lua:396: attempt to perform arithmetic on a nil value"
  -- twice: a hint at or past the last point (a car at the end of its path) read P.s[P.n + 1]. Clamp it.
  if P.n < 2 then return P.x[1] or 0, P.y[1] or 0, 0, 1, 1 end
  local i = clamp(hint or 1, 1, P.n - 1)
  if s <= 0 then i = 1 end
  while i < P.n - 1 and P.s[i + 1] < s do i = i + 1 end
  while i > 1 and P.s[i] > s do i = i - 1 end
  local ds = P.s[i + 1] - P.s[i]
  local t = ds > 1e-6 and clamp((s - P.s[i]) / ds, 0, 1) or 0
  local dx, dy = P.x[i + 1] - P.x[i], P.y[i + 1] - P.y[i]
  local L = sqrt(dx * dx + dy * dy)
  if L < 1e-6 then dx, dy, L = 0, 1, 1 end
  return P.x[i] + dx * t, P.y[i] + dy * t, dx / L, dy / L, i
end

-- ---------------------------------------------------------------------------------
-- perception: every vehicle near the car, with its size and how it is turning
-- ---------------------------------------------------------------------------------
local sizes = {}        -- [id] = {hl, hw, ox, oy}: half length / width, box centre offset from the reference node
local motion = {}       -- [id] = {fx, fy, t, yaw, speed, acc}

local function sizeOf(id)
  local s = sizes[id]
  if s then return s end
  s = {hl = 2.3, hw = 0.95, ox = 0, oy = 0}
  local v = getObjectByID(id)
  if v then
    local ok, bb = pcall(function() return v:getSpawnWorldOOBB() end)
    if ok and bb then
      local he = bb:getHalfExtents()
      if he and he.y > 0.3 then
        s.hl, s.hw = max(he.x, he.y), min(he.x, he.y)
        local c, p, d = bb:getCenter(), v:getPosition(), v:getDirectionVector()
        local dx, dy = d.x, d.y
        local L = sqrt(dx * dx + dy * dy)
        if L > 1e-3 then
          dx, dy = dx / L, dy / L
          local rx, ry = right2(dx, dy)
          s.oy = (c.x - p.x) * dx + (c.y - p.y) * dy
          s.ox = (c.x - p.x) * rx + (c.y - p.y) * ry
        end
      end
    end
  end
  sizes[id] = s
  return s
end

-- range: optional, metres (lumen/nn.lua's collision shield looks 200 m out; Lumen Drive uses SENSE_RANGE)
local function sense(selfId, pos, now, range)
  local out = {}
  local r2 = (range or SENSE_RANGE) ^ 2
  for id, o in pairs(map.objects) do
    if id ~= selfId and o.pos and (o.pos.x - pos.x) ^ 2 + (o.pos.y - pos.y) ^ 2 < r2 then
      local v = getObjectByID(id)
      if v and v:getJBeamFilename() ~= "unicycle" then
        local d = o.dirVec or vec3(0, 1, 0)
        local fx, fy = d.x, d.y
        local L = sqrt(fx * fx + fy * fy)
        if L > 1e-3 then fx, fy = fx / L, fy / L else fx, fy = 0, 1 end
        local vx, vy = o.vel and o.vel.x or 0, o.vel and o.vel.y or 0
        local speed = sqrt(vx * vx + vy * vy)
        local m = motion[id]
        local yaw, acc = 0, 0
        if m and now > m.t then
          local dt = now - m.t
          local cross = m.fx * fy - m.fy * fx
          local raw = math.asin(clamp(cross, -1, 1)) / dt
          yaw = m.yaw + (raw - m.yaw) * min(1, dt * 3)
          acc = m.acc + (((speed - m.speed) / dt) - m.acc) * min(1, dt * 2)
        end
        motion[id] = {fx = fx, fy = fy, t = now, yaw = yaw, speed = speed, acc = acc}
        local sz = sizeOf(id)
        local rx, ry = right2(fx, fy)
        out[#out + 1] = {id = id, x = o.pos.x + fx * sz.oy + rx * sz.ox, y = o.pos.y + fy * sz.oy + ry * sz.ox, z = o.pos.z,
                         fx = fx, fy = fy, vx = vx, vy = vy, speed = speed, yaw = yaw, acc = clamp(acc, -7, 2),
                         hl = sz.hl, hw = sz.hw}
      end
    end
  end
  return out
end

-- ---------------------------------------------------------------------------------
-- prediction: where each vehicle will be over the next 4 s
-- ---------------------------------------------------------------------------------
-- Along its road when it is driving along one (keeping its offset in its lane, straight on at
-- junctions), otherwise by its current turn rate, easing off. stopAt: metres to a red light or a
-- stop sign in front of it, so traffic that will stop is not treated as coming through.
local function predict(ag, stopAt)
  local poses = {}
  local function travel(t)
    local a = ag.acc
    local s = ag.speed * t + 0.5 * a * t * t
    if a < 0 and ag.speed + a * t < 0 then s = ag.speed * ag.speed / (-2 * a) end
    s = max(0, s)
    if stopAt then s = min(s, max(0, stopAt)) end
    return s
  end
  if ag.speed < 0.4 then
    for k = 1, PRED_STEPS do poses[k] = {ag.x, ag.y, ag.fx, ag.fy} end
    return poses, "still"
  end
  local mn = map.getMap().nodes
  local L = roads.closestLink(vec3(ag.x, ag.y, ag.z), {maxDist = 6})
  local alongRoad = false
  if L then
    local dx, dy = L.pb.x - L.pa.x, L.pb.y - L.pa.y
    local len = sqrt(dx * dx + dy * dy)
    if len > 0.5 then
      dx, dy = dx / len, dy / len
      local al = dx * ag.fx + dy * ag.fy
      if abs(al) > 0.86 and abs(ag.yaw) < 0.25 then
        alongRoad = true
        local from, to = L.a, L.b
        if al < 0 then from, to, dx, dy = L.b, L.a, -dx, -dy end
        local rx, ry = right2(dx, dy)
        local A = (al < 0) and L.pb or L.pa
        local lat = (ag.x - A.x) * rx + (ag.y - A.y) * ry
        local px, py = ag.x - rx * lat, ag.y - ry * lat
        local poly = {{px, py}}
        local prev, cur = from, to
        local need = travel(PRED_STEPS * PRED_DT) + 4
        local acc = 0
        while mn[cur] and acc < need and #poly < 24 do
          local c = mn[cur].pos
          acc = acc + sqrt((c.x - poly[#poly][1]) ^ 2 + (c.y - poly[#poly][2]) ^ 2)
          poly[#poly + 1] = {c.x, c.y}
          local nxt = roads.straightOn(prev, cur)
          if not nxt then break end
          prev, cur = cur, nxt
        end
        -- walk the polyline, keeping the lateral offset
        local segI, segS = 1, 0
        for k = 1, PRED_STEPS do
          local want = travel(k * PRED_DT)
          local x, y, fx, fy = poly[#poly][1], poly[#poly][2], ag.fx, ag.fy
          while segI < #poly do
            local a, b = poly[segI], poly[segI + 1]
            local sl = sqrt((b[1] - a[1]) ^ 2 + (b[2] - a[2]) ^ 2)
            if segS + sl >= want or segI == #poly - 1 then
              local t = sl > 1e-6 and clamp((want - segS) / sl, 0, 1.5) or 0
              fx, fy = sl > 1e-6 and (b[1] - a[1]) / sl or ag.fx, sl > 1e-6 and (b[2] - a[2]) / sl or ag.fy
              local qx, qy = right2(fx, fy)
              x, y = a[1] + (b[1] - a[1]) * t + qx * lat, a[2] + (b[2] - a[2]) * t + qy * lat
              break
            end
            segS = segS + sl
            segI = segI + 1
          end
          poses[k] = {x, y, fx, fy}
        end
      end
    end
  end
  if not alongRoad then
    -- constant turn rate, easing off
    local x, y, h = ag.x, ag.y, math.atan2(ag.fy, ag.fx)
    local w = clamp(ag.yaw, -0.9, 0.9)
    local sPrev = 0
    for k = 1, PRED_STEPS do
      local s = travel(k * PRED_DT)
      local ds = s - sPrev
      sPrev = s
      h = h + w * PRED_DT
      w = w * 0.92
      x, y = x + math.cos(h) * ds, y + math.sin(h) * ds
      poses[k] = {x, y, math.cos(h), math.sin(h)}
    end
  end
  return poses, alongRoad and "road" or "turn"
end

-- 2D oriented boxes overlap (separating axis test)
local function boxesOverlap(ax, ay, afx, afy, ahl, ahw, bx, by, bfx, bfy, bhl, bhw)
  local arx, ary = right2(afx, afy)
  local brx, bry = right2(bfx, bfy)
  local dx, dy = bx - ax, by - ay
  local axes = {{afx, afy}, {arx, ary}, {bfx, bfy}, {brx, bry}}
  for i = 1, 4 do
    local ux, uy = axes[i][1], axes[i][2]
    local ra = ahl * abs(afx * ux + afy * uy) + ahw * abs(arx * ux + ary * uy)
    local rb = bhl * abs(bfx * ux + bfy * uy) + bhw * abs(brx * ux + bry * uy)
    if abs(dx * ux + dy * uy) > ra + rb then return false end
  end
  return true
end

-- ---------------------------------------------------------------------------------
-- traffic signals near a point: {pos, dir, action}
-- ---------------------------------------------------------------------------------
local HOLD = {stop = true, briefStop = true}
local function signalsNear(pos, range)
  local ts = core_trafficSignals
  if not ts or not ts.getSignals then return {} end
  local ok, list = pcall(ts.getSignals)
  if not ok or type(list) ~= "table" then return {} end
  local out, r2 = {}, range * range
  for _, sig in pairs(list) do
    if type(sig) == "table" and not sig._invalid and sig.pos and sig.dir and (sig.pos.x - pos.x) ^ 2 + (sig.pos.y - pos.y) ^ 2 < r2 then
      local oks, _, data = pcall(sig.getState, sig)
      local action = oks and type(data) == "table" and data.action or "none"
      out[#out + 1] = {pos = sig.pos, dir = sig.dir, action = action, sig = sig}
    end
  end
  return out
end

-- metres from a vehicle (x, y heading fx, fy) to the stop line of a light or stop sign holding it
local function heldAt(sigs, x, y, fx, fy)
  local best
  for _, s in ipairs(sigs) do
    if HOLD[s.action] and s.dir.x * fx + s.dir.y * fy > 0.7 then
      local ahead = (s.pos.x - x) * fx + (s.pos.y - y) * fy
      local side = abs((s.pos.x - x) * fy - (s.pos.y - y) * fx)
      if ahead > -1 and ahead < 60 and side < 6 and (not best or ahead < best) then best = ahead end
    end
  end
  return best
end

-- ---------------------------------------------------------------------------------
-- the driver
-- ---------------------------------------------------------------------------------
-- car.drv = {mode = "follow" | "hold" | "pull" | "off", path, idx, s, arrived, reason, ...}
local reasonHook = nil   -- function(car, text, kind) set by the service for the thinking feed

local function D(car)
  car.drv = car.drv or {mode = "off", sendT = 0, idx = 1}
  return car.drv
end

local function say(car, d, key, text, kind)
  if d.reasonKey ~= key then
    d.reasonKey = key
    d.reason = text
    if reasonHook and text then pcall(reasonHook, car, text, kind or "go") end
  end
end

local function command(car, code)
  local v = car.id and getObjectByID(car.id)
  if v then v:queueLuaCommand("local c = controller.getController('lumenRobotaxi') if c and c.lumenDrive then " .. code .. " end") end
end

local function sendHold(car, halt)
  command(car, halt and "c.lumenDrive({halt = true})" or "c.lumenDrive({hold = true})")
end

-- the plan as a Lua table literal for the car: x, y every sample, target speeds
local function sendPlan(car, P, i0, i1, vp, brake)
  local xs, ys, vs = {}, {}, {}
  local ox, oy = P.x[i0], P.y[i0]
  for i = i0, i1 do
    xs[#xs + 1] = string.format("%.2f", P.x[i] - ox)
    ys[#ys + 1] = string.format("%.2f", P.y[i] - oy)
    vs[#vs + 1] = string.format("%.2f", vp[i])
  end
  command(car, string.format("c.lumenDrive({ox = %.3f, oy = %.3f, x = {%s}, y = {%s}, v = {%s}%s})",
    ox, oy, table.concat(xs, ","), table.concat(ys, ","), table.concat(vs, ","), brake and ", brake = true" or ""))
end

-- Guard decisions (commit to a yellow, go through one, never stand in a junction, gap accepted or refused)
-- are logged once per decision per car, at most 6 lines a second for the whole fleet, so they can be read
-- back from beamng.log.
local guardBudget = {t = 0, n = 0}
local function glog(car, d, key, text)
  local g = d.glog
  if not g then
    g = {}
    d.glog = g
  end
  local t = os.clock()
  if g[key] and t - g[key] < 12 then return end
  if t - guardBudget.t > 1 then guardBudget.t, guardBudget.n = t, 0 end
  if guardBudget.n >= 6 then return end
  guardBudget.n = guardBudget.n + 1
  g[key] = t
  log("I", logTag, string.format("guard: car %s: %s", tostring(car.slot), text))
end

-- one planning step for a car that is following a path
local function plan(car, veh, d, now)
  -- NN HOOK: Lumen NN plans this car when the brain is "nn" (lumen/nn.lua). It returns true when it
  -- handled the step; anything else (no weights, an error) falls through to the rules below.
  -- (a custom path from lumen/parking.lua is a maneuver, not a lane: the rules drive it whichever brain is selected)
  if not d.custom and (car.brain or M.brain) == "nn" then
    local okn, nnm = pcall(require, "lumen/nn")
    if okn and type(nnm) == "table" and nnm.plan then
      local okp, handled = pcall(nnm.plan, M, car, veh, d, now)
      if okp and handled then return end
      if not okp and not d.nnErr then
        d.nnErr = true
        log("E", logTag, "nn.plan failed, the rules drive this car: " .. tostring(handled))
      end
    end
  end
  local P = d.path
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  local vel = veh:getVelocity()
  local v = max(0, vel:dot(dir))
  local s, idx, lat = project(P, pos.x, pos.y, d.idx, 80)
  d.idx, d.s, d.lat = idx, s, lat

  -- the stretch to plan over
  local horizon = clamp(v * HORIZON_T + 25, HORIZON_MIN, HORIZON_MAX)
  local i0 = max(1, idx - 1)
  local i1 = i0
  while i1 < P.n and P.s[i1] < s + horizon do i1 = i1 + 1 end

  -- free speed: limit and curves
  local vp = {}
  for i = i0, i1 do
    local lim = P.vl[i]
    local k = P.k[i]
    if k > 1e-4 then lim = min(lim, sqrt(A_LAT / k)) end
    vp[i] = lim
  end

  local stopS, stopWhy, stopKind, brake = nil, nil, nil, false
  local function stopBefore(sv, why, kind)
    if not stopS or sv < stopS then stopS, stopWhy, stopKind = sv, why, kind end
  end
  -- the end of the path is a stop unless we are cruising on
  if d.stopAtEnd and P.s[i1] >= P.len - 0.01 then stopBefore(P.len, "stop", "end") end

  -- traffic lights and stop signs on the path ahead
  local sigs = signalsNear(pos, horizon + 30)
  d.sigs = sigs
  local sigNote
  local latch = d.latch                   -- committed yellow decisions (lumen/guard.lua)
  if not latch then
    latch = {}
    d.latch = latch
  end
  local seenSig, sigCentres = {}, {}
  for _, sg in ipairs(sigs) do
    local ss, si, sl = project(P, sg.pos.x, sg.pos.y, idx, (i1 - idx) + 4)
    local _, _, tx, ty = poseAt(P, ss, si)
    if ss > s + FRONT - 0.5 and abs(sl) < 7 and tx * sg.dir.x + ty * sg.dir.y > 0.6 then
      local lineS = ss - FRONT - LINE_GAP
      local key = tostring(sg.sig)
      seenSig[key] = true
      sigCentres[#sigCentres + 1] = ss + guard.LINE_BACK
      if sg.action == "stop" then
        stopBefore(lineS, "red", "signal")
        sigNote = sigNote or {key = "red", text = string.format("Red light in %d m. Stopping at the line.", floor(ss - s - FRONT + 0.5))}
      elseif sg.action == "alert" then
        -- Yellow, committed (lumen/guard.lua). The plain "is there room to stop" test flipped to "go" once
        -- the car had braked more gently than planned, and the car then ran the red in the dilemma zone.
        -- Once it has decided to stop it keeps stopping until the light is green.
        local need = v * v / (2 * A_DEC) + 1
        local room = lineS - s > need
        local was = latch[key]
        if guard.yellowHold(latch, key, true, room) then
          stopBefore(lineS, "yellow", "signal")
          sigNote = sigNote or {key = "yellow", text = "Light turning red. Stopping at the line."}
          if not was then
            glog(car, d, "yellow" .. key, string.format("commits to stopping for the yellow: %.0f m to the line at %.1f m/s, stopping needs %.0f m; the decision is kept until green",
              lineS - s, v, need))
          elseif not room then
            glog(car, d, "kept" .. key, string.format("stays committed to the yellow although the room test now says no (%.0f m to the line at %.1f m/s, %.0f m needed)",
              lineS - s, v, need))
          end
        else
          glog(car, d, "go" .. key, string.format("goes through the yellow: %.0f m to the line at %.1f m/s, stopping would need %.0f m", lineS - s, v, need))
        end
      elseif sg.action == "briefStop" then
        if d.stopSignDone ~= key then
          stopBefore(lineS, "stopsign", "signal")
          if lineS - s < 1.5 and v < 0.3 then
            d.stopSignT = (d.stopSignKey == key) and (d.stopSignT or 0) + PLAN_RATE or 0
            d.stopSignKey = key
            if d.stopSignT >= STOP_SIGN_WAIT then d.stopSignDone = key end
          end
          sigNote = sigNote or {key = "stopsign", text = "Stop sign ahead. Coming to a full stop, then going when the way is clear."}
        end
      elseif sg.action == "slow" then
        for i = i0, i1 do if P.s[i] > ss - 30 and P.s[i] < ss + 60 then vp[i] = min(vp[i], 8) end end
      else
        guard.greenReset(latch, key)        -- a green light ends a commitment to stop
      end
    end
  end
  guard.prune(latch, seenSig)

  -- vehicles: the one ahead in our lane, and everyone's predicted path
  local agents = sense(car.id, pos, now)
  local lead, leadGap, leadV
  local ego = {}
  local cross = {}                        -- vehicles crossing our path, for the junction gap test (lumen/guard.lua)
  for _, ag in ipairs(agents) do
    local as, ai, al = project(P, ag.x, ag.y, idx, (i1 - idx) + 4)
    ag.s, ag.lat = as, al
    local _, _, tx, ty = poseAt(P, as, ai)
    ag.along = ag.vx * tx + ag.vy * ty
    do
      local vd = ag.vx * ty - ag.vy * tx                       -- speed to the right of our travel
      local cosv
      if ag.speed > 1.0 then cosv = abs(ag.along) / ag.speed else cosv = abs(ag.fx * tx + ag.fy * ty) end
      -- (the window ends at the horizon: a vehicle beyond it would project onto the path's end)
      if cosv < 0.5 and as - s > -10 and as - s < 150 and abs(al) < 40 and as < P.s[i1] + 1 then
        local latDir = vd
        if abs(latDir) < 0.3 then latDir = ag.fx * ty - ag.fy * tx end     -- waiting: the way it is facing
        cross[#cross + 1] = {rs = as - s, rd = al - lat, vc = ag.speed, a = clamp(ag.acc or 0, -7, 2), len = 2 * ag.hl, wid = 2 * ag.hw,
                             dirSign = latDir >= 0 and 1 or -1, id = ag.id}
      end
    end
    local inLane = abs(al) < LANE_HALF + max(0, ag.hw - 0.95) and as > s + 1 and as < P.s[i1] + 5
    if inLane and (ag.speed < 0.5 or ag.fx * tx + ag.fy * ty > 0.3) then
      local gap = as - s - ag.hl - FRONT
      if not lead or gap < leadGap then lead, leadGap, leadV = ag, gap, max(0, ag.along) end
    end
  end

  -- A STANDING vehicle that reaches into the path (2026-11-01, test instance: a Lumen sent to a pickup stood
  -- five minutes at the start of its path, "yielding" to the rider's own parked pickup truck 2.1 m to the side,
  -- because the prediction below treated it as a car about to cross). Standing vehicles do not cross. The
  -- nearest one that intrudes is passed slowly when 0.3 m of real clearance remain, passed with a sideways nudge
  -- when the road leaves room for it (and no oncoming car is near when the nudge crosses the centre line), and
  -- otherwise it is a parked car blocking the way: stop before it as "parked", which the health check turns into
  -- "take another way" after a few seconds instead of a wait that never ends.
  local nudge, stillHandled = nil, {}
  if not d.custom then
    local best
    for _, ag in ipairs(agents) do
      if ag.speed < 0.4 and ag.s and ag.s < P.s[i1] then
        local _, _, tx, ty = poseAt(P, ag.s, idx)
        local cr, dt_ = abs(ag.fx * ty - ag.fy * tx), abs(ag.fx * tx + ag.fy * ty)
        local across = ag.hl * cr + ag.hw * dt_              -- its half extent across our path
        local along = ag.hl * dt_ + ag.hw * cr                -- and along it
        local gap = abs(ag.lat) - across - HALF_W             -- body to body, with us on the path line
        if gap < 0.45 and abs(ag.lat) > 0.6 and ag.s + along > s - HALF_LEN then
          stillHandled[ag.id] = true                          -- never a "car about to cross"; the nearest is dealt with now
          if not best or ag.s < best.ag.s then best = {ag = ag, gap = gap, across = across, along = along} end
        end
      end
    end
    if best then
      local ag = best.ag
      local sg = P.segs and P.segs[P.seg[idx] or 1] or nil
      local rad, off = (sg and sg.rad) or 3.6, (sg and sg.off) or 0
      local side = ag.lat > 0 and -1 or 1                    -- move away from it (right of travel is positive)
      local room = side > 0 and (rad - off - HALF_W - 0.3) or min(1.3, rad + off - HALF_W - 0.3)
      local need = best.gap >= 0.3 and 0 or (0.6 - best.gap)
      local oncoming = false
      if side < 0 and abs(off) > 0.3 then
        for _, o in ipairs(agents) do
          if o.speed > 1 and o.along and o.along < -1 and o.s and o.s - s > -5 and o.s - s < 60 + 3 * v and o.lat < -0.5 then oncoming = true break end
        end
      end
      local s0, s1 = ag.s - best.along - 3, ag.s + best.along + 3
      if need <= 0 or (need <= room and not oncoming) then
        nudge = {off = side * need, s0 = s0, s1 = s1, id = ag.id}
        stillHandled[ag.id] = true
        if lead == ag then lead = nil end
        for i = i0, i1 do if P.s[i] > s0 - 8 and P.s[i] < s1 + 2 then vp[i] = min(vp[i], (best.gap + need) < 0.9 and 4 or 7) end end
        say(car, d, "nudge" .. tostring(ag.id), need > 0 and "A parked car reaches into my lane. Moving over a little and passing it slowly."
          or "Passing close to a parked car, slowly.", "go")
      elseif oncoming and need <= room then
        stillHandled[ag.id] = true                         -- waiting for the other lane to clear, an ordinary wait
        stopBefore(s0 - FRONT - STOP_GAP + 3, "parked", "lead")
      else
        stillHandled[ag.id] = true
        if lead == ag then lead = nil end
        stopBefore(s0 - FRONT - STOP_GAP + 3, "parked", "parked")
        say(car, d, "parked" .. tostring(ag.id), "A parked car leaves no room to pass. If it does not move I will take another way.", "wait")
      end
      d.nudge = nudge
    else
      d.nudge = nil
    end
  end

  -- Parked-car caution (2026-11-01, training/caution.py "flat", Storm: "a car pulling out"): while a stationary vehicle
  -- stands beside our lane (not in it: that one the lead logic stops behind), within 3.5 m of the lane centre line, from
  -- alongside to 30 m ahead, the target is at most 0.65 x the posted limit. Simulator, 512 pull-out episodes: Drive's
  -- collision episodes 39.5% -> 13.7%; on the ordinary traffic distribution it changed nothing.
  local caution = false
  if not d.custom then
    for _, ag in ipairs(agents) do
      if ag.speed <= 1.0 and ag.s and abs(ag.lat) < 3.5 and abs(ag.lat) >= LANE_HALF + max(0, ag.hw - 0.95) then
        local rs = ag.s - s
        if rs + ag.hl > -2.4 and rs - ag.hl - FRONT < 30 then
          caution = true
          break
        end
      end
    end
    if caution then
      for i = i0, i1 do vp[i] = min(vp[i], 0.65 * P.vl[i] / LIMIT_SHARE) end
      say(car, d, "caution", "Passing parked cars slowly, in case one pulls out.", "go")
    end
  end
  d.caution = caution
  if lead then
    if leadV < 0.5 then
      -- leadGap is front bumper to its rear already. 2026-11-01: this added FRONT again and aimed
      -- the car 0.1 m INTO the stopped car (found by the training simulator's crash breakdown)
      stopBefore(s + leadGap - STOP_GAP, "stopped", "lead")
    else
      -- intelligent driver model toward the free speed, gap and closing speed
      local v0 = max(vp[i0], 1)
      local sStar = IDM_S0 + max(0, v * IDM_T + v * (v - leadV) / (2 * sqrt(IDM_A * IDM_B)))
      local acc = IDM_A * (1 - (v / v0) ^ 4 - (sStar / max(leadGap, 0.5)) ^ 2)
      local cap = max(0, v + clamp(acc, -A_HARD, IDM_A) * 0.6)
      for i = i0, i1 do if P.s[i] <= s + leadGap then vp[i] = min(vp[i], max(cap, 0)) end end
      d.following = lead.id
    end
  else
    d.following = nil
  end

  -- the plan so far, as time: where the Lumen will be at each prediction step
  local function egoTimeline(prof)
    local tl = {}
    local tAcc, sAcc, vAcc = 0, s, v
    local i = idx
    for k = 1, PRED_STEPS do
      local tEnd = k * PRED_DT
      while tAcc < tEnd and i < i1 do
        local target = prof[i + 1] or 0
        local ds = P.s[i + 1] - max(P.s[i], sAcc)
        if ds <= 0 then i = i + 1 else
          local vNext = target
          if vNext > vAcc then vNext = min(vNext, sqrt(vAcc * vAcc + 2 * A_ACC * ds)) end
          local vAvg = max(0.25, (vAcc + vNext) * 0.5)
          local dtSeg = ds / vAvg
          if tAcc + dtSeg > tEnd then
            local frac = (tEnd - tAcc) / dtSeg
            sAcc = sAcc + ds * frac
            vAcc = vAcc + (vNext - vAcc) * frac
            tAcc = tEnd
          else
            tAcc = tAcc + dtSeg
            sAcc = P.s[i + 1]
            vAcc = vNext
            i = i + 1
          end
        end
      end
      local x, y, fx, fy = poseAt(P, sAcc, i)
      tl[k] = {x, y, fx, fy, sAcc}
    end
    return tl
  end

  -- prediction: yield to anyone whose path crosses ours at the same time
  local conflict
  do
    local prof = {}
    for i = i0, i1 do prof[i] = vp[i] end
    if stopS then for i = i0, i1 do if P.s[i] >= stopS then prof[i] = 0 end end end
    local tl = egoTimeline(prof)
    for _, ag in ipairs(agents) do
      local behindUs = ag.s < s - 3 and abs(ag.lat) < 2.6
      local isLead = lead and ag.id == lead.id
      if not behindUs and not isLead and not stillHandled[ag.id] and (ag.speed > 0.4 or abs(ag.lat) < 3.5) then
        local stopAt = heldAt(sigs, ag.x, ag.y, ag.fx, ag.fy)
        local poses = predict(ag, stopAt and max(0, stopAt - ag.hl - 1) or nil)
        for k = 1, PRED_STEPS do
          local e, p = tl[k], poses[k]
          local margin = 0.5 + 0.05 * v
          if boxesOverlap(e[1], e[2], e[3], e[4], HALF_LEN + margin, HALF_W + 0.25,
                          p[1], p[2], p[3], p[4], ag.hl + 0.3, ag.hw + 0.2) then
            if not conflict or k < conflict.k then conflict = {k = k, s = e[5], ag = ag} end
            break
          end
        end
      end
    end
  end
  -- Committed: already inside a junction (the front bumper is a metre or more past the stop line and
  -- the body has not left the box). 2026-11-01, Storm: "a car sped through the intersection and lumen
  -- waited before completing the turn even though it was already mid turn". A stop in the box blocks
  -- everyone and is no safer, so a predicted conflict no longer stops the car there: it completes the
  -- turn. Only a hit predicted within 0.75 s that braking can still avoid brakes it.
  local inBox = false
  for _, jc in ipairs(guard.centres(sigCentres, guard.pathJunctions(P, project), s)) do
    if s + FRONT > jc - guard.LINE_BACK + 1.0 and s - HALF_LEN < jc + guard.BOX_AFTER then
      inBox = true
      break
    end
  end
  d.inBox = inBox
  if conflict then
    local tc = conflict.k * PRED_DT
    local needStop = v * v / (2 * A_HARD)
    local room = conflict.s - s
    if inBox then
      if tc <= 0.75 and room > needStop * 0.7 then
        stopBefore(conflict.s - 2.5, "yield", "conflict")
        brake = true
        d.yieldTo, d.yieldT = conflict.ag.id, now
      else
        glog(car, d, "commit" .. tostring(conflict.ag.id), string.format("inside the junction, so I finish the turn: a car (%.1f m/s) is predicted to cross my path in %.1f s",
          conflict.ag.speed, tc))
      end
    elseif room > needStop * 0.7 then
      stopBefore(conflict.s - 2.5, "yield", "conflict")
      if tc <= 1.0 and room < v * v / (2 * A_DEC) then brake = true end
      d.yieldTo, d.yieldT = conflict.ag.id, now
    end
  end

  -- Junction discipline (lumen/guard.lua): never plan to stand inside a junction (a stop for a car queued
  -- beyond it is moved back to the line in front of it), and hold at the line when a crossing vehicle would
  -- be in the junction while we cross.
  local guardNote
  do
    local jcs = guard.centres(sigCentres, guard.pathJunctions(P, project), s)
    if stopS and stopKind ~= "end" then
      local ns, jc = guard.boxStop(stopS, s, jcs, FRONT, LINE_GAP)
      if jc then
        glog(car, d, "box" .. floor(jc), string.format("will not stop inside the junction %.0f m ahead: the stop for %s is moved back to the line, %.0f m from the car",
          jc - s, tostring(stopKind), ns - s))
        stopS = ns
        guardNote = {key = "box" .. floor(jc), text = "I will not wait inside the junction. Stopping at the line before it."}
      end
    end
    local cs, refused, accepted = guard.crossStop(cross, s, v, FRONT, LINE_GAP)
    if cs and (not stopS or cs < stopS) then
      stopS, stopWhy, stopKind = cs, "junction", "junction"
      guardNote = {key = "gap" .. tostring(refused.c.id), text = string.format("A car crossing ahead would be in the junction when I get there. Holding at the line until it has passed.")}
      glog(car, d, "gap" .. tostring(refused.c.id), string.format("holds at the line, gap refused: a crossing car (%.1f m/s, %.0f m to the side) is in the junction %.1f to %.1f s from now, we would be there %.1f to %.1f s (margin %.1f s)",
        refused.c.vc, abs(refused.c.rd), refused.tIn, refused.tOut, refused.teIn, refused.teOut, guard.GAP_MARGIN))
    elseif accepted and not cs and accepted.c.rs < 40 then
      glog(car, d, "ok" .. tostring(accepted.c.id), string.format("gap accepted: a crossing car (%.1f m/s, %.0f m to the side) is in the junction %.1f to %.1f s from now, we would be there %.1f to %.1f s",
        accepted.c.vc, abs(accepted.c.rd), accepted.tIn, accepted.tOut, accepted.teIn, accepted.teOut))
    end
  end

  -- the profile: stops, then braking (backwards) and acceleration (forwards) limits
  if stopS then for i = i0, i1 do if P.s[i] >= stopS then vp[i] = 0 end end end
  for i = i1 - 1, i0, -1 do
    local ds = P.s[i + 1] - P.s[i]
    local lim = sqrt(vp[i + 1] * vp[i + 1] + 2 * A_DEC * ds)
    if vp[i] > lim then vp[i] = lim end
  end
  for i = idx + 1, i1 do
    local ds = max(0, P.s[i] - s)
    local lim = sqrt(v * v + 2 * A_ACC * ds) + 0.5
    if vp[i] > lim then vp[i] = lim end
  end

  -- the nudge around a parked car: the same line shifted sideways, easing in over 10 m before it and out after
  local line = P
  if nudge and nudge.off ~= 0 then
    line = {x = {}, y = {}}
    for i = i0, i1 do
      local a, c = max(1, i - 1), min(P.n, i + 1)
      local dx, dy = P.x[c] - P.x[a], P.y[c] - P.y[a]
      local L = sqrt(dx * dx + dy * dy)
      local tx, ty = 0, 1
      if L > 1e-6 then tx, ty = dx / L, dy / L end
      local si, w = P.s[i], 1
      if si < nudge.s0 then w = clamp((si - (nudge.s0 - 10)) / 10, 0, 1)
      elseif si > nudge.s1 then w = clamp(((nudge.s1 + 10) - si) / 10, 0, 1) end
      w = w * w * (3 - 2 * w)
      line.x[i], line.y[i] = P.x[i] + ty * nudge.off * w, P.y[i] - tx * nudge.off * w
    end
  end
  sendPlan(car, line, i0, i1, vp, brake)
  d.vp, d.i0, d.i1 = vp, i0, i1
  d.stopS = stopS
  d.stopKind = stopKind
  d.conflict = conflict

  -- what it is doing, in words (only when it changes)
  local stopNear = stopS and stopS - s < max(25, v * v / (2 * A_DEC) + 10)
  if conflict and stopKind == "conflict" and stopNear then
    local ag = conflict.ag
    local side = ""
    local rx, ry = right2(dir.x, dir.y)
    local rel = (ag.x - pos.x) * rx + (ag.y - pos.y) * ry
    if abs(rel) > 3 then side = rel > 0 and " from the right" or " from the left" end
    say(car, d, "yield" .. tostring(ag.id), string.format("A car coming%s will cross my path in %.1f s. Yielding until it has passed.",
      side, conflict.k * PRED_DT), "wait")
    d.state = "wait"
  elseif stopKind == "signal" and stopNear and sigNote then
    say(car, d, sigNote.key, sigNote.text, "wait")
    d.state = "wait"
  elseif guardNote and stopNear then
    say(car, d, guardNote.key, guardNote.text, "wait")
    d.state = "wait"
  elseif stopKind == "lead" and stopNear then
    say(car, d, "leadstop", "The car ahead has stopped. Stopping behind it with a gap.", "wait")
    d.state = "wait"
  elseif lead and leadGap < 40 and leadV > 0.5 then
    say(car, d, "follow", string.format("Following the car ahead at %.1f s (%d m).", leadGap / max(v, 1), floor(leadGap + 0.5)), "go")
    d.state = "go"
  else
    if stopKind ~= "end" then say(car, d, "clear", nil) end
    d.state = "go"
  end

  -- arrived: at the end of the path and stopped
  if d.stopAtEnd and P.len - s < (d.arriveTol or 2.5) and v < 0.35 then d.arrived = true end
end

-- ---------------------------------------------------------------------------------
-- public
-- ---------------------------------------------------------------------------------
function M.setReasonHook(fn) reasonHook = fn end

-- NN HOOK: which brain plans (the N switch). "drive" = Lumen Drive's rules, "nn" = Lumen NN.
M.brain = "drive"
function M.setBrain(name)
  if name ~= "drive" and name ~= "nn" then return false end
  M.brain = name
  return true
end
function M.getBrain() return M.brain end
-- which brain plans this car: a test can give single cars their own (car.brain); otherwise the fleet's
function M.brainOf(car) return (car and car.brain) or M.brain end

-- drive this trip and stop at its end (opts.curb pulls the stop in to the curb)
function M.follow(car, trip, opts)
  local veh = car.id and getObjectByID(car.id)
  if not veh or not trip then return false end
  local ok, P = pcall(buildPath, trip, veh, {curb = opts and opts.curb})
  if not ok or not P or P.n < 2 then
    log("W", logTag, "could not build a path: " .. tostring(P))
    return false
  end
  local d = D(car)
  d.mode, d.path, d.idx, d.arrived = "follow", P, 1, false
  d.vp, d.i0, d.i1, d.stopS, d.stopKind, d.conflict = nil, nil, nil, nil, nil, nil
  d.stopAtEnd = not (opts and opts.through)
  d.cruising = opts and opts.cruise or false
  d.custom, d.noSolids, d.arriveTol = nil, nil, nil
  d.stopSignDone, d.stopSignKey, d.stopSignT = nil, nil, nil
  d.sendT = 0
  return true
end

-- Drive a line made by the caller and stop at its end: the parking maneuver (lumen/parking.lua). pts is a list
-- of {x, y, z} for the REAR AXLE (the controller steers the rear axle along the line). opts.speed is the speed
-- on the line and opts.slowSpeed the speed over its last opts.slowLen metres. The rules drive it whichever
-- brain is selected (plan() leaves custom paths alone), and the AR view draws it like any other path.
function M.followCustom(car, pts, opts)
  local veh = car.id and getObjectByID(car.id)
  if not veh or not pts or #pts < 2 then return false end
  opts = opts or {}
  local z0 = veh:getPosition().z
  local raw = {}
  for i, p in ipairs(pts) do raw[i] = {p[1], p[2], p[3] or z0, 1} end
  local P = finalize(raw, {{vl = opts.speed or 2.5}})
  if P.n < 2 then return false end
  local slowFrom = P.len - (opts.slowLen or 6)
  for i = 1, P.n do
    if P.s[i] >= slowFrom then P.vl[i] = min(P.vl[i], opts.slowSpeed or 1.4) end
  end
  local d = D(car)
  d.mode, d.path, d.idx, d.arrived = "follow", P, 1, false
  d.vp, d.i0, d.i1, d.stopS, d.stopKind, d.conflict = nil, nil, nil, nil, nil, nil
  d.stopAtEnd, d.cruising = true, false
  d.custom, d.noSolids, d.arriveTol, d.nnAr = true, true, opts.arriveTol, nil
  d.stopSignDone, d.stopSignKey, d.stopSignT = nil, nil, nil
  d.sendT = 0
  return true, P
end

-- come to a gentle stop where it is and hold there (parking brake, never the pedal)
function M.stop(car)
  local d = D(car)
  d.mode, d.path, d.arrived = "hold", nil, false
  d.custom, d.noSolids = nil, nil
  d.sendT = 0
end

-- stop now, firmly (going to service, an emergency)
function M.halt(car)
  local d = D(car)
  d.mode, d.path = "hold", nil
  d.custom, d.noSolids = nil, nil
  sendHold(car, true)
  d.sendT = 0.5
end

-- nothing drives the car (it is backing up under the controller's reverse loop, or being moved)
function M.release(car)
  local d = D(car)
  d.mode, d.path = "off", nil
  d.custom, d.noSolids = nil, nil
  command(car, "c.lumenDrive(nil)")
end

-- Back at `speed` m/s (0 lets go) with a steering value from -1 to 1, right positive: the controller's reverse
-- speed loop, steered by lumen/parking.lua while a Lumen backs out of a space. Call release() first.
function M.reverseCmd(car, speed, steer)
  local v = car.id and getObjectByID(car.id)
  if not v then return end
  v:queueLuaCommand(string.format("local c = controller.getController('lumenRobotaxi') if c and c.reverseAt then c.reverseAt(%.2f, %.3f) end",
    speed or 0, steer or 0))
end

function M.forget(car) car.drv = nil end

-- roam on its own: random public roads within reach of the player, one after another
function M.cruise(car, centre)
  local veh = car.id and getObjectByID(car.id)
  if not veh then return false end
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  local c = centre or pos
  -- a way-back trip (below) runs until it ends: re-planning it every second only thrashes the path
  local d0 = D(car)
  if d0.backUntil and os.clock() < d0.backUntil and d0.mode == "follow" and (M.remaining(car) or 0) > 3 then return true end
  -- targets on the car's own piece of the served network: it can always route to them
  local myPiece = roads.componentAt(pos, dir)
  local anyLink = false
  -- Candidate links first (cheap: no route planning), then tried in the order that spreads the fleet out:
  -- the target farthest from where the other cars are heading. With 20 Lumens they would otherwise pile
  -- onto the same few roads near the rider.
  for _ = 1, 2 do
    local cands = {}
    for attempt = 1, 12 do
      -- the first 6 tries are around the player (the fleet stays near them); when the player is outside
      -- the service area nothing may be covered there, so the rest are around the car itself
      local L = roads.randomLink(attempt <= 6 and c or pos, 250, 900, myPiece)
      if L then
        local mx, my = (L.pa.x + L.pb.x) * 0.5, (L.pa.y + L.pb.y) * 0.5
        local gap = 1e9
        for other, t in pairs(roamTargets) do
          if other ~= car then
            local dd = sqrt((mx - t[1]) ^ 2 + (my - t[2]) ^ 2)
            if dd < gap then gap = dd end
          end
        end
        cands[#cands + 1] = {L = L, mx = mx, my = my, gap = gap}
        if #cands >= 6 then break end
      end
    end
    if #cands > 0 then anyLink = true end
    table.sort(cands, function(x, y) return x.gap > y.gap end)
    for _, k in ipairs(cands) do
      local stop = roads.stopPoint(k.L, 0.5)
      local trip = roads.planTrip(pos, dir, k.L, stop, {speed = veh:getVelocity():length(), noReversal = true})
      if trip and trip.remaining and trip.remaining > 150 and not roads.hasReversal(trip.nodes) then
        if M.follow(car, trip, {through = true, cruise = true}) then
          roamTargets[car] = {k.mx, k.my}
          if geofence and geofence.hasZones() then
            local z = geofence.where(k.mx, k.my)
            log("I", logTag, string.format("roaming: car %s heads for (%.0f, %.0f) in %s, %d m of route, %s", tostring(car.slot), k.mx, k.my,
              z and string.format("%s (tier %s)", z.name, tostring(z.tier)) or "no open zone", math.floor(trip.remaining),
              k.gap < 1e8 and string.format("%.0f m from the nearest other target", k.gap) or "no other target yet"))
          end
          return true
        end
      end
    end
  end
  -- Nothing to roam to from here: the car is not on a served road (or on a piece with nowhere to go).
  -- It takes the shortest legal way back to the nearest served road instead of standing on a road the
  -- service does not use.
  local rt = roads.reenterTrip and roads.reenterTrip(pos, dir)
  if rt and M.follow(car, rt, {through = true, cruise = true}) then
    d0.backUntil = os.clock() + 8
    local pc = roads.pieceInfo and roads.pieceInfo(myPiece)
    log("I", logTag, string.format("roaming: car %s has no roaming route from (%.0f, %.0f) (%s); driving to the nearest served road over %d nodes, %d m",
      tostring(car.slot), pos.x, pos.y, myPiece and string.format("on piece %s, %.1f km", tostring(myPiece), pc and pc.km or 0) or "not on a served road",
      #rt.nodes, math.floor(rt.remaining or 0)))
    return true
  end
  local d = D(car)
  if not anyLink and geofence and geofence.hasZones() and (not d.noRoamLogT or os.clock() - d.noRoamLogT > 30) then
    d.noRoamLogT = os.clock()
    log("I", logTag, string.format("roaming: no served road between 250 and 900 m of (%.0f, %.0f); car %s waits", c.x, c.y, tostring(car.slot)))
  end
  M.stop(car)
  return false
end

-- Pull over: onto the edge of the road ahead, clear of junctions, on the shoulder when the rays
-- find flat open ground there. Returns false when there is no path to pull off from.
function M.pullOver(car)
  local d = D(car)
  local P = d.path
  local veh = car.id and getObjectByID(car.id)
  if not P or not veh or d.mode ~= "follow" then return false end
  local v = max(0, veh:getVelocity():dot(veh:getDirectionVector()))
  local s0 = (d.s or 0) + max(12, v * v / (2 * A_DEC) * 0.8 + 6)
  local ramp, hold = 22, 8
  -- keep the whole maneuver clear of junctions
  for _ = 1, 20 do
    local clear = true
    for _, js in ipairs(P.junctions) do
      if js > s0 - 12 and js < s0 + ramp + hold + 15 then clear = false s0 = js + 15 break end
    end
    if clear then break end
  end
  if s0 + ramp + hold > P.len then return false end
  -- how far right: to the road edge, plus the shoulder if it is flat and open
  local seg = P.trip and P.trip.nodes and 1
  local sx, sy, fx, fy, si = poseAt(P, s0 + ramp, 1)
  local mid = P.seg[si] or 1
  local _ = seg
  local rad, off = 3.5, 1.8
  do
    local trip = P.trip
    local nn = trip and trip.nodes
    if nn and nn[mid - 1] and nn[mid] then
      local o, r = laneFor(nn[mid - 1], nn[mid])
      off, rad = o, r
    end
  end
  local rx, ry = right2(fx, fy)
  local edge = rad - off                 -- metres from our lane line to the road edge
  local z0 = P.z[si] or veh:getPosition().z
  local shoulder = 0
  for k = 1, 6 do
    local o = edge + k * 0.25
    local gx, gy = sx + rx * o, sy + ry * o
    local hit = castRayStatic(vec3(gx, gy, z0 + 1.5), vec3(0, 0, -1), 3)
    local gz = z0 + 1.5 - hit
    local side = castRayStatic(vec3(sx + rx * (edge - 0.3), sy + ry * (edge - 0.3), z0 + 0.5), vec3(rx, ry, 0), o + 0.3)
    if hit >= 3 or abs(gz - z0) > 0.18 or side < o then break end
    shoulder = k * 0.25
  end
  local shift = max(0.4, edge - HALF_W - CURB_MARGIN + shoulder)
  -- the new path: the old one, eased sideways by `shift`, ending at the stop
  local raw = {}
  local _, i0 = project(P, veh:getPosition().x, veh:getPosition().y, d.idx, 40)
  for i = max(1, i0 - 1), P.n do
    local sv = P.s[i]
    if sv > s0 + ramp + hold then break end
    local f = 0
    if sv > s0 then f = sv >= s0 + ramp and 1 or (1 - math.cos(math.pi * (sv - s0) / ramp)) * 0.5 end
    local tx, ty = 0, 1
    if i < P.n then
      local dx, dy = P.x[i + 1] - P.x[i], P.y[i + 1] - P.y[i]
      local L = sqrt(dx * dx + dy * dy)
      if L > 1e-6 then tx, ty = dx / L, dy / L end
    end
    local qx, qy = right2(tx, ty)
    raw[#raw + 1] = {P.x[i] + qx * shift * f, P.y[i] + qy * shift * f, P.z[i], P.seg[i]}
  end
  local segs = {}
  for i = 1, max(1, P.seg[P.n] or 1) do segs[i] = {vl = 8} end
  local Q = finalize(raw, segs)
  for i = 1, Q.n do Q.vl[i] = min(8, P.vl[min(P.n, i0 + i)] or 8) end
  Q.junctions, Q.trip = {}, P.trip
  d.mode, d.path, d.idx, d.arrived, d.stopAtEnd = "follow", Q, 1, false, true
  d.vp, d.i0, d.i1, d.stopS, d.stopKind, d.conflict = nil, nil, nil, nil, nil, nil
  d.pull = {shift = shift, shoulder = shoulder}
  d.sendT = 0
  return true, shift, shoulder
end

-- Cruising on: before the path runs out, extend the ROUTE from its last node to another random
-- road (never straight back) and rebuild the path from the car along the rest of the old route,
-- so a moving car never improvises a fresh line from wherever it is. 2026-11-01: fresh plans
-- from the car's spot made it cut corners and once took it through a gap into a wall.
function M.keepCruising(car, centre)
  local d = car.drv
  local veh = car.id and getObjectByID(car.id)
  if not d or not veh or d.mode ~= "follow" or not d.cruising or not d.path or not d.path.trip then return false end
  local P, trip = d.path, d.path.trip
  local nodes = trip.nodes
  local sg = P.segs and P.segs[P.seg[d.idx] or 1]
  local ti = sg and sg.ti or 1
  -- the first route node actually ahead of the car (a junction curve is labelled with the segment
  -- it leaves, so the car can be past nodes[ti] already)
  local mn = map.getMap().nodes
  local cp, cd = veh:getPosition(), veh:getDirectionVector()
  while ti < #nodes and mn[nodes[ti]] and (mn[nodes[ti]].pos - cp):dot(cd) < 1 do ti = ti + 1 end
  if #nodes < 2 or ti > #nodes then return false end
  local last, prev = nodes[#nodes], nodes[#nodes - 1]
  for _ = 1, 8 do
    local L = roads.randomLink(centre or veh:getPosition(), 250, 900, roads.componentAt(veh:getPosition(), veh:getDirectionVector()))
    if L then
      for _, o in ipairs({{L.a, L.b}, {L.b, L.a}}) do
        if roads.legal(L, o[1], o[2]) then
          local ext = roads.path(last, o[1], {[roads.linkKey(last, prev)] = true})
          if ext and #ext >= 2 then
            local list = {}
            for q = ti, #nodes do list[#list + 1] = nodes[q] end
            for q = 2, #ext do list[#list + 1] = ext[q] end
            list[#list + 1] = o[2]
            if roads.hasReversal(list) then goto continue end
            local stop = roads.stopPoint(L, 0.5)
            local t2 = roads.tripFromNodes(list, veh:getPosition(), stop)
            if t2 and M.follow(car, t2, {through = true, cruise = true}) then return true end
          end
        end
        ::continue::
      end
    end
  end
  return false
end

function M.arrived(car) return car.drv and car.drv.arrived or false end
function M.stopKind(car) return car.drv and car.drv.mode == "follow" and car.drv.stopKind or nil end

-- test dump: the whole path, the route's nodes, the car and the slice last sent, for top-down plots
function M.debugDump(car)
  local d = car.drv
  local v = car.id and getObjectByID(car.id)
  if not d or not d.path or not v then return nil end
  local P = d.path
  local px, py = {}, {}
  for i = 1, P.n do px[i], py[i] = P.x[i], P.y[i] end
  local mn = map.getMap().nodes
  local nx, ny = {}, {}
  for _, n in ipairs(P.trip and P.trip.nodes or {}) do
    if mn[n] then nx[#nx + 1], ny[#ny + 1] = mn[n].pos.x, mn[n].pos.y end
  end
  local segs = {}
  for _, sg in ipairs(P.segs or {}) do segs[#segs + 1] = {sg.ax, sg.ay, sg.bx, sg.by, sg.off, sg.ti} end
  local pos, dir = v:getPosition(), v:getDirectionVector()
  return {px = px, py = py, nx = nx, ny = ny, segs = segs, car = {pos.x, pos.y, dir.x, dir.y}, idx = d.idx, s = d.s, lat = d.lat,
          i0 = d.i0, i1 = d.i1, cruising = d.cruising, len = P.len}
end

-- test overlay: the plan the car is following (green), where it will stop (red), the yield target
M.debug = false
function M.drawDebug(car, veh)
  local d = car.drv
  if not d or not d.path or not d.i0 then return end
  local P = d.path
  local col = ColorF(0.1, 1, 0.5, 0.9)
  for i = d.i0, math.min(d.i1 or P.n, P.n) - 1 do
    debugDrawer:drawLine(vec3(P.x[i], P.y[i], P.z[i] + 0.4), vec3(P.x[i + 1], P.y[i + 1], P.z[i + 1] + 0.4), col)
  end
  if d.stopS then
    local x, y = poseAt(P, d.stopS + FRONT, d.idx)
    local z = veh:getPosition().z
    debugDrawer:drawSphere(vec3(x, y, z + 0.6), 0.5, ColorF(1, 0.2, 0.2, 0.8))
  end
  if d.conflict and d.conflict.ag then
    local ag = d.conflict.ag
    debugDrawer:drawSphere(vec3(ag.x, ag.y, ag.z + 1.8), 0.6, ColorF(1, 0.5, 0, 0.9))
  end
end
function M.state(car) return car.drv and car.drv.state or "go" end
function M.mode(car) return car.drv and car.drv.mode or "off" end
function M.reason(car) return car.drv and car.drv.reason end
function M.remaining(car)
  local d = car.drv
  if not d or not d.path then return nil end
  return max(0, d.path.len - (d.s or 0))
end

-- the plan as drawn by the AR view: the path ahead with speeds, where it will stop and why, and
-- the predicted path of the car it is yielding to
function M.arPlan(car)
  local d = car.drv
  -- NN HOOK: with the NN driving, the AR path is the NN's own offset line
  if d and M.brainOf(car) == "nn" and d.mode == "follow" and d.nnAr then return d.nnAr end
  if not d or d.mode ~= "follow" or not d.path or not d.vp then return nil end
  local P = d.path
  -- i0 / i1 / vp were made for the path before this one when a new, shorter path has just been set
  local i0, i1 = d.i0 or 0, min(d.i1 or 0, P.n)
  if i0 < 1 or i1 < i0 or not P.x[i0] then return nil end
  local pts, spd = {}, {}
  for i = i0, i1 do
    pts[#pts + 1] = vec3(P.x[i], P.y[i], P.z[i])
    spd[#spd + 1] = d.vp[i] or 0
  end
  local out = {points = pts, speeds = spd, startS = P.s[i0], s = d.s, stopS = d.stopS, stopKind = d.stopKind,
               endS = d.stopAtEnd and P.len or nil, state = d.state}
  if d.conflict then
    local ag = d.conflict.ag
    local poses = predict(ag, nil)
    local line = {vec3(ag.x, ag.y, ag.z)}
    for k = 2, #poses, 2 do line[#line + 1] = vec3(poses[k][1], poses[k][2], ag.z) end
    out.threat = line
  end
  return out
end

-- ---------------------------------------------------------------------------------
-- emergency layer: people and solid obstacles on the car's own path, for every brain
-- ---------------------------------------------------------------------------------
-- Storm, 2026-11-01: "Lumen will just run into people" and "it tried to go into a park and it sped up and
-- hit one of the walls". sense() leaves walking characters out (the rider beside the pickup must not
-- block it) and nothing looked at walls, pillars or rails at all. This runs ten times a second for every
-- car that is following a path, whichever brain wrote the plan:
--   people   an active walking character inside the corridor of the path ahead (or, at its own speed,
--            about to be) and closer than the car can stop in, plus a margin
--   solids   rays along the path 0.6 m above the road: forward along each 3 m piece, and sideways at its
--            end, so a gap narrower than the car (gate pillars, a wall corner) is seen too
-- Either one gets the car a firm halt, with the reason in the thinking feed. A car held by a solid
-- obstacle for 5 s counts as blocked in robotaxi.lua health(), which backs it up and takes another way.
local GUARD_RATE = 0.1
local PERSON_CORRIDOR = HALF_W + 1.1     -- m either side of the path centre
local SOLID_SIDE = 0.9                   -- m: a post this close to the path touches the body (half width 0.95)
local SCAN_STEP = 3.0                    -- m between rays along the path
local walkerIds, walkerT, walkerList, walkerIdsT = {}, -1, {}, -1e9

-- every active walking character, refreshed 10 times a second and shared by all cars
local function walkers(now)
  if now - walkerT < 0.1 then return walkerList end
  walkerT = now
  if now - walkerIdsT > 30 then walkerIds, walkerIdsT = {}, now end        -- vehicle ids get reused
  walkerList = {}
  for id, o in pairs(map.objects) do
    if o.pos then
      local u = walkerIds[id]
      if u == nil then
        local v = getObjectByID(id)
        u = (v ~= nil and v:getJBeamFilename() == "unicycle") and true or false
        walkerIds[id] = u
      end
      if u then
        local v = getObjectByID(id)
        if v and v:getActive() then
          walkerList[#walkerList + 1] = {id = id, x = o.pos.x, y = o.pos.y, vx = o.vel and o.vel.x or 0, vy = o.vel and o.vel.y or 0}
        end
      end
    end
  end
  return walkerList
end

-- metres from the front bumper to the first solid thing (terrain, walls, posts, rails: whatever
-- castRayStatic hits, never vehicles) in the corridor of the path within `range`, or nil
local function solidAhead(P, s, idx, range)
  -- 2026-11-02: the rays used the path's recorded height, and where the real ground was higher (a crest,
  -- a junction apron, a curb cut) they hit the road itself: 367 false halts in a few minutes. Each sample
  -- now finds the real ground first (a ray down from 0.8 m above the path) and scans 0.55 m above THAT.
  local function point(sv)
    local x, y, tx, ty, i = poseAt(P, min(sv, P.len), idx)
    local zr = (P.z[i] + P.z[min(i + 1, P.n)]) * 0.5
    local h = castRayStatic(vec3(x, y, zr + 0.8), vec3(0, 0, -1), 2.5)
    local gz = h < 2.5 and (zr + 0.8 - h) or zr
    return x, y, gz + 0.55, tx, ty
  end
  local ax, ay, az = point(s + FRONT)
  local k = 1
  while k * SCAN_STEP <= range + SCAN_STEP and (s + FRONT + (k - 1) * SCAN_STEP) < P.len do
    local sv = s + FRONT + k * SCAN_STEP
    local bx, by, bz, btx, bty = point(sv)
    local dx, dy, dz = bx - ax, by - ay, bz - az
    local len = sqrt(dx * dx + dy * dy + dz * dz)
    if len > 0.5 then
      local hit = castRayStatic(vec3(ax, ay, az), vec3(dx / len, dy / len, dz / len), len + 0.2)
      if hit < len + 0.2 then return (k - 1) * SCAN_STEP + hit end
      local o = vec3(bx, by, bz)
      local rx, ry = bty, -btx
      if castRayStatic(o, vec3(rx, ry, 0), SOLID_SIDE) < SOLID_SIDE or castRayStatic(o, vec3(-rx, -ry, 0), SOLID_SIDE) < SOLID_SIDE then
        return k * SCAN_STEP
      end
    end
    ax, ay, az = bx, by, bz
    k = k + 1
  end
  return nil
end

-- returns "person" or "obstacle" and how far ahead it is, or nil when the way is clear
local function emergency(car, veh, d, now)
  local P = d.path
  if not P or P.n < 2 then return nil end
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  local v = veh:getVelocity():dot(dir)
  if v < -0.3 then return nil end                      -- backing up runs on the controller's own loop
  v = max(0, v)
  local s, idx = project(P, pos.x, pos.y, d.idx or 1, 80)
  local stopDist = v * v / (2 * 6.0) + v * 0.25 + 1.0
  for _, p in ipairs(walkers(now)) do
    local ps, pi, plat = project(P, p.x, p.y, idx, 80)
    local ahead = ps - s - FRONT
    if ahead > -1.0 and ahead < 45 and ahead < stopDist + 3.0 then
      local _, _, tx, ty = poseAt(P, ps, pi)
      local latV = p.vx * ty - p.vy * tx               -- sideways speed, right of the path positive
      local lat2 = plat + latV * clamp(ahead / max(v, 1.5), 0, 2.5)
      if abs(plat) < PERSON_CORRIDOR or abs(lat2) < PERSON_CORRIDOR then
        d.solidHits = 0
        return "person", ahead
      end
    end
  end
  -- Solid obstacles. Never for a car that is stopped or stopping (nothing moves into anything), never
  -- while a maneuver says so (d.noSolids: parking goes within centimetres of walls), and only when it is
  -- seen on several checks in a row: 0.3 s at speed, 0.5 s when creeping.
  local stopping = v < 0.5 and d.stopS and d.stopS - s < 2.5
  if d.noSolids or stopping then
    d.solidHits = 0
    return nil
  end
  local ahead = solidAhead(P, s, idx, stopDist + 3)
  if ahead and ahead < stopDist + 1.0 then
    d.solidHits = (d.solidHits or 0) + 1
    if d.solidHits >= (v > 1.0 and 3 or 5) then return "obstacle", ahead end
  else
    d.solidHits = 0
  end
  return nil
end

-- ---------------------------------------------------------------------------------
-- A/B test bookkeeping (2026-11-02): only while robotaxi.abTest() has switched it on. Every car
-- counts what it did under its own brain, the same way for both: distance, time, time standing still,
-- collisions, red-light and stop-sign runs (from the car and the signals alone, so the NN's plan is
-- counted exactly like Drive's), emergency halts, blocked recoveries, service retirements.
-- ---------------------------------------------------------------------------------
M.ab = nil
function M.abReset() M.ab = {drive = {}, nn = {}} end
function M.abCount(car, key, n)
  if not M.ab then return end
  local t = M.ab[M.brainOf(car)]
  if t then t[key] = (t[key] or 0) + (n or 1) end
end

local function abWatch(car, veh, d, now)
  local P = d.path
  if not P or d.mode ~= "follow" then d.abSig = nil return end
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  local v = veh:getVelocity():dot(dir)
  local s, idx = project(P, pos.x, pos.y, d.idx or 1, 80)
  local watch = d.abSig or {}
  d.abSig = watch
  local seen = {}
  for _, sg in ipairs(signalsNear(pos, 60)) do
    local ss, si, sl = project(P, sg.pos.x, sg.pos.y, idx, 80)
    local _, _, tx, ty = poseAt(P, ss, si)
    if abs(sl) < 7 and tx * sg.dir.x + ty * sg.dir.y > 0.6 and ss > s + FRONT - 0.5 then
      local key = tostring(sg.sig)
      local gap = ss - s - FRONT
      local w = watch[key] or {}
      seen[key] = true
      w.action, w.gap, w.v = sg.action, gap, v
      w.still = w.still or (sg.action == "briefStop" and v < 0.6 and gap < 10)
      watch[key] = w
    end
  end
  -- a signal that was just ahead and is no longer: the front bumper crossed its line
  for key, w in pairs(watch) do
    if not seen[key] then
      if w.gap < 5 then
        if w.action == "stop" and w.v > 1.5 then M.abCount(car, "red_run") end
        if w.action == "briefStop" and not w.still and w.v > 0.6 then M.abCount(car, "stopsign_run") end
      end
      watch[key] = nil
    end
  end
end

function M.abReport()
  if not M.ab then return nil end
  local out = {}
  for name, t in pairs(M.ab) do
    local km, time = (t.dist or 0) / 1000, t.time or 0
    local inc = (t.hit or 0) + (t.red_run or 0) + (t.stopsign_run or 0) + (t.svc_flipped or 0)
    out[name] = {km = km, hours = time / 3600, mean_speed = time > 0 and (t.dist or 0) / time or 0,
      still_share = time > 0 and (t.still or 0) / time or 0, hits = t.hit or 0, red_run = t.red_run or 0,
      stopsign_run = t.stopsign_run or 0, flipped = t.svc_flipped or 0, stuck = t.svc_stuck or 0, damaged = t.svc_damaged or 0,
      blocked = t.blocked or 0, halt_person = t.halt_person or 0, halt_obstacle = t.halt_obstacle or 0,
      relief_s = t.nn_relief_s or 0, relief_share = time > 0 and (t.nn_relief_s or 0) / time or 0,
      incidents = inc, incidents_per_100km = km > 0 and inc / km * 100 or nil}
  end
  return out
end

-- every frame for every fleet car on the road
function M.update(car, veh, dt, now)
  local d = car.drv
  if not d or d.mode == "off" then return end
  if M.ab and d.mode == "follow" then
    local sp = veh:getVelocity():length()
    M.abCount(car, "dist", sp * dt)
    M.abCount(car, "time", dt)
    if sp < 0.5 then M.abCount(car, "still", dt) end
    d.abT = (d.abT or 0) - dt
    if d.abT <= 0 then
      d.abT = 0.2
      local okw, errw = pcall(abWatch, car, veh, d, now)
      if not okw and not d.abErr then
        d.abErr = true
        log("E", logTag, "abWatch failed: " .. tostring(errw))
      end
    end
  end
  if d.mode == "follow" and d.path then
    d.guardT = (d.guardT or 0) - dt
    if d.guardT <= 0 then
      d.guardT = GUARD_RATE
      local okg, why, ahead = pcall(emergency, car, veh, d, now)
      if okg and not why then d.guardOn = false end
      if okg and why then
        if not d.guardOn then
          d.guardOn = true
          M.abCount(car, "halt_" .. why)
        end
        sendHold(car, true)
        d.state, d.stopKind = "wait", why
        say(car, d, why, why == "person" and "A person is close to my path. Stopping until they are clear."
          or "Something solid is on my path ahead, a wall or a post. Stopping before it.", why == "person" and "wait" or "warn")
        if now - (d.guardLogT or -1e9) > 3 then
          d.guardLogT = now
          log("I", logTag, string.format("emergency: car %d halts for %s, %.1f m ahead at %.1f m/s", car.slot, why, ahead or -1,
            veh:getVelocity():length()))
        end
        return
      elseif not okg then
        if now - (d.guardLogT or -1e9) > 5 then
          d.guardLogT = now
          log("E", logTag, "emergency layer failed: " .. tostring(why))
        end
      end
    end
  end
  d.sendT = (d.sendT or 0) - dt
  if d.sendT > 0 then return end
  -- Cars far from the camera plan less often: the controller follows the last plan and only holds after
  -- 1.5 s without one. 5 Hz near, 2.5 Hz past 250 m, 1.25 Hz past 600 m (20 cars: the plan strings are
  -- the main per-car cost).
  local far = veh:getPosition():distance(core_camera.getPosition())
  d.sendT = PLAN_RATE * ((far > 600) and 4 or (far > 250) and 2 or 1)
  if d.mode == "hold" then
    sendHold(car, false)
    d.sendT = 1.0                       -- a keep-alive; the controller holds on its own
    d.state = "wait"
    return
  end
  local ok, err = pcall(plan, car, veh, d, now)
  if not ok then
    log("E", logTag, "planning failed: " .. tostring(err))
    sendHold(car, false)
    d.sendT = 1.0
  end
end

M._buildPath, M._project, M._predict, M._boxesOverlap, M._laneFor = buildPath, project, predict, boxesOverlap, laneFor
-- NN HOOK exports (lumen/nn.lua builds on these)
M._poseAt, M._sense, M._signalsNear, M._command, M._sendPlan, M._say, M._sendHold = poseAt, sense, signalsNear, command, sendPlan, say, sendHold
M._heldAt = heldAt
-- lumen/parking.lua looks for people with the same list the emergency layer uses
M._walkers = walkers

return M
