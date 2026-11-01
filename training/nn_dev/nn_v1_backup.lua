-- Lumen NN (Frontier Studios): the Lumen's learned driver.
-- Storm, 2026-11-01: "Make the N button switch from Lumen Drive to Lumen NN. Train the neural net
-- now. Make it 3x better than the rule based button."
--
-- A small neural network (83 inputs, two hidden layers of 64, 2 outputs) trained in a simulator
-- (training/) chooses two things five times a second: the speed to drive at, and how far to one side
-- of its lane to drive. Everything else stays shared with Lumen Drive: the lane line and its smooth
-- turns, the pure-pursuit steering and the pedals in the vehicle controller, the end-of-trip stop,
-- pull over, backing away from a reversing car, recovery.
--
-- It plugs into drive.lua through the "NN HOOK" blocks: when drive.brain == "nn", drive.plan() calls
-- nn.plan(drive, car, veh, d, now) first, and falls back to the rules when this returns false (no
-- weights, an error). The observation below is the simulator's (training/sim.py observe()), the spec:
--   8  signed curvature of the lane ahead at 5, 10, 20, 30, 45, 60, 80, 100 m (right turns positive, x50, clipped to +-3)
--   1  safest speed over those points (curve speed plus braking in 2.2 m/s2), /20, at most 1.5
--   1  posted limit /20
--   5  speed /20, lateral offset in lane /2, heading error /(0.5), previous target speed /20, previous offset /2
--   4  room to the right edge /4, room to the left edge /4, two-way road, shoulder /2
--   1  distance from the front bumper to the next stop line or junction /100 (clipped -0.1..1.5)
--   5  that line's state, one-hot: none or open junction, green, yellow, red, stop sign
--   1  stop sign: seconds stood still at the line /3 (0..1)
--   56 the 8 nearest vehicles, each: ahead /50, right /5, speed along the lane minus ours /15,
--      sideways speed /10, length (width if crossing) /5, crossing, present
--   1  distance left on the path /200 (0..2)
-- Outputs: out[1] -> target speed = sigmoid(out[1]) * 1.1 * limit (at most 25 m/s); out[2] -> offset = tanh(out[2]) * 4 m.

local M = {}
local logTag = "lumen_nn"

local OBS_DIM, HIDDEN = 83, 64
local FRONT, HALF_LEN, LIMIT_SHARE, V_MAX = 2.6, 2.4, 0.95, 25
local A_LAT, A_DEC, A_ACC = 2.2, 2.2, 1.6
local EGO_W = 1.9
local AHEAD = {5, 10, 20, 30, 45, 60, 80, 100}
local BLEND = 10                               -- m over which the offset line leaves the car's own position
local WEIGHTS_FILE = "/lua/ge/extensions/lumen/nn_weights.json"
local HORIZON_T, HORIZON_MIN, HORIZON_MAX = 9, 45, 200

local W, loadErr = nil, nil
local sqrt, abs, min, max, floor, exp = math.sqrt, math.abs, math.min, math.max, math.floor, math.exp

local function clampf(x, lo, hi) if x < lo then return lo elseif x > hi then return hi end return x end

local function tanh(x)
  if x > 20 then return 1 elseif x < -20 then return -1 end
  local e = exp(2 * x)
  return (e - 1) / (e + 1)
end

-- ---------------------------------------------------------------------------------
-- the network
-- ---------------------------------------------------------------------------------
function M.load()
  if W or loadErr then return W end
  local j = jsonReadFile(WEIGHTS_FILE)
  if type(j) ~= "table" or type(j.w1) ~= "table" then
    loadErr = "no weights at " .. WEIGHTS_FILE
    log("E", logTag, loadErr)
    return nil
  end
  if #j.w1 ~= OBS_DIM * HIDDEN or #j.b1 ~= HIDDEN or #j.w2 ~= HIDDEN * HIDDEN or #j.b2 ~= HIDDEN or #j.w3 ~= HIDDEN * 2 or #j.b3 ~= 2 then
    loadErr = "weights have the wrong shape"
    log("E", logTag, loadErr)
    return nil
  end
  W = {w1 = j.w1, b1 = j.b1, w2 = j.w2, b2 = j.b2, w3 = j.w3, b3 = j.b3, info = j.info or {}}
  log("I", logTag, "Lumen NN weights loaded: " .. tostring(W.info.note or W.info.trained or "no note"))
  return W
end

function M.reload() W, loadErr = nil, nil return M.load() end
function M.info() return W and W.info or nil, loadErr end

local h1, h2, out = {}, {}, {0, 0}
-- x: array 1..83; returns the two raw outputs
local function forward(x)
  local w1, b1, w2, b2, w3, b3 = W.w1, W.b1, W.w2, W.b2, W.w3, W.b3
  for j = 1, HIDDEN do h1[j] = b1[j] end
  for i = 1, OBS_DIM do
    local xi = x[i]
    if xi ~= 0 then
      local base = (i - 1) * HIDDEN
      for j = 1, HIDDEN do h1[j] = h1[j] + xi * w1[base + j] end
    end
  end
  for j = 1, HIDDEN do h1[j] = tanh(h1[j]) end
  for j = 1, HIDDEN do h2[j] = b2[j] end
  for i = 1, HIDDEN do
    local hi = h1[i]
    local base = (i - 1) * HIDDEN
    for j = 1, HIDDEN do h2[j] = h2[j] + hi * w2[base + j] end
  end
  for j = 1, HIDDEN do h2[j] = tanh(h2[j]) end
  local o1, o2 = b3[1], b3[2]
  for i = 1, HIDDEN do
    local hi = h2[i]
    o1 = o1 + hi * w3[(i - 1) * 2 + 1]
    o2 = o2 + hi * w3[(i - 1) * 2 + 2]
  end
  out[1], out[2] = o1, o2
  return o1, o2
end

-- for the numeric equivalence test against numpy
function M.forwardTest(x)
  if not M.load() then return nil end
  local a, b = forward(x)
  return {a, b}
end

-- ---------------------------------------------------------------------------------
-- per-path caches: signed curvature and the junction nodes along the route
-- ---------------------------------------------------------------------------------
local function ensurePath(drive, P)
  if P.nnk then return end
  local n = P.n
  local ks = {}
  for i = 1, n do
    local a, c = max(1, i - 2), min(n, i + 2)
    local k = P.k[i] or 0
    local sgn = 1
    if c - a >= 2 then
      local cross = (P.x[i] - P.x[a]) * (P.y[c] - P.y[i]) - (P.y[i] - P.y[a]) * (P.x[c] - P.x[i])
      sgn = cross > 0 and -1 or 1      -- counter-clockwise (a left turn) is negative: right turns are positive
    end
    ks[i] = sgn * k
  end
  P.nnk = ks
  local J = {}
  local nodes = P.trip and P.trip.nodes
  if nodes then
    local mn = map.getMap().nodes
    for _, nd in ipairs(nodes) do
      local node = mn[nd]
      if node and map.getNodeLinkCount(nd) > 2 then
        local s, _, lat = drive._project(P, node.pos.x, node.pos.y, 1, n)
        if abs(lat) < 8 then J[#J + 1] = s end
      end
    end
  end
  P.nnJ = J
end

-- like drive.project but searching a window behind the car too (vehicles behind count)
local function projectWide(P, px, py, idx)
  local best, bestD2, bestT = idx, math.huge, 0
  local i0, i1 = max(1, idx - 60), min(P.n - 1, idx + 90)
  for i = i0, i1 do
    local ax, ay = P.x[i], P.y[i]
    local dx, dy = P.x[i + 1] - ax, P.y[i + 1] - ay
    local L2 = dx * dx + dy * dy
    local t = 0
    if L2 > 1e-8 then t = clampf(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1) end
    local qx, qy = ax + dx * t, ay + dy * t
    local d2 = (px - qx) ^ 2 + (py - qy) ^ 2
    if d2 < bestD2 then best, bestD2, bestT = i, d2, t end
  end
  local dx, dy = P.x[best + 1] - P.x[best], P.y[best + 1] - P.y[best]
  local L = sqrt(dx * dx + dy * dy)
  if L < 1e-6 then return P.s[best], 0, 1, 0, 1 end
  local tx, ty = dx / L, dy / L
  local s = P.s[best] + L * bestT
  local lat = (px - (P.x[best] + dx * bestT)) * ty - (py - (P.y[best] + dy * bestT)) * tx   -- right of travel is (ty, -tx)
  return s, lat, tx, ty, best
end

-- value of a per-point array at path coordinate sx (0 beyond the end)
local function sampleAt(P, arr, sx, hint)
  if sx >= P.len then return 0, hint end
  local i = hint
  while i < P.n and P.s[i + 1] <= sx do i = i + 1 end
  return arr[i] or 0, i
end

-- ---------------------------------------------------------------------------------
-- the observation
-- ---------------------------------------------------------------------------------
local STATE = {none = 1, slow = 1, alert = 2, stop = 3, briefStop = 4}   -- one-hot slot (0 none/open, 1 green, 2 yellow, 3 red, 4 stop sign)
local x = {}
local info = {}

-- builds the 83 inputs into x; info gets the extras the plan needs. Returns false when the car has no path.
local function features(drive, car, veh, d, now, s, idx, lat, v)
  local P = d.path
  ensurePath(drive, P)
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  local k = 0
  -- curvature and the safest speed
  local vsafeMin = 1e9
  local hint = idx
  for _, a in ipairs(AHEAD) do
    local kap
    kap, hint = sampleAt(P, P.nnk, s + a, hint)
    k = k + 1
    x[k] = clampf(kap * 50, -3, 3)
    local vcurve2 = A_LAT / max(abs(kap), 1e-4)
    vsafeMin = min(vsafeMin, sqrt(vcurve2 + 2 * A_DEC * a))
  end
  info.vsafe = vsafeMin
  k = k + 1; x[k] = min(vsafeMin, 30) / 20
  local vlim = (P.vl[idx] or 13.9) / LIMIT_SHARE
  info.vlim = vlim
  k = k + 1; x[k] = vlim / 20
  -- own state
  local tx, ty = 0, 1
  do
    local i = min(idx, P.n - 1)
    local dx, dy = P.x[i + 1] - P.x[i], P.y[i + 1] - P.y[i]
    local L = sqrt(dx * dx + dy * dy)
    if L > 1e-6 then tx, ty = dx / L, dy / L end
  end
  local hl = sqrt(dir.x * dir.x + dir.y * dir.y)
  local hx, hy = dir.x / max(hl, 1e-6), dir.y / max(hl, 1e-6)
  local crossTD = tx * hy - ty * hx                 -- > 0: heading is to the left of the lane
  local epsi = math.atan2(-crossTD, tx * hx + ty * hy)   -- right of the lane is positive
  k = k + 1; x[k] = v / 20
  k = k + 1; x[k] = lat / 2
  k = k + 1; x[k] = epsi * 2
  k = k + 1; x[k] = (d.nnPrevT or 0) / 20
  k = k + 1; x[k] = (d.nnPrevD or 0) / 2
  -- the road
  local sg = P.segs and P.segs[P.seg[idx] or 1] or nil
  local rad, off = 3.6, 0
  if sg then rad, off = sg.rad or 3.6, sg.off or 0 end
  local edgeR, edgeL = rad - off, -(rad + off)
  info.edgeR, info.edgeL = edgeR, edgeL
  k = k + 1; x[k] = (edgeR - lat) / 4
  k = k + 1; x[k] = (lat - edgeL) / 4
  k = k + 1; x[k] = abs(off) > 0.3 and 1 or 0       -- two-way road (the lane sits to one side of the centre line)
  k = k + 1; x[k] = 0                               -- shoulder: not known in the game
  -- the next stop line or junction ahead
  local sigs = drive._signalsNear(pos, 170)
  local bestDist, bestState, bestKind = 1e9, 0, nil
  for _, sgl in ipairs(sigs) do
    local ss, _, stx, sty, sl = nil, nil, nil, nil, nil
    local ps, plat, ptx, pty = projectWide(P, sgl.pos.x, sgl.pos.y, idx)
    ss, sl, stx, sty = ps, plat, ptx, pty
    if ss > s + FRONT - 0.5 and abs(sl) < 7 and stx * sgl.dir.x + sty * sgl.dir.y > 0.6 then
      local dist = ss - (s + FRONT)
      if dist < bestDist then bestDist, bestState, bestKind = dist, STATE[sgl.action] or 1, "signal" end
      if STATE[sgl.action] == 4 then info.signKey = tostring(sgl.sig) end
    end
  end
  if bestKind == nil then
    for _, js in ipairs(P.nnJ or {}) do
      local dist = (js - 6) - (s + FRONT)
      if dist > -2 and dist < bestDist then bestDist, bestState, bestKind = dist, 0, "open" end
    end
  end
  if bestDist > 150 then bestDist, bestState, bestKind = 1e9, 0, nil end
  info.lineDist, info.lineState, info.lineKind = bestDist, bestState, bestKind
  k = k + 1; x[k] = bestKind and clampf(bestDist / 100, -0.1, 1.5) or 1.5
  for st = 0, 4 do k = k + 1; x[k] = (bestKind and bestState == st) and 1 or 0 end
  if not bestKind then x[k - 4] = 1 end             -- no junction at all counts as "none", like the simulator
  -- stop sign: how long it has stood still at the line
  local waited = 0
  if bestState == 4 then
    if bestDist < 8 and v < 0.3 then
      if d.nnSignKey ~= info.signKey then d.nnSignKey, d.nnSignAt = info.signKey, now end
      waited = clampf((now - (d.nnSignAt or now)) / 3, 0, 1)
    elseif d.nnSignKey == info.signKey and d.nnSignAt then
      waited = clampf((now - d.nnSignAt) / 3, 0, 1)
    end
  end
  k = k + 1; x[k] = waited
  info.waited = waited
  -- vehicles
  local agents = drive._sense(car.id, pos, now)
  local cand = {}
  for _, ag in ipairs(agents) do
    local as, al, atx, aty = projectWide(P, ag.x, ag.y, idx)
    local rs, rd = as - s, al - lat
    if abs(rs) <= 80 and abs(rd) <= 30 then
      local speed = ag.speed or sqrt(ag.vx * ag.vx + ag.vy * ag.vy)
      local vs = ag.vx * atx + ag.vy * aty
      local vd = ag.vx * aty - ag.vy * atx                       -- right of travel
      local cosv
      if speed > 1.0 then cosv = abs(vs) / speed else cosv = abs(ag.fx * atx + ag.fy * aty) end
      local crossing = cosv < 0.5
      if crossing then vs = 0 end
      cand[#cand + 1] = {d2 = rs * rs + (rd * 1.5) ^ 2, rs = rs, rd = rd, vs = vs, vd = crossing and vd or 0,
                         size = crossing and (2 * ag.hw) or (2 * ag.hl), cross = crossing and 1 or 0, speed = speed, id = ag.id}
    end
  end
  table.sort(cand, function(a, b) return a.d2 < b.d2 end)
  info.near = {}
  for i = 1, 8 do
    local c = cand[i]
    if c then
      k = k + 1; x[k] = c.rs / 50
      k = k + 1; x[k] = c.rd / 5
      k = k + 1; x[k] = (c.vs - v) / 15
      k = k + 1; x[k] = c.vd / 10
      k = k + 1; x[k] = c.size / 5
      k = k + 1; x[k] = c.cross
      k = k + 1; x[k] = 1
      info.near[i] = c
    else
      for _ = 1, 7 do k = k + 1; x[k] = 0 end
    end
  end
  -- distance left on the path
  local left = d.stopAtEnd and (P.len - s) or 1e9
  k = k + 1; x[k] = clampf(left / 200, 0, 2)
  return k == OBS_DIM
end

-- the observation of a car as a plain array (tests compare this against the simulator's layout)
function M.observe(drive, car)
  local veh = car.id and getObjectByID(car.id)
  local d = car.drv
  if not veh or not d or not d.path then return nil end
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  local v = max(0, veh:getVelocity():dot(dir))
  local s, idx, lat = drive._project(d.path, pos.x, pos.y, d.idx, 80)
  if not features(drive, car, veh, d, 0, s, idx, lat, v) then return nil end
  local copy = {}
  for i = 1, OBS_DIM do copy[i] = x[i] end
  return copy
end

-- ---------------------------------------------------------------------------------
-- one planning step (called by drive.plan() through the NN HOOK)
-- ---------------------------------------------------------------------------------
local function smoothstep(t) t = clampf(t, 0, 1) return t * t * (3 - 2 * t) end

function M.plan(drive, car, veh, d, now)
  if not M.load() then return false end
  local P = d.path
  if not P or d.mode ~= "follow" or P.n < 3 then return false end
  local pos, dir = veh:getPosition(), veh:getDirectionVector()
  local v = max(0, veh:getVelocity():dot(dir))
  local s, idx, lat = drive._project(P, pos.x, pos.y, d.idx, 80)
  d.idx, d.s, d.lat = idx, s, lat
  if not features(drive, car, veh, d, now, s, idx, lat, v) then return false end

  local o1, o2 = forward(x)
  local share = 1 / (1 + exp(-o1))
  local vT = min(share * 1.1 * info.vlim, V_MAX)
  -- shared safety: never faster than the curves ahead allow (the simulator's roads have no tight junction
  -- turns, so the network has not met them; Lumen Drive caps its speed the same way)
  vT = min(vT, info.vsafe)
  local dT = tanh(o2) * 4
  dT = clampf(dT, info.edgeL + EGO_W / 2 + 0.2, info.edgeR - EGO_W / 2 - 0.2)
  d.nnPrevT, d.nnPrevD = vT, dT

  -- the stretch to plan over, and the offset line
  local horizon = clampf(v * HORIZON_T + 25, HORIZON_MIN, HORIZON_MAX)
  local i0 = max(1, idx - 1)
  local i1 = i0
  while i1 < P.n and P.s[i1] < s + horizon do i1 = i1 + 1 end
  local Q = {x = {}, y = {}}
  local vp = {}
  local pts = {}
  for i = i0, i1 do
    local a, c = max(1, i - 1), min(P.n, i + 1)
    local dx, dy = P.x[c] - P.x[a], P.y[c] - P.y[a]
    local L = sqrt(dx * dx + dy * dy)
    local tx, ty = 0, 1
    if L > 1e-6 then tx, ty = dx / L, dy / L end
    local o = lat + (dT - lat) * smoothstep((P.s[i] - s) / BLEND)
    Q.x[i], Q.y[i] = P.x[i] + ty * o, P.y[i] - tx * o
    local sp = vT
    -- the trip's own end: a braking-limited stop at the stop point (the same as the rules)
    if d.stopAtEnd then sp = min(sp, sqrt(2 * A_DEC * max(0, P.len - P.s[i]))) end
    -- and the forward limit the rules use, so acceleration stays comfortable
    sp = min(sp, sqrt(v * v + 2 * A_ACC * max(0, P.s[i] - s)) + 0.5)
    vp[i] = sp
    pts[#pts + 1] = vec3(Q.x[i], Q.y[i], P.z[i])
  end
  drive._sendPlan(car, Q, i0, i1, vp, false)
  d.vp, d.i0, d.i1 = vp, i0, i1

  -- why it is slowing, for the health checks and the thinking feed
  local stopKind
  local nearLine = info.lineKind == "signal" and (info.lineState == 2 or info.lineState == 3 or (info.lineState == 4 and info.waited < 0.55))
  if vT < 0.6 * max(v, 1) or vT < 1.0 then
    if nearLine and info.lineDist < 60 then
      stopKind = "signal"
    else
      for _, c in pairs(info.near) do
        if c.cross == 1 and abs(c.rs) < 40 and abs(c.rd) < 12 then stopKind = "conflict" break end
      end
      if not stopKind then
        for _, c in pairs(info.near) do
          if c.cross == 0 and c.rs > 0 and c.rs < 40 and abs(c.rd) < 2.2 then stopKind = "lead" break end
        end
      end
    end
  end
  d.stopKind = stopKind
  d.stopS = (stopKind == "signal" and info.lineDist < 60) and (s + max(0, info.lineDist - 1.2)) or nil
  d.conflict = nil
  d.state = (vT < 1.0 and v < 2.5) and "wait" or "go"

  d.nnAr = {points = pts, speeds = {}, startS = P.s[i0], s = s, stopS = d.stopS, stopKind = stopKind,
            endS = d.stopAtEnd and P.len or nil, state = d.state}
  for i = i0, i1 do d.nnAr.speeds[#d.nnAr.speeds + 1] = vp[i] end

  -- the thinking feed: only when the behaviour changes
  if stopKind == "signal" then
    local what = info.lineState == 3 and "Red light" or info.lineState == 2 and "Yellow light" or "Stop sign"
    drive._say(car, d, "nn_sig" .. info.lineState, string.format("%s ahead in %d m. Lumen NN is slowing to stop at the line.", what, floor(info.lineDist + 0.5)), "wait")
  elseif stopKind == "conflict" then
    drive._say(car, d, "nn_conf", "Cross traffic ahead. Lumen NN is slowing to let it pass.", "wait")
  elseif stopKind == "lead" then
    drive._say(car, d, "nn_lead", "Lumen NN is holding back behind the car ahead.", "wait")
  elseif abs(dT) > 1.0 then
    drive._say(car, d, dT > 0 and "nn_right" or "nn_left", string.format("Lumen NN is driving %.1f m to the %s of the lane to make room.", abs(dT), dT > 0 and "right" or "left"), "go")
  else
    drive._say(car, d, "nn_go", nil)
  end

  if d.stopAtEnd and P.len - s < 2.5 and v < 0.35 then d.arrived = true end
  return true
end

return M
