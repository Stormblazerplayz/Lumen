-- Lumen NN predicted-collision shield for lateral manoeuvres (attempt 2, 2026-11-01). Line-by-line port of
-- training/shield.py (the reference); training/lua_shield_test.py checks the two against each other under LuaJIT.
--
-- Pure computation, no game calls: nn.lua senses the vehicles (a 200 m scan of map.objects, projected on the path) and the
-- guards, and calls  shield.plan(st, agents, capFn). See the header of shield.py for the design (a pass is checked as a
-- whole: go out, hold until the vehicles that block the lane are behind, come back; against 9 s of predicted traffic).
--
--   st      = { v0, ed, epsi, vA, dT, lo, hi, near }
--               v0 own speed (m/s); ed lateral offset of the car from its lane centre (right positive, m); epsi heading
--               relative to the lane (rad, right positive); vA the speed the network asked for (m/s); dT the offset it
--               asked for (m); lo, hi the drivable offsets (edge of the road minus half the car and 0.2 m); near the
--               distance to the next junction or stop line (1e9 when none)
--   agents  = array of { rs, rd, ad, vs, vd, acc, aL, cross, onc }  (every vehicle within 200 m along the path and 30 m
--               across): rs, rd position relative to the car along the path and to the right (m), ad its lateral
--               position relative to OUR LANE CENTRE (m), vs, vd its velocity along and across the path (m/s; for an
--               oncoming car vs is negative), acc its smoothed acceleration along its motion (clamped to -7..2), aL its
--               extent along the path (a crossing car: its width), cross, onc booleans; optional aW its extent
--               across the path (default: 5 m for a crossing car, 2.1 m otherwise, what the simulator assumes);
--               optional held (a crossing car waiting for its light or sign) with heldD, the distance sideways to its
--               line: it is predicted to stop there
--   capFn   = function(offset) -> vcap, dstop : the lane guards' speed cap and the distance to the point the car must
--               stop at (1e9 when none) for a car that plans to be at that lateral offset
--   returns speed, offset, mode  (mode 0 requested plan, 1 lane plan, 2 continue the manoeuvre, 3 least bad,
--               4 lane plan because the junction gate refused a new manoeuvre, 5 stays in its lane under the guards)
local M = {}

local min, max, abs, sqrt, exp, floor = math.min, math.max, math.abs, math.sqrt, math.exp, math.floor

local H, DTS = 36, 0.25
local A_ACC, A_PLAN_DEC, STOP_DECEL = 1.6, 3.0, 2.2
local M_S0, M_SV, M_D = 0.6, 0.06, 0.35
local M_ONC_T = 0.6              -- an oncoming car's box is also longer by 0.6 s of closing speed
local AGENT_ACC_T, AGENT_LAT_T = 2.0, 2.5
local EGO_L, EGO_W = 4.8, 1.9
local SPEED_TRIES = {1.0, 0.8, 0.6, 0.4, 0.25, 0.1}
local J_GATE_START, J_GATE_T = 50.0, 4.0
local RANGE = 200.0
local OUT_LANE = 0.45
local AGENT_W, CROSS_LEN = 2.1, 5.0
local KS = 12
local CLUSTER_GAP = 22.0
local RET_M0, RET_MV = 1.0, 0.15
local LANE_BAND = 2.5
local CK_MAX = 24                -- a pass must have the group behind it within 6 s
local CLEAR_M = 0.6              -- a pass clears the group's outermost edge by this much (m)
-- sideways models: delay s, time constant s, overshoot gain, drift time constant (0: none), reference speed m/s
local FAST = {0.20, 0.33, 1.12, 0.8, 3.0}
local SLOW = {0.40, 0.55, 1.00, 0.0, 10.0}

M.H, M.DTS, M.KS, M.OUT_LANE = H, DTS, KS, OUT_LANE

local function sign(x) if x > 0 then return 1 elseif x < 0 then return -1 else return 0 end end

-- the KS nearest visible vehicles, with the extents the car assumes
local function view(agents)
  local list = {}
  for _, a in ipairs(agents) do
    if abs(a.rs) < RANGE and abs(a.rd) < 30 then list[#list + 1] = a end
  end
  table.sort(list, function(p, q) return abs(p.rs) < abs(q.rs) end)
  local ag = {}
  for i = 1, min(KS, #list) do
    local a = list[i]
    ag[i] = {rs = a.rs, rd = a.rd, ad = a.ad, vs = a.vs, vd = a.vd, acc = a.acc, aL = a.aL,
             aW = a.aW or (a.cross and CROSS_LEN or AGENT_W), cross = a.cross and true or false, onc = a.onc and true or false,
             held = (a.cross and a.held) and true or false, heldD = a.heldD or 0}
  end
  return ag
end

-- predicted displacement of each vehicle at step k: travS[i][k] along, travD[i][k] across
local function predict(ag)
  local travS, travD = {}, {}
  for i, a in ipairs(ag) do
    local vs, acc = a.vs, a.acc
    local aS = acc * sign(vs)
    local tstop = 1e9
    if aS * vs < 0 then tstop = -vs / aS end
    local tLim = aS < 0 and 1e9 or AGENT_ACC_T
    local vd = a.vd
    local aD = a.cross and acc * sign(vd) or 0
    local tstopD = 1e9
    if aD * vd < 0 then tstopD = -vd / aD end
    local ts, td = {}, {}
    for k = 1, H do
      local t = k * DTS
      local tacc = min(min(t, tLim), tstop)
      local vEnd = vs + aS * tacc
      local s = vs * tacc + 0.5 * aS * tacc * tacc + vEnd * (t - tacc)
      if abs(vs) < 0.4 then s = 0 end
      local ttd
      if a.cross then ttd = min(t, tstopD) else ttd = min(t, AGENT_LAT_T) end
      local d = vd * ttd + 0.5 * aD * ttd * ttd
      if abs(vd) < 0.3 then d = 0 end
      if a.held then d = max(-a.heldD, min(a.heldD, d)) end      -- waiting for its light or sign: stops at its line
      ts[k], td[k] = s, d
    end
    travS[i], travD[i] = ts, td
  end
  return travS, travD
end

-- progress and speed at each step: toward vT at <= 1.6 m/s2 (braking 3 m/s2), and never faster than lets the car stop
-- (2.2 m/s2) at the point dstop metres ahead
local function speedProfile(v0, vT, dstop)
  local v, ds = v0, 0
  local prog, vk = {}, {}
  for k = 1, H do
    local tgt = min(vT, sqrt(2 * STOP_DECEL * max(0, dstop - ds - (1.0 + 0.6 * v))))
    local dv
    if tgt > v then dv = min(A_ACC * DTS, tgt - v) else dv = -min(A_PLAN_DEC * DTS, v - tgt) end
    local vn = v + dv
    ds = ds + 0.5 * (v + vn) * DTS
    v = vn
    prog[k], vk[k] = ds, v
  end
  return prog, vk
end

-- the vehicles that block my own lane and that I have not passed, grouped: the nearest one and those within CLUSTER_GAP
-- of the group's front
local function clusterOf(ag)
  local obs = {}
  for i, a in ipairs(ag) do
    if not a.onc and not a.cross and abs(a.ad) < LANE_BAND and a.rs + a.aL / 2 > -EGO_L / 2 then
      obs[#obs + 1] = {i = i, rear = a.rs - a.aL / 2, front = a.rs + a.aL / 2}
    end
  end
  table.sort(obs, function(p, q) return p.rear < q.rear end)
  local cl, frontC, any = {}, -1e9, false
  for _, o in ipairs(obs) do
    if frontC < -1e8 or o.rear - frontC < CLUSTER_GAP then
      cl[o.i] = true
      any = true
      frontC = max(frontC, o.front)
    end
  end
  return cl, any
end

-- first step (0-based) at which the rear of the car is past the front of every vehicle of the group; H if never
local function clearStep(prog, vk, ag, travS, cl)
  for k = 1, H do
    local rear = prog[k] - EGO_L / 2 - (RET_M0 + RET_MV * vk[k])
    local ok = true
    for i, a in ipairs(ag) do
      if cl[i] and not (rear > a.rs + travS[i][k] + a.aL / 2) then ok = false break end
    end
    if ok then return k - 1 end
  end
  return H
end

-- lowest and highest lateral position of the car's centre at each step: command dA now, 0 from Th (s) on
local function latEnvelope(vk, ed, vd0, dA, Th)
  local xs = {}
  for m, mod in ipairs({FAST, SLOW}) do
    local delay, tau, gain, drift, vref = mod[1], mod[2], mod[3], mod[4], mod[5]
    local teq, cum = {}, {0}
    local acc = 0
    for k = 1, H do
      acc = acc + min(vk[k], vref) / vref * DTS
      teq[k] = acc
      cum[k + 1] = acc
    end
    local stepsH = max(0, min(Th / DTS, H))
    local loI = min(floor(stepsH), H - 1)
    local frac = stepsH - loI
    local tRet = cum[loI + 1] + frac * (cum[loI + 2] - cum[loI + 1])
    if Th >= 1e2 then tRet = 1e3 end
    local x = {}
    for k = 1, H do
      local z1 = max(teq[k] - delay, 0) / tau
      local s1 = 1 - (1 + z1) * exp(-z1)
      local z2 = max(teq[k] - tRet - delay, 0) / tau
      local s2 = 1 - (1 + z2) * exp(-z2)
      local xv = ed + gain * ((dA - ed) * s1 - dA * s2)
      if drift ~= 0 then xv = xv + vd0 * drift * (1 - exp(-(k * DTS) / drift)) end
      x[k] = xv
    end
    xs[m] = x
  end
  local lo, hi = {}, {}
  for k = 1, H do
    lo[k] = min(xs[1][k], xs[2][k])
    hi[k] = max(xs[1][k], xs[2][k])
  end
  return lo, hi
end

-- first step (0-based) at which the car's inflated box overlaps a vehicle's predicted box; H if none does
local function firstConflict(prog, vk, lo, hi, ed, ag, travS, travD, skip)
  for k = 1, H do
    local mid = (lo[k] + hi[k]) / 2 - ed
    local half = (hi[k] - lo[k]) / 2
    for i, a in ipairs(ag) do
      if skip and skip[i] then goto continue end
      local ps = a.rs + travS[i][k] - prog[k]
      local pd = a.rd + travD[i][k] - mid
      local hs = (EGO_L + a.aL) / 2 + M_S0 + M_SV * vk[k]
      if a.onc then hs = hs + M_ONC_T * (abs(a.vs) + vk[k]) end
      local hd = (EGO_W + a.aW) / 2 + M_D + half
      if abs(ps) < hs and abs(pd) < hd then return k - 1 end
      ::continue::
    end
  end
  return H
end

-- a vehicle entirely ahead of the car in its lane: the lane guards follow it or stop behind it
local function aheadInLane(a)
  return not a.onc and not a.cross and abs(a.ad) < LANE_BAND and a.rs - a.aL / 2 > EGO_L / 2
end

function M.plan(st, agentsIn, capFn)
  local v0, ed = st.v0, st.ed
  local vd0 = v0 * math.sin(st.epsi or 0)
  local dA = max(st.lo, min(st.hi, st.dT))
  local vA = st.vA
  local ag = view(agentsIn)
  local travS, travD = predict(ag)
  local cl, needs = clusterOf(ag)
  -- the size of a pass is the shield's: the network chooses whether and on which side; a smaller request is widened to
  -- the offset that clears the group (its outermost edge + half the car + CLEAR_M)
  if needs and abs(dA) > OUT_LANE then
    local el, er = 1e9, -1e9
    for i, a in ipairs(ag) do
      if cl[i] then el = min(el, a.ad - a.aW / 2); er = max(er, a.ad + a.aW / 2) end
    end
    if dA < 0 then dA = min(dA, el - EGO_W / 2 - CLEAR_M) elseif dA > 0 then dA = max(dA, er + EGO_W / 2 + CLEAR_M) end
    dA = max(st.lo, min(st.hi, dA))
  end
  local outNow = abs(ed) > OUT_LANE
  local allowed = outNow or (st.near > J_GATE_START + J_GATE_T * v0)
  local cont = max(st.lo, min(st.hi, ed))
  local laneSkip = {}                       -- only a car still in its lane leaves the car ahead to the guards
  if not outNow then
    for i, a in ipairs(ag) do if aheadInLane(a) then laneSkip[i] = true end end
  end
  local caps = {}
  local function cap(name, offset)
    local c = caps[name]
    if not c then
      local v, s = capFn(offset)
      c = {v, s}
      caps[name] = c
    end
    return c[1], c[2]
  end

  local bestFirst, outV, outD, outM, resolved = -1, 0, ed, 3, false
  local function try(name, dTarget, hold, mode, f)
    local capV, capS = cap(name, name == "lane" and 0 or dTarget)
    local vcmd = min(f * vA, capV)
    local prog, vk = speedProfile(v0, vcmd, capS)
    local Th, ck = 0, 0
    if hold then
      ck = clearStep(prog, vk, ag, travS, cl)
      if ck < H then Th = (ck + 1) * DTS else Th = 1e3 end
    end
    local lo, hi = latEnvelope(vk, ed, vd0, dTarget, Th)
    local fc = firstConflict(prog, vk, lo, hi, ed, ag, travS, travD, (not hold) and laneSkip or nil)
    if hold and ck > CK_MAX then fc = min(fc, H - 1) end      -- a pass not over well inside the horizon is not known safe
    if fc > bestFirst then
      bestFirst, outV, outD, outM = fc, vcmd, hold and dTarget or 0, mode
    end
    if fc >= H then resolved = true end
  end

  if needs and allowed then
    for _, f in ipairs(SPEED_TRIES) do if not resolved then try("net", dA, true, 0, f) end end
  end
  if needs and allowed and outNow then
    for _, f in ipairs(SPEED_TRIES) do if not resolved then try("cont", cont, true, 2, f) end end
  end
  for _, f in ipairs(SPEED_TRIES) do if not resolved then try("lane", 0, false, 1, f) end end

  local mode
  if resolved then
    mode = outM
  else
    local stay = not outNow
    if stay then
      local capV = cap("lane", 0)
      outV, outD = min(vA, capV), 0
      mode = 5
    else
      mode = 3
    end
  end
  if not allowed and mode == 1 then mode = 4 end
  return outV, outD, mode
end

-- The check for a car that stays in its lane: the lane plan rolled out against the vehicles the lane guards do not handle
-- (crossing traffic, cars moving sideways, cars beside the lane; not the car ahead in the lane, not oncoming cars, nothing
-- behind). Returns speed, mode: 6 clear at the returned speed, 7 nothing clear (the speed that stays clear longest).
-- aheadRel = true: the car left to the guards is the one in the lane the car is in NOW (shield.py ahead_rel), for a car
-- that may start far off its lane centre (the shipped network with this check, no lateral shield).
function M.inlane(st, agentsIn, dstop, aheadRel)
  local v0, ed = st.v0, st.ed
  local vd0 = v0 * math.sin(st.epsi or 0)
  local vA = st.vA
  local all = view(agentsIn)
  local ag = {}
  for _, a in ipairs(all) do
    local inlaneAhead
    if aheadRel then
      -- the guards' own lead test, from where the car is now (a car far off its lane centre checks the lane centre)
      inlaneAhead = not a.onc and not a.cross and abs(a.rd) < 1.9 + max(0, a.aW / 2 - 0.95) and a.rs - a.aL / 2 > EGO_L / 2
    else
      inlaneAhead = aheadInLane(a)
    end
    local behind = a.rs + a.aL / 2 < -EGO_L / 2
    if not a.onc and not inlaneAhead and not behind and (a.cross or abs(a.vd) >= 0.3 or abs(a.rd) < 4.0) then
      ag[#ag + 1] = a
    end
  end
  if #ag == 0 then return vA, 6 end
  local travS, travD = predict(ag)
  local bestFc, bestV = -1, vA
  for _, f in ipairs(SPEED_TRIES) do
    local vcmd = f * vA
    local prog, vk = speedProfile(v0, vcmd, dstop)
    local lo, hi = latEnvelope(vk, ed, vd0, 0, 0)
    local fc = firstConflict(prog, vk, lo, hi, ed, ag, travS, travD)
    if fc > bestFc then bestFc, bestV = fc, vcmd end
    if fc >= H then return bestV, 6 end
  end
  return bestV, 7
end

return M
