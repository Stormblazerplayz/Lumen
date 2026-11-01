-- Lumen guard rules (Frontier Studios): the yellow-light and junction rules that Lumen Drive
-- (drive.lua) and Lumen NN (nn.lua) share. One copy of each.
-- Reference implementation: training/brains.py (_junction_discipline, and rules(w, latch=True,
-- junction=True)). On 1024 held-out simulator episodes these two rules alone take the rule brain from
-- 4.47 to 1.33 incidents per 100 km at the same speed (training/README.md, 2026-11-01).
--
--   committed yellow     Once the car has decided to stop for a yellow light it keeps stopping until the
--                        light is green or the line is behind it. The plain test "is there room to stop"
--                        flips to "go" once the car has braked more gently than planned, and the car then
--                        runs the red in the dilemma zone.
--   junction discipline  (1) Never plan to stand inside a junction: a stop point that falls in a
--                        junction's box (from 7.5 m past its centre back to its stop line) is moved back
--                        to the line in front of it, so a car queued beyond a junction does not leave us
--                        parked across it. (2) Gap acceptance: hold at the line when a crossing vehicle
--                        would be in the junction while we cross, with GAP_MARGIN seconds of margin.
--
-- Plain module: require("lumen/guard"). Every position is an arc length along the car's path, in metres.

local M = {}
local sqrt, abs, max = math.sqrt, math.abs, math.max

M.GAP_MARGIN = 0.5        -- s of margin in the junction gap test
M.EGO_W, M.EGO_L = 1.9, 4.8
M.LINE_BACK = 6           -- a signal's stop line lies this far before its junction's centre
M.BOX_AFTER = 7.5         -- the junction box reaches this far past its centre
M.ACCEL = 1.6             -- m/s2 the car is assumed to accelerate at while it crosses

-- seconds to cover d metres from speed v, accelerating at a
function M.timeToCover(v, d, a)
  a = a or M.ACCEL
  return (-v + sqrt(v * v + 2 * a * max(d, 0))) / a
end

-- ---------------------------------------------------------------------------------
-- committed yellow
-- ---------------------------------------------------------------------------------
-- latch: a table kept per car; key: the signal; before: the line is still ahead; room: the car can stop
-- for it comfortably from here. True when the car must hold at this line for the yellow.
function M.yellowHold(latch, key, before, room)
  if before and room then latch[key] = true end
  return before and (room or latch[key] == true)
end

-- a green light ends the commitment
function M.greenReset(latch, key) latch[key] = nil end

-- forget commitments to signals no longer ahead (the line is behind the car, or out of range): without
-- this a stale commitment would make the next yellow at the same signal look decided already
function M.prune(latch, seen)
  for k in pairs(latch) do
    if not seen[k] then latch[k] = nil end
  end
end

-- ---------------------------------------------------------------------------------
-- junction discipline
-- ---------------------------------------------------------------------------------
-- Arc length of every junction node on the path's route, cached on the path as P.jc.
-- project(P, x, y, idx, ahead) is drive.lua's projection (returns s, index, lateral).
function M.pathJunctions(P, project)
  if P.jc then return P.jc end
  local J = {}
  local nodes = P.trip and P.trip.nodes
  if nodes and map and map.getMap then
    local mn = map.getMap().nodes
    for _, nd in ipairs(nodes) do
      local node = mn[nd]
      if node and map.getNodeLinkCount(nd) > 2 then
        local s, _, lat = project(P, node.pos.x, node.pos.y, 1, P.n)
        if abs(lat) < 8 then J[#J + 1] = s end
      end
    end
  end
  P.jc = J
  return J
end

-- Junction centres ahead: the ones the signals imply (a signal's line is LINE_BACK before its centre),
-- plus the path's junction nodes not within 8 m of one of those.
function M.centres(sigCentres, pathJ, s)
  local out = {}
  for _, c in ipairs(sigCentres) do out[#out + 1] = c end
  for _, js in ipairs(pathJ or {}) do
    if js - s > -10 and js - s < 170 then
      local dup = false
      for _, jc in ipairs(out) do
        if abs(jc - js) < 8 then
          dup = true
          break
        end
      end
      if not dup then out[#out + 1] = js end
    end
  end
  return out
end

-- (1) Never plan to stand inside a junction. stopS: the planned stop point; s: the car's arc length;
-- jcs: junction centres; front: reference node to front bumper; lineGap: space left before a stop line.
-- Returns the stop point (moved back to the line in front of a junction when it was inside one) and the
-- centre of the junction it was moved out of (nil when it stayed).
function M.boxStop(stopS, s, jcs, front, lineGap)
  local moved
  for _, jc in ipairs(jcs) do
    local lineAbs = jc - M.LINE_BACK - front - lineGap
    local before = (s + front) < jc - M.LINE_BACK - 0.3
    if before and stopS > lineAbs + 0.2 and stopS < jc + M.BOX_AFTER then
      stopS = lineAbs
      moved = jc
    end
  end
  return stopS, moved
end

-- (2) Gap acceptance. cross: the crossing vehicles, each {rs (arc length ahead of us), rd (lateral offset
-- from us), vc (speed), a (acceleration), len, wid, dirSign (+1 when it moves to the right of our travel)};
-- v: our speed. Returns the stop point in front of the line the first conflicting vehicle crosses at (nil
-- when every crossing vehicle clears the junction before we get there), then the conflict record
-- {c, tIn, tOut, teIn, teOut} (c in the junction between tIn and tOut seconds, we between teIn and teOut),
-- then the nearest vehicle whose gap was accepted, for the log.
function M.crossStop(cross, s, v, front, lineGap)
  local best, refused, accepted
  for _, c in ipairs(cross) do
    if c.rs > front + 6.3 then                              -- still before its line
      local dd = abs(c.rd)
      local zone = (M.EGO_W + c.len) / 2 + 0.4              -- half-extent across our path: its length lies along it
      local toward = (-c.rd * c.dirSign) > -zone
      local dz = max(dd - zone, 0)
      local stopsShort = c.a < -1.0 and c.vc * c.vc / (2 * max(-c.a, 1e-3)) < dz - 1.0
      if toward and not stopsShort and (c.vc > 0.4 or c.a > 0.5) then
        local veff = max(c.vc + max(c.a, 0), 0.5)
        local tIn = dz / veff
        local tOut = tIn + 2 * zone / veff
        local sz = (M.EGO_L + c.wid) / 2 + 0.4              -- our extent along the path around its crossing
        local dIn, dOut = max(c.rs - sz, 0), max(c.rs + sz, 0)
        local teIn, teOut = M.timeToCover(v, dIn), M.timeToCover(v, dOut)
        local rec = {c = c, tIn = tIn, tOut = tOut, teIn = teIn, teOut = teOut}
        if tIn - M.GAP_MARGIN < teOut and teIn - M.GAP_MARGIN < tOut and dOut > 0 then
          local stopAbs = s + (c.rs - M.LINE_BACK - front - lineGap)
          if not best or stopAbs < best then best, refused = stopAbs, rec end
        elseif not accepted or c.rs < accepted.c.rs then
          accepted = rec
        end
      end
    end
  end
  return best, refused, accepted
end

return M
