# Lumen NN attempt 2: what nn.lua needs to run it

Written by the NN fork on 2026-11-01. nn.lua belongs to the parent session, so nothing in `src/` was edited. Everything
here is staged in `training/staged/`.

## Files

| file | what it is | where it goes |
|---|---|---|
| `staged/shield.lua` | the predicted-collision shield: a line-by-line port of `training/shield.py`, pure computation with no game calls | `src/lua/ge/extensions/lumen/shield.lua` |
| `staged/nn_weights.json`, `staged/nn_testvectors.json` | the attempt-2 network (91 inputs, the filtered observation), exported by `export_weights.py --stage` | replace the files in `src/.../lumen/` |
| `training/old/nn_weights_v1.json` | the weights that are shipped now, kept for the A/B | stays outside `src` |
| `staged/shield_vectors.json`, `training/lua_shield_test.py` | 400 lateral and 311 in-lane states from simulator rollouts, each with the Python shield's answer; the test runs `shield.lua` under LuaJIT and compares (0 mismatches at 12:22) | run `py -3.12 training/lua_shield_test.py` (with `LUPA_PATH` set) after any edit to `shield.lua` |

The observation is unchanged: the same 91 inputs, with the vehicle filter that is in nn.lua now. `sim.observe()` applies
the identical rule (`BEHIND_DROP`, `SHOW_RD_STILL`, `SHOW_RD_MOVING`; a stationary vehicle is never flagged as crossing).

## Changes to `M.plan` (the block from `local share = ...` to the road-edge clamp of `dT`)

```lua
local shield = require("lumen/shield")
...
local share = 1 / (1 + exp(-o1))
local vNet = min(share * 1.1 * info.vlim, info.vlim, V_MAX)  -- never above the posted limit (the output reaches 1.1 x)
local lo, hi = info.edgeL + EGO_W / 2 + 0.2, info.edgeR - EGO_W / 2 - 0.2
local dNet = clampf(tanh(o2) * 4, lo, hi)
local vcap, capWhy, dstop = guards(d, now, s, v)             -- the car's own lateral position; updates the yellow latch
local lateral = abs(dNet - lat) > shield.OUT_LANE or abs(lat) > shield.OUT_LANE or abs(dNet) > shield.OUT_LANE
local vT, dT, smode
if not lateral then
  dT = clampf(dNet, -shield.OUT_LANE, shield.OUT_LANE)
  -- the in-lane check: Drive's own prediction (drive.lua predict()) as a 9 s roll-out with the speed ladder
  vT, smode = shield.inlane({v0 = v, ed = lat, epsi = epsi, vA = min(vNet, vcap)}, sense200(car, pos, now, P, idx, s, lat), dstop or 1e9)
else
  local st = {v0 = v, ed = lat, epsi = epsi, vA = vNet, dT = dNet, lo = lo, hi = hi, near = nearestLine}
  vT, dT, smode = shield.plan(st, sense200(car, pos, now, P, idx, s, lat), function(offset)
    local c, _, ds = guards(d, now, s, v, leadFor(offset))
    return c, ds or 1e9
  end)
end
d.nnPrevT, d.nnPrevD = vT, dT                 -- the FINAL speed and offset, after the guards, the clamp and the shield
```

This differs from attempt 1. The simulator stores what `brains2.nn_actions2` returns: the in-lane offset clamped to
±0.45 m, or the shield's offset. The network was trained on that input. The watchdog block can stay where it is: it
only raises `vT`. If it does raise `vT`, store the raised value, as now.

## Changes to `guards(d, now, s, v)`

1. `YELLOW_DECEL` is 2.2 for this network. That is Drive's own "room to stop" test, and it is what
   `brains2.guards2` uses. The 3.5 that is there now belongs to the attempt-1 network.
2. Return `stopGap` as a third value: the distance to the point where the car must stop, or `nil` when there is none.
   The shield needs it to model the car braking in time for a stopped car or a line.
3. Take an optional `leadOverride` that replaces `info.lead`.

## New helpers in nn.lua

`leadFor(offset)` is the lead car for a car that plans to be at lateral offset `offset`. It is the same test as the
lead in `features()`, measured from the planned offset instead of from `lat`, and it ignores oncoming cars, because the
shield predicts those:

```lua
local function leadFor(offset)
  local best
  for _, a in ipairs(lastAgents) do         -- the projected agents of this step (rs, al, vs, hw, crossing)
    if not a.crossing and a.vs >= -1.5 and a.rs > 1 and a.rs < 150
       and abs(a.al - offset) < LANE_HALF + max(0, a.hw - 0.95) then
      local gap = a.rs - a.hl - FRONT
      if not best or gap < best.gap then best = {gap = gap, v = max(0, a.vs)} end
    end
  end
  return best or false                     -- false = "no lead" (not nil, so it overrides info.lead)
end
```

`sense200(...)` returns every vehicle within 200 m along the path and 30 m across, as the table that `shield.plan`
expects. `drive._sense` only reaches 75 m, and a car in the opposing lane closes 200 m in about 8 s. Make it a copy of
`drive.lua`'s `sense()` with `SENSE_RANGE` = 200, then project each vehicle like `features()` does with `projectWide`:

```lua
{ rs = rs, rd = al - lat, ad = al, vs = vs, vd = crossing and moving and vd or 0, acc = clampf(ag.acc or 0, -7, 2),
  aL = crossing and 2 * ag.hw or 2 * ag.hl,
  aW = crossing and 2 * ag.hl or 2 * ag.hw,     -- the real width (the simulator assumes 5 m and 2.1 m)
  cross = crossing and moving,                 -- a stationary sideways car is an obstacle to pass, not cross traffic
  onc = vs < -1.5,
  held = crossing and stopAt ~= nil,           -- waiting for its red light or stop sign (drive.lua predict()'s stopAt)
  heldD = stopAt or 0 }                        -- how far it is from its line: it is predicted to stop there
```

`nearestLine` is the distance from the front bumper to the nearest stop line OR junction ahead (1e9 when there is
none). `info.lineDist` is not quite this, because it ignores an open junction whenever a signal is in range. Compute
the minimum of the signal lines in `info.lines` and `js - 6 - (s + FRONT)` over `P.nnJ`.

## The pass commitment (pipeline state in nn.lua, added 2026-11-01 12:20)

`brains2.nn_actions2` keeps one number per car: the offset of the last pass the shield accepted (`smode` 0 or 2),
else 0. A weaker request to the same side keeps that offset, so a network that wavers between -2.7 and -1.5 m does not
abort the pass it has started.

```lua
local commit = d.nnCommit or 0
if commit ~= 0 and (dNet > 0) == (commit > 0) and abs(dNet) < abs(commit) and abs(dNet) > shield.OUT_LANE then dNet = commit end
...                                                  -- lateral test and shield call as above, with this dNet
if lateral then d.nnCommit = (smode == 0 or smode == 2) and dT or 0 else d.nnCommit = 0 end
```

`shield.plan` itself widens a request that is too small to clear the group of blocking vehicles. The network chooses
whether to pass and on which side; the shield chooses how far out to go. That logic is inside `shield.lua`, so the
pipeline does not need to repeat it.

## The shipped network + in-lane check (the parent's safety-fix candidate, 2026-11-01 14:15)

`shield.inlane(st, agents, dstop, true)`: with `aheadRel = true`, the car left to the guards is the one in the lane the
car is in NOW. That is the guards' own lead test, measured from `lat`. A car that starts 2 to 5 m off its lane centre
(a take-over in a lot) therefore still checks a car in the lane centre. Without this flag it leaves that car to the
guards, which cannot see it from where it is. Call it on `min(vNet, vcap)` every step, with no lateral shield, for the
attempt-1 weights.

Checked:
* Lua parity: `staged/shield_vectors_rel.json`, 18 vectors, 0 mismatches.
* Development results: `training/README.md`, section "The parent's extra arm".

## Attempt 4: the 5-output network (staged/nn_weights_a4.json, written when training ends)

The layer sizes are 91, 64, 64, then **5** outputs: `w3` is 64 x 5 and `b3` has 5 entries. The `info.outputs_meaning`
field in the file lists them. The self-test vectors have 5 numbers per output. What `M.plan` does with them
(`training/brains4.py nn_actions4`):

```lua
local vNet = min(sigmoid(o[1]) * 1.1 * info.vlim, info.vlim, V_MAX)
vNet = min(vNet, cautionCap)                                   -- the flat parked-car caution, as in the shipped nn.lua
if inJunctionZone then                                         -- 60 m before the next stop line or junction until 11 m past its line
  vNet = min(vNet, (0.3 + 0.7 * sigmoid(o[2])) * info.vlim)   -- the network's junction-approach speed
end
local intent = argmax(o[3], o[4], o[5])                        -- stay, pass left, pass right
local dNet = clampf(intent == 2 and -0.5 or intent == 3 and 0.5 or 0, lo, hi)
-- then exactly the attempt-3 block: the commitment, the lateral test, shield.inlane / shield.plan
```

The guards use `YELLOW_DECEL` = **3.5**, the shipped network's rule. In attempt 4 Storm left Drive as it is, at 2.2.

`inJunctionZone` is `nearestLine < 60 and nearestLine > -11`, using the same `nearestLine` as the shield: the nearest
stop line or junction (`js - 6`) ahead of the front bumper. For this test it must also keep a line up to 11 m behind
the front bumper, because the car is then still inside the junction box. The shield's own gate only needs lines ahead.

## Thinking feed

Use `smode`. 6 and 7 come from the in-lane check: 7 means it is slowing for a predicted conflict that the guards do
not see, so say "Lumen NN is slowing for the car crossing ahead". 0 is a pass ("Lumen NN is passing the stopped car"). 1 and 4 are the car going back to or staying in its
lane. 2 is finishing a pass already under way. 3 means no option stayed clear: the car takes the one that stays clear
longest, and this should be logged with the inputs. 5 means the car stays in its lane under the guards.

## Simulator-only assumptions to check in the game

* The shield's model of how fast the car moves sideways was fitted to the simulator's copy of the controller
  (`training/lat_probe.py`). It has a fast and a slow bound around that controller. Check one real lane change at
  5, 10 and 15 m/s against `FAST`/`SLOW` in `shield.lua`.
* The predictions assume a vehicle keeps its measured acceleration for 2 s (braking until it stops). The simulator's
  oncoming cars never brake, but real traffic does: that error is on the safe side.
* `V_MAX` is 25 in the simulator and 20 in nn.lua. The shield's speed ladder is relative to `vA`, so this only changes
  how fast a pass is driven.
