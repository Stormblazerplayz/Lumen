# Lumen NN: training, the "3x better" test and what the result means

2026-11-01. The metric was written before any training run and has not moved.

## What "3x better than the rule-based button" means here

Storm: "Make it 3x better than the rule based button." The rule-based button is **Lumen Drive**
(`lua/ge/extensions/lumen/drive.lua`). "Better" needs a number, so:

**Primary metric: incidents per 100 km** in the simulator (`sim.py`) on **1024 held-out episodes** (seed
424242; no training run, selection or tuning step used it for training), identical roads and traffic for every
brain:

    incidents = collisions + red-light runs + stop-sign runs + off-road + seconds spent within 0.6 m (along)
    and 0.35 m (across) of touching another vehicle at more than 1.5 m/s

**3x reached** means `incidents/100 km (Lumen Drive) / incidents/100 km (NN) >= 3.0` **and both guards hold**:
mean speed of the NN >= the rules' mean speed, and metres driven per episode >= the rules'. (A brain that stands
still or crashes early cannot win; the "never moves" control row shows it.)

## Result (held-out seed 424242, 1024 episodes of 60 s, 461 km, `py evaluate.py lumen_nn_final.npz --raw`)

| brain | collisions | red | stop sign | near-miss s | **incidents / 100 km** | mean speed | m / episode |
|---|---|---|---|---|---|---|---|
| Lumen Drive (rules) | 9 | 11 | 0 | 0.6 | **4.47** | 7.65 m/s | 450 |
| rules + yellow-light commitment | 7 | 3 | 0 | 0.6 | 2.30 | 7.63 | 450 |
| rules + commitment + junction discipline (the NN's teacher) | 2 | 4 | 0 | 0.1 | 1.33 | 7.60 | 449 |
| **Lumen NN** (network + guardrails) | 4 | 0 | 0 | 0.3 | **0.93** | 7.64 | 450 |
| Lumen NN without the guardrails | 688 | 58 | 56 | 198.6 | 262.68 | 10.16 | 373 |
| never moves (control) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

* **Incidents per 100 km: 4.47 -> 0.93, 4.8x fewer than Lumen Drive**, 2.5x fewer than the rules with the
  yellow-light commitment.
* **Guards: mean speed 7.64 against 7.65 m/s and 450 against 450 m per episode. Strictly, both guards fail**
  (by 0.01 m/s and by under a metre, 0.2%). By my own definition "3x reached" is therefore **NO**, by a hair, even
  though the primary ratio is 4.8x. I did not tune further against the held-out seed to close that gap.
* Selection: three candidates were scored on the held-out seed during the session (two warm starts and the final
  one); the final configuration (speed bias, junction margin) was chosen on seeds 888 and 555, not on 424242.
* Absolute numbers are small (4 and 9 collisions in 1024 episodes), so the ratio carries real noise.
  The same comparison on the two other seeds (512 episodes each, junction margin 0.3 to 0.8 s) gave 2.4x to 3.2x
  (seed 888) and 2.2x to 34x (seed 555).

## What Lumen NN is, honestly

* **The network** (91 inputs, two hidden layers of 64, two outputs) chooses the **speed** to drive at and the
  **position in the lane**, five times a second. The steering, pedals, stop at the end of a trip, pull over,
  backing away and recovery are shared with Lumen Drive.
* **The guardrails** (`brains.guards` in Python, `guards()` in `nn.lua`, the same logic) cap the speed the network
  may ask for: curves ahead, red and yellow lights (a yellow is stopped for if it can be stopped for at 3.5 m/s2,
  and the decision is kept), stop signs, a car stopped or slower in the lane, **junction discipline** (never plan
  to stand inside a junction, hold at the line when a crossing vehicle would be in the way, 0.5 s margin).
  **Without the guardrails the network is unsafe** (table row above): it learned the free-flow speed and the lane
  position, and it has never needed to learn to stop, because in training the guardrails always did.
* **Warm start: yes.** The network was trained by behaviour cloning (DAgger) of the rule brain with the speed
  set to the limit instead of 95% of it (`brains.rules_fast`), with the guardrails in the loop, then one speed-bias
  shift (+0.15 on the speed output, chosen on a selection set). **The gains over Lumen Drive do not come from
  outcome-based fine-tuning.** They come mostly from the guardrails (junction discipline, the committed and firmer
  yellow rule), which are hand-written rules. Outcome-based fine-tuning (evolution strategies, runs 1 and 2) did
  not improve on the warm start in the time available; see below.
* **Never BeamNG's AI**: nothing in training or at run time uses it.

## What this means for Lumen Drive (the finding that matters most)

The same junction discipline and yellow-light commitment can simply be added to `drive.lua`. In the simulator
that alone takes the rule brain from 4.47 to 1.33 incidents per 100 km (rules_v3 above, 3.4x), without a
network. Lumen Drive today runs red lights in the dilemma zone (its "room to stop" test flips to "go" once it has
braked more gently than planned) and can sit inside a junction waiting for a stopped car beyond it.

## Runs

1. `train.py` BC + ES, no guardrails: the warm start collided in 33% of episodes; ES (Adam lr 0.01, 32 pairs) did
   not improve it (a random walk), reward 365-385 against 450 for the rules.
2. `train2.py` with guardrails in the loop, vehicle acceleration inputs, junction-heavy training scenarios, per-
   layer ES steps: warm start 420 reward, ES did not improve it after 40 iterations. Diagnosis: collisions are
   ~2% of episodes and the progress term dominates the gradient; remaining failures were junction right-of-way.
3. `train3.py` distillation of a stronger rule teacher (`rules_v3`): stopped, its students were trained under the
   old guardrails.
4. `train4.py` behaviour cloning with the final guardrails and the faster free-flow teacher, then
   `tune_bias.py`. This is `lumen_nn_final.npz`, the checkpoint in the game files.

## Caveats

* Simulator numbers. The road is one-dimensional, the traffic is scripted, there is no BeamNG physics and the
  in-game controller has other gains. **Nothing here was run in the game**: the in-game test copy was withdrawn
  before it could be used (it disturbed Storm's running game). `nn.lua` checks itself when it loads (below).
* The network's inputs in the game are approximations of the simulator's (shoulder width unknown, junction nodes
  from the road graph, vehicle acceleration from smoothed speed).

## In the game

* The brain always starts as **Lumen Drive**. The app's `Driver: Drive | NN` toggle, the radial-menu entry and the
  input action `lumen_robotaxi_brain` (unbound: plain `N` is the headlights in the default keyboard map; `alt n`
  fog lights, `shift n` lightbar, `ctrl n` debug) switch every fleet car live. The choice is not remembered.
* `nn.lua` loads `nn_weights.json`, runs the 14 test vectors inside it (numpy's answers from the same rounded
  weights) and **refuses to load, logging why, if they do not match to 1e-3**; then Lumen Drive keeps driving. It
  logs the first plan of every car (feature summary, outputs, caps), a line a minute, every brain switch with the car
  ids, and any NaN or exception, to `beamng.log` under `lumen_nn` and `lumen_robotaxi`. A failing plan falls back to
  the rules for that step.
* `nn_testvectors.json` (same vectors) sits next to the weights for a later in-game check:
  `py training/nn_ingame.py equiv | obs | fleet` (needs the test copy).

## Files

`sim.py` world and the 91-input observation (the spec of `nn.lua`) · `brains.py` rule brains, guardrails, MLP ·
`evaluate.py` the test above · `train.py` `train2.py` `train3.py` `train4.py` the runs · `tune_bias.py` ·
`export_weights.py` (writes `nn_weights.json` with self-test vectors and `nn_testvectors.json`) ·
`diag_nn.py` `diag_viol.py` `diag_cross.py` diagnosis · `nn_ingame.py` in-game checks · `nn_dev/` working copies
of `nn.lua`.

---

# Attempt 2: "Make the NN actually better than Drive" (Storm's /goal, 2026-11-01 evening)

## PRE-REGISTRATION (written 2026-11-01 before any training of attempt 2 and before any held-out seed was run)

**Why attempt 1 does not count as "actually better".** (1) The game's Drive now has the committed yellow and the
junction discipline (`lumen/guard.lua`), so Drive = `brains.rules_v3` = 1.33 incidents/100 km in the table above; the
NN (0.93) was a behaviour-cloned copy of that teacher with the same guards in the loop, on 4 to 9 collisions: not
significant, and nothing the net itself learned. (2) The evolution-strategies fine-tuning never improved on the
warm start. (3) The net had nothing to learn that the rules cannot already do.

**Arms.**
* **Drive** = `brains.rules_v3` exactly (the Python mirror of the shipped `drive.lua plan()` + `guard.lua`).
* **NN** = a network that outputs a target speed and a lateral offset, under (a) the SAME safety guards as Drive for
  signals, stop signs, the junction box and cross-traffic gap acceptance, the curve speed, and following a moving
  car in the lane the plan occupies; and (b) a general **predicted-collision shield** (`shield.py`, ported to
  `nn.lua`) that replaces "a stopped car in my lane, so stop behind it": it rolls the planned (speed, offset)
  forward 6 s against the predicted vehicles and only lets a plan through that is predicted collision-free and
  stoppable; otherwise it falls back to the in-lane plan. The shield lets the net pass a stopped or slow car when the
  road beside it is predicted free. Drive has no passing at all (that is the thing a policy can learn that the rules
  lack).

**Held-out data.** Never used for training, tuning, selection or early stopping (development seeds are 300_000+ for
training and 400_000+ for selection):
* ORIGINAL distribution: `sim.Scenarios(256, seed)` for seeds 900_001 ... 900_016 = **4096 episodes**.
* OVERTAKE-HEAVY distribution: `sim.Scenarios(256, seed, overtake=True)` for seeds 950_001 ... 950_016 = **4096
  episodes**: 1 to 2 parked or broken-down cars in the lane in 65% of episodes (instead of one in 25%), slow movers
  (3 to 7 m/s) in front, fewer junctions (0 or 1), the same oncoming traffic, cut-ins and cross traffic.

**Metrics** (per distribution, per arm, paired by seed and episode: both arms see identical roads and traffic):
incidents per 100 km (collision + red-light run + stop-sign run + off road + seconds within 0.6 m along and 0.35 m
across of another vehicle at more than 1.5 m/s), mean speed, metres per episode, share of time standing still,
jerk, 95th percentile lateral acceleration. **Confidence intervals:** bootstrap over episodes (10 000 resamples,
paired), 95%.

**Decision rule, fixed now.** "NN actually better than Drive" holds on a distribution if EITHER
 (i) incidents not worse: the upper 95% bound of (NN incidents per 100 km minus Drive's) is at or below 0, AND metres
 per episode higher by at least 8% with the 95% interval of the relative difference excluding 0; OR
 (ii) incidents at most 0.5x Drive's with the 95% interval of the ratio excluding 1 AND metres per episode not lower
 (the lower 95% bound of the relative difference at or above 0).
The goal needs it on the ORIGINAL distribution (the one the game resembles); the overtake-heavy result is reported
beside it and cannot rescue a failure on the original. I will not change this rule, the seeds or the metrics after
seeing a held-out number. Selection of the final checkpoint uses only the development seeds.

**Also reported:** what the net learned (share of episodes in which it passes a stopped or slow car, the offsets it
uses, its speed profile against Drive), the ablation "same net without the shield", a control "Drive + a hand-written
pass rule" so a reader can see whether the learning added anything beyond the shield, and which numbers are
simulator-only.

**Constraints I am keeping:** no BeamNG, no shipping, training workers below normal priority (at most 5 while
Storm's game runs), the observation stays the 91 inputs the game computes.

## Attempt 2: what was built (written after the pre-registration, before the held-out run)

Everything below was decided on development seeds only. The held-out seeds were run once, at the end.

**What changed from the pre-registration's one-sentence description of the shield, and why.**
* The first shield matched that description: a 6 to 9 s roll-out of the requested (speed, offset), required to be
  collision-free and stoppable, with "stop where you are" as the fallback. In development it parked the car in the
  opposing lane: 12 of the 14 collisions of a scripted pass requester. The cause was the check itself. It assumed the
  car keeps the offset for the whole horizon, so as soon as an oncoming car came into range the check failed, and the
  fallback stopped the car beside the vehicle it was passing.
* `shield.py` now checks the whole manoeuvre:
  1. Go to the offset.
  2. Hold it until the group of vehicles that blocks the lane is behind. The hold time is computed from their
     predicted positions, not chosen by the network.
  3. Come back to the middle of the lane.
* The model of the car's sideways motion is an envelope of a fast and a slow model fitted to the controller copy
  (`lat_probe.py`). It runs on distance at low speed, because a car cannot steer while standing.
* The speed profile brakes for the guards' own stop point.
* A pass must be over within 6 s of the 9 s checked. Crawling passes were caught by oncoming cars that came into range
  after the check.
* Fallbacks, in order:
  1. Continue the manoeuvre the car is already in.
  2. Return to the lane.
  3. The candidate that stays clear longest.
  It never stops in the opposing lane by default.
* No new manoeuvre is started within 50 m + 4 s × speed of a junction or stop line.
* **In-lane check.** Drive's plan contains a 4 s prediction of the other vehicles; the network's guards did not.
  Without that prediction, a network optimised for progress hit crossing cars 5 times in 128 development episodes.
  `shield_inlane` is the same roll-out for a car staying in its lane. It checks the vehicles the guards do not handle:
  crossing traffic, cars moving sideways, cars beside the lane. It does not check the car ahead in the lane, oncoming
  cars, or anything behind. Speed factors are tried from 1.0 down to 0.1.
* Like Drive's prediction, it predicts that a crossing car waiting for its light stops at its line.

**Training.**
1. `a2_bc.py`: behaviour cloning of both heads (DAgger, 7 rounds, the shield in the loop) on the filtered
   observation.
   * The speed teacher is `rules_fast`, the hand-written rules at the posted limit.
   * The offset teacher is `rulepass.py`, a hand-written pass rule.
   * Neither teacher is a model, so the "no distillation" rule holds.
2. `a2_es.py`: antithetic ES on the 130 output parameters only, with the hidden layers frozen.
   * Each candidate drives the same 72 scenarios per generation: junction-heavy, plain, overtake-heavy and the two
     diagnostic classes.
   * The incident penalties are heavy (collision 3000 m, red light 2000 m), because the goal needs fewer incidents
     than Drive.
   * Selection is on the original-distribution development reward.

**The speed limit.** The network's speed output reaches 1.1 × the posted limit, as in attempt 1. The attempt-2
pipeline caps it at the limit, because a gain that comes from speeding is not "better". Drive drives at 95% of the
limit; the network may drive at up to 100%.

**The observation** was changed during the attempt by the in-game finding (the stall: "Only the NN sat there"). It now
uses the same vehicle filter as nn.lua `features()`: nothing behind the car, stationary vehicles only within 2.6 m of
the lane, moving ones within 4.5 m, and moving crossing traffic as before. Behaviour cloning was redone on it. With the
old observation the simulator now reproduces the stall: arm `old_unfiltered` in the diagnostic classes.

## Attempt 2: RESULT on the held-out seeds (run once, 2026-11-01 12:41 to 12:56, rules and seeds as pre-registered)

**Verdict: the goal is NOT met.**
* On the ORIGINAL distribution the new network is clearly faster: +11.3% metres per episode, 95% CI +10.4 to
  +12.1.
* Its incident rate is not lower than Drive's: 1.75 against 1.56 per 100 km, difference +0.19 with 95% CI -0.59 to
  +0.94.
* Rule (i) therefore fails on its incident part. Rule (ii) fails too.
* On the overtake-heavy distribution it is much faster (+49.5%) and significantly less safe (2.08 against 0.71).
  That result could not have rescued the original anyway.
* The shipped attempt-1 network does not meet the goal either: 1.21 against 1.56 (CI of the difference -1.11 to
  +0.37), and no progress gain.

What the table says about where the gain comes from:
* **The shield is what makes passing possible at all.** The same network without it has 174 incidents per 100 km.
* **The hand-written pass rule under the same shield (the control) gets +8.2%.** The learned network gets +11.3%.
  So learning added about 3 points of progress over a rule plus shield, with no measurable change in incidents
  (1.75 against 1.85).
* Part of the gain is speed choice. In free flow the network drives 12.54 m/s against Drive's 11.80 (+6%): Drive
  drives at 95% of the limit, and the network may use 100%, never more. The rest comes from passing: 1.4 s per
  episode out of the lane, at a mean offset of 3.7 m while out. This split was not measured separately.
* **Red lights.** The network ran 11 reds in the overtake-heavy set, where Drive ran none. They are not analysed yet,
  and they are a large part of why it loses there. On the original set it ran 10 against Drive's 13.

Tables (incidents = collisions + red lights + stop signs + off road + seconds too close; CIs are paired bootstrap
over episodes, 10 000 resamples):

**original** (4096 episodes per arm, paired)

| arm | km | coll | red | stop | off | near s | incidents/100 km | m/episode | mean m/s | standing | first 20 s m/s | free-flow m/s | out of lane s/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Drive (rules_v3) | 1848 | 15 | 13 | 0 | 0 | 0.8 | 1.56 | 451 | 7.63 | 20.2% | 8.85 | 11.80 | 0.00 |
| old NN (shipped, attempt 1) | 1849 | 20 | 0 | 0 | 0 | 2.4 | 1.21 | 451 | 7.65 | 20.7% | 8.97 | 12.10 | 0.00 |
| old NN, old observation | 1849 | 21 | 0 | 0 | 0 | 2.7 | 1.28 | 451 | 7.65 | 20.6% | 8.96 | 12.10 | 0.00 |
| **new NN (attempt 2)** | 2057 | 20 | 10 | 0 | 3 | 2.9 | 1.75 | 502 | 8.59 | 16.3% | 9.16 | 12.54 | 1.40 |
| new NN without the shield | 1450 | 2178 | 14 | 0 | 166 | 169.5 | 174.33 | 354 | 10.00 | 7.4% | 7.95 | 12.34 | 33.31 |
| Drive speed + hand-written pass rule + shield | 1999 | 17 | 15 | 0 | 2 | 2.9 | 1.85 | 488 | 8.29 | 16.5% | 9.03 | 12.09 | 0.72 |

| arm vs Drive | incidents/100 km (arm vs Drive) | difference, 95% CI | ratio, 95% CI | m/episode change, 95% CI | rule (i) | rule (ii) |
|---|---|---|---|---|---|---|
| old NN (shipped, attempt 1) | 1.21 vs 1.56 | -0.35 (-1.11 to +0.37) | 0.78 (0.44 to 1.33) | +0.1% (-0.1 to +0.2) | not met | not met |
| old NN, old observation | 1.28 vs 1.56 | -0.28 (-1.07 to +0.48) | 0.82 (0.46 to 1.43) | +0.0% (-0.2 to +0.2) | not met | not met |
| **new NN (attempt 2)** | 1.75 vs 1.56 | +0.19 (-0.59 to +0.94) | 1.12 (0.70 to 1.84) | +11.3% (+10.4 to +12.1) | not met | not met |
| new NN without the shield | 174.33 vs 1.56 | +172.77 (+165.11 to +181.00) | 111.87 (77.93 to 177.60) | -21.6% (-23.2 to -19.9) | not met | not met |
| Drive speed + hand-written pass rule + shield | 1.85 vs 1.56 | +0.29 (-0.32 to +0.88) | 1.18 (0.83 to 1.75) | +8.2% (+7.4 to +9.0) | not met | not met |

**overtake** (4096 episodes per arm, paired)

| arm | km | coll | red | stop | off | near s | incidents/100 km | m/episode | mean m/s | standing | first 20 s m/s | free-flow m/s | out of lane s/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Drive (rules_v3) | 1416 | 10 | 0 | 0 | 0 | 0.1 | 0.71 | 346 | 5.82 | 25.9% | 8.01 | 12.04 | 0.00 |
| old NN (shipped, attempt 1) | 1416 | 12 | 0 | 0 | 0 | 1.2 | 0.93 | 346 | 5.84 | 26.2% | 8.09 | 12.42 | 0.00 |
| old NN, old observation | 1416 | 12 | 0 | 0 | 0 | 0.8 | 0.90 | 346 | 5.84 | 26.2% | 8.09 | 12.41 | 0.00 |
| **new NN (attempt 2)** | 2116 | 24 | 11 | 0 | 2 | 7.0 | 2.08 | 517 | 9.00 | 13.9% | 9.04 | 13.42 | 5.22 |
| new NN without the shield | 1504 | 2022 | 7 | 0 | 144 | 167.1 | 155.59 | 367 | 10.45 | 6.4% | 8.18 | 12.97 | 34.64 |
| Drive speed + hand-written pass rule + shield | 2003 | 25 | 2 | 0 | 3 | 9.1 | 1.95 | 489 | 8.42 | 15.3% | 8.84 | 12.91 | 3.97 |

| arm vs Drive | incidents/100 km (arm vs Drive) | difference, 95% CI | ratio, 95% CI | m/episode change, 95% CI | rule (i) | rule (ii) |
|---|---|---|---|---|---|---|
| old NN (shipped, attempt 1) | 0.93 vs 0.71 | +0.22 (-0.18 to +0.64) | 1.31 (0.78 to 2.40) | +0.0% (-0.0 to +0.1) | not met | not met |
| old NN, old observation | 0.90 vs 0.71 | +0.19 (-0.26 to +0.67) | 1.27 (0.69 to 2.48) | +0.0% (-0.1 to +0.1) | not met | not met |
| **new NN (attempt 2)** | 2.08 vs 0.71 | +1.37 (+0.72 to +2.03) | 2.91 (1.72 to 6.36) | +49.5% (+47.4 to +51.6) | not met | not met |
| new NN without the shield | 155.59 vs 0.71 | +154.87 (+147.48 to +162.63) | 218.08 (132.10 to 522.98) | +6.2% (+3.7 to +8.9) | not met | not met |
| Drive speed + hand-written pass rule + shield | 1.95 vs 0.71 | +1.24 (+0.62 to +1.89) | 2.74 (1.63 to 5.83) | +41.5% (+39.5 to +43.4) | not met | not met |

**start_stop** (1024 episodes per arm, paired)

| arm | km | coll | red | stop | off | near s | incidents/100 km | m/episode | mean m/s | standing | first 20 s m/s | free-flow m/s | out of lane s/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Drive (rules_v3) | 645 | 28 | 0 | 0 | 0 | 1.8 | 4.62 | 630 | 10.97 | 4.0% | 5.73 | 13.23 | 0.00 |
| old NN (shipped, attempt 1) | 641 | 58 | 0 | 0 | 0 | 5.7 | 9.94 | 626 | 11.27 | 4.1% | 5.67 | 13.77 | 0.00 |
| old NN, old observation | 128 | 38 | 0 | 0 | 0 | 2.5 | 31.75 | 125 | 2.11 | 59.7% | 2.27 | 7.01 | 0.00 |
| **new NN (attempt 2)** | 646 | 47 | 0 | 0 | 0 | 3.9 | 7.87 | 631 | 11.26 | 4.1% | 5.77 | 13.68 | 0.00 |
| new NN without the shield | 471 | 342 | 0 | 0 | 34 | 51.3 | 90.75 | 460 | 10.33 | 6.9% | 5.09 | 13.28 | 42.81 |
| Drive speed + hand-written pass rule + shield | 642 | 35 | 0 | 0 | 0 | 5.2 | 6.26 | 627 | 10.98 | 4.0% | 5.75 | 13.24 | 0.00 |

| arm vs Drive | incidents/100 km (arm vs Drive) | difference, 95% CI | ratio, 95% CI | m/episode change, 95% CI | rule (i) | rule (ii) |
|---|---|---|---|---|---|---|
| old NN (shipped, attempt 1) | 9.94 vs 4.62 | +5.32 (+3.13 to +7.68) | 2.15 (1.60 to 3.11) | -0.6% (-1.8 to +0.5) | - | - |
| old NN, old observation | 31.75 vs 4.62 | +27.13 (+17.53 to +37.85) | 6.87 (4.31 to 11.10) | -80.2% (-81.8 to -78.7) | - | - |
| **new NN (attempt 2)** | 7.87 vs 4.62 | +3.25 (+1.39 to +5.27) | 1.70 (1.27 to 2.42) | +0.2% (-0.8 to +1.2) | - | - |
| new NN without the shield | 90.75 vs 4.62 | +86.13 (+76.44 to +96.57) | 19.65 (14.14 to 30.02) | -27.0% (-29.6 to -24.5) | - | - |
| Drive speed + hand-written pass rule + shield | 6.26 vs 4.62 | +1.64 (+0.18 to +3.19) | 1.36 (1.04 to 1.82) | -0.5% (-1.2 to +0.2) | - | - |

**start_crowd** (1024 episodes per arm, paired)

| arm | km | coll | red | stop | off | near s | incidents/100 km | m/episode | mean m/s | standing | first 20 s m/s | free-flow m/s | out of lane s/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Drive (rules_v3) | 512 | 0 | 0 | 0 | 0 | 0.0 | 0.00 | 500 | 8.34 | 0.5% | 8.40 | 9.81 | 0.00 |
| old NN (shipped, attempt 1) | 492 | 0 | 0 | 0 | 0 | 0.0 | 0.00 | 481 | 8.01 | 6.8% | 6.70 | 7.38 | 0.00 |
| old NN, old observation | 58 | 0 | 0 | 0 | 0 | 0.0 | 0.00 | 57 | 0.95 | 84.9% | 0.76 | 0.54 | 0.00 |
| **new NN (attempt 2)** | 617 | 5 | 0 | 0 | 0 | 0.9 | 0.96 | 603 | 10.50 | 0.6% | 9.12 | 13.84 | 4.57 |
| new NN without the shield | 348 | 542 | 0 | 0 | 0 | 105.4 | 186.11 | 340 | 9.69 | 11.0% | 6.95 | 10.20 | 33.61 |
| Drive speed + hand-written pass rule + shield | 580 | 1 | 0 | 0 | 0 | 1.9 | 0.50 | 566 | 9.56 | 0.5% | 8.90 | 12.55 | 1.86 |

| arm vs Drive | incidents/100 km (arm vs Drive) | difference, 95% CI | ratio, 95% CI | m/episode change, 95% CI | rule (i) | rule (ii) |
|---|---|---|---|---|---|---|
| old NN (shipped, attempt 1) | 0.00 vs 0.00 | +0.00 (+0.00 to +0.00) | n/a | -4.0% (-4.9 to -3.1) | - | - |
| old NN, old observation | 0.00 vs 0.00 | +0.00 (+0.00 to +0.00) | n/a | -88.6% (-90.2 to -87.0) | - | - |
| **new NN (attempt 2)** | 0.96 vs 0.00 | +0.96 (+0.29 to +1.77) | n/a | +20.5% (+18.9 to +22.0) | - | - |
| new NN without the shield | 186.11 vs 0.00 | +186.11 (+168.36 to +206.16) | n/a | -32.1% (-35.5 to -28.7) | - | - |
| Drive speed + hand-written pass rule + shield | 0.50 vs 0.00 | +0.50 (+0.00 to +1.21) | n/a | +13.2% (+11.8 to +14.6) | - | - |

**How the final network was chosen.**
* The final ES run (`a2_es_log.jsonl`) used the pipeline as it stood at 12:20, frozen from then on: the shield sizes a
  pass, a pass once accepted is kept, and lane plans leave the car ahead to the guards.
* It started from the behaviour-cloned network `a2_bc.npz` and ran 28 generations of 16 candidates × 40 scenarios.
  Iteration 27 had the best original-distribution development reward: 478.6, against 438.4 at the start.
* On those development episodes (192 original): 506.8 m per episode and 2.06 incidents per 100 km (1 collision).
  Drive on the same episodes: 448.3 m and 1.16 (0 collisions).
* There were six earlier ES runs. The power failure cut one, and I stopped five when the pipeline changed. Their
  logs are kept as `a2_es_stdout_partial_0210.log`, `a2_es_stdout_run2..5_killed.log`,
  `a2_es_stdout_run6_oldpipeline.log` and `old_es/`. None of them touched a held-out seed.

**What is simulator-only.**
* The sideways response model was fitted to the simulator's copy of the controller.
* Oncoming cars never brake, and cross traffic follows the simulator's rules.
* Sensing is perfect: no noise, no latency.
* The scenario mix is the simulator's.
* `V_MAX` is 25 m/s here and 20 m/s in the game.
* The shield assumes 5 m and 2.1 m for the extent of other vehicles across the road; the game passes their real size.
* The in-game A/B is the parent session's job.

**Follow-ups (not done).**
1. The offset head is a regression and averages "pass" and "stay". An intent output would let the network itself
   decide the size: pass left, pass right or stay, with the shield sizing the pass.
2. More optimisation. 28 ES generations on 130 parameters is a small search. CMA-ES, or unfreezing the last hidden
   layer, are the next steps.
3. The shield only plans passes. It has no evasive manoeuvre for a car merging into our side (`side_merge`): it can
   only slow down.
4. The in-lane check uses the shield's 9 s roll-out on every step. Its cost is half of the simulator time per step.

## Hard scenarios

These are diagnostic classes Storm asked for: "Have you tested it through hard scenarios? Like a car pulling out and a
car swerving and a car changing lanes right next to Lumen?". They are NOT part of the pre-registered test.

Setup: `sim.Scenarios(mode=...)` with fixed seeds 970_001+, 971_001+ and 972_001+, 512 episodes per class. Every brain
drives the same cars, under the guards and shield it ships with: Drive = `rules_v3`; shipped = the weights in
`src/.../nn_weights.json` now, with attempt-1 guards and the filtered observation; new = attempt 2. Measured from the
moment the actor starts. The runner is `a2_hard.py`; the raw numbers are in `a2_hard_results.json`.

The three actors:
* **pull_out**: a car parked at the curb (2.4 to 3.2 m right of our lane centre) pulls into our lane when we are 12 to
  35 m behind it, arriving at 8 to 15 m/s. It accelerates at 2 to 3 m/s² while its offset reaches 0 over 1.5 to 3 s.
* **swerve**: a car in the lane to our left, 0 to 15 m ahead or right beside us, swerves 1.0 to 1.8 m toward our lane
  and back over 1.5 to 2.5 s.
* **side_merge**: a car alongside (-4 to +6 m, either side) at our speed ±2 m/s changes into our lane over 2 to 3 s.

**Read the absolute rates with care.** Many of these episodes cannot be avoided by braking at all: a car that merges
into our side with no gap, or one that pulls out 12 m ahead of us at 15 m/s. That is why every brain crashes in 18 to
77% of them. The paired differences between brains are the informative part.

**pull_out** (512 episodes per brain, the same cars for every brain)

| brain | actor started | collisions | rate (exact 95% CI) | mean min gap m | min TTC median s | min TTC 5th pct s | hard braking (>4 m/s2) | s/ep |
|---|---|---|---|---|---|---|---|---|
| drive | 512 | 176 | 34.4% (30.3 to 38.7) | 4.85 | 0.93 | 0.00 | 77.3% | 0.81 |
| shipped | 512 | 225 | 43.9% (39.6 to 48.4) | 3.66 | 0.49 | 0.00 | 79.1% | 0.82 |
| new | 512 | 230 | 44.9% (40.6 to 49.3) | 3.95 | 0.51 | 0.00 | 75.2% | 0.68 |

shipped minus Drive (paired): collision rate +9.6 points (95% CI +7.0 to +12.3); mean min gap -1.20 m (-1.32 to -1.07); hard-braking rate +1.8 points (+0.0 to +3.5)

new minus Drive (paired): collision rate +10.5 points (95% CI +7.4 to +13.9); mean min gap -0.91 m (-1.10 to -0.72); hard-braking rate -2.1 points (-4.1 to -0.2)

**swerve** (512 episodes per brain, the same cars for every brain)

| brain | actor started | collisions | rate (exact 95% CI) | mean min gap m | min TTC median s | min TTC 5th pct s | hard braking (>4 m/s2) | s/ep |
|---|---|---|---|---|---|---|---|---|
| drive | 512 | 93 | 18.2% (14.9 to 21.8) | 2.11 | 0.59 | 0.00 | 10.2% | 0.03 |
| shipped | 512 | 89 | 17.4% (14.2 to 20.9) | 2.09 | 0.60 | 0.00 | 10.2% | 0.03 |
| new | 512 | 91 | 17.8% (14.6 to 21.4) | 2.12 | 0.60 | 0.00 | 11.5% | 0.03 |

shipped minus Drive (paired): collision rate -0.8 points (95% CI -1.6 to -0.2); mean min gap -0.02 m (-0.08 to +0.03); hard-braking rate +0.0 points (-1.2 to +1.0)

new minus Drive (paired): collision rate -0.4 points (95% CI -1.0 to +0.0); mean min gap +0.02 m (-0.03 to +0.06); hard-braking rate +1.4 points (+0.0 to +2.7)

**side_merge** (512 episodes per brain, the same cars for every brain)

| brain | actor started | collisions | rate (exact 95% CI) | mean min gap m | min TTC median s | min TTC 5th pct s | hard braking (>4 m/s2) | s/ep |
|---|---|---|---|---|---|---|---|---|
| drive | 512 | 393 | 76.8% (72.9 to 80.4) | 0.32 | 0.00 | 0.00 | 22.3% | 0.14 |
| shipped | 512 | 392 | 76.6% (72.6 to 80.2) | 0.32 | 0.00 | 0.00 | 22.3% | 0.14 |
| new | 512 | 373 | 72.9% (68.8 to 76.7) | 0.42 | 0.00 | 0.00 | 28.9% | 0.15 |

shipped minus Drive (paired): collision rate -0.2 points (95% CI -1.8 to +1.4); mean min gap +0.01 m (-0.01 to +0.02); hard-braking rate +0.0 points (-1.6 to +1.8)

new minus Drive (paired): collision rate -3.9 points (95% CI -6.1 to -2.0); mean min gap +0.10 m (+0.07 to +0.13); hard-braking rate +6.6 points (+4.3 to +9.2)

What the hard scenarios say:
* **Pull-out: the new network is worse than Drive**, +10.5 collision points (CI +7.4 to +13.9). The shipped
  network is worse by about the same amount (+9.6). Both networks drive faster than Drive's 95% of the limit, so
  they arrive at the curb car with less margin. Drive keeps 0.9 m more gap on average.
* **Swerve: no measurable difference**: new minus Drive is -0.4 points (CI -1.0 to 0.0).
* **Side merge: the new network is slightly better**, -3.9 points (CI -6.1 to -2.0). It keeps 0.10 m more gap and
  brakes hard more often (+6.6 points). That is the in-lane check seeing the car's sideways motion, which Drive's own
  prediction does not use.
* **No brain swerves away from a car merging into it.** That is the largest gap for all three, and it is in the
  follow-ups.

# Attempt 3: PRE-REGISTRATION ADDENDUM (written 2026-11-01 13:05, before any attempt-3 analysis or training)

Attempt 2 did not meet the goal. Its held-out seeds (900_xxx, 950_xxx, 920_xxx, 930_xxx, 970-972_xxx) have now been
seen once, so attempt 3 is judged on NEW seeds that no run has used.

* **Decision rule, metrics, confidence intervals:** exactly as in the attempt-2 pre-registration above. Original
  distribution decides; overtake-heavy is reported beside it and cannot rescue a failure.
* **New held-out seeds:**
  * ORIGINAL: `sim.Scenarios(256, seed)` for seeds 910_001 to 910_016 (4096 episodes).
  * OVERTAKE-HEAVY: `sim.Scenarios(256, seed, overtake=True)` for seeds 960_001 to 960_016 (4096 episodes).
  * Diagnostics, reported separately:
    * start_stop: 925_001 to 925_004.
    * start_crowd: 935_001 to 935_004.
    * Hard scenarios: pull_out 975_001+, swerve 976_001+, side_merge 977_001+, 512 episodes each.
* **Arms:**
  * Drive (`rules_v3`).
  * The attempt-2 network under the attempt-2 pipeline, frozen as `a2_final.npz` with `shield_a2.py` / `brains2_a2.py`.
  * The attempt-3 network under the attempt-3 pipeline.
  * The control: Drive's speed plus the hand-written pass rule, under the attempt-3 shield.
  * The attempt-3 network without the shield.
* **Discipline:**
  * Analysis, shield changes, training and checkpoint selection use development seeds only: training 300_000+,
    selection 400_000+ and 450_000+, analysis 400_000 to 499_999.
  * The new held-out seeds are run once, at the end, and reported as they come out, including a loss to Drive.
* **Plan, in order:**
  1. Break down the incidents of Drive, the attempt-2 network and the control by cause and phase, on development seeds:
     during a pass, returning from a pass, red lights during or after a pass, start_stop, cut-ins, cross traffic.
  2. Fix the shield where passes go wrong, re-checking `shield.lua` parity under LuaJIT after each change.
  3. Make incidents dominate the ES objective.
  4. Select on development seeds and run the new held-out seeds once.

## Attempt 3: development findings (dev seeds only)

**Where the attempt-2 incidents came from** (`a3_breakdown.py`, attempt-2 pipeline). Phases: in lane; going out
(out of the lane, moving away or holding); returning (out of the lane, moving back); just back (within 4 s of
returning).

| set (episodes) | Drive | attempt-2 NN | control (rule pass + shield) |
|---|---|---|---|
| original (1024) | 5 cross collisions, 3 reds, 0.3 s near: 1.81/100 km | 5 cross, 4 reds, **2 lead collisions while returning**, 1.2 s near (0.5 returning): 2.37 | 4 cross, 5 reds, 1 lead returning, 1.6 s near: 2.34 |
| overtake-heavy (1024) | nothing: 0.00 | **2 lead collisions returning, 2 oncoming collisions going out**, 2 reds, 1 cross, 1.8 s near (1.3 returning): 1.64 | 2 lead returning, 1 cross, 4.0 s near (2.6 returning): 1.38 |
| start_stop (512) | 10 cross collisions: 3.35 | **18 cross collisions**: 6.12 | 12 cross: 4.38 |

What this shows:
* Every red light was run in lane, none during a pass. Every cross-traffic collision was in lane.
* The passes themselves fail in two ways:
  * The return: the attempt-2 lane plan left the car ahead to the guards, even for a car still out in the other
    lane.
  * Oncoming cars when going out.

**Shield changes for attempt 3** (staged/shield.lua matches again: 0 mismatches on 400 lateral and 311 in-lane
vectors):
1. A returning car checks the car ahead in its lane. Only a car still in its lane leaves it to the guards.
2. The box of an oncoming car is 0.6 s of closing speed longer.

**Parked-car caution** (`caution.py`, the parent's proposal, a rule for both brains). Cap the speed at 65% of the
limit while a stationary car is ahead or alongside and meets all of these:
* its centre is within 3.5 m of our lane centre, measured in the lane frame;
* it is outside our lane, which is the guards' "in lane" test applied in the lane frame;
* its rear is less than 30 m ahead of our front bumper.

| set | Drive | Drive + caution | attempt-2 NN | NN + caution |
|---|---|---|---|---|
| pull_out (512, seeds 470_001+) collision episodes | 39.5% | **13.7%** | 44.5% | **15.2%** |
| original (1024) m/episode and incidents/100 km | 446.7, 1.82 | 446.7, 1.82 (no change) | 501.0, 2.07 | 501.0, 2.07 (no change) |

The variant "stopping distance to the parked car" was better for Drive (10.0%) and worse for the network (22.3%), so
the flat rule was chosen. The attempt-3 pipeline uses the flat rule on the network's speed. Drive in the
pre-registered arm stays `rules_v3` without it; "Drive + caution" is reported beside it in the hard scenarios.

**ES objective for attempt 3** (`a3_es.py`): collision 10 000 m, red light 6 000, stop sign 3 000, off road 6 000,
1 000 per second too close, 2 per second in the opposing lane. It starts from the attempt-2 network. The pipeline,
selection rule and search size are the same as in attempt 2.

## Attempt 3: RESULT on the NEW held-out seeds (run once, 2026-11-01 13:47 to 14:10)

**Verdict: the goal is NOT met.** On the original distribution, on seeds no run had used before, the attempt-3
network is again about 11% faster: +11.2% metres per episode, 95% CI +10.4 to +12.1. But its incidents are
significantly WORSE than Drive's: 2.02 against 1.27 per 100 km, difference +0.75 with CI +0.08 to +1.45.

* **What the ES did.** It found nothing better than its starting point under the incident-dominant objective. The
  attempt-3 network therefore has the attempt-2 weights; it differs only in the pipeline (shield fixes plus the
  parked-car caution).
* **The pipeline fixes helped where the passes were the problem.**
  * Overtake-heavy: 1.81 down to 1.49 incidents per 100 km, and near-miss seconds 6.5 down to 3.4.
  * Original: no effect (2.04 to 2.02).
* **The extra incidents on the original distribution are not in passes.**
  * The development breakdown shows every red light and every cross-traffic collision happening in lane.
  * These counts agree: 24 collisions and 14 reds, against Drive's 14 and 9.
  * The network drives at the limit (12.5 m/s in free flow, Drive 11.7), so it reaches junctions faster. The shared
    junction guards then have less room.
* **Drive's own rate is lower on these seeds:** 1.27, against 1.56 on the attempt-2 seeds. Against it, the attempt-2
  network's "no significant difference" from attempt 2 is now a significant loss (2.04, CI of the difference +0.08 to
  +1.50). One held-out set was not enough to tell the network and Drive apart.


**original** (4096 episodes per arm, paired)

| arm | km | coll | red | stop | off | near s | incidents/100 km | m/episode | mean m/s | standing | first 20 s m/s | free-flow m/s | out of lane s/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Drive (rules_v3) | 1841 | 14 | 9 | 0 | 0 | 0.4 | 1.27 | 449 | 7.59 | 20.1% | 8.78 | 11.73 | 0.00 |
| shipped NN (attempt 1, capped at the limit) | 1841 | 20 | 1 | 0 | 0 | 1.2 | 1.21 | 449 | 7.61 | 20.6% | 8.89 | 11.97 | 0.00 |
| attempt-2 NN (frozen pipeline) | 2053 | 23 | 15 | 0 | 1 | 2.9 | 2.04 | 501 | 8.57 | 16.1% | 9.11 | 12.51 | 1.39 |
| **attempt-3 NN** | 2048 | 24 | 14 | 0 | 1 | 2.3 | 2.02 | 500 | 8.55 | 16.2% | 9.11 | 12.50 | 1.38 |
| attempt-3 NN without the shield | 1447 | 2140 | 16 | 0 | 160 | 172.8 | 171.95 | 353 | 9.90 | 7.8% | 7.87 | 12.27 | 33.64 |
| Drive speed + hand-written pass rule + attempt-3 shield | 1997 | 18 | 13 | 0 | 1 | 3.0 | 1.75 | 488 | 8.28 | 16.2% | 8.97 | 12.03 | 0.76 |

| arm vs Drive | incidents/100 km (arm vs Drive) | difference, 95% CI | ratio, 95% CI | m/episode change, 95% CI | rule (i) | rule (ii) |
|---|---|---|---|---|---|---|
| shipped NN (attempt 1, capped at the limit) | 1.21 vs 1.27 | -0.06 (-0.73 to +0.59) | 0.95 (0.55 to 1.66) | -0.0% (-0.2 to +0.1) | not met | not met |
| attempt-2 NN (frozen pipeline) | 2.04 vs 1.27 | +0.77 (+0.08 to +1.50) | 1.61 (1.05 to 2.63) | +11.5% (+10.6 to +12.4) | not met | not met |
| **attempt-3 NN** | 2.02 vs 1.27 | +0.75 (+0.08 to +1.45) | 1.59 (1.05 to 2.60) | +11.2% (+10.4 to +12.1) | not met | not met |
| attempt-3 NN without the shield | 171.95 vs 1.27 | +170.68 (+163.01 to +178.74) | 135.29 (90.26 to 233.61) | -21.4% (-23.1 to -19.7) | not met | not met |
| Drive speed + hand-written pass rule + attempt-3 shield | 1.75 vs 1.27 | +0.48 (-0.02 to +1.01) | 1.38 (0.98 to 2.05) | +8.5% (+7.7 to +9.3) | not met | not met |

**overtake** (4096 episodes per arm, paired)

| arm | km | coll | red | stop | off | near s | incidents/100 km | m/episode | mean m/s | standing | first 20 s m/s | free-flow m/s | out of lane s/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Drive (rules_v3) | 1414 | 5 | 2 | 0 | 0 | 0.2 | 0.51 | 345 | 5.82 | 25.4% | 8.00 | 12.08 | 0.00 |
| shipped NN (attempt 1, capped at the limit) | 1414 | 13 | 1 | 0 | 0 | 0.8 | 1.05 | 345 | 5.83 | 25.7% | 8.07 | 12.35 | 0.00 |
| attempt-2 NN (frozen pipeline) | 2129 | 27 | 5 | 0 | 0 | 6.5 | 1.81 | 520 | 9.06 | 13.6% | 9.03 | 13.41 | 5.42 |
| **attempt-3 NN** | 2112 | 23 | 5 | 0 | 0 | 3.4 | 1.49 | 516 | 8.97 | 13.9% | 8.98 | 13.40 | 5.38 |
| attempt-3 NN without the shield | 1503 | 2022 | 3 | 0 | 144 | 139.6 | 153.58 | 367 | 10.45 | 6.6% | 8.19 | 13.01 | 34.61 |
| Drive speed + hand-written pass rule + attempt-3 shield | 2028 | 14 | 7 | 0 | 1 | 3.2 | 1.24 | 495 | 8.52 | 14.6% | 8.81 | 12.92 | 4.26 |

| arm vs Drive | incidents/100 km (arm vs Drive) | difference, 95% CI | ratio, 95% CI | m/episode change, 95% CI | rule (i) | rule (ii) |
|---|---|---|---|---|---|---|
| shipped NN (attempt 1, capped at the limit) | 1.05 vs 0.51 | +0.54 (+0.03 to +1.07) | 2.06 (1.05 to 5.42) | -0.0% (-0.1 to +0.1) | not met | not met |
| attempt-2 NN (frozen pipeline) | 1.81 vs 0.51 | +1.30 (+0.67 to +1.98) | 3.55 (1.86 to 10.64) | +50.6% (+48.5 to +52.7) | not met | not met |
| **attempt-3 NN** | 1.49 vs 0.51 | +0.98 (+0.39 to +1.61) | 2.92 (1.54 to 8.73) | +49.4% (+47.3 to +51.4) | not met | not met |
| attempt-3 NN without the shield | 153.58 vs 0.51 | +153.07 (+145.85 to +160.65) | 301.58 (167.92 to 1021.58) | +6.3% (+3.8 to +8.8) | not met | not met |
| Drive speed + hand-written pass rule + attempt-3 shield | 1.24 vs 0.51 | +0.73 (+0.22 to +1.31) | 2.44 (1.30 to 6.76) | +43.5% (+41.5 to +45.4) | not met | not met |

**start_stop** (1024 episodes per arm, paired)

| arm | km | coll | red | stop | off | near s | incidents/100 km | m/episode | mean m/s | standing | first 20 s m/s | free-flow m/s | out of lane s/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Drive (rules_v3) | 647 | 25 | 0 | 0 | 0 | 1.6 | 4.11 | 632 | 11.00 | 4.0% | 5.72 | 13.27 | 0.00 |
| shipped NN (attempt 1, capped at the limit) | 642 | 52 | 0 | 0 | 0 | 5.4 | 8.94 | 627 | 11.22 | 4.0% | 5.67 | 13.67 | 0.00 |
| attempt-2 NN (frozen pipeline) | 652 | 37 | 0 | 0 | 0 | 4.4 | 6.35 | 637 | 11.30 | 4.0% | 5.79 | 13.72 | 0.00 |
| **attempt-3 NN** | 652 | 37 | 0 | 0 | 0 | 4.4 | 6.35 | 637 | 11.30 | 4.0% | 5.79 | 13.72 | 0.00 |
| attempt-3 NN without the shield | 478 | 317 | 0 | 0 | 30 | 62.6 | 85.62 | 467 | 10.30 | 7.7% | 5.09 | 13.23 | 43.80 |
| Drive speed + hand-written pass rule + attempt-3 shield | 646 | 29 | 0 | 0 | 0 | 5.0 | 5.27 | 631 | 11.01 | 3.9% | 5.75 | 13.28 | 0.00 |

| arm vs Drive | incidents/100 km (arm vs Drive) | difference, 95% CI | ratio, 95% CI | m/episode change, 95% CI | rule (i) | rule (ii) |
|---|---|---|---|---|---|---|
| shipped NN (attempt 1, capped at the limit) | 8.94 vs 4.11 | +4.84 (+2.81 to +6.96) | 2.18 (1.59 to 3.17) | -0.9% (-2.0 to +0.2) | - | - |
| attempt-2 NN (frozen pipeline) | 6.35 vs 4.11 | +2.24 (+0.53 to +4.02) | 1.54 (1.11 to 2.22) | +0.8% (-0.2 to +1.7) | - | - |
| **attempt-3 NN** | 6.35 vs 4.11 | +2.24 (+0.53 to +4.02) | 1.54 (1.11 to 2.22) | +0.8% (-0.2 to +1.7) | - | - |
| attempt-3 NN without the shield | 85.62 vs 4.11 | +81.51 (+72.03 to +91.60) | 20.84 (14.69 to 32.74) | -26.1% (-28.7 to -23.5) | - | - |
| Drive speed + hand-written pass rule + attempt-3 shield | 5.27 vs 4.11 | +1.16 (-0.36 to +2.73) | 1.28 (0.92 to 1.82) | -0.3% (-1.1 to +0.5) | - | - |

**start_crowd** (1024 episodes per arm, paired)

| arm | km | coll | red | stop | off | near s | incidents/100 km | m/episode | mean m/s | standing | first 20 s m/s | free-flow m/s | out of lane s/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Drive (rules_v3) | 514 | 0 | 0 | 0 | 0 | 0.0 | 0.00 | 502 | 8.36 | 0.5% | 8.41 | 9.84 | 0.00 |
| shipped NN (attempt 1, capped at the limit) | 494 | 0 | 0 | 0 | 0 | 0.0 | 0.00 | 482 | 8.04 | 6.5% | 6.68 | 7.37 | 0.00 |
| attempt-2 NN (frozen pipeline) | 611 | 10 | 0 | 0 | 0 | 2.3 | 2.01 | 597 | 10.41 | 0.6% | 9.04 | 13.68 | 4.49 |
| **attempt-3 NN** | 610 | 10 | 0 | 0 | 0 | 1.3 | 1.85 | 595 | 10.37 | 0.6% | 8.99 | 13.67 | 4.50 |
| attempt-3 NN without the shield | 347 | 558 | 0 | 0 | 0 | 106.6 | 191.33 | 339 | 9.71 | 10.6% | 7.00 | 10.31 | 33.52 |
| Drive speed + hand-written pass rule + attempt-3 shield | 578 | 2 | 0 | 0 | 0 | 0.0 | 0.35 | 564 | 9.55 | 0.5% | 8.88 | 12.49 | 1.94 |

| arm vs Drive | incidents/100 km (arm vs Drive) | difference, 95% CI | ratio, 95% CI | m/episode change, 95% CI | rule (i) | rule (ii) |
|---|---|---|---|---|---|---|
| shipped NN (attempt 1, capped at the limit) | 0.00 vs 0.00 | +0.00 (+0.00 to +0.00) | n/a | -3.9% (-4.8 to -3.0) | - | - |
| attempt-2 NN (frozen pipeline) | 2.01 vs 0.00 | +2.01 (+0.94 to +3.24) | n/a | +19.0% (+17.4 to +20.5) | - | - |
| **attempt-3 NN** | 1.85 vs 0.00 | +1.85 (+0.83 to +3.04) | n/a | +18.7% (+17.1 to +20.3) | - | - |
| attempt-3 NN without the shield | 191.33 vs 0.00 | +191.33 (+173.02 to +210.90) | n/a | -32.4% (-35.6 to -29.1) | - | - |
| Drive speed + hand-written pass rule + attempt-3 shield | 0.35 vs 0.00 | +0.35 (+0.00 to +0.87) | n/a | +12.5% (+11.1 to +13.9) | - | - |


## Hard scenarios, attempt 3 (new seeds 975/976/977_001+, with the ideal driver)

Setup:
* The brains: Drive, Drive + the parked-car caution, the shipped network (attempt-1 weights capped at the limit,
  without the caution), the attempt-2 network (frozen pipeline) and the attempt-3 network.
* **Ideal driver:** it approaches exactly like Drive. From the first step the actor moves it brakes fully (the
  controller's limit is 6.5 m/s²), either staying in its lane or with the best of six swerve targets within the road
  edges.
* An episode is "avoidable" when at least one of those reactions ends without a collision.
* Code in `a3_ideal.py` and `a3_hard.py`; raw numbers in `a3_hard_results.json`.

**pull_out** (512 episodes per brain, the same cars for every brain). An ideal driver avoids **88.7%** by full braking at the first moment the actor moves, and **94.5%** by braking plus the best swerve within the road edges.

| brain | collisions (rate, exact 95% CI) | on brake-avoidable episodes | on swerve-avoidable episodes | mean min gap m | min TTC median / 5th pct s | hard braking |
|---|---|---|---|---|---|---|
| drive | 203 (39.6%, 35.4 to 44.0) | 146/454 (32.2%, 27.9 to 36.7) | 175/484 (36.2%, 31.9 to 40.6) | 4.62 | 0.94 / 0.00 | 73.4% |
| drive_caution | 65 (12.7%, 9.9 to 15.9) | 14/454 (3.1%, 1.7 to 5.1) | 42/484 (8.7%, 6.3 to 11.5) | 7.48 | 1.59 / 0.00 | 76.6% |
| shipped | 232 (45.3%, 40.9 to 49.7) | 175/454 (38.5%, 34.0 to 43.2) | 204/484 (42.1%, 37.7 to 46.7) | 3.54 | 0.45 / 0.00 | 74.2% |
| a2 | 234 (45.7%, 41.3 to 50.1) | 182/454 (40.1%, 35.5 to 44.8) | 210/484 (43.4%, 38.9 to 47.9) | 3.66 | 0.47 / 0.00 | 72.1% |
| a3 | 72 (14.1%, 11.2 to 17.4) | 24/454 (5.3%, 3.4 to 7.8) | 50/484 (10.3%, 7.8 to 13.4) | 7.63 | 1.79 / 0.00 | 83.2% |

drive_caution minus Drive (paired): collision rate -27.0 points (95% CI -30.9 to -23.2); mean min gap +2.86 m (+2.69 to +3.03)

shipped minus Drive (paired): collision rate +5.7 points (95% CI +3.7 to +7.6); mean min gap -1.09 m (-1.20 to -0.97)

a2 minus Drive (paired): collision rate +6.1 points (95% CI +3.1 to +9.0); mean min gap -0.96 m (-1.16 to -0.77)

a3 minus Drive (paired): collision rate -25.6 points (95% CI -29.5 to -21.9); mean min gap +3.00 m (+2.75 to +3.25)

**swerve** (512 episodes per brain, the same cars for every brain). An ideal driver avoids **86.5%** by full braking at the first moment the actor moves, and **99.8%** by braking plus the best swerve within the road edges.

| brain | collisions (rate, exact 95% CI) | on brake-avoidable episodes | on swerve-avoidable episodes | mean min gap m | min TTC median / 5th pct s | hard braking |
|---|---|---|---|---|---|---|
| drive | 93 (18.2%, 14.9 to 21.8) | 24/443 (5.4%, 3.5 to 8.0) | 92/511 (18.0%, 14.8 to 21.6) | 1.83 | 0.60 / 0.00 | 10.2% |
| drive_caution | 93 (18.2%, 14.9 to 21.8) | 24/443 (5.4%, 3.5 to 8.0) | 92/511 (18.0%, 14.8 to 21.6) | 1.83 | 0.60 / 0.00 | 10.2% |
| shipped | 91 (17.8%, 14.6 to 21.4) | 22/443 (5.0%, 3.1 to 7.4) | 90/511 (17.6%, 14.4 to 21.2) | 1.83 | 0.61 / 0.00 | 10.2% |
| a2 | 93 (18.2%, 14.9 to 21.8) | 24/443 (5.4%, 3.5 to 8.0) | 92/511 (18.0%, 14.8 to 21.6) | 1.84 | 0.60 / 0.00 | 12.9% |
| a3 | 93 (18.2%, 14.9 to 21.8) | 24/443 (5.4%, 3.5 to 8.0) | 92/511 (18.0%, 14.8 to 21.6) | 1.84 | 0.60 / 0.00 | 12.9% |

drive_caution minus Drive (paired): collision rate +0.0 points (95% CI +0.0 to +0.0); mean min gap +0.00 m (+0.00 to +0.00)

shipped minus Drive (paired): collision rate -0.4 points (95% CI -1.0 to +0.0); mean min gap +0.01 m (-0.01 to +0.02)

a2 minus Drive (paired): collision rate +0.0 points (95% CI +0.0 to +0.0); mean min gap +0.01 m (-0.01 to +0.03)

a3 minus Drive (paired): collision rate +0.0 points (95% CI +0.0 to +0.0); mean min gap +0.01 m (-0.01 to +0.03)

**side_merge** (512 episodes per brain, the same cars for every brain). An ideal driver avoids **68.6%** by full braking at the first moment the actor moves, and **85.2%** by braking plus the best swerve within the road edges.

| brain | collisions (rate, exact 95% CI) | on brake-avoidable episodes | on swerve-avoidable episodes | mean min gap m | min TTC median / 5th pct s | hard braking |
|---|---|---|---|---|---|---|
| drive | 378 (73.8%, 69.8 to 77.6) | 225/351 (64.1%, 58.8 to 69.1) | 308/436 (70.6%, 66.1 to 74.9) | 0.36 | 0.00 / 0.00 | 24.6% |
| drive_caution | 378 (73.8%, 69.8 to 77.6) | 225/351 (64.1%, 58.8 to 69.1) | 308/436 (70.6%, 66.1 to 74.9) | 0.36 | 0.00 / 0.00 | 24.6% |
| shipped | 373 (72.9%, 68.8 to 76.7) | 219/351 (62.4%, 57.1 to 67.5) | 303/436 (69.5%, 64.9 to 73.8) | 0.37 | 0.00 / 0.00 | 25.4% |
| a2 | 351 (68.6%, 64.3 to 72.6) | 196/351 (55.8%, 50.5 to 61.1) | 280/436 (64.2%, 59.5 to 68.7) | 0.47 | 0.00 / 0.00 | 32.2% |
| a3 | 351 (68.6%, 64.3 to 72.6) | 196/351 (55.8%, 50.5 to 61.1) | 280/436 (64.2%, 59.5 to 68.7) | 0.47 | 0.00 / 0.00 | 32.2% |

drive_caution minus Drive (paired): collision rate +0.0 points (95% CI +0.0 to +0.0); mean min gap +0.00 m (+0.00 to +0.00)

shipped minus Drive (paired): collision rate -1.0 points (95% CI -2.5 to +0.6); mean min gap +0.02 m (+0.01 to +0.02)

a2 minus Drive (paired): collision rate -5.3 points (95% CI -7.6 to -2.9); mean min gap +0.11 m (+0.09 to +0.15)

a3 minus Drive (paired): collision rate -5.3 points (95% CI -7.6 to -2.9); mean min gap +0.11 m (+0.09 to +0.15)


What the ideal driver adds:
* **Pull-out.** 88.7% of the episodes can be avoided by braking alone, yet Drive collides in 32.2% of those. The
  caution rule brings that to 3.1% for Drive and 5.3% for the attempt-3 network. Neither network learned that
  caution; the rule supplies it.
* **Swerve.** 86.5% can be avoided by braking and 99.8% by braking plus a swerve. Every brain still collides in 18%
  of the swerve-avoidable episodes: none of them swerves.
* **Side merge.** 68.6% can be avoided by braking. Drive collides in 64.1% of those, the attempt-3 network in 55.8%.
  The in-lane check brakes for the car's sideways motion, but too late for most.

## The parent's extra arm: shipped weights + the attempt-3 in-lane check, no lateral shield (dev seeds and hard classes)

The arm (`a3_extra.py`):
* The shipped network: attempt-1 weights and guards, the filtered observation, the limit cap and the flat caution.
* Its speed, min(vNet, vcap), goes through `shield.shield_inlane(..., ahead_rel=True)`.
* `ahead_rel` hands the guards only the car in the lane the car is in now, so a car that starts off its lane centre
  still checks a car in the lane centre.
* `staged/shield.lua` has this as `M.inlane(st, agents, dstop, true)`. It matches Python on 18 extra vectors: 4 of
  them slowed by the check, 15 starting more than 1 m off the lane centre.
* These are development and hard-class numbers, not a held-out rule.

| set (episodes per arm) | arm | collisions | reds | near s | incidents/100 km | m/episode | hard-class collision episodes, mean min gap |
|---|---|---|---|---|---|---|---|
| original (1024) | Drive (rules_v3) | 5 | 3 | 0.3 | 1.81 | 447 | - |
| original (1024) | shipped NN as in the game now (limit cap + caution) | 4 | 1 | 0.3 | 1.16 | 448 | - |
| original (1024) | **shipped NN + attempt-3 in-lane check** | 2 | 1 | 0.1 | 0.68 | 448 | - |
| overtake (1024) | Drive (rules_v3) | 0 | 0 | 0.0 | 0.00 | 344 | - |
| overtake (1024) | shipped NN as in the game now (limit cap + caution) | 1 | 0 | 0.0 | 0.28 | 344 | - |
| overtake (1024) | **shipped NN + attempt-3 in-lane check** | 0 | 0 | 0.0 | 0.00 | 344 | - |
| start_stop (512) | Drive (rules_v3) | 10 | 0 | 0.9 | 3.35 | 636 | - |
| start_stop (512) | shipped NN as in the game now (limit cap + caution) | 22 | 0 | 2.3 | 7.52 | 631 | - |
| start_stop (512) | **shipped NN + attempt-3 in-lane check** | 12 | 0 | 1.1 | 3.98 | 643 | - |
| start_crowd (512) | Drive (rules_v3) | 0 | 0 | 0.0 | 0.00 | 503 | - |
| start_crowd (512) | shipped NN as in the game now (limit cap + caution) | 0 | 0 | 0.0 | 0.00 | 483 | - |
| start_crowd (512) | **shipped NN + attempt-3 in-lane check** | 0 | 0 | 0.0 | 0.00 | 483 | - |
| pull_out (512) | Drive (rules_v3) | 203 | 0 | 26.3 | 117.60 | 381 | 39.6%, 4.62 m |
| pull_out (512) | shipped NN as in the game now (limit cap + caution) | 64 | 0 | 21.3 | 33.51 | 497 | 12.5%, 6.86 m |
| pull_out (512) | **shipped NN + attempt-3 in-lane check** | 43 | 0 | 9.9 | 20.07 | 515 | 8.4%, 9.25 m |
| swerve (512) | Drive (rules_v3) | 93 | 0 | 103.8 | 62.70 | 613 | 18.2%, 1.83 m |
| swerve (512) | shipped NN as in the game now (limit cap + caution) | 91 | 0 | 102.7 | 61.35 | 617 | 17.8%, 1.83 m |
| swerve (512) | **shipped NN + attempt-3 in-lane check** | 91 | 0 | 102.4 | 61.26 | 617 | 17.8%, 1.85 m |
| side_merge (512) | Drive (rules_v3) | 378 | 0 | 80.0 | 237.56 | 377 | 73.8%, 0.36 m |
| side_merge (512) | shipped NN as in the game now (limit cap + caution) | 373 | 0 | 79.2 | 228.26 | 387 | 72.9%, 0.37 m |
| side_merge (512) | **shipped NN + attempt-3 in-lane check** | 349 | 0 | 69.9 | 202.50 | 404 | 68.2%, 0.48 m |


On these development numbers, the shipped network with the in-lane check is the safest arm.
* Original: 0.68 incidents per 100 km, against 1.16 for the shipped network and 1.81 for Drive, at the same
  progress. These are small counts (3 against 5 and 8.3 incidents in 1024 episodes), so no confidence interval
  separates them.
* start_stop: 12 collisions, against 22 for the shipped network and 10 for Drive.
* pull_out: 8.4% collision episodes, against 12.5% and 39.6%.
* side_merge: 68.2% against 72.9% and 73.8%.
* Progress: start_crowd is 4% slower than Drive, which comes from the shipped weights. Elsewhere it equals Drive.

**Starting 2 to 5 m off the lane centre** (`a3_offlane.py`). The in-lane check's model of the car's sideways motion
contains the simulated car's real offset at every step of the 9 s. The worst miss is 0.12 m (2 m off at 2 m/s), well
inside the check's 0.35 m margin. Two caveats:
* The game converges to the path over BLEND = 10 m of travel rather than through this controller. A hand check of that
  rule at 2 m/s also stays inside the envelope.
* The real gap was elsewhere. With the lane-frame test, a car far off its lane left the car in the lane centre to
  guards that cannot see it. `ahead_rel` closes that gap.

## Attempt 4 (proposal only; not started)

1. **Junction speed is a decision, not the limit.** The network keeps the attempt-3 shield. A junction-approach cap
   becomes part of its output, and the ES objective is scored on junction-heavy scenarios. Target: fewer red lights
   and cross-traffic collisions than Drive.
2. **Keep the passing pipeline.** Its incidents per km in the overtake-heavy set are within about 1 per 100 km of
   Drive's.
3. **Replace the offset regression with an intent head** (stay, pass left, pass right), so the network itself
   decides when to pass.
4. **Optimise with more compute.** CMA-ES over the output layer and the last hidden layer, about 100 generations of
   64 scenarios.
5. **Evaluate once on a third, unused seed set,** with the same rule. Cost: about 4 to 5 hours on this PC while the
   game runs, most of it ES and evaluation time.


# Safety-fix arm: PRE-REGISTRATION (written 2026-11-01 14:18, before this arm was run on any of the seeds below)

**The arm** ("shipped + in-lane check"), exactly as `a3_extra.policy("shipped_inlane")` is now:
* The shipped attempt-1 weights (`lumen_nn_final.npz`) on the filtered observation.
* The network's speed: `min(share × 1.1 × limit, limit, 25)`, then the flat parked-car caution (`caution.py`).
* The attempt-1 guards (`a3_extra.guards_a1`, with the firm yellow at 3.5 m/s²), giving the cap and the stop distance.
* Then `shield.shield_inlane(..., ahead_rel=True)` on `min(vNet, vcap)` every step. The offset is the network's own;
  there is no lateral shield.
* It is frozen by file hash. sha256 prefixes:

  | file | sha256 prefix |
  |---|---|
  | `a3_extra.py` | 3dc5aac252570729 |
  | `shield.py` | 654f61b4db39ff22 |
  | `caution.py` | e9ea728b8ff2d23b |
  | `sim.py` | 0f8ef101a3e19b95 |
  | `brains.py` | a0fda8fbc24ebf28 |
  | `lumen_nn_final.npz` | 2cb55b1d65450346 |

  Nothing in them changes after any number below is seen.

**The comparison.** Drive (`brains.rules_v3`), the arm, and a reference without the check:
`a3_extra.policy("shipped")`, the shipped network as it is in the game now (limit cap and caution).

**Seeds** never used for anything. I checked every script and log in `training/`.
* The parent's example range 930_001 to 930_016 overlaps the start_crowd seeds 930_001 to 930_004 used in attempt 2,
  so it is not used.
* ORIGINAL: `sim.Scenarios(256, seed)` for seeds 940_001 to 940_016 (4096 episodes).
* OVERTAKE-HEAVY: `sim.Scenarios(256, seed, overtake=True)` for seeds 990_001 to 990_016 (4096 episodes).

**Decision rule, metrics, confidence intervals:** the same as the attempt-2 pre-registration. The original
distribution decides; overtake-heavy is reported beside it and cannot rescue a failure. The run happens once, and the
result is reported as it comes out, including a failure, with the rule it meets or misses.

## Safety-fix arm: RESULT (run once, 2026-11-01 14:19 to 14:21, seeds 940_xxx and 990_xxx)

**Verdict: neither rule is met.** On the original distribution the arm is significantly SAFER than Drive, at the same
progress.
* Incidents: 1.00 against 1.85 per 100 km, difference -0.85 (95% CI -1.54 to -0.21), ratio 0.54 (0.32 to 0.85).
* Progress: metres per episode -0.0% (-0.2 to +0.2).

Why that is still not a "met":
* **Rule (i)** also needs +8% progress, and the arm has none.
* **Rule (ii)** needs incidents at most half of Drive's (the ratio here is 0.54), plus a progress interval that does
  not go below 0. It goes to -0.2%.

What the safety comes from:
* **Mostly red lights:** 17 for Drive against 1 for both networks. That is the attempt-1 guards' firm yellow (stop
  whenever 3.5 m/s² is enough), which Drive could be given as well.
* **The in-lane check's own share** is the step from the reference to the arm: 1.12 to 1.00, with 18 to 16 collisions
  and 1.6 to 1.3 s too close. That step is small, and no interval was computed for it.

**Overtake-heavy:** 0.52 against 0.89 (CI -0.92 to +0.15), so no significant difference, at equal progress.

These numbers are simulator-only: perfect sensing, and the simulator's traffic and controller.


**original** (4096 episodes per arm, paired)

| arm | km | coll | red | stop | off | near s | incidents/100 km | m/episode | mean m/s | standing | first 20 s m/s | free-flow m/s | out of lane s/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Drive (rules_v3) | 1836 | 16 | 17 | 0 | 0 | 0.9 | 1.85 | 448 | 7.58 | 20.7% | 8.85 | 11.79 | 0.00 |
| shipped NN as in the game (limit cap + caution) | 1835 | 18 | 1 | 0 | 0 | 1.6 | 1.12 | 448 | 7.59 | 21.2% | 8.96 | 12.04 | 0.00 |
| **shipped NN + in-lane check (the safety-fix arm)** | 1835 | 16 | 1 | 0 | 0 | 1.3 | 1.00 | 448 | 7.59 | 21.2% | 8.96 | 12.04 | 0.00 |

| arm vs Drive | incidents/100 km (arm vs Drive) | difference, 95% CI | ratio, 95% CI | m/episode change, 95% CI | rule (i) | rule (ii) |
|---|---|---|---|---|---|---|
| shipped NN as in the game (limit cap + caution) | 1.12 vs 1.85 | -0.72 (-1.43 to -0.07) | 0.61 (0.36 to 0.95) | -0.0% (-0.2 to +0.1) | not met | not met |
| **shipped NN + in-lane check (the safety-fix arm)** | 1.00 vs 1.85 | -0.85 (-1.54 to -0.21) | 0.54 (0.32 to 0.85) | -0.0% (-0.2 to +0.2) | not met | not met |

**overtake** (4096 episodes per arm, paired)

| arm | km | coll | red | stop | off | near s | incidents/100 km | m/episode | mean m/s | standing | first 20 s m/s | free-flow m/s | out of lane s/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Drive (rules_v3) | 1398 | 9 | 3 | 0 | 0 | 0.4 | 0.89 | 341 | 5.75 | 25.9% | 7.96 | 12.05 | 0.00 |
| shipped NN as in the game (limit cap + caution) | 1397 | 11 | 0 | 0 | 0 | 0.6 | 0.83 | 341 | 5.75 | 26.2% | 8.02 | 12.33 | 0.00 |
| **shipped NN + in-lane check (the safety-fix arm)** | 1398 | 7 | 0 | 0 | 0 | 0.3 | 0.52 | 341 | 5.75 | 26.2% | 8.02 | 12.32 | 0.00 |

| arm vs Drive | incidents/100 km (arm vs Drive) | difference, 95% CI | ratio, 95% CI | m/episode change, 95% CI | rule (i) | rule (ii) |
|---|---|---|---|---|---|---|
| shipped NN as in the game (limit cap + caution) | 0.83 vs 0.89 | -0.06 (-0.68 to +0.55) | 0.94 (0.43 to 2.06) | -0.0% (-0.2 to +0.1) | not met | not met |
| **shipped NN + in-lane check (the safety-fix arm)** | 0.52 vs 0.89 | -0.36 (-0.92 to +0.15) | 0.59 (0.22 to 1.26) | -0.0% (-0.1 to +0.1) | not met | not met |


# Attempt 4: PRE-REGISTRATION ADDENDUM (written 2026-11-01 14:55, before any attempt-4 code, training or evaluation)

Storm approved attempt 4 ("ok keep going"). The parent's brief: the network decides its junction-approach speed, and
it gets an intent head. It keeps the shipped network's guards, including the 3.5 m/s² yellow; Drive stays as it is.
The attempt-3 passing pipeline and in-lane check stay. CMA-ES on the output layer and the last hidden layer.

**Decision rule, metrics, confidence intervals:** exactly as in the attempt-2 pre-registration. The original
distribution decides; overtake-heavy is reported beside it and cannot rescue a failure.

**A FOURTH seed set, never used for anything.** I checked every seed literal in `training/*.py`, the scripts and the
logs: 900, 910, 920, 925, 930, 935, 940, 950, 960, 970 to 977 and 990 (thousands, `_001` onward) are taken, plus
development 400_xxx to 470_xxx and training 10_000+, 60_000+, 200_000+, 300_000 to 310_001.
* ORIGINAL: `sim.Scenarios(256, seed)` for seeds 945_001 to 945_016 (4096 episodes).
* OVERTAKE-HEAVY: `sim.Scenarios(256, seed, overtake=True)` for seeds 995_001 to 995_016 (4096 episodes).

**Arms in the one held-out run:**
* Drive (`brains.rules_v3`).
* The attempt-4 network under its pipeline (`brains4.py`).
* The same network without the shield (ablation).
* The safety-fix arm as frozen above (`a3_extra.policy("shipped_inlane")`), as the reference for "what the game would
  get without learning".

**Network and pipeline (attempt 4):**
* 91 inputs, then 64, then 64, then 5 outputs: speed share; junction-approach share; three intent logits (stay, pass
  left, pass right).
* Speed: `min(share × 1.1 × limit, limit, 25)`, then the flat parked-car caution.
* Junction approach: from 60 m before the next stop line or junction until the car is through it, the speed is also
  capped at `(0.3 + 0.7 × sigmoid(junction output)) × limit`.
* Intent: the argmax of the three logits.
  * Stay asks for offset 0.
  * Pass left asks for -0.5 m and pass right for +0.5 m. The shield widens a pass request to the offset that clears the
    group it passes, or refuses it.
* The guards are guards2 with the yellow at 3.5 m/s², as the shipped network has. The attempt-3 shield, the in-lane
  check, the commitment and the limit cap are unchanged.

**Training (`a4_train.py`, one detached process, below normal priority, 5 workers):**
* Warm start from `a2_final.npz`:
  * The hidden layers as they are.
  * Speed head = the old speed head.
  * Intent logits built from the old offset output o2 as (0, -10 o2 - 1.13, 10 o2 - 1.13), which reproduces its
    pass decisions.
  * Junction head starts with no effect: bias +3, weights 0.
* sep-CMA-ES over the last hidden layer and the output layer (4485 parameters):
  * 16 candidates per generation, mirrored, 100 generations.
  * 64 training scenarios per generation, all candidates on the same ones: 24 junction-heavy, 20 plain,
    12 overtake-heavy, 4 standing starts at a stop sign, 4 stopped cars around a standing start.
  * Seeds 320_001 + 100 × generation.
* Objective: the attempt-3 objective, incidents dominate (collision 10 000 m, red 6 000, stop sign 3 000, off road
  6 000, 1 000 per second too close, 2 per second in the opposing lane, jerk 0.004).
* Selection: every 5 generations the CMA mean drives development seeds (original 400_001 and 400_002, 128 each;
  overtake-heavy 450_001, 128). The best by original-development reward is the candidate.
* Then the held-out run, once, in the same process. It is reported as it comes out.

The code files are frozen by sha256 below, before training starts.

**Frozen code (sha256 prefixes, computed 2026-11-01 14:58, before training starts; a4_train.py logs the same hashes at its start):**

* `a4_train.py` c9620a2b749bc75e
* `brains4.py` 55a0b7ac6e80db22
* `brains2.py` c4eeca6f3d933362
* `shield.py` 654f61b4db39ff22
* `caution.py` e9ea728b8ff2d23b
* `sim.py` 0f8ef101a3e19b95
* `brains.py` a0fda8fbc24ebf28
* `eval2.py` ef73114bdc5865e9
* `a3_extra.py` 3dc5aac252570729
* `train_es.py` c9c819879931b798
* `a2_final.npz` 05cc23788d4aee64
* `lumen_nn_final.npz` 2cb55b1d65450346

Training command: `py a4_train.py 100 5 170`, one detached process (100 generations or 170 minutes).


## Attempt 4: RESULT on the fourth seed set (run once, 2026-11-01 16:59 to 17:10, seeds 945_xxx and 995_xxx)

**Verdict: the goal is NOT met.**
* Original distribution: the attempt-4 network is +10.8% metres per episode (95% CI +9.9 to +11.7). It has 1.54
  incidents per 100 km against Drive's 0.93; difference +0.61 (CI -0.01 to +1.25), ratio 1.66 (0.99 to 3.13).
  Neither rule is met.
* Overtake-heavy: +45.6%, but significantly more incidents: 1.05 against 0.35, CI +0.17 to +1.22.

**What happened:**
* **The training barely moved.** The development reward went from 481.0 at the warm start to 483.9 after 100
  generations (best at generation 99). The candidate is essentially the attempt-2 decisions in the attempt-4
  pipeline.
* **The yellow at 3.5 m/s² did its job:** 1 red light against Drive's 6.
* **Collisions and near misses went up:** 23 collisions against 10, and 6.2 s too close against 1.0. Most of them
  come from the network's own driving, not from the passes. The same network without the shield, which cannot pass
  at all with its 0.5 m requests, has 25 collisions. The shipped weights under the same in-lane check have 10.
* **The selection was too small to see this.** The 256 original development episodes had 0 collisions for every
  checked mean. At the held-out rate of 0.56% per episode, about 1.4 were expected. The selection set could not tell
  the candidates apart on collisions.
* **The safety-fix arm** (shipped weights plus the in-lane check) is again the safest network arm: 0.68 against 0.93,
  difference -0.25 (CI -0.80 to +0.28). It is not significant on this seed set, at equal progress.

Staged: `staged/nn_weights_a4.json` (5 outputs, numpy self-test vectors), and `staged/NN_INTEGRATION.md`, section
"Attempt 4". The shield is unchanged from attempt 3.


**original** (4096 episodes per arm, paired)

| arm | km | coll | red | stop | off | near s | incidents/100 km | m/episode | mean m/s | standing | first 20 s m/s | free-flow m/s | out of lane s/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Drive (rules_v3) | 1834 | 10 | 6 | 0 | 0 | 1.0 | 0.93 | 448 | 7.57 | 20.6% | 8.83 | 11.82 | 0.00 |
| **attempt-4 NN** | 2032 | 23 | 1 | 0 | 1 | 6.2 | 1.54 | 496 | 8.50 | 16.9% | 9.16 | 12.56 | 1.37 |
| attempt-4 NN without the shield | 1837 | 25 | 0 | 0 | 0 | 1.9 | 1.46 | 448 | 7.61 | 21.0% | 9.00 | 12.14 | 58.78 |
| **shipped NN + in-lane check (the safety-fix arm)** | 1837 | 10 | 1 | 0 | 0 | 1.5 | 0.68 | 448 | 7.60 | 21.0% | 8.95 | 12.08 | 0.00 |

| arm vs Drive | incidents/100 km (arm vs Drive) | difference, 95% CI | ratio, 95% CI | m/episode change, 95% CI | rule (i) | rule (ii) |
|---|---|---|---|---|---|---|
| **attempt-4 NN** | 1.54 vs 0.93 | +0.61 (-0.01 to +1.25) | 1.66 (0.99 to 3.13) | +10.8% (+9.9 to +11.7) | not met | not met |
| attempt-4 NN without the shield | 1.46 vs 0.93 | +0.54 (-0.11 to +1.19) | 1.58 (0.91 to 3.00) | +0.1% (-0.0 to +0.3) | not met | not met |
| **shipped NN + in-lane check (the safety-fix arm)** | 0.68 vs 0.93 | -0.25 (-0.80 to +0.28) | 0.73 (0.35 to 1.47) | +0.2% (+0.0 to +0.3) | not met | not met |

**overtake** (4096 episodes per arm, paired)

| arm | km | coll | red | stop | off | near s | incidents/100 km | m/episode | mean m/s | standing | first 20 s m/s | free-flow m/s | out of lane s/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Drive (rules_v3) | 1412 | 2 | 3 | 0 | 0 | 0.0 | 0.35 | 345 | 5.81 | 25.6% | 7.98 | 12.06 | 0.00 |
| **attempt-4 NN** | 2057 | 15 | 0 | 0 | 1 | 5.5 | 1.05 | 502 | 8.72 | 15.6% | 8.96 | 13.36 | 4.95 |
| attempt-4 NN without the shield | 1413 | 11 | 0 | 0 | 0 | 0.8 | 0.84 | 345 | 5.83 | 25.9% | 8.06 | 12.39 | 59.13 |
| **shipped NN + in-lane check (the safety-fix arm)** | 1413 | 4 | 0 | 0 | 0 | 0.1 | 0.29 | 345 | 5.82 | 25.8% | 8.04 | 12.33 | 0.00 |

| arm vs Drive | incidents/100 km (arm vs Drive) | difference, 95% CI | ratio, 95% CI | m/episode change, 95% CI | rule (i) | rule (ii) |
|---|---|---|---|---|---|---|
| **attempt-4 NN** | 1.05 vs 0.35 | +0.69 (+0.17 to +1.22) | 2.95 (1.32 to 13.60) | +45.6% (+43.7 to +47.7) | not met | not met |
| attempt-4 NN without the shield | 0.84 vs 0.35 | +0.48 (+0.01 to +1.01) | 2.36 (1.02 to 9.70) | +0.0% (-0.1 to +0.1) | not met | not met |
| **shipped NN + in-lane check (the safety-fix arm)** | 0.29 vs 0.35 | -0.06 (-0.40 to +0.23) | 0.82 (0.17 to 2.70) | +0.1% (-0.0 to +0.1) | not met | not met |
