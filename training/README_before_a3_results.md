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
