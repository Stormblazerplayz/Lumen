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
