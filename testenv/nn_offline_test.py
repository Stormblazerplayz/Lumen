"""Offline checks of lumen/nn.lua: the real module and the shipped weights under LuaJIT, no game needed.
usage: py -3.12 testenv/nn_offline_test.py        (exit code 1 when anything fails)

Why it exists (2026-11-01): the network passed every simulator test and then sat still for two minutes in the game,
because the game shows it vehicles the simulator never does (parked cars beside the road, cars behind it). These
checks pin down what the network is shown, that the progress watchdog works, and that the stall cannot return.
"""
import os
import random
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lua_harness import Harness as _Harness, break_policy  # noqa: E402


def Harness():
    """the module under test: the source tree's nn.lua and weights, or a candidate named by the environment:
    NN_FILE (an nn.lua), NN_WEIGHTS (a weights json), NN_SHIELD (a lumen/shield.lua not in the source tree yet)"""
    f, w, sh = os.environ.get("NN_FILE"), os.environ.get("NN_WEIGHTS"), os.environ.get("NN_SHIELD")
    return _Harness(file=f, module="lumen/nn_candidate" if f else "lumen/nn", weights=w,
                    preload={"lumen/shield": sh} if sh else None)

CROSS_SLOT0 = 26 + 5          # 0-based index of the "crossing" flag of the nearest vehicle slot in the 91 inputs
PRESENT_SLOT0 = 26 + 7        # ... and of its "present" flag; each vehicle slot is 8 wide
VD_SLOT0 = 26 + 3

TESTS = []


def test(f):
    TESTS.append(f)
    return f


def present_count(obs):
    return sum(1 for i in range(8) if obs[PRESENT_SLOT0 + 8 * i] > 0.5)


def same(a, b, tol=1e-12):
    return max(abs(x - y) for x, y in zip(a, b)) <= tol


@test
def loads_and_self_tests():
    h = Harness()
    assert h.nn.load(), "nn.lua did not load: " + str(h.nn.info())
    assert any("self-test passed" in l for l in h.logs()), "the numpy self-test did not run"


@test
def the_original_incident_now_drives():
    """Storm's log: standing, stop sign 56 m ahead, parked cars around and one behind. Same inputs as an empty road."""
    h = Harness()
    base = {"v": 0.0, "signs": [(56, "briefStop")]}
    cars = [{"rs": 8, "rd": 27.5, "face": "across"}, {"rs": -5, "rd": 9, "face": "across"}, {"rs": 22, "rd": -14, "face": "across"},
            {"rs": 40, "rd": 22}, {"rs": 15, "rd": 12, "face": "across"}, {"rs": -6, "rd": 0.2}]
    v0, cap0, obs0, _ = h.request(base)
    v1, cap1, obs1, _ = h.request(dict(base, agents=cars))
    assert same(obs0, obs1) and v0 == v1, "the network still sees the parked cars"
    assert v1 > 8.0, f"the network asks only {v1:.1f} m/s on an open road with a stop sign 56 m away"


@test
def stationary_cars_beside_the_road_are_not_shown():
    h = Harness()
    rng = random.Random(1)
    for sc0 in ({"v": 0.0}, {"v": 8.0}, {"v": 5.0, "signs": [(60, "briefStop")]}):
        v0, c0, obs0, _ = h.request(sc0)
        for _ in range(120):
            a = {"rs": rng.uniform(-20, 80), "rd": rng.choice([-1, 1]) * rng.uniform(2.7, 30), "speed": 0.0,
                 "face": rng.choice(["along", "across"])}
            v1, c1, obs1, _ = h.request(dict(sc0, agents=[a]))
            assert same(obs0, obs1), f"a parked car at {a} changed the observation"


@test
def vehicles_behind_are_not_shown():
    h = Harness()
    rng = random.Random(2)
    sc0 = {"v": 3.0}
    v0, c0, obs0, _ = h.request(sc0)
    for _ in range(150):
        a = {"rs": rng.uniform(-80, -4.0), "rd": rng.uniform(-30, 30), "speed": rng.choice([0.0, rng.uniform(1.5, 15)]),
             "face": rng.choice(["along", "across", "against"])}
        v1, c1, obs1, _ = h.request(dict(sc0, agents=[a]))
        assert same(obs0, obs1), f"a vehicle behind at {a} changed the observation"


@test
def vehicles_that_matter_are_still_shown():
    h = Harness()
    v0, _, obs0, _ = h.request({"v": 0.0})
    # a stopped car in the lane ahead: shown, and the car will not drive into it. What it PLANS (after the guards and, for
    # attempt 2, the shield) stays within the rules' cap, and 2.5 m behind it the plan is to stand. The network's own request
    # is not asserted: attempt 1 held by itself, attempt 2 keeps asking for speed while a pass may be possible and leaves the
    # stop to the guards (this mock road is too narrow for any pass: its drivable offsets end at +-2.45 m).
    v1, cap1, obs1, _ = h.request({"v": 0.0, "agents": [{"rs": 20, "rd": 0.0}]})
    assert not same(obs0, obs1) and present_count(obs1) == 1, f"lane car: present={present_count(obs1)}"
    rows, _ = h.plan_steps({"v": 0.0, "agents": [{"rs": 20, "rd": 0.0}]}, 3.0)
    assert all(r["target"] <= cap1 + 1e-6 for r in rows), f"planned {max(r['target'] for r in rows):.2f} m/s over the rules' cap {cap1:.2f}"
    rows, _ = h.plan_steps({"v": 0.0, "agents": [{"rs": 7.4, "rd": 0.0}]}, 2.0)    # front bumper 2.5 m from its rear
    assert rows[-1]["target"] < 0.3, f"2.5 m behind a stopped car it plans {rows[-1]['target']:.2f} m/s"
    # a car at the lane edge (rd 2.0) is shown too
    _, _, obs2, _ = h.request({"v": 0.0, "agents": [{"rs": 20, "rd": 2.0}]})
    assert present_count(obs2) == 1
    # oncoming traffic in the other lane is shown (moving, within 4.5 m)
    _, _, obs3, _ = h.request({"v": 8.0, "agents": [{"rs": 60, "rd": -3.5, "speed": 12.0, "face": "against"}]})
    assert present_count(obs3) == 1
    # moving cross traffic is shown and flagged crossing
    _, _, obs4, _ = h.request({"v": 0.0, "agents": [{"rs": 25, "rd": 12, "face": "across", "speed": 10.0}]})
    assert present_count(obs4) == 1 and obs4[CROSS_SLOT0] == 1.0 and obs4[VD_SLOT0] != 0.0, "moving cross traffic lost its crossing flag"


@test
def a_parked_car_turned_sideways_is_an_obstacle_not_cross_traffic():
    h = Harness()
    _, _, obs, _ = h.request({"v": 0.0, "agents": [{"rs": 15, "rd": 1.0, "face": "across", "speed": 0.0}]})
    assert present_count(obs) == 1, "the sideways car in the lane is not shown"
    assert obs[CROSS_SLOT0] == 0.0 and obs[VD_SLOT0] == 0.0, "a standing sideways car is still flagged as crossing traffic"


@test
def simulator_style_scenes_lose_nothing():
    """LEAD, PARKED (rd <= 1.1), ONCOMING, CUTIN and CROSS vehicles in range are all shown."""
    h = Harness()
    rng = random.Random(3)
    U = rng.uniform
    for _ in range(300):
        agents = []
        for _ in range(rng.choice([1, 2, 3, 4])):
            k = rng.choice(["lead", "parked", "oncoming", "cross", "cutin"])
            if k == "lead":
                agents.append({"rs": U(6, 80), "rd": U(-0.35, 0.35), "speed": rng.choice([0.0, U(1.5, 14)])})
            elif k == "parked":
                agents.append({"rs": U(30, 80), "rd": rng.choice([0, .3, -.3, .6, -.5, 1.1]), "speed": 0.0})
            elif k == "oncoming":
                agents.append({"rs": U(10, 80), "rd": -3.5 + U(-0.2, 0.2), "speed": U(8, 18), "face": "against"})
            elif k == "cross":
                agents.append({"rs": U(8, 60), "rd": rng.choice([-1, 1]) * U(3, 30), "speed": U(7, 14), "face": "across"})
            else:
                agents.append({"rs": U(5, 40), "rd": rng.choice([-1, 1]) * U(1, 3.5), "speed": U(5, 12)})
        _, _, obs, _ = h.request({"v": U(0, 15), "agents": agents})
        assert present_count(obs) == len(agents), f"{len(agents)} vehicles in the scene, the network sees {present_count(obs)}: {agents}"


@test
def the_standing_trap_is_gone():
    """With a car parked right behind it, or clutter around it, a standing car used to hold forever (old nn.lua:
    150/150 and 49/150 scenes). Now it must always settle to driving."""
    h = Harness()
    rng = random.Random(4)
    U = rng.uniform
    for kind in ("queue_behind", "clutter"):
        holds = 0
        for _ in range(60):
            if kind == "queue_behind":
                ags = [{"rs": -U(4, 12), "rd": U(-0.3, 0.3)}]
            else:
                ags = [{"rs": U(-20, 70), "rd": rng.choice([-1, 1]) * U(3, 28), "face": rng.choice(["along", "across"])}
                       for _ in range(rng.randint(2, 6))]
            vn, cap, _ = h.settle({"v": 0.0, "vl": rng.choice([11.2, 15.0, 20.0]), "agents": ags})
            if cap > 3 and vn < 0.5 * min(cap, 20):
                holds += 1
        assert holds == 0, f"{kind}: {holds}/60 standing scenes hold"


@test
def the_rules_lead_detection_is_unchanged():
    h = Harness()
    r = h.debug({"v": 0.0, "agents": [{"rs": 15, "rd": 0.0}]})
    assert r.why == "lead" and r.vcap < 12, f"a stopped car 15 m ahead in the lane: guard says {r.why} cap {r.vcap}"
    r = h.debug({"v": 0.0, "agents": [{"rs": 15, "rd": 3.0}]})
    assert r.why != "lead", "a parked car 3 m to the side became a lead"


@test
def watchdog_fires_lasts_and_logs():
    h = Harness()
    break_policy(h)
    rows, d = h.plan_steps({"v": 0.0, "vl": 20.0}, 40.0)
    starts = [r["t"] for i, r in enumerate(rows) if r["relief"] and (i == 0 or not rows[i - 1]["relief"])]
    ends = [r["t"] for i, r in enumerate(rows) if not r["relief"] and i and rows[i - 1]["relief"]]
    assert starts and 3.8 <= starts[0] <= 4.4, f"first relief at {starts[:1]} s, expected about 4"
    assert 7.6 <= ends[0] - starts[0] <= 8.4, f"relief lasted {ends[0] - starts[0]:.1f} s, expected 8"
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert len(starts) in (3, 4) and all(11.5 <= g <= 12.6 for g in gaps), f"relief starts {starts}: expected one every ~12 s (8 s relief + 4 s hold)"
    first = next(r for r in rows if r["relief"])
    assert abs(first["target"] - 0.9 * 20.0 / 0.95) < 0.3 or first["target"] > 15, f"relief target {first['target']}"
    assert rows[0]["target"] < 0.2, "the broken network is not asking for ~0"
    logs = h.logs("watchdog")
    assert 1 <= len(logs) <= 2, f"{len(logs)} watchdog log lines in 40 s (rate limit is 30 s)"
    nums = logs[0].split("obs=[")[1].rstrip("]").split(",")
    assert len(nums) == 91, f"the log carries {len(nums)} inputs, expected 91"


@test
def watchdog_leaves_vehicles_and_stops_alone():
    h = Harness()
    break_policy(h)
    for name, sc in (("a parked car at the lane edge 20 m ahead", {"v": 0.0, "agents": [{"rs": 20, "rd": 2.0}]}),
                     ("a stopped lead 15 m ahead", {"v": 0.0, "agents": [{"rs": 15, "rd": 0.0}]}),
                     ("moving cross traffic 30 m ahead", {"v": 0.0, "agents": [{"rs": 30, "rd": 10, "face": "across", "speed": 10.0}]}),
                     ("a red light 2.5 m ahead", {"v": 0.0, "signs": [(2.5, "stop")]})):
        rows, d = h.plan_steps(sc, 20.0)
        assert d.nnReliefs is None or int(d.nnReliefs) == 0, f"watchdog overrode the network for {name}"


@test
def watchdog_does_not_count_a_moving_car():
    h = Harness()
    break_policy(h)
    rows, d = h.plan_steps({"v": 6.0, "vl": 20.0}, 20.0)
    assert d.nnReliefs is None or int(d.nnReliefs) == 0, "a car at 6 m/s is not stalled"


def main():
    failed = 0
    for f in TESTS:
        try:
            f()
            print("PASS  " + f.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL  " + f.__name__ + ": " + str(e))
        except Exception:
            failed += 1
            print("ERROR " + f.__name__ + "\n" + traceback.format_exc())
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
