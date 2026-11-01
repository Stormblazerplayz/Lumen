"""In-game checks of Lumen NN on the test instance (the test copy must be running with the current mod).

  py nn_ingame.py equiv            the Lua network against numpy on real simulator states and random vectors
  py nn_ingame.py obs [seconds]    the observation and decision of each fleet car on the NN, named by feature
  py nn_ingame.py fleet [seconds] [nn|drive]   a fleet run on west_coast_usa with the chosen brain

Reads the weights from src/lua/ge/extensions/lumen/nn_weights.json, the file the game loads.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "testenv"))
import mcp  # noqa: E402

import sim  # noqa: E402

WEIGHTS = os.path.join(HERE, "..", "src", "lua", "ge", "extensions", "lumen", "nn_weights.json")


def load_weights():
    j = json.load(open(WEIGHTS))
    h = j["info"]["hidden"]
    n = j["info"]["obs_dim"]
    return (np.array(j["w1"]).reshape(n, h), np.array(j["b1"]), np.array(j["w2"]).reshape(h, h), np.array(j["b2"]),
            np.array(j["w3"]).reshape(h, 2), np.array(j["b3"]), j["info"])


def numpy_forward(W, x):
    W1, b1, W2, b2, W3, b3, _ = W
    return np.tanh(np.tanh(x @ W1 + b1) @ W2 + b2) @ W3 + b3


def equiv():
    W = load_weights()
    n = W[6]["obs_dim"]
    assert n == sim.OBS_DIM, f"weights have {n} inputs, the simulator {sim.OBS_DIM}"
    sc = sim.Scenarios(32, 31337, hard=True)
    w = sim.World(sc)
    vecs = []
    for k in range(220):
        if k % 20 == 0 and k:
            vecs.extend(w.observe()[:3].astype(np.float64))
        w.step(np.full(32, 8.0), np.zeros(32))
    rng = np.random.default_rng(5)
    vecs.extend(rng.standard_normal((12, n)) * 0.7)
    worst = 0.0
    for x in vecs:
        lit = ",".join(f"{v:.9g}" for v in x)
        out = json.loads(mcp.lua("return jsonEncode(require('lumen/nn').forwardTest({" + lit + "}))"))
        ref = numpy_forward(W, x)
        worst = max(worst, float(np.abs(np.array(out) - ref).max()))
    print(f"{len(vecs)} vectors, largest difference between the Lua network and numpy: {worst:.2e}  ->  {'OK' if worst < 1e-4 else 'FAIL'}")
    return worst < 1e-4


NAMES = (["kap@%d" % a for a in (5, 10, 20, 30, 45, 60, 80, 100)] + ["vsafe/20", "limit/20", "v/20", "lat/2", "heading*2", "prevT/20", "prevD/2",
         "roomR/4", "roomL/4", "twoWay", "shoulder/2", "lineDist/100", "line:none", "line:green", "line:yellow", "line:red", "line:stopsign", "waited"]
         + [f"veh{i}:{f}" for i in range(1, 9) for f in ("ahead/50", "right/5", "dv/15", "vside/10", "size/5", "cross", "acc/5", "present")]
         + ["pathLeft/200"])


def obs(seconds=20):
    t0 = time.time()
    mcp.lua("extensions.lumen_robotaxi.setBrain('nn') return 1")
    seen = 0
    while time.time() - t0 < seconds and seen < 6:
        for slot in (1, 2, 3, 4):
            r = mcp.lua(f"return jsonEncode(extensions.lumen_robotaxi.debugNN({slot}))")
            if r in ("null", "", None):
                continue
            d = json.loads(r)
            if not d:
                continue
            o = d["obs"]
            assert len(o) == len(NAMES), (len(o), len(NAMES))
            shown = {NAMES[i]: round(o[i], 3) for i in range(len(o)) if abs(o[i]) > 1e-9 and not NAMES[i].startswith("kap@")}
            kap = [round(o[i], 2) for i in range(8)]
            print(f"car {slot}: v={d['v']:.1f} lat={d['lat']:+.2f} limit={d['vlim']:.1f} vcap={d['vcap']:.1f} ({d['why']}) out={[round(a, 2) for a in d['out']]}")
            print("   curvature x50:", kap)
            print("   ", shown)
            seen += 1
        time.sleep(2)


def fleet(seconds=180, brain="nn"):
    import drive_test as dt
    print("loading west_coast_usa", dt.load_level("west_coast_usa"))
    mcp.lua(f"extensions.lumen_robotaxi.setBrain('{brain}') return extensions.lumen_robotaxi.getState().brain")
    t0 = time.time()
    stats = {}
    services = []
    while time.time() - t0 < seconds:
        f = dt.fleet()
        for c in f:
            st = stats.setdefault(c["slot"], {"n": 0, "lat2": 0.0, "v": 0.0, "moving": 0, "start": None, "last": None, "dist": 0.0})
            if c.get("x") is not None:
                if st["last"] is not None:
                    st["dist"] += ((c["x"] - st["last"][0]) ** 2 + (c["y"] - st["last"][1]) ** 2) ** 0.5
                st["last"] = (c["x"], c["y"])
            if c.get("mode") == "follow":
                st["n"] += 1
                st["lat2"] += (c.get("lat") or 0) ** 2
                st["v"] += c.get("speed") or 0
            if c.get("service") and not any(s[0] == c["slot"] and s[1] == c["service"] for s in services):
                services.append((c["slot"], c["service"], round(time.time() - t0)))
        time.sleep(2)
    print(f"\nbrain {brain}, {seconds:.0f} s on west_coast_usa")
    tot_d = sum(s["dist"] for s in stats.values())
    tot_n = sum(s["n"] for s in stats.values())
    for k, s in sorted(stats.items()):
        print(f" car {k}: {s['dist']:.0f} m driven, mean speed {s['v'] / max(1, s['n']):.1f} m/s, lateral rms {(s['lat2'] / max(1, s['n'])) ** 0.5:.2f} m")
    print(f" fleet: {tot_d / 1000:.2f} km in total; service events: {services or 'none'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "equiv"
    if cmd == "equiv":
        sys.exit(0 if equiv() else 1)
    elif cmd == "obs":
        obs(float(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif cmd == "fleet":
        fleet(float(sys.argv[2]) if len(sys.argv) > 2 else 180, sys.argv[3] if len(sys.argv) > 3 else "nn")
