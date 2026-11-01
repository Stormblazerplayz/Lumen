"""Write a checkpoint as the game's weight file: src/lua/ge/extensions/lumen/nn_weights.json,
plus nn_testvectors.json next to it (the same test vectors, for a later one-go check).

Flat row-major arrays, 6 significant digits. w1 is (OBS_DIM, 64) so w1[(i - 1) * 64 + j] in Lua is the
weight from input i to hidden unit j; w3 is (64, 2). The file also carries "selftest": observation
vectors from real simulator states (and a few random ones) with numpy's outputs for them, computed from
the ROUNDED weights exactly as the game will read them. nn.lua runs them when it loads, logs the largest
difference to Storm's beamng.log, and refuses to drive if they do not match.

usage: py export_weights.py [checkpoint.npz] [note] [--stage]
  --stage writes into training/staged/ instead of src/ (to be copied over together with nn.lua)
"""
import json
import os
import sys

import numpy as np

import brains
import sim

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "..", "src", "lua", "ge", "extensions", "lumen")
STAGE_DIR = os.path.join(HERE, "staged")


def r6(a):
    return [float(f"{x:.6g}") for x in np.asarray(a, dtype=np.float64).ravel()]


def test_vectors(n_sim=10, n_rand=4):
    sc = sim.Scenarios(32, 31337, hard=True)
    w = sim.World(sc)
    xs = []
    for k in range(400):
        if k in (60, 120, 180, 240, 300) and len(xs) < n_sim:
            obs = w.observe()
            xs.append(obs[0].astype(np.float64))
            xs.append(obs[7].astype(np.float64))
        w.step(np.full(32, 8.0), np.zeros(32))
    xs = xs[:n_sim]
    rng = np.random.default_rng(5)
    xs.extend(rng.standard_normal(sim.OBS_DIM) * 0.7 for _ in range(n_rand))
    return [np.array(r6(x)) for x in xs]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stage = "--stage" in sys.argv
    path = args[0] if args else os.path.join(HERE, "lumen_nn_best.npz")
    note = args[1] if len(args) > 1 else ""
    z = np.load(path)
    theta = z["theta"]
    net = brains.MLP()
    assert theta.shape == (net.n,), (theta.shape, net.n)
    W1, b1, W2, b2, W3, b3 = [a[0] for a in net.unpack(theta[None, :])]
    # what the game will read: the rounded numbers
    W1r, b1r, W2r, b2r, W3r, b3r = [np.array(r6(a)).reshape(a.shape) for a in (W1, b1, W2, b2, W3, b3)]
    tests = []
    for x in test_vectors():
        y = np.tanh(np.tanh(x @ W1r + b1r) @ W2r + b2r) @ W3r + b3r
        tests.append({"x": [float(v) for v in x], "y": [float(f"{v:.9g}") for v in y]})
    info = {"obs_dim": sim.OBS_DIM, "hidden": 64, "checkpoint": os.path.basename(path), "note": note,
            "iteration": int(z["it"]) if "it" in z.files else None, "check_eval": float(z["eval"]) if "eval" in z.files else None}
    out = {"info": info, "w1": r6(W1), "b1": r6(b1), "w2": r6(W2), "b2": r6(b2), "w3": r6(W3), "b3": r6(b3), "selftest": tests}
    dest = STAGE_DIR if stage else SRC_DIR
    os.makedirs(dest, exist_ok=True)
    for name, obj in (("nn_weights.json", out), ("nn_testvectors.json", {"info": info, "tests": tests})):
        target = os.path.join(dest, name)
        tmp = target + ".part"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, separators=(",", ":"))
        os.replace(tmp, target)
        print("wrote", os.path.normpath(target), os.path.getsize(target) // 1024, "KB")
    print(info, "| test vectors:", len(tests))


if __name__ == "__main__":
    main()
