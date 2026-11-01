"""Lumen NN vs Lumen Drive on held-out scenarios: the "3x better" test (definition in README.md).

usage: py evaluate.py [weights.npz [more.npz ...]] [--episodes N] [--seed S] [--raw]
  several checkpoints are listed side by side; --raw also evaluates each without the guardrails

The scenarios come from one seed that no training run ever uses (training: 10_000+, check set 777,
behaviour cloning 50_000+). Every brain sees identical roads and traffic.
"""
import os
import sys

import numpy as np

import brains
import sim

HERE = os.path.dirname(os.path.abspath(__file__))
NET = brains.MLP()
SEED, EPISODES = 424242, 1024
NAMES = {0: "lead", 1: "oncoming", 2: "cross", 3: "parked", 4: "cutin", 5: "reverser"}


def run_policy(policy, N, seed):
    """policy(world) -> (target speed, offset). Returns metrics and collision causes."""
    sc = sim.Scenarios(N, seed)
    w = sim.World(sc)
    causes = {}
    for _ in range(sim.STEPS):
        vT, dT = policy(w)
        alive = ~w.done
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        kind = sc.kind
        active = kind >= 0
        cross = kind == sim.CROSS
        aL = np.where(cross, sc.W, sc.L)
        aW = np.where(cross, sc.L, sc.W)
        hit = active & (np.abs(w.agS - w.es[:, None]) < (sim.EGO_L + aL) / 2) & (np.abs(w.agD - w.ed[:, None]) < (sim.EGO_W + aW) / 2)
        for e in np.where(hit.any(axis=1) & alive)[0]:
            n = NAMES[int(kind[e, np.where(hit[e])[0][0]])]
            causes[n] = causes.get(n, 0) + 1
        if w.done.all():
            break
    return w.summary(), causes


def nn_policy(theta):
    def pol(w):
        out = NET.forward(theta[None, :], w.observe()[None, :, :])[0]
        return brains.nn_actions(out, w)
    return pol


def stats(m, N):
    km = m["dist"].sum() / 1000
    near_events = m["near"].sum() / 1.0               # one near-miss event per second spent too close
    incidents = m["collision"].sum() + m["red"].sum() + m["stopsign"].sum() + m["offroad"].sum() + near_events
    return {
        "km": km,
        "episodes": N,
        "collisions": int(m["collision"].sum()),
        "red": int(m["red"].sum()),
        "stopsign": int(m["stopsign"].sum()),
        "offroad": int(m["offroad"].sum()),
        "near_s": float(m["near"].sum()),
        "incidents": float(incidents),
        "per100km": float(incidents / max(km, 1e-9) * 100),
        "speed": float(m["dist"].sum() / max(1.0, m["time"].sum())),
        "m_per_ep": float(m["dist"].sum() / N),
        "stopped": float(m["blocked"].sum() / max(1.0, m["time"].sum())),
        "jerk": float(m["jerk2"].sum() / max(1.0, m["time"].sum())),
        "latmax": float(np.percentile(m["latmax"], 95)),
    }


def row(name, st):
    return (f"{name:<22} {st['km']:7.1f} {st['collisions']:5d} {st['red']:4d} {st['stopsign']:4d} {st['offroad']:4d} {st['near_s']:6.1f} "
            f"{st['per100km']:9.2f} {st['speed']:6.2f} {st['m_per_ep']:7.0f} {100 * st['stopped']:6.1f}% {st['latmax']:6.2f}")


def main():
    args = sys.argv[1:]
    N, seed, raw = EPISODES, SEED, False
    paths = []
    i = 0
    while i < len(args):
        if args[i] == "--episodes":
            N = int(args[i + 1]); i += 2
        elif args[i] == "--seed":
            seed = int(args[i + 1]); i += 2
        elif args[i] == "--raw":
            raw = True; i += 1
        else:
            paths.append(args[i]); i += 1
    if not paths:
        paths = [os.path.join(HERE, "lumen_nn_best.npz")]
    res = {}
    res["Lumen Drive (rules)"] = stats(*run_policy(brains.rules, N, seed)[:1], N)
    res["rules + yellow latch"] = stats(*run_policy(brains.rules_latched, N, seed)[:1], N)
    res["rules_v3 (the NN's teacher)"] = stats(*run_policy(brains.rules_v3, N, seed)[:1], N)
    for wpath in paths:
        if not os.path.exists(wpath):
            continue
        theta = np.load(wpath)["theta"]
        label = os.path.basename(wpath).replace(".npz", "")
        m, causes = run_policy(nn_policy(theta), N, seed)
        res["NN " + label] = stats(m, N)
        print(f"NN {label} collision causes:", causes)
        if raw:
            def raw_pol(w, th=theta):
                out = NET.forward(th[None, :], w.observe()[None, :, :])[0]
                return brains.nn_actions(out, w, shield=False)
            res["NN " + label + " (no guardrails)"] = stats(run_policy(raw_pol, N, seed)[0], N)
    res["never moves"] = stats(*run_policy(lambda w: (np.zeros(w.E), np.zeros(w.E)), N, seed)[:1], N)
    print(f"\nheld-out seed {seed}, {N} episodes of {sim.STEPS * sim.DT:.0f} s")
    print(f"{'brain':<34} {'km':>7} {'coll':>5} {'red':>4} {'stop':>4} {'off':>4} {'near_s':>6} {'inc/100km':>9} {'m/s':>6} {'m/ep':>7} {'stopped':>7} {'latg95':>6}")
    for k, v in res.items():
        print(row(k[:33].ljust(33), v))
    base = res["Lumen Drive (rules)"]
    for k, nn in res.items():
        if not k.startswith("NN ") or "no guardrails" in k:
            continue
        ratio = base["per100km"] / max(nn["per100km"], 1e-9)
        guard_speed = nn["speed"] >= base["speed"] - 1e-9
        guard_prog = nn["m_per_ep"] >= base["m_per_ep"] - 1e-9
        rl = res["rules + yellow latch"]["per100km"] / max(nn["per100km"], 1e-9)
        print(f"\n{k}: incidents per 100 km {base['per100km']:.2f} (rules) -> {nn['per100km']:.2f}  =>  {ratio:.2f}x better than the rules, {rl:.2f}x vs rules+latch")
        print(f"   guards: mean speed {nn['speed']:.2f} vs {base['speed']:.2f} m/s: {'ok' if guard_speed else 'FAIL'}; progress {nn['m_per_ep']:.0f} vs {base['m_per_ep']:.0f} m/episode: {'ok' if guard_prog else 'FAIL'}")
        print("   3x reached:", "YES" if (ratio >= 3.0 and guard_speed and guard_prog) else "NO")


if __name__ == "__main__":
    main()
