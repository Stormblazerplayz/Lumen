"""Outcome-based tuning of ONE parameter: the bias of the network's speed output. A bigger bias drives closer to
the speed limit; the guardrails keep stops, junctions and following unchanged. Evaluates each shift on a
selection set (seed 888, never the held-out test seed) and reports incidents and speed next to the rules.
usage: py tune_bias.py checkpoint.npz [out.npz] [episodes=384]
With out.npz: writes the checkpoint with the smallest shift whose speed is at least the rules' speed + 0.05 m/s."""
import sys

import numpy as np

import brains
import evaluate

NET = brains.MLP()


def policy(theta):
    return lambda w: brains.nn_actions(NET.forward(theta[None, :], w.observe()[None, :, :])[0], w)


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].isdigit() else None
    N = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 384
    theta0 = np.load(path)["theta"]
    base = evaluate.stats(evaluate.run_policy(brains.rules, N, 888)[0], N)
    print(f"rules on the selection set: {base['per100km']:.2f} incidents/100 km, {base['speed']:.2f} m/s, {base['m_per_ep']:.0f} m/episode")
    chosen = None
    for delta in (0.0, 0.15, 0.3, 0.45, 0.6, 0.8):
        th = theta0.copy()
        th[-2] += delta                       # b3[0]: the speed logit
        st = evaluate.stats(evaluate.run_policy(policy(th), N, 888)[0], N)
        ok = st["speed"] >= base["speed"] + 0.05 and st["m_per_ep"] >= base["m_per_ep"]
        print(f"  bias +{delta:.2f}: {st['per100km']:.2f} incidents/100 km ({st['collisions']} collisions, {st['red']} red), {st['speed']:.2f} m/s, {st['m_per_ep']:.0f} m/episode  {'<- meets the speed guard' if ok else ''}")
        if ok and chosen is None:
            chosen = (delta, th)
    if out and chosen:
        np.savez(out, theta=chosen[1], bias_shift=chosen[0], source=path)
        print(f"wrote {out} with bias +{chosen[0]:.2f}")
    elif out:
        print("no shift met the speed guard; nothing written")


if __name__ == "__main__":
    main()
