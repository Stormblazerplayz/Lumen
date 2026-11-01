"""Test vectors for staged/shield.lua: states from simulator rollouts (development seeds) with the inputs the shield
receives (own state, vehicles, the guards' caps at the offsets it asks about) and the Python shield's answer."""
import json
import sys

import numpy as np

import brains2
import rulepass
import shield
import sim

N = 96
rec_rows = []
rec_inlane = []
orig = shield.shield
orig_in = shield.shield_inlane


def agents_of(w, r):
    sc = w.sc
    agents = []
    for k in range(sim.K):
        kind = int(sc.kind[r, k])
        if kind < 0:
            continue
        cross = kind == sim.CROSS
        onc = kind == sim.ONCOMING
        held = bool(cross and w.agHeld[r, k])
        held_d = float(max(0.0, abs(-sc.side[r, k] * (abs(sc.edgeL[r]) + 2.0) - w.agD[r, k]) - 1.0))
        agents.append({"rs": float(w.agS[r, k] - w.es[r]), "rd": float(w.agD[r, k] - w.ed[r]), "ad": float(w.agD[r, k]),
                       "vs": float(-w.agV[r, k] if onc else (0.0 if cross else w.agV[r, k])), "vd": float(w.agVd[r, k]),
                       "acc": float(np.clip(w.agAcc[r, k], -7, 2)), "aL": float(sc.W[r, k] if cross else sc.L[r, k]),
                       "cross": bool(cross), "onc": bool(onc), "held": held, "heldD": held_d})
    return agents


def wrapped_in(w, vT, dstop, rows):
    v, mode = orig_in(w, vT, dstop, rows)
    for i, r in enumerate(rows):
        r = int(r)
        rec_inlane.append({"st": {"v0": float(w.ev[r]), "ed": float(w.ed[r]), "epsi": float(w.epsi[r]), "vA": float(vT[r])},
                           "dstop": float(dstop[r]), "agents": agents_of(w, r), "out": {"v": float(v[i]), "mode": int(mode[i])}})
    return v, mode


shield.shield_inlane = wrapped_in


def wrapped(w, vT_plan, dT_plan, rows, cap_fn):
    log = {int(r): [] for r in rows}

    def cap2(r, offs):
        cap, dstop = cap_fn(r, offs)
        for i, rr in enumerate(r):
            log[int(rr)].append({"offset": float(offs[i]), "cap": float(cap[i]), "dstop": float(dstop[i])})
        return cap, dstop
    v, d, mode = orig(w, vT_plan, dT_plan, rows, cap2)
    sc = w.sc
    for i, r in enumerate(rows):
        r = int(r)
        agents = agents_of(w, r)
        lo = float(sc.edgeL[r] + sim.EGO_W / 2 + 0.2)
        hi = float(sc.edgeR[r] - sim.EGO_W / 2 - 0.2)
        line = sc.jS[r] - 6.0
        dist = line - (w.es[r] + 2.6)
        valid = (dist > -2.0) & (sc.jKind[r] != sim.J_NONE)
        near = float(np.where(valid, dist, 1e9).min())
        rec_rows.append({"st": {"v0": float(w.ev[r]), "ed": float(w.ed[r]), "epsi": float(w.epsi[r]), "vA": float(vT_plan[r]),
                                "dT": float(dT_plan[r]), "lo": lo, "hi": hi, "near": near},
                         "agents": agents, "caps": log[r], "out": {"v": float(v[i]), "d": float(d[i]), "mode": int(mode[i])}})
    return v, d, mode


shield.shield = wrapped
rng = np.random.default_rng(3)
for overtake, seed in ((False, 400_501), (True, 450_501), ("hard", 400_601)):
    sc = sim.Scenarios(N, seed, hard=True) if overtake == "hard" else sim.Scenarios(N, seed, overtake=overtake)
    w = sim.World(sc)
    for step in range(sim.STEPS):
        out = rulepass.rule_out(w)
        # half the environments ask for 60% of the rule's offset, so the widening of small requests is exercised
        out[::2, 1] = np.arctanh(np.clip(0.6 * np.tanh(out[::2, 1]), -0.99, 0.99))
        before = len(rec_rows)
        before_in = len(rec_inlane)
        vT, dT = brains2.nn_actions2(out, w)
        if step % 12 != 5:
            del rec_rows[before:]
        if step % 25 != 7:
            # keep every in-lane call that the check slowed (the branch that matters), sample the rest
            keep = [r for r in rec_inlane[before_in:] if r["out"]["mode"] == 7 or r["out"]["v"] < r["st"]["vA"] - 1e-9]
            del rec_inlane[before_in:]
            rec_inlane.extend(keep[:3])
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
sel = rng.permutation(len(rec_rows))[:400]
rows = [rec_rows[i] for i in sorted(sel)]
modes = np.bincount([r["out"]["mode"] for r in rows], minlength=6).tolist()
# in-lane vectors: all the "nothing clear" ones and the slowed ones, and a sample of the rest
slow = [r for r in rec_inlane if r["out"]["mode"] == 7 or r["out"]["v"] < r["st"]["vA"] - 1e-9]
rest = [r for r in rec_inlane if not (r["out"]["mode"] == 7 or r["out"]["v"] < r["st"]["vA"] - 1e-9) and any(a["cross"] or abs(a["rd"]) < 4 for a in r["agents"])]
pick = slow[:300] + [rest[i] for i in rng.permutation(len(rest))[:200]]
with open("staged/shield_vectors.json", "w") as f:
    json.dump({"lateral": rows, "inlane": pick}, f, separators=(",", ":"))
print(len(rows), "lateral vectors; modes", modes, "|", len(pick), "in-lane vectors (", len(slow), "slowed or not clear )")
