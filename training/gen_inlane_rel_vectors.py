"""Parity vectors for shield.inlane with aheadRel = true (the shipped network + in-lane check, a3_extra.py): states from
development rollouts in which half of the cars start 2 to 4 m off their lane centre (the game's take-over in a lot).
Writes staged/shield_vectors_rel.json; lua_shield_test.py --rel checks staged/shield.lua against it."""
import json

import numpy as np

import a3_extra
import shield
import sim


class G:
    @staticmethod
    def agents_of(w, r):
        """the vehicles of environment r as shield.lua receives them (the same fields as gen_shield_vectors.py)"""
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


rec = []
orig = shield.shield_inlane


def wrapped(w, vT, dstop, rows, ahead_rel=False):
    v, mode = orig(w, vT, dstop, rows, ahead_rel=ahead_rel)
    for i, r in enumerate(rows):
        r = int(r)
        rec.append({"st": {"v0": float(w.ev[r]), "ed": float(w.ed[r]), "epsi": float(w.epsi[r]), "vA": float(vT[r])},
                    "dstop": float(dstop[r]), "agents": G.agents_of(w, r), "out": {"v": float(v[i]), "mode": int(mode[i])}})
    return v, mode


shield.shield_inlane = wrapped
pol = a3_extra.policy("shipped_inlane")
rng = np.random.default_rng(9)
picked = []
for seed, hard in ((400_901, False), (400_902, True)):
    sc = sim.Scenarios(64, seed, hard=hard)
    sc.edgeR[:] = np.maximum(sc.edgeR, 6.0)
    w = sim.World(sc)
    w.ed[::2] = rng.uniform(2.0, 4.0, 32)
    for step in range(sim.STEPS):
        before = len(rec)
        vT, dT = pol(w)
        new = rec[before:]
        del rec[before:]
        if step % 10 == 3:
            slow = [r for r in new if r["out"]["mode"] == 7 or r["out"]["v"] < r["st"]["vA"] - 1e-9]
            off = [r for r in new if abs(r["st"]["ed"]) > 1.0]
            picked.extend(slow[:4] + off[:2])
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
with open("staged/shield_vectors_rel.json", "w") as f:
    json.dump(picked, f, separators=(",", ":"))
print(len(picked), "in-lane vectors with aheadRel; slowed or not clear:",
      sum(1 for r in picked if r["out"]["mode"] == 7 or r["out"]["v"] < r["st"]["vA"] - 1e-9),
      "| starting more than 1 m off the lane centre:", sum(1 for r in picked if abs(r["st"]["ed"]) > 1.0))
