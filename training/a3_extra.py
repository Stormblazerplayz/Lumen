"""The parent's extra arm (2026-11-01 14:00, after game evidence that the shipped network damaged 5 Lumens in 7 minutes):
"shipped attempt-1 weights + the attempt-3 in-lane check (shield.inlane on min(vNet, vcap)), no lateral shield",
against the shipped network as it is in the game now (attempt-1 weights and guards, filtered observation, the limit cap,
the flat parked-car caution) and Drive. Development seeds and the attempt-3 hard classes only (not a held-out rule).
The in-lane check here hands the guards only the car in the lane the car is in now (ahead_rel=True), so a car that
starts off its lane centre still checks one in the lane centre.

usage: py a3_extra.py [workers=5]
"""
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

import brains
import caution
import shield
import sim
import train_es
from brains import (A_DEC, A_HARD, FRONT, IDM_A, IDM_B, IDM_S0, IDM_T, LANE_HALF, LINE_GAP, STOP_GAP, V_MAX,
                    _junction_discipline)

HERE = os.path.dirname(os.path.abspath(__file__))
NET = brains.MLP()
THETA = np.load(os.path.join(HERE, "lumen_nn_final.npz"))["theta"]


def guards_a1(w):
    """brains.guards (the shipped network's guards: firm yellow at 3.5 m/s2) with the distance to the stop point"""
    sc, E = w.sc, w.E
    s, v = w.es, w.ev
    ahead = np.arange(0, 120, 3.0)
    kap = np.abs(w.kap_at(s[:, None] + ahead[None, :]))
    vcurve = np.sqrt(sim.A_LAT / np.maximum(kap, 1e-4))
    vsafe = np.minimum(np.sqrt(vcurve ** 2 + 2 * A_DEC * ahead[None, :]).min(axis=1), V_MAX)
    stopS = np.full(E, 1e9)
    sig = sim.signal_state(w.t, sc)
    for j in range(sim.J):
        line = sc.jS[:, j] - 6.0
        lineS = line - FRONT - LINE_GAP
        before = (s + FRONT) < line + 0.5
        room = (lineS - s) > v * v / (2 * brains.YELLOW_DECEL) + 1
        red = before & (sig[:, j] == sim.S_RED)
        yellow_now = before & (sig[:, j] == sim.S_YELLOW) & room
        w.latchStop[:, j] = np.where(sig[:, j] == sim.S_GREEN, False, w.latchStop[:, j] | yellow_now)
        w.latchStop[:, j] &= before
        yellow = yellow_now | (w.latchStop[:, j] & (sig[:, j] == sim.S_YELLOW))
        stopsign = before & (sig[:, j] == sim.S_STOPSIGN)
        done = (w.stopped_at[:, j] >= 0) & (w.t - w.stopped_at[:, j] >= 1.6)
        hold = red | yellow | (stopsign & ~done)
        stopS = np.where(hold & (lineS < stopS), lineS, stopS)
    kind = sc.kind
    active = kind >= 0
    cross = kind == sim.CROSS
    rs = w.agS - s[:, None]
    rd = w.agD - w.ed[:, None]
    along = np.where(kind == sim.ONCOMING, -w.agV, np.where(cross, 0.0, w.agV))
    inlane = active & ~cross & (np.abs(rd) < LANE_HALF + np.maximum(0, sc.W / 2 - 0.95)) & (rs > 1) & (rs < 150)
    inlane &= (kind != sim.ONCOMING) | (np.abs(rd) < 1.0)
    gap = np.where(inlane, rs - sc.L / 2 - FRONT, 1e9)
    li = np.argmin(gap, axis=1)
    lg = gap[np.arange(E), li]
    lv = np.maximum(0, along[np.arange(E), li])
    has = lg < 1e8
    stoppedLead = has & (lv < 0.5)
    stopS = np.where(stoppedLead & (s + lg - STOP_GAP < stopS), s + lg - STOP_GAP, stopS)
    v0 = np.maximum(np.minimum(vsafe, 1.05 * w.vlim_at(s)), 1)
    sstar = IDM_S0 + np.maximum(0, v * IDM_T + v * (v - lv) / (2 * np.sqrt(IDM_A * IDM_B)))
    acc = IDM_A * (1 - (v / v0) ** 4 - (sstar / np.maximum(lg, 0.5)) ** 2)
    idmCap = np.where(has & ~stoppedLead, np.maximum(0, v + np.clip(acc, -A_HARD, IDM_A) * 0.6), 1e9)
    stopS = _junction_discipline(w, stopS)
    lead = 1.0 + 0.6 * v
    vcap = np.minimum(vsafe, idmCap)
    vcap = np.minimum(vcap, np.sqrt(2 * A_DEC * np.maximum(0, stopS - s - lead)))
    vcap = np.where(stopS - s < 0.3, 0.0, vcap)
    return vcap, np.where(stopS < 1e8, stopS - s, 1e9)


def policy(arm):
    if arm == "drive":
        return brains.rules_v3

    def pol(w):
        out = NET.forward(THETA[None, :], w.observe()[None, :, :])[0]
        share = 1 / (1 + np.exp(-out[:, 0]))
        vlim = w.vlim_at(w.es)
        vNet = np.minimum(np.minimum(share * 1.1 * vlim, vlim), V_MAX)
        vNet = np.minimum(vNet, caution.cap(w, "flat"))
        vcap, dstop = guards_a1(w)
        vT = np.minimum(vNet, vcap)
        if arm == "shipped_inlane":
            rows = np.where(~w.done)[0]
            vI, _ = shield.shield_inlane(w, vT, dstop, rows, ahead_rel=True)
            vT[rows] = vI
        return vT, np.tanh(out[:, 1]) * 4.0
    return pol


def run(task):
    arm, cls, seed, n = task
    train_es.lower_priority()
    if cls == "original":
        sc = sim.Scenarios(n, seed)
    elif cls == "overtake":
        sc = sim.Scenarios(n, seed, overtake=True)
    else:
        sc = sim.Scenarios(n, seed, mode=cls)
    w = sim.World(sc)
    pol = policy(arm)
    for _ in range(sim.STEPS):
        vT, dT = pol(w)
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
    m = w.summary()
    out = {k: m[k] for k in ("dist", "collision", "red", "stopsign", "offroad", "near")}
    if getattr(sc, "hazard", False):
        out["hz_coll"] = (m["collision"] > 0).astype(float)
        out["hz_gap"] = np.where(w.hz_t0 >= 0, w.hz_mingap, np.nan)
    return task[:3], out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    arms = ("drive", "shipped", "shipped_inlane")
    sets = {"original": [400_001 + i for i in range(4)], "overtake": [450_001 + i for i in range(4)],
            "start_stop": [420_001, 420_002], "start_crowd": [430_001, 430_002],
            "pull_out": [975_001 + i for i in range(4)], "swerve": [976_001 + i for i in range(4)],
            "side_merge": [977_001 + i for i in range(4)]}
    big = ("original", "overtake", "start_stop", "start_crowd")
    tasks = [(a, cls, seed, 256 if cls in big else 128) for cls, seeds in sets.items() for seed in seeds for a in arms]
    with Pool(workers) as pool:
        res = pool.map(run, tasks, chunksize=1)
    store = {}
    for (a, cls, seed), out in sorted(res, key=lambda r: r[0][2]):
        d = store.setdefault((a, cls), {})
        for k, v in out.items():
            d.setdefault(k, []).append(v)
    result = {}
    for cls in sets:
        print("")
        print(f"{cls}:")
        for a in arms:
            m = {k: np.concatenate(v) for k, v in store[(a, cls)].items()}
            km = m["dist"].sum() / 1000
            inc = (m["collision"] + m["red"] + m["stopsign"] + m["offroad"] + m["near"]).sum()
            r = {"episodes": len(m["dist"]), "km": float(km), "collisions": int(m["collision"].sum()), "red": int(m["red"].sum()),
                 "stopsign": int(m["stopsign"].sum()), "offroad": int(m["offroad"].sum()), "near_s": float(m["near"].sum()),
                 "inc100": float(inc / max(km, 1e-9) * 100), "m_per_ep": float(m["dist"].mean())}
            if "hz_coll" in m:
                r["hard_collision_rate"] = float(m["hz_coll"].mean())
                r["hard_mean_min_gap"] = float(np.nanmean(m["hz_gap"]))
            result.setdefault(cls, {})[a] = r
            extra = (f"  collision episodes {100 * r['hard_collision_rate']:.1f}%, mean min gap {r['hard_mean_min_gap']:.2f} m"
                     if "hard_collision_rate" in r else "")
            print(f"  {a:<15} {r['episodes']:5d} eps {r['km']:7.1f} km  coll {r['collisions']:4d} red {r['red']:3d} stop {r['stopsign']:2d} "
                  f"off {r['offroad']:2d} near {r['near_s']:5.1f}s  inc/100km {r['inc100']:6.2f}  m/ep {r['m_per_ep']:6.1f}{extra}")
    with open(os.path.join(HERE, "a3_extra_results.json"), "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
