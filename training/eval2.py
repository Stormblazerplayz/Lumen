"""Attempt 2 evaluation: paired comparison of arms on identical scenarios, with bootstrap confidence intervals and the
pre-registered decision rule (see README.md, "Attempt 2: PRE-REGISTRATION").

  py eval2.py --split dev|heldout [--arms drive,old,old_unfiltered,new,noshield,control] [--theta a2_final.npz]
              [--classes original,overtake,start_stop,start_crowd] [--workers 5] [--out results.json]

Splits (scenario seeds; training used 60_000+ and 300_000+, selection 400_000+):
  dev     : original 400_001..400_004, overtake 450_001..450_004, start_stop 420_001..420_002, start_crowd 430_001..430_002
  heldout : original 900_001..900_016, overtake-heavy 950_001..950_016 (each 256 episodes, as pre-registered);
            start_stop 920_001..920_004 and start_crowd 930_001..930_004 (added diagnostics)
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import brains
import brains2
import rulepass
import sim
import train_es

HERE = os.path.dirname(os.path.abspath(__file__))
NET = brains.MLP()
EPISODES = 256
SPLITS = {
    "dev": {"original": [400_001 + i for i in range(4)], "overtake": [450_001 + i for i in range(4)],
            "start_stop": [420_001, 420_002], "start_crowd": [430_001, 430_002]},
    "heldout": {"original": [900_001 + i for i in range(16)], "overtake": [950_001 + i for i in range(16)],
                "start_stop": [920_001 + i for i in range(4)], "start_crowd": [930_001 + i for i in range(4)]},
    # attempt 3 (README "Attempt 3: PRE-REGISTRATION ADDENDUM"): seeds never used before
    "heldout3": {"original": [910_001 + i for i in range(16)], "overtake": [960_001 + i for i in range(16)],
                 "start_stop": [925_001 + i for i in range(4)], "start_crowd": [935_001 + i for i in range(4)]},
    # the safety-fix arm (README "Safety-fix arm: PRE-REGISTRATION"): a third seed set never used before
    "heldout4": {"original": [940_001 + i for i in range(16)], "overtake": [990_001 + i for i in range(16)]},
    # attempt 4 (README "Attempt 4: PRE-REGISTRATION ADDENDUM"): the fourth seed set
    "heldout5": {"original": [945_001 + i for i in range(16)], "overtake": [995_001 + i for i in range(16)]},
}
METRICS = ("dist", "time", "collision", "near", "red", "stopsign", "offroad", "jerk2", "latmax", "blocked", "dist20", "wronglane")


def scenarios(cls, seed, n):
    if cls == "original":
        return sim.Scenarios(n, seed)
    if cls == "overtake":
        return sim.Scenarios(n, seed, overtake=True)
    return sim.Scenarios(n, seed, mode=cls)


# ---------------------------------------------------------------------------------------------
# the arms: every one is a function world -> (target speed, offset)
# ---------------------------------------------------------------------------------------------
def make_policy(arm, theta_path):
    if arm == "drive":
        return brains.rules_v3
    if arm in ("old", "old_unfiltered"):
        theta = np.load(os.path.join(HERE, "lumen_nn_final.npz"))["theta"]

        def pol(w):
            out = NET.forward(theta[None, :], w.observe()[None, :, :])[0]
            return brains.nn_actions(out, w)
        return pol
    if arm == "shipped":
        # the network in the game since 2026-11-01 13:10: attempt 1's weights and guards, the filtered observation, and
        # the target capped at the posted limit (nn.lua: min(share * 1.1 * vlim, vlim, V_MAX))
        theta = np.load(os.path.join(HERE, "lumen_nn_final.npz"))["theta"]

        def pol(w):
            out = NET.forward(theta[None, :], w.observe()[None, :, :])[0]
            vT, dT = brains.nn_actions(out, w)
            return np.minimum(vT, w.vlim_at(w.es)), dT
        return pol
    if arm in ("a4", "a4_noshield"):
        import brains4
        theta = np.load(theta_path)["theta"]
        use = arm == "a4"

        def pol(w):
            out = brains4.Net5.forward(theta[None, :], w.observe().astype(np.float64)[None, :, :])[0]
            return brains4.nn_actions4(out, w, use_shield=use)
        return pol
    if arm in ("shipped_c", "shipped_inlane"):
        import a3_extra          # frozen by hash in the pre-registration
        return a3_extra.policy("shipped" if arm == "shipped_c" else "shipped_inlane")
    if arm == "a2":
        import brains2_a2   # the attempt-2 pipeline exactly as it was run (shield_a2.py)
        theta = np.load(os.path.join(HERE, "a2_final.npz"))["theta"]

        def pol(w):
            out = NET.forward(theta[None, :], w.observe()[None, :, :])[0]
            return brains2_a2.nn_actions2(out, w)
        return pol
    if arm in ("new", "noshield"):
        theta = np.load(theta_path)["theta"]
        use = arm == "new"

        def pol(w):
            out = NET.forward(theta[None, :], w.observe()[None, :, :])[0]
            return brains2.nn_actions2(out, w, use_shield=use)
        return pol
    if arm == "control":
        # Drive's speed choice (95% of the limit) and the hand-written pass rule, under the same guards and shield
        def pol(w):
            out = rulepass.rule_out(w, free_logit=float(np.log(0.95 / 1.1 / (1 - 0.95 / 1.1))))
            return brains2.nn_actions2(out, w)
        return pol
    if arm == "inlane":
        def pol(w):
            out = np.zeros((w.E, 2))
            out[:, 0] = 6.0
            return brains2.nn_actions2(out, w)
        return pol
    raise ValueError(arm)


def run_task(task):
    arm, cls, seed, n, theta_path = task
    train_es.lower_priority()
    sim.OBS_FILTER = arm != "old_unfiltered"
    sc = scenarios(cls, seed, n)
    w = sim.World(sc)
    pol = make_policy(arm, theta_path)
    modes = np.zeros(8)
    out_steps = np.zeros(n)          # steps spent with an executed offset beyond the lane band
    out_abs = np.zeros(n)
    free_v = np.zeros(n)             # speed in free flow: nothing within 80 m ahead in the lane, no line within 120 m
    free_t = np.zeros(n)
    for _ in range(sim.STEPS):
        brains2.LAST["mode"] = np.zeros(8, dtype=int)
        vT, dT = pol(w)
        alive = ~w.done
        modes += brains2.LAST["mode"]
        lat = alive & (np.abs(dT) > 0.45)
        out_steps += lat
        out_abs += np.where(lat, np.abs(dT), 0.0)
        rs = w.agS - w.es[:, None]
        ahead = ((sc.kind >= 0) & (sc.kind != sim.CROSS) & (rs > 0) & (rs < 80) & (np.abs(w.agD) < 2.0)).any(axis=1)
        line = sc.jS - 6.0 - (w.es[:, None] + 2.6)
        near_line = ((line > -2) & (line < 120) & (sc.jKind != sim.J_NONE)).any(axis=1)
        free = alive & ~ahead & ~near_line
        free_v += np.where(free, w.ev, 0.0)
        free_t += free
        w.prevT, w.prevD = vT, dT
        w.step(vT, dT)
        if w.done.all():
            break
    m = w.summary()
    out = {k: m[k] for k in METRICS}
    out["out_steps"], out["out_abs"], out["free_v"], out["free_t"] = out_steps, out_abs, free_v, free_t
    out["modes"] = modes[None, :].repeat(n, axis=0) / n
    out["passedParked"] = w.passedParked.astype(float)
    out["passedLead"] = w.passedLead.astype(float)
    out["hasParked"] = (sc.kind == sim.PARKED).any(axis=1).astype(float)
    out["hasLead"] = (sc.kind == sim.LEAD).any(axis=1).astype(float)
    return task[:3], out


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------
def incidents(m):
    return m["collision"] + m["red"] + m["stopsign"] + m["offroad"] + m["near"]


def boot_compare(a, b, B=10000, seed=1):
    """a, b: per-episode metric dicts of two arms on the same episodes (b = reference, Drive). Bootstrap over episodes."""
    N = len(a["dist"])
    rng = np.random.default_rng(seed)
    ia, ib = incidents(a), incidents(b)
    da, db = a["dist"], b["dist"]
    inc_a, inc_b = np.empty(B), np.empty(B)
    mep_a, mep_b = np.empty(B), np.empty(B)
    chunk = 500
    for s in range(0, B, chunk):
        idx = rng.integers(0, N, (min(chunk, B - s), N))
        sa, sb = ia[idx].sum(1), ib[idx].sum(1)
        ka, kb = da[idx].sum(1) / 1000, db[idx].sum(1) / 1000
        inc_a[s:s + len(idx)] = sa / np.maximum(ka, 1e-9) * 100
        inc_b[s:s + len(idx)] = sb / np.maximum(kb, 1e-9) * 100
        mep_a[s:s + len(idx)] = da[idx].mean(1)
        mep_b[s:s + len(idx)] = db[idx].mean(1)
    diff = inc_a - inc_b
    ratio = inc_a / np.maximum(inc_b, 1e-9)
    rel = mep_a / mep_b - 1
    pt = {"inc_a": ia.sum() / (da.sum() / 1000) * 100, "inc_b": ib.sum() / (db.sum() / 1000) * 100,
          "mep_a": da.mean(), "mep_b": db.mean()}
    pt["rel"] = pt["mep_a"] / pt["mep_b"] - 1
    pt["ratio"] = pt["inc_a"] / max(pt["inc_b"], 1e-9)
    ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    res = {"point": pt, "diff_ci": ci(diff), "ratio_ci": ci(ratio), "rel_ci": ci(rel), "diff_upper95": float(np.percentile(diff, 97.5))}
    # the pre-registered rule
    i_ok = res["diff_upper95"] <= 0 and pt["rel"] >= 0.08 and res["rel_ci"][0] > 0
    ii_ok = pt["inc_a"] <= 0.5 * pt["inc_b"] and res["ratio_ci"][1] < 1 and res["rel_ci"][0] >= 0
    res["rule_i"], res["rule_ii"], res["goal_met"] = bool(i_ok), bool(ii_ok), bool(i_ok or ii_ok)
    return res


def summarise(m, N):
    km = m["dist"].sum() / 1000
    return {"episodes": N, "km": float(km), "collisions": int(m["collision"].sum()), "red": int(m["red"].sum()),
            "stopsign": int(m["stopsign"].sum()), "offroad": int(m["offroad"].sum()), "near_s": float(m["near"].sum()),
            "inc100": float(incidents(m).sum() / max(km, 1e-9) * 100), "m_per_ep": float(m["dist"].mean()),
            "speed": float(m["dist"].sum() / max(1.0, m["time"].sum())), "stopped_share": float(m["blocked"].sum() / max(1.0, m["time"].sum())),
            "jerk": float(m["jerk2"].sum() / max(1.0, m["time"].sum())), "latg95": float(np.percentile(m["latmax"], 95)),
            "first20s_speed": float(m["dist20"].mean() / 20.0), "wronglane_s": float(m["wronglane"].mean()),
            "passed_parked": float(m["passedParked"].sum() / max(1.0, m["hasParked"].sum())),
            "passed_lead": float(m["passedLead"].sum() / max(1.0, m["hasLead"].sum())),
            "out_of_lane_s": float(m["out_steps"].mean() * sim.DT), "mean_abs_offset_out": float(m["out_abs"].sum() / max(1.0, m["out_steps"].sum())),
            "free_flow_speed": float(m["free_v"].sum() / max(1.0, m["free_t"].sum())),
            "shield_modes": [float(x) for x in m["modes"].sum(axis=0)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--arms", default="drive,old,new,noshield,control")
    ap.add_argument("--theta", default=os.path.join(HERE, "a2_final.npz"))
    ap.add_argument("--classes", default="original,overtake,start_stop,start_crowd")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    args = ap.parse_args()
    arms = args.arms.split(",")
    classes = args.classes.split(",")
    tasks = [(arm, cls, seed, args.episodes, args.theta) for cls in classes for seed in SPLITS[args.split][cls] for arm in arms]
    t0 = time.time()
    with Pool(args.workers) as pool:
        res = pool.map(run_task, tasks, chunksize=1)
    print(f"{len(tasks)} runs in {time.time() - t0:.0f}s")
    store = {}
    for (arm, cls, seed), out in res:
        d = store.setdefault((arm, cls), {})
        for k, v in out.items():
            d.setdefault(k, []).append(v)
    result = {}
    for cls in classes:
        print(f"\n=== {cls} ({args.split}) ===")
        print(f"{'arm':<15} {'eps':>5} {'km':>7} {'coll':>4} {'red':>3} {'stop':>4} {'off':>3} {'near_s':>6} {'inc/100km':>9} {'m/ep':>6} {'m/s':>5} {'stand%':>6} {'20s m/s':>7} {'wrong s':>7} {'pass P':>6} {'pass L':>6}")
        merged = {}
        for arm in arms:
            m = {k: np.concatenate(v) for k, v in store[(arm, cls)].items()}
            merged[arm] = m
            s = summarise(m, len(m["dist"]))
            result.setdefault(cls, {})[arm] = s
            print(f"{arm:<15} {s['episodes']:5d} {s['km']:7.1f} {s['collisions']:4d} {s['red']:3d} {s['stopsign']:4d} {s['offroad']:3d} {s['near_s']:6.1f} {s['inc100']:9.2f} "
                  f"{s['m_per_ep']:6.0f} {s['speed']:5.2f} {100 * s['stopped_share']:6.1f} {s['first20s_speed']:7.2f} {s['wronglane_s']:7.2f} {100 * s['passed_parked']:5.0f}% {100 * s['passed_lead']:5.0f}%"
                  f"   free-flow {s['free_flow_speed']:5.2f} m/s, out of lane {s['out_of_lane_s']:5.2f} s/ep at |offset| {s['mean_abs_offset_out']:.2f} m, shield modes/ep {[round(x, 1) for x in s['shield_modes']]}")
        if "drive" in arms:
            for arm in arms:
                if arm == "drive":
                    continue
                r = boot_compare(merged[arm], merged["drive"])
                result[cls][arm]["vs_drive"] = r
                p = r["point"]
                tag = f"  RULE: (i) {'MET' if r['rule_i'] else 'no'}  (ii) {'MET' if r['rule_ii'] else 'no'}" if cls in ("original", "overtake") else ""
                rtxt = f"ratio CI {r['ratio_ci'][0]:.2f}..{r['ratio_ci'][1]:.2f}" if p['inc_b'] > 0 else "ratio n/a"
                print(f"  {arm} vs Drive: incidents {p['inc_a']:.2f} vs {p['inc_b']:.2f} per 100 km (diff 95% CI {r['diff_ci'][0]:+.2f}..{r['diff_ci'][1]:+.2f}; {rtxt}); "
                      f"m/episode {100 * p['rel']:+.1f}% (95% CI {100 * r['rel_ci'][0]:+.1f}..{100 * r['rel_ci'][1]:+.1f}){tag}")
                if cls in ("start_stop", "start_crowd"):
                    d20 = merged[arm]["dist20"].mean() / max(merged["drive"]["dist20"].mean(), 1e-9)
                    print(f"      first-20-s progress ratio vs Drive: {d20:.2f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1, default=float)


if __name__ == "__main__":
    main()
