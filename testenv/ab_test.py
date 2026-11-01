"""In-game A/B of Lumen NN against Lumen Drive on the test instance (Storm's /goal, 2026-11-01: "Make the NN
actually better than Drive"). Half the fleet drives on each brain, the halves swap halfway (so a slot's luck with
its roads cancels out), and drive.lua counts both brains the same way (robotaxi.abTest / drive.abReport).

usage: py -u ab_test.py [minutes_per_half=10] [label=run] [--traffic N] [--parked N]

PRE-REGISTERED (written before any in-game run, 2026-11-01):
  primary   incidents per 100 km driven = collisions + red-light runs + stop-sign runs + rollovers, per brain,
            pooled over both halves. "NN better" means the upper end of the exact 95% interval of the rate ratio
            NN / Drive is below 1.0.
  guards    (a) NN mean speed while driving >= 90% of Drive's (a timid car that never moves has no incidents);
            (b) NN blocked recoveries per hour not above Drive's by more than its own 95% interval allows;
            (c) the NN watchdog's relief share is reported: time the rules drove an NN car because it held back.
  The result is reported as measured, including "not distinguishable" when the intervals overlap 1.0.
"""
import json
import math
import os
import sys
import time

import mcp

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "game"))
KEYS = ("km", "hours", "hits", "red_run", "stopsign_run", "flipped", "stuck", "damaged", "blocked", "halt_person",
        "halt_obstacle", "incidents", "relief_s")


# ---------------------------------------------------------------------------------------------------------------
# exact intervals (no scipy): Poisson (Garwood) for one rate, conditional binomial (Clopper-Pearson) for a ratio
# ---------------------------------------------------------------------------------------------------------------
def pois_cdf(k, mu):
    if k < 0:
        return 0.0
    term = math.exp(-mu)
    s = term
    for i in range(1, k + 1):
        term *= mu / i
        s += term
    return min(1.0, s)


def bisect(f, lo, hi, it=200):
    """root of an increasing f on [lo, hi]"""
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def garwood(k, alpha=0.05):
    lo = 0.0 if k == 0 else bisect(lambda mu: (1 - pois_cdf(k - 1, mu)) - alpha / 2, 0.0, 10 * k + 50)
    hi = bisect(lambda mu: alpha / 2 - pois_cdf(k, mu), 0.0, 10 * k + 50)
    return lo, hi


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def rate_ratio(k1, e1, k2, e2, alpha=0.05):
    """ratio of rates k1/e1 over k2/e2 with the exact conditional interval; None when there are no events at all"""
    n = k1 + k2
    if n == 0 or e1 <= 0 or e2 <= 0:
        return None
    pl = 0.0 if k1 == 0 else bisect(lambda p: (1 - binom_cdf(k1 - 1, n, p)) - alpha / 2, 0.0, 1.0)
    pu = 1.0 if k1 == n else bisect(lambda p: alpha / 2 - binom_cdf(k1, n, p), 0.0, 1.0)
    conv = e2 / e1
    est = (k1 / e1) / (k2 / e2) if k2 > 0 else math.inf
    lo = pl / (1 - pl) * conv if pl < 1 else math.inf
    hi = pu / (1 - pu) * conv if pu < 1 else math.inf
    return est, lo, hi


# ---------------------------------------------------------------------------------------------------------------
# the game
# ---------------------------------------------------------------------------------------------------------------
def lua_json(code):
    return json.loads(mcp.lua(code))


def fleet_ready(minimum=16, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            f = lua_json("return jsonEncode(extensions.lumen_robotaxi.debugFleet())")
        except (ValueError, mcp.McpError):
            f = []
        follow = sum(1 for c in f if c.get("mode") == "follow")
        if follow >= minimum:
            return len(f), follow
        time.sleep(5)
    return None


def spawn_world(traffic, parked):
    if traffic:
        print("traffic:", mcp.lua(f"""
if not gameplay_traffic then extensions.load('gameplay_traffic') end
local ok, err = pcall(gameplay_traffic.setupTraffic, {traffic}, 0)
return tostring(ok) .. ' ' .. tostring(err)"""))
    if parked:
        print("parked:", mcp.lua(f"""
if not gameplay_parking then extensions.load('gameplay_parking') end
local ok, err = pcall(gameplay_parking.setupVehicles, {parked})
return tostring(ok) .. ' ' .. tostring(err)"""))


def run_half(flip, minutes):
    ok = mcp.lua(f"return tostring(extensions.lumen_robotaxi.abTest(true, {str(bool(flip)).lower()}))")
    if ok != "true":
        raise SystemExit("abTest refused to start (is the NN loaded?): " + ok)
    t0 = time.time()
    last = None
    while time.time() - t0 < minutes * 60:
        time.sleep(30)
        try:
            last = lua_json("return jsonEncode(extensions.lumen_robotaxi.abStats() or {})")
        except (ValueError, mcp.McpError) as e:
            print("  stats read failed:", e)
            continue
        line = []
        for b in ("nn", "drive"):
            s = last.get(b) or {}
            line.append(f"{b}: {s.get('km', 0):6.2f} km {s.get('incidents', 0)} inc {s.get('blocked', 0)} blk "
                        f"v={s.get('mean_speed', 0):4.1f}")
        print(f"  {(time.time() - t0) / 60:5.1f} min  " + " | ".join(line))
    last = lua_json("return jsonEncode(extensions.lumen_robotaxi.abStats() or {})")
    return last


def pool(halves):
    out = {}
    for b in ("nn", "drive"):
        tot = {k: 0.0 for k in KEYS}
        for h in halves:
            s = h.get(b) or {}
            for k in KEYS:
                tot[k] += float(s.get(k, 0) or 0)
        out[b] = tot
    return out


def report(halves, label, minutes):
    P = pool(halves)
    nn, dr = P["nn"], P["drive"]
    res = {"label": label, "minutes_per_half": minutes, "halves": halves, "pooled": P}
    lines = []
    for b, s in (("NN", nn), ("Drive", dr)):
        k, km = int(s["incidents"]), s["km"]
        lo, hi = garwood(k)
        rate = k / km * 100 if km > 0 else float("nan")
        speed = s["km"] * 1000 / (s["hours"] * 3600) if s["hours"] > 0 else 0
        res[b + "_rate"] = {"incidents": k, "km": km, "per100km": rate, "ci95": [lo / km * 100 if km else None, hi / km * 100 if km else None],
                            "mean_speed": speed, "blocked_per_h": s["blocked"] / s["hours"] if s["hours"] else None,
                            "relief_share": s["relief_s"] / (s["hours"] * 3600) if s["hours"] else 0}
        lines.append(f"{b:5}: {km:7.2f} km, {s['hours']:.2f} h, mean {speed:4.1f} m/s | incidents {k} "
                     f"(hits {int(s['hits'])}, red {int(s['red_run'])}, stop sign {int(s['stopsign_run'])}, flipped {int(s['flipped'])}) "
                     f"= {rate:.2f} per 100 km [95% {lo / km * 100 if km else 0:.2f}, {hi / km * 100 if km else 0:.2f}] | "
                     f"blocked {int(s['blocked'])} ({s['blocked'] / s['hours'] if s['hours'] else 0:.1f}/h) | "
                     f"halts person {int(s['halt_person'])} obstacle {int(s['halt_obstacle'])} | relief {s['relief_s']:.0f} s")
    rr = rate_ratio(int(nn["incidents"]), nn["km"], int(dr["incidents"]), dr["km"])
    res["rate_ratio"] = rr
    vn = res["NN_rate"]["mean_speed"]
    vd = res["Drive_rate"]["mean_speed"]
    speed_ok = vd > 0 and vn >= 0.9 * vd
    if rr is None:
        verdict = "no incidents for either brain: not distinguishable on the primary metric"
    elif rr[2] < 1.0 and speed_ok:
        verdict = f"NN BETTER: rate ratio {rr[0]:.2f} [95% {rr[1]:.2f}, {rr[2]:.2f}] and speed {vn / vd:.0%} of Drive's"
    elif rr[1] > 1.0:
        verdict = f"NN WORSE: rate ratio {rr[0]:.2f} [95% {rr[1]:.2f}, {rr[2]:.2f}]"
    else:
        verdict = f"not distinguishable: rate ratio {rr[0]:.2f} [95% {rr[1]:.2f}, {rr[2]:.2f}]" + ("" if speed_ok else f"; NN speed {vn / vd:.0%} of Drive's FAILS the 90% guard")
    if rr is not None and rr[2] < 1.0 and not speed_ok:
        verdict = f"NOT BETTER: fewer incidents but NN speed is {vn / vd:.0%} of Drive's (guard: 90%)"
    res["verdict"] = verdict
    print("\n".join(lines))
    print("VERDICT:", verdict)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"ab_{label}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("saved", path)
    return res


def main():
    args = [a for a in sys.argv[1:]]
    traffic = int(args[args.index("--traffic") + 1]) if "--traffic" in args else 0
    parked = int(args[args.index("--parked") + 1]) if "--parked" in args else 0
    pos = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    minutes = float(pos[0]) if pos else 10.0
    label = pos[1] if len(pos) > 1 else "run"
    if mcp.lua("return tostring(extensions.lumen_robotaxi ~= nil)") != "true":
        raise SystemExit("the Lumen mod is not loaded in the test instance")
    mcp.lua("local s = extensions.lumen_robotaxi.getState() if s and s.state ~= 'idle' then extensions.lumen_robotaxi.cancel() end return 1")
    spawn_world(traffic, parked)
    r = fleet_ready()
    print("fleet:", r)
    halves = []
    for flip in (False, True):
        print(f"half {'B (swapped)' if flip else 'A'}: odd slots on {'Drive' if flip else 'NN'}, {minutes} min")
        halves.append(run_half(flip, minutes))
    mcp.lua("extensions.lumen_robotaxi.abTest(false) return 1")
    report(halves, label, minutes)


if __name__ == "__main__":
    main()
