"""Markdown tables for README.md from a2_heldout_results.json (eval2.py --out) and a2_hard_results.json (a2_hard.py)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"drive": "Drive (rules_v3)", "old": "old NN (shipped, attempt 1)", "old_unfiltered": "old NN, old observation",
         "new": "**new NN (attempt 2)**", "noshield": "new NN without the shield", "control": "Drive speed + hand-written pass rule + shield",
         "shipped": "shipped NN (attempt 1, capped at the limit)", "a2": "attempt-2 NN (frozen pipeline)",
         "shipped_c": "shipped NN as in the game (limit cap + caution)", "shipped_inlane": "**shipped NN + in-lane check (the safety-fix arm)**",
         "a4": "**attempt-4 NN**", "a4_noshield": "attempt-4 NN without the shield"}
if len(sys.argv) > 2 and sys.argv[2] == "a3":
    NAMES.update({"new": "**attempt-3 NN**", "noshield": "attempt-3 NN without the shield",
                  "control": "Drive speed + hand-written pass rule + attempt-3 shield"})


def heldout(path):
    r = json.load(open(path))
    out = []
    for cls in ("original", "overtake", "start_stop", "start_crowd"):
        if cls not in r:
            continue
        rows = r[cls]
        out.append(f"\n**{cls}** ({rows['drive']['episodes']} episodes per arm, paired)\n")
        out.append("| arm | km | coll | red | stop | off | near s | incidents/100 km | m/episode | mean m/s | standing | first 20 s m/s | free-flow m/s | out of lane s/ep |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm, s in rows.items():
            out.append(f"| {NAMES.get(arm, arm)} | {s['km']:.0f} | {s['collisions']} | {s['red']} | {s['stopsign']} | {s['offroad']} | {s['near_s']:.1f} | {s['inc100']:.2f} | "
                       f"{s['m_per_ep']:.0f} | {s['speed']:.2f} | {100 * s['stopped_share']:.1f}% | {s['first20s_speed']:.2f} | {s['free_flow_speed']:.2f} | {s['out_of_lane_s']:.2f} |")
        out.append("")
        out.append("| arm vs Drive | incidents/100 km (arm vs Drive) | difference, 95% CI | ratio, 95% CI | m/episode change, 95% CI | rule (i) | rule (ii) |")
        out.append("|---|---|---|---|---|---|---|")
        for arm, s in rows.items():
            v = s.get("vs_drive")
            if not v:
                continue
            p = v["point"]
            ratio = f"{p['ratio']:.2f} ({v['ratio_ci'][0]:.2f} to {v['ratio_ci'][1]:.2f})" if p["inc_b"] > 0 else "n/a"
            prim = cls in ("original", "overtake")
            out.append(f"| {NAMES.get(arm, arm)} | {p['inc_a']:.2f} vs {p['inc_b']:.2f} | {p['inc_a'] - p['inc_b']:+.2f} ({v['diff_ci'][0]:+.2f} to {v['diff_ci'][1]:+.2f}) | {ratio} | "
                       f"{100 * p['rel']:+.1f}% ({100 * v['rel_ci'][0]:+.1f} to {100 * v['rel_ci'][1]:+.1f}) | {('MET' if v['rule_i'] else 'not met') if prim else '-'} | {('MET' if v['rule_ii'] else 'not met') if prim else '-'} |")
    return "\n".join(out)


def hard(path):
    r = json.load(open(path))
    out = []
    for cls, rows in r.items():
        out.append(f"\n**{cls}** ({rows['drive']['n']} episodes per brain, the same cars for every brain)\n")
        out.append("| brain | actor started | collisions | rate (exact 95% CI) | mean min gap m | min TTC median s | min TTC 5th pct s | hard braking (>4 m/s2) | s/ep |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for b, s in rows.items():
            out.append(f"| {b} | {s['started']} | {s['collisions']} | {100 * s['rate']:.1f}% ({100 * s['ci'][0]:.1f} to {100 * s['ci'][1]:.1f}) | {s['mean_min_gap']:.2f} | "
                       f"{s['ttc_median']:.2f} | {s['ttc_p5']:.2f} | {100 * s['hard_brake_rate']:.1f}% | {s['hard_brake_s']:.2f} |")
        for b in ("shipped", "new"):
            v = rows[b].get("vs_drive")
            if v:
                c, g, h = v["collision_rate"], v["min_gap"], v["hard_brake_rate"]
                out.append(f"\n{b} minus Drive (paired): collision rate {100 * c[0]:+.1f} points (95% CI {100 * c[1]:+.1f} to {100 * c[2]:+.1f}); "
                           f"mean min gap {g[0]:+.2f} m ({g[1]:+.2f} to {g[2]:+.2f}); hard-braking rate {100 * h[0]:+.1f} points ({100 * h[1]:+.1f} to {100 * h[2]:+.1f})")
    return "\n".join(out)


def hard3(path):
    r = json.load(open(path))
    out = []
    for cls, rows in r.items():
        ideal = rows["ideal"]
        out.append("")
        out.append(f"**{cls}** ({ideal['n']} episodes per brain, the same cars for every brain). An ideal driver avoids "
                   f"**{100 * ideal['avoid_brake']:.1f}%** by full braking at the first moment the actor moves, and "
                   f"**{100 * ideal['avoid_swerve']:.1f}%** by braking plus the best swerve within the road edges.")
        out.append("")
        out.append("| brain | collisions (rate, exact 95% CI) | on brake-avoidable episodes | on swerve-avoidable episodes | mean min gap m | min TTC median / 5th pct s | hard braking |")
        out.append("|---|---|---|---|---|---|---|")
        for b, s_ in rows.items():
            if b == "ideal":
                continue
            kb, nb, cib = s_["on_brake_avoidable"]
            ks, ns, cis = s_["on_swerve_avoidable"]
            fb = f"{kb}/{nb} ({100 * kb / max(nb, 1):.1f}%, {100 * cib[0]:.1f} to {100 * cib[1]:.1f})" if nb else "-"
            fs = f"{ks}/{ns} ({100 * ks / max(ns, 1):.1f}%, {100 * cis[0]:.1f} to {100 * cis[1]:.1f})" if ns else "-"
            out.append(f"| {b} | {s_['collisions']} ({100 * s_['rate']:.1f}%, {100 * s_['ci'][0]:.1f} to {100 * s_['ci'][1]:.1f}) | {fb} | {fs} | "
                       f"{s_['mean_min_gap']:.2f} | {s_['ttc_median']:.2f} / {s_['ttc_p5']:.2f} | {100 * s_['hard_brake_rate']:.1f}% |")
        for b in ("drive_caution", "shipped", "a2", "a3"):
            v = rows[b].get("vs_drive")
            if v:
                c, g = v["collision_rate"], v["min_gap"]
                out.append("")
                out.append(f"{b} minus Drive (paired): collision rate {100 * c[0]:+.1f} points (95% CI {100 * c[1]:+.1f} to {100 * c[2]:+.1f}); "
                           f"mean min gap {g[0]:+.2f} m ({g[1]:+.2f} to {g[2]:+.2f})")
    return chr(10).join(out)


if __name__ == "__main__":
    what = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "a2"
    if what == "heldout":
        print(heldout(os.path.join(HERE, f"{tag}_heldout_results.json")))
    elif tag == "a3":
        print(hard3(os.path.join(HERE, "a3_hard_results.json")))
    else:
        print(hard(os.path.join(HERE, f"{tag}_hard_results.json")))
