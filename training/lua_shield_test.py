"""Checks staged/shield.lua (LuaJIT, through lupa) against the Python shield on the vectors from gen_shield_vectors.py.
usage: set LUPA_PATH=<folder containing lupa>; py -3.12 lua_shield_test.py"""
import json
import os
import sys

for extra in (os.environ.get("LUPA_PATH"),):
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
from lupa import luajit21 as lupa  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
allv = json.load(open(os.path.join(HERE, "staged", "shield_vectors.json")))
vectors = allv["lateral"]
lua = lupa.LuaRuntime(unpack_returned_tuples=True)
src = open(os.path.join(HERE, "staged", "shield.lua"), encoding="utf-8").read()
shield = lua.execute("local f = ...; return load(f)()", src)
setup = lua.eval("""function(vec)
  local st = vec.st
  local agents = {}
  for i = 1, #vec.agents do agents[i] = vec.agents[i] end
  return st, agents
end""")
worst_v = worst_d = 0.0
bad = 0
modes_bad = 0
import time
t0 = time.time()
for n, vec in enumerate(vectors):
    # build Lua tables
    st = lua.table_from(vec["st"])
    agents = lua.table_from([lua.table_from(a) for a in vec["agents"]])
    caps = vec["caps"]
    offs = [c["offset"] for c in caps]

    def capFn(offset, caps=caps):
        # the recorded call whose offset is closest
        best = min(caps, key=lambda c: abs(c["offset"] - offset))
        assert abs(best["offset"] - offset) < 1e-6, (best["offset"], offset)
        return best["cap"], best["dstop"]
    v, d, mode = shield.plan(st, agents, capFn)
    ref = vec["out"]
    dv, dd = abs(v - ref["v"]), abs(d - ref["d"])
    worst_v, worst_d = max(worst_v, dv), max(worst_d, dd)
    if mode != ref["mode"]:
        modes_bad += 1
        if modes_bad <= 5:
            print("MODE differs on vector", n, "lua", mode, "python", ref["mode"], "v", v, ref["v"], "d", d, ref["d"])
    elif dv > 1e-6 or dd > 1e-6:
        bad += 1
        if bad <= 5:
            print("value differs on vector", n, "dv", dv, "dd", dd)
print(f"lateral: {len(vectors)} vectors in {time.time() - t0:.1f}s: mode mismatches {modes_bad}, value mismatches {bad}, largest |dv| {worst_v:.2e}, |dd| {worst_d:.2e}")
bad_in = 0
worst_in = 0.0
for n, vec in enumerate(allv["inlane"]):
    st = lua.table_from(vec["st"])
    agents = lua.table_from([lua.table_from(a) for a in vec["agents"]])
    v, mode = shield.inlane(st, agents, vec["dstop"])
    ref = vec["out"]
    worst_in = max(worst_in, abs(v - ref["v"]))
    if mode != ref["mode"] or abs(v - ref["v"]) > 1e-6:
        bad_in += 1
        if bad_in <= 5:
            print("IN-LANE differs on vector", n, "lua", v, mode, "python", ref)
print(f"in-lane: {len(allv['inlane'])} vectors: mismatches {bad_in}, largest |dv| {worst_in:.2e}")
rel_path = os.path.join(HERE, "staged", "shield_vectors_rel.json")
if os.path.exists(rel_path):
    rel = json.load(open(rel_path))
    for n, vec in enumerate(rel):
        st = lua.table_from(vec["st"])
        agents = lua.table_from([lua.table_from(a) for a in vec["agents"]])
        v, mode = shield.inlane(st, agents, vec["dstop"], True)
        ref = vec["out"]
        if mode != ref["mode"] or abs(v - ref["v"]) > 1e-6:
            bad_in += 1
            if bad_in <= 5:
                print("IN-LANE (aheadRel) differs on vector", n, "lua", v, mode, "python", ref)
    print(f"in-lane with aheadRel: {len(rel)} vectors checked (mismatches so far {bad_in})")
sys.exit(0 if modes_bad == 0 and bad == 0 and bad_in == 0 else 1)
