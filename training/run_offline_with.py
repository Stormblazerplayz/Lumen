"""Runs testenv/nn_offline_test.py (the real nn.lua under LuaJIT) with another weights file, without touching the test.
usage: py -3.12 run_offline_with.py training/staged/nn_weights.json    (LUPA_PATH as for the test)"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "testenv"))
WPATH = os.path.abspath(sys.argv[1])
import lua_harness  # noqa: E402

_orig = lua_harness.Harness.__init__


def _init(self, file=None, module="lumen/nn", weights=None):
    _orig(self, file, module, weights or WPATH)


lua_harness.Harness.__init__ = _init
import nn_offline_test  # noqa: E402

print("weights:", WPATH)
sys.exit(nn_offline_test.main())
