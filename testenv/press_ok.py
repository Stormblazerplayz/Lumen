"""Presses OK on the test BeamNG instance's native dialog (0.40's "Third-party library warning",
OK/Cancel, class #32770) by posting WM_COMMAND IDOK. Only touches windows owned by a BeamNG process
whose command line has the test profile; never Cancel (that saves a stop-warning preference).
usage: py press_ok.py [max_seconds]"""
import ctypes
import json
import subprocess
import sys
import time
from ctypes import wintypes

PROFILE = "beamngtest"
user32 = ctypes.WinDLL("user32", use_last_error=True)
WM_COMMAND, IDOK = 0x0111, 1
EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def test_pids():
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'BeamNG*' } | "
          "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.strip()
    if not out:
        return set()
    data = json.loads(out)
    data = data if isinstance(data, list) else [data]
    return {d["ProcessId"] for d in data if PROFILE in (d.get("CommandLine") or "").lower()}


def dialogs(pids):
    found = []

    def cb(hwnd, _):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids and user32.IsWindowVisible(hwnd):
            cls = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(hwnd, cls, 64)
            if cls.value == "#32770":
                title = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, title, 256)
                found.append((hwnd, title.value))
        return True

    user32.EnumWindows(EnumWindowsProc(cb), 0)
    return found


def main():
    limit = float(sys.argv[1]) if len(sys.argv) > 1 else 120
    t0 = time.time()
    while time.time() - t0 < limit:
        pids = test_pids()
        for hwnd, title in dialogs(pids):
            user32.PostMessageW(hwnd, WM_COMMAND, IDOK, 0)
            print(f"pressed OK on '{title}' after {time.time() - t0:.0f} s")
            return 0
        time.sleep(2)
    print("no dialog appeared")
    return 1


if __name__ == "__main__":
    sys.exit(main())
