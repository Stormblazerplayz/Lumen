"""Installs the staged Lumen build into Storm's mods folder as soon as BeamNG has closed.

Replaces out/release/install_when_closed.py. 2026-11-01, Storm: "THESE TERMINALS KEEP POPPING UP MY
SCREEN": that script was started detached (no console) and polled with a child `powershell` every 10 s,
and Windows gives a console child of a console-less parent its own visible window. Three copies were
running (one per ship while the game was open). This one:
  * is started with pythonw.exe and CREATE_NO_WINDOW (tools/ship.py does that),
  * polls through the Win32 process snapshot (ctypes), so it never starts a child process,
  * allows one instance only (a named mutex),
  * installs with an atomic rename, logs to out/release/watcher.log, and gives up after 4 hours.
Only ever copies out/release/lumen_robotaxi.zip over mods/lumen_robotaxi.zip.
"""
import ctypes
import os
import shutil
import sys
import time
from ctypes import wintypes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "out", "release", "lumen_robotaxi.zip")
DST = os.path.join(os.environ["LOCALAPPDATA"], "BeamNG", "BeamNG.drive", "current", "mods", "lumen_robotaxi.zip")
LOG = os.path.join(ROOT, "out", "release", "watcher.log")

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
TH32CS_SNAPPROCESS = 0x2
ERROR_ALREADY_EXISTS = 183


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
k32.CloseHandle.argtypes = [wintypes.HANDLE]


def beamng_running():
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or snap == wintypes.HANDLE(-1).value:
        return True                          # cannot tell: do nothing
    try:
        e = PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            if e.szExeFile.lower().startswith("beamng.drive"):
                return True
            ok = k32.Process32NextW(snap, ctypes.byref(e))
        return False
    finally:
        k32.CloseHandle(snap)


def log(text):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + text + "\n")
    except OSError:
        pass


def main():
    mutex = k32.CreateMutexW(None, False, "Local\\LumenInstallWatcher")
    if k32.GetLastError() == ERROR_ALREADY_EXISTS:
        return 0                             # another watcher is already waiting
    log("watcher started, waiting for BeamNG to close")
    deadline = time.time() + 4 * 3600
    while time.time() < deadline:
        if not beamng_running():
            time.sleep(5)                    # let the game finish writing and release the zip
            if not beamng_running():
                try:
                    shutil.copyfile(SRC, DST + ".part")
                    os.replace(DST + ".part", DST)
                    log("installed the staged build")
                except OSError as e:
                    log("install failed: %s" % e)
                return 0
        time.sleep(5)
    log("gave up waiting for BeamNG to close")
    return 1


if __name__ == "__main__":
    sys.exit(main())
