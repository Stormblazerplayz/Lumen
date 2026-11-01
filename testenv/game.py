"""Start/stop/status for the isolated BeamNG test instance.

    py game.py start [--owner NAME]   refuse when ANY other BeamNG runs (Storm's game, always: on 2026-11-01 a test
                          copy beside it replaced their textures with placeholders), when another agent holds the
                          lock, or when the PC is busy; otherwise launch and start the watchdog
    py game.py status     list BeamNG processes (test instance or not) and the lock
    py game.py stop [--owner NAME] [--force]
                          stop ONLY the test instance (its command line must contain the test profile path, plus
                          its child processes) and release the lock; never touches Storm's own game
    py game.py watch      the watchdog, started by `start`: it kills ONLY the test instance the moment a BeamNG
                          that is not part of it appears, i.e. Storm launched their game

One test instance at a time. The lock (testenv/test_copy.lock) names its owner; an agent that did not take it can
neither start nor stop the instance. Log of the watchdog: out/game/watchdog.log.

Why Python and not a .ps1: Norton's script protection removed the PowerShell launcher (2026-10-31) and the game it
started exited at once.
Storm's real profile (%LOCALAPPDATA%\\BeamNG\\BeamNG.drive) is never used: the game runs with -userpath pointing at
%LOCALAPPDATA%\\BeamNGTest\\robotaxi.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GAME_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive"
LAUNCHER = os.path.join(GAME_DIR, "BeamNG.drive.exe")
PROFILE = os.path.join(os.environ["LOCALAPPDATA"], "BeamNGTest", "robotaxi")
MCP_PORT = 29393
# optional: the Nimbus project's "is this PC busy" check (GPU load, competing work); without it the PC counts as idle
BUSY = os.path.join(os.environ.get("USERPROFILE", ""), "OneDrive", "Documents", "Projects", "Nimbus", "tools", "dev", "busy.ps1")
PID_FILE = os.path.join(HERE, "game.pid")
LOCK_FILE = os.path.join(HERE, "test_copy.lock")
WATCH_LOG = os.path.normpath(os.path.join(HERE, "..", "out", "game", "watchdog.log"))
CREATE_NO_WINDOW = 0x08000000
DETACHED = 0x00000008 | 0x00000200          # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
STALE_LOCK_S = 600                           # a lock with no running test instance older than this is stale


def write_profile(low=False):
    """Minimal settings: windowed, Storm's quality levels, MCP on, consent = disable.
    low=True (env LUMEN_LOWVRAM=1): the smallest video-memory footprint."""
    local = {
        "GraphicDisplayModes": "Window",
        "GraphicDisplayResolutions": "1600 900",
        "GraphicBorderless": False,
        "GraphicMeshQuality": "Ultra",
        "GraphicTextureQuality": "Normal",
        "GraphicLightingQuality": "Ultra",
        "GraphicShadowsQuality": "High",
        "enableMcp": True,
        "trafficAmount": int(os.environ.get("LUMEN_TRAFFIC", "0")),
        "trafficParkedAmount": int(os.environ.get("LUMEN_PARKED", "0")),
    }
    if low:
        local.update({"GraphicDisplayResolutions": "1360 768", "GraphicMeshQuality": "Low", "GraphicTextureQuality": "Lowest",
                      "GraphicLightingQuality": "Low", "GraphicShadowsQuality": "Low"})
    # 'disable' is the privacy-preserving answer; the consent wizard then has nothing
    # to ask, and nothing is accepted on Storm's behalf.
    cloud = {"onlineFeatures": "disable", "telemetry": "disable"}
    for root in (PROFILE, os.path.join(PROFILE, "current")):
        os.makedirs(os.path.join(root, "settings", "cloud"), exist_ok=True)
        os.makedirs(os.path.join(root, "mods", "unpacked"), exist_ok=True)
        with open(os.path.join(root, "settings", "settings.json"), "w", encoding="utf-8") as f:
            json.dump(local, f, indent=2)
        with open(os.path.join(root, "settings", "cloud", "settings.json"), "w", encoding="utf-8") as f:
            json.dump(cloud, f, indent=2)


def processes():
    """BeamNG processes with their command lines and parents (via CIM)."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'BeamNG*' } | "
          "Select-Object ProcessId, ParentProcessId, Name, CommandLine | ConvertTo-Json -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                         creationflags=CREATE_NO_WINDOW).stdout.strip()
    if not out:
        return []
    data = json.loads(out)
    return data if isinstance(data, list) else [data]


def is_test(proc):
    return PROFILE.lower() in (proc.get("CommandLine") or "").lower()


def tree_pids(procs):
    """pids of the test instance: every process whose command line carries the test profile, plus all descendants"""
    mine = {p["ProcessId"] for p in procs if is_test(p)}
    changed = True
    while changed:
        changed = False
        for p in procs:
            if p["ProcessId"] not in mine and p.get("ParentProcessId") in mine:
                mine.add(p["ProcessId"])
                changed = True
    return mine


def busy():
    if not os.path.exists(BUSY):
        return False
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", BUSY], capture_output=True,
                       text=True, creationflags=CREATE_NO_WINDOW)
    print(r.stdout.strip())
    return r.returncode != 0


def note(msg):
    os.makedirs(os.path.dirname(WATCH_LOG), exist_ok=True)
    with open(WATCH_LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def read_lock():
    try:
        with open(LOCK_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def lock_age_s(lock):
    try:
        return time.time() - time.mktime(time.strptime(lock["since"], "%Y-%m-%d %H:%M:%S"))
    except (KeyError, ValueError, TypeError):
        return 1e9


def drop_lock():
    try:
        os.remove(LOCK_FILE)
        return True
    except OSError:
        return False


def start(owner="parent"):
    running = processes()
    if any(is_test(p) for p in running):
        print("the test instance is already running:", [(p["ProcessId"], p["Name"]) for p in running if is_test(p)])
        return 1
    storm = [p for p in running if p["ProcessId"] not in tree_pids(running)]
    if storm:
        # Never beside Storm's game (their textures were replaced by placeholders on 2026-11-01), whatever they said
        # in passing: the rule is "only while their BeamNG is closed", and this is where it is enforced.
        print("REFUSED: BeamNG is running (pids %s) and it is not the test instance. The test copy only runs while Storm's game is closed." %
              [p["ProcessId"] for p in storm])
        return 1
    lock = read_lock()
    if lock and lock.get("owner") != owner and lock_age_s(lock) < STALE_LOCK_S:
        print("REFUSED: the test copy is reserved by %r since %s. Ask them to run `py game.py stop`, or wait." % (lock.get("owner"), lock.get("since")))
        return 1
    if busy():
        print("PC is busy; not starting the test instance.")
        return 1
    with open(LOCK_FILE, "w", encoding="utf-8") as f:
        json.dump({"owner": owner, "since": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    write_profile(low=os.environ.get("LUMEN_LOWVRAM") == "1")
    args = [LAUNCHER, "-nosteam", "-userpath", PROFILE, "-mcpport", str(MCP_PORT), "-notelemetry"]
    open(PID_FILE + ".starting", "w").close()          # keeps the watchdog from quitting while the game is still booting
    p = subprocess.Popen(args, cwd=GAME_DIR, creationflags=DETACHED, close_fds=True)
    with open(PID_FILE, "w") as f:
        f.write(str(p.pid))
    print("launcher started pid", p.pid, "owner", owner)
    # the watchdog: a separate, windowless python that outlives this command
    subprocess.Popen([sys.executable, os.path.abspath(__file__), "watch"], cwd=HERE, close_fds=True,
                     creationflags=CREATE_NO_WINDOW | DETACHED)
    time.sleep(20)
    try:
        os.remove(PID_FILE + ".starting")
    except OSError:
        pass
    return 0


def status():
    procs = processes()
    if not procs:
        print("no BeamNG process")
    mine = tree_pids(procs)
    for p in procs:
        print(p["ProcessId"], p["Name"], "TEST" if p["ProcessId"] in mine else "NOT-TEST (Storm's?)")
    lock = read_lock()
    print("lock:", lock if lock else "free")
    return 0


def stop(owner="parent", force=False):
    lock = read_lock()
    if lock and lock.get("owner") != owner and not force and lock_age_s(lock) < STALE_LOCK_S:
        print("REFUSED: the test copy belongs to %r. Use --force only when they are finished." % lock.get("owner"))
        return 1
    n = 0
    procs = processes()
    mine = tree_pids(procs)
    for p in procs:
        if p["ProcessId"] in mine:
            subprocess.run(["taskkill", "/PID", str(p["ProcessId"]), "/T", "/F"], capture_output=True, creationflags=CREATE_NO_WINDOW)
            print("stopped test instance pid", p["ProcessId"])
            n += 1
        else:
            print("left alone (not the test instance):", p["ProcessId"], p["Name"])
    if drop_lock():
        print("lock released")
    return 0 if n else 1


def watch(poll=2.0, max_hours=6.0):
    """Kill the test instance, and only it, when a BeamNG that is not part of it appears. Cheap polling with tasklist;
    the process tree is read (CIM) only when the set of BeamNG pids changes."""
    def pids():
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq BeamNG*", "/FO", "CSV", "/NH"], capture_output=True, text=True,
                             creationflags=CREATE_NO_WINDOW).stdout
        return {int(l.split(",")[1].strip('"')) for l in out.splitlines() if l.startswith('"BeamNG')}

    note("watchdog started, pid %d" % os.getpid())
    t0, seen, empty = time.time(), None, 0
    while time.time() - t0 < max_hours * 3600:
        cur = pids()
        if cur != seen:
            seen = cur
            procs = processes()
            mine = tree_pids(procs)
            intruders = [p for p in procs if p["ProcessId"] not in mine]
            if intruders and mine:
                note("a BeamNG that is not the test instance appeared (pids %s): stopping the test instance %s" %
                     ([p["ProcessId"] for p in intruders], sorted(mine)))
                for pid in sorted(mine):
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=CREATE_NO_WINDOW)
                drop_lock()
                note("test instance stopped; the other BeamNG was left alone")
                return 0
        empty = empty + 1 if not cur else 0
        if empty * poll > 20 and not os.path.exists(PID_FILE + ".starting"):
            note("no BeamNG running any more; watchdog done")
            return 0
        time.sleep(poll)
    note("watchdog timed out after %.0f h" % max_hours)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    owner = sys.argv[sys.argv.index("--owner") + 1] if "--owner" in sys.argv else "parent"
    sys.exit({"start": lambda: start(owner), "status": status, "stop": lambda: stop(owner, "--force" in sys.argv), "watch": watch}[cmd]())
