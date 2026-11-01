"""Ship the Lumen mod: check, build, verify, install. Parent session only (the forks never run this).

usage:  py tools/ship.py --check-only     parse every Lua file, load every json, list suspicious globals
        py tools/ship.py                  check + build (scripts only) + verify + install
        py tools/ship.py --vehicle        same, but rebuild the vehicle too (generator quick, a few minutes)
        py tools/ship.py --no-install     check + build + verify, leave out/release/lumen_robotaxi.zip

Install: the zip is copied to a temporary name inside Storm's mods folder and renamed over
lumen_robotaxi.zip (an atomic replace; measured 2026-11-01: the running game does not hold a write
lock on it). If the replace is refused, a detached watcher installs it when BeamNG closes.
A running game keeps its already loaded Lua: Storm presses Ctrl+L (reload Lua) or restarts; a changed
vehicle needs a restart or a respawn.
"""
import json
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import luaparse  # noqa: E402

SRC = os.path.join(ROOT, "src")
MOD = os.path.join(ROOT, "mod")
RELEASE = os.path.join(ROOT, "out", "release", "lumen_robotaxi.zip")
MODS_ZIP = os.path.join(os.environ["LOCALAPPDATA"], "BeamNG", "BeamNG.drive", "current", "mods", "lumen_robotaxi.zip")
WATCHER = os.path.join(ROOT, "tools", "install_watcher.py")
NO_WINDOW = 0x08000000                       # CREATE_NO_WINDOW: no console window for any child process

# names Lua code in this mod may read as globals: Lua's own, and BeamNG's (all checked against the game's lua/ folder)
KNOWN = set("""require math map table string tonumber tostring type ipairs pairs pcall select next unpack os io assert error
setmetatable getmetatable rawget rawset print log clamp vec3 quat quatFromDir be extensions scenetree spawn guihooks jsonReadFile
jsonWriteFile jsonEncode jsonDecode getObjectByID getPlayerVehicle getAllVehicles vehiclesIterator getCurrentLevelIdentifier
core_camera core_gamestate core_groundMarkers core_quickAccess core_trafficSignals core_vehicles core_modmanager career_career
gameplay_traffic gameplay_missions_missionManager ui_appLayouts ui_message castRay castRayStatic ColorF Engine debugDrawer
setExtensionUnloadMode electrics input ai controller obj tableKeys deepcopy lerp sign square round""".split())


def lua_files():
    for dp, _, fs in os.walk(SRC):
        for f in fs:
            if f.endswith(".lua"):
                yield os.path.join(dp, f)


def check():
    bad = 0
    for p in sorted(lua_files()):
        rel = os.path.relpath(p, ROOT)
        try:
            g = luaparse.check_file(p)
        except luaparse.LuaError as e:
            print(f"FAIL  {rel}: {e}")
            bad += 1
            continue
        odd = sorted(n for n in g if n not in KNOWN)
        print(f"ok    {rel}" + (f"   unknown globals: {', '.join(odd)}" if odd else ""))
    for dp, _, fs in os.walk(SRC):
        for f in fs:
            if f.endswith(".json"):
                p = os.path.join(dp, f)
                try:
                    json.load(open(p, encoding="utf-8"))
                except Exception as e:                                   # noqa: BLE001
                    print(f"FAIL  {os.path.relpath(p, ROOT)}: not valid json ({e})")
                    bad += 1
    return bad


def beamng_running():
    out = subprocess.run(["tasklist"], capture_output=True, text=True, creationflags=NO_WINDOW).stdout.lower()
    return "beamng.drive.x64.exe" in out


def build(vehicle):
    cmd = [sys.executable, "build.py"] + (["quick"] if vehicle else ["scripts"]) + ["--release"]
    print("build:", " ".join(cmd[1:]))
    return subprocess.run(cmd, cwd=os.path.join(ROOT, "generator")).returncode


def build_keep_vehicle():
    """Scripts, UI and art from src/ (through build.py scripts), but the vehicle files taken from the
    zip Storm already has, never from mod/vehicles: a vehicle rebuild in progress (or one Storm has not
    seen yet) cannot reach their game by accident."""
    r = subprocess.run([sys.executable, "build.py", "scripts"], cwd=os.path.join(ROOT, "generator"))
    if r.returncode:
        return 1
    old = zipfile.ZipFile(MODS_ZIP)
    if old.testzip():
        print("FAIL  the installed zip is damaged; cannot keep its vehicle files")
        return 1
    tmp = RELEASE + ".part"
    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for dp, dirs, fs in os.walk(MOD):
            dirs.sort()
            if os.path.abspath(dp) == os.path.abspath(MOD):
                dirs[:] = [d for d in dirs if d != "vehicles"]
            for f in sorted(fs):
                p = os.path.join(dp, f)
                z.write(p, os.path.relpath(p, MOD).replace(os.sep, "/"))
                n += 1
        for info in old.infolist():
            if info.filename.startswith("vehicles/") and not info.filename.endswith("/"):
                z.writestr(info.filename, old.read(info.filename))
                n += 1
    os.replace(tmp, RELEASE)
    print(f"release (vehicle kept from the installed zip): {n} files")
    return 0


def verify():
    z = zipfile.ZipFile(RELEASE)
    if z.testzip():
        print("FAIL  zip CRC:", z.testzip())
        return 1
    names = set(z.namelist())
    bad = 0
    for dp, _, fs in os.walk(SRC):
        for f in fs:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, SRC).replace(os.sep, "/")
            if rel not in names:
                print("FAIL  missing from the zip:", rel)
                bad += 1
            elif z.read(rel) != open(p, "rb").read():
                print("FAIL  zip differs from src:", rel)
                bad += 1
    for must in ("vehicles/lumen/lumen.jbeam", "vehicles/lumen/lumen.dae", "vehicles/lumen/info.json"):
        if must not in names:
            print("FAIL  vehicle file missing:", must)
            bad += 1
    print(f"zip: {len(names)} files, {os.path.getsize(RELEASE) / 1e6:.1f} MB, CRC ok" + ("" if not bad else f", {bad} problems"))
    return bad


def start_watcher():
    """One windowless watcher (pythonw, no console, named mutex inside the script). 2026-11-01: the old
    watcher was started once per ship and each copy opened a visible PowerShell window every 10 s."""
    exe = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if not os.path.exists(exe):
        exe = sys.executable
    flags = 0x00000008 | 0x00000200 | NO_WINDOW       # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    subprocess.Popen([exe, WATCHER], creationflags=flags, close_fds=True, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def install():
    # Measured 2026-11-01 01:44 in Storm's own beamng.log: replacing the zip under a RUNNING game makes
    # the mod manager re-mount the mod mid-session ("added > zip", activateMod) and the mod's own start
    # script failed to load. The swap itself succeeds (no write lock), which is what made it look safe.
    # So: never while BeamNG runs, unless --now. The watcher installs the moment the game closes.
    if beamng_running() and "--now" not in sys.argv:
        start_watcher()
        print("BeamNG is running, so the zip is NOT replaced now (that re-mounts the mod under the running game).")
        print("A watcher is waiting: it installs the new build as soon as BeamNG closes. Wait a few seconds after closing before relaunching.")
        return 2
    tmp = MODS_ZIP + ".new"
    shutil.copyfile(RELEASE, tmp)
    try:
        os.replace(tmp, MODS_ZIP)
    except OSError as e:
        print("the game refused the replace:", e)
        try:
            os.remove(tmp)
        except OSError:
            pass
        start_watcher()
        print("started the watcher: it installs when BeamNG closes")
        return 2
    same = open(MODS_ZIP, "rb").read() == open(RELEASE, "rb").read()
    print("installed:", MODS_ZIP, "(identical to the release)" if same else "(MISMATCH)")
    if beamng_running():
        print("BeamNG is running: press Ctrl+L in game to reload Lua (a changed vehicle needs a restart).")
    return 0 if same else 1


def main():
    a = set(sys.argv[1:])
    bad = check()
    if bad:
        print(f"\n{bad} problem(s): not building.")
        return 1
    if "--check-only" in a:
        return 0
    if "--keep-vehicle" in a:
        if build_keep_vehicle():
            return 1
    elif build("--vehicle" in a):
        return 1
    if verify():
        return 1
    if "--no-install" in a:
        return 0
    return install()


if __name__ == "__main__":
    sys.exit(main())
