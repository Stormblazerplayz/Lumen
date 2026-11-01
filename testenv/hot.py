"""Hot-reload the Lumen mod's Lua on the running test instance without a full GE Lua reload
(that takes the MCP server down on 0.40): copy src into the test profile, remove the fleet's
vehicles, unload the extensions, forget the required modules, load them again.
usage: py hot.py"""
import os
import subprocess
import sys

import mcp

HERE = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, "build.py", "scripts", "--install-test"], cwd=os.path.join(HERE, "..", "generator"), check=True)
print(mcp.lua("""
local n = 0
for _, v in ipairs(getAllVehicles()) do
  if v:getJBeamFilename() == 'lumen' then v:delete() n = n + 1 end
end
for _, e in ipairs({'lumen_robotaxi', 'lumen_arpath', 'lumen_lidar'}) do extensions.unload(e) end
for k in pairs(package.loaded) do if k:find('^lumen/') then package.loaded[k] = nil end end
extensions.load('lumen_robotaxi')
extensions.load('lumen_arpath')
extensions.load('lumen_lidar')
return 'removed ' .. n .. ' lumens; reloaded: ' .. tostring(extensions.lumen_robotaxi ~= nil)"""))
