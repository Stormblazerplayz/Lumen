-- Lumen service area (Frontier Studios).
-- Storm, 2026-11-01: "Make it have a geofence on east and west coast" and "Make the geofence
-- expand as the capabilities expand." The service grows in TIERS, one number per map:
--   tier 1  main roads
--   tier 2  plus residential streets and parking lots
--   tier 3  plus highways
-- A zone (a polygon on the map) belongs to a tier and is served once its map has opened that tier;
-- a road belongs to a tier by its class, and is used once its tier is open. So the service area
-- is the zones that are open AND the roads that are open, and bumping OPEN_TIER below (or adding
-- a zone) changes the map, the fleet, pickups, destinations and routes together. Nothing else
-- needs to change. Open a tier only when the driver has been shown to handle it on that map.
--
-- Zones are polygons in world metres taken from the road layout of each level (2026-11-01, from
-- the DecalRoads in the level files). They have not been walked in game yet: a first cut.
-- Plain module: require("lumen/geofence").

local M = {}
local logTag = "lumen_geofence"

-- ---------------------------------------------------------------------------------
-- DATA 1: what each tier means, and how far each map has opened
-- ---------------------------------------------------------------------------------
M.TIERS = {
  {id = 1, name = "Main roads",
   note = "traffic lights, stop signs, following, yielding and turns on the main roads"},
  {id = 2, name = "Residential streets and parking lots",
   note = "narrower streets, parking-lot pickups and drop-offs, hill and country roads"},
  {id = 3, name = "Highways and country roads",
   note = "freeways and winding mountain roads without shoulders. A Lumen lost control at 25 m/s on a divided road (2026-11-01): opens after the high-speed work"},
}

-- the highest tier open on each map (a map not listed is open to no tier)
M.OPEN_TIER = {
  west_coast_usa = 2,   -- the fleet ran across the whole map on 2026-11-01 (lights, stop signs, turns, yielding)
  east_coast_usa = 2,   -- the Harbor town only (its streets are tier 2 roads). The mountain roads are a tier 3 zone: set 3 to open them
}

-- A road with a posted limit at or above this (m/s) needs tier 3 whatever zone it is in. Off (99) since
-- 2026-11-01: BeamNG gives a road without a speed limit one worked out from its width alone (19.4 m/s
-- x (half-width + 5) / 8, rounded down to 20/35/40/50/60/70 mph), so a wide city boulevard gets 60 to 70
-- mph and 50 mph at 9 m wide. Closing "fast" roads cut Downtown Belasco into pieces no car could route
-- between (Storm: "No Lumen can reach the roads right here"). Highways are fenced by PLACE instead:
-- the tier 3 zones below.
M.FAST = 99.0

-- ---------------------------------------------------------------------------------
-- DATA 2: zones per level. `poly` is one polygon, `polys` several (a list of {x, y} corners).
-- A point in several zones is served if ANY of them is open.
-- ---------------------------------------------------------------------------------
M.ZONES = {
  west_coast_usa = {
    {id = "downtown", name = "Downtown Belasco", tier = 1, polys = {
      {{-1040, 960}, {-360, 960}, {-300, 700}, {-480, 450}, {-700, 200}, {-860, -40}, {-1040, -40}},
      {{-380, 1000}, {250, 1000}, {250, 560}, {-300, 560}},
    }},
    {id = "south_port", name = "South Belasco and the port", tier = 2, polys = {
      {{-1660, -1100}, {-1000, -1100}, {-1000, -40}, {-1660, -40}},
      {{-1000, -640}, {-250, -640}, {-250, -40}, {-1000, -40}},
    }},
    {id = "east_hills", name = "East hills", tier = 2, poly =
      {{-150, -1100}, {1100, -1100}, {1100, 1050}, {-150, 1050}}},
    {id = "us110", name = "US 110 corridor", tier = 3, poly =
      {{-1029, -229}, {878, 1108}, {1130, 748}, {-777, -589}}},
    {id = "north_island", name = "Northern island", tier = 3, poly =
      {{-1150, 1150}, {1050, 1150}, {1050, 2250}, {-1150, 2250}}},
  },
  east_coast_usa = {
    -- 2026-11-01: the Harbor town alone has about 0.4 km of road a car can use (its streets are decals
    -- with no drivability), so the east side of the map is the first zone, and it grows westwards.
    {id = "east_valley", name = "Harbor valley", tier = 1, poly =
      {{150, -1000}, {1000, -1000}, {1000, 1000}, {150, 1000}}},
    {id = "east_ridge", name = "Ridge roads", tier = 2, poly =
      {{-300, -1000}, {150, -1000}, {150, 1000}, {-300, 1000}}},
    {id = "west_mountains", name = "Western mountains", tier = 3, poly =
      {{-1000, -1000}, {-300, -1000}, {-300, 1000}, {-1000, 1000}}},
  },
}

-- ---------------------------------------------------------------------------------
-- logic
-- ---------------------------------------------------------------------------------
local override = {}           -- [level] = tier, set at run time (tests, a debug switch)
local version = 1             -- bumped on every change, so caches elsewhere refresh

local function level()
  local l = getCurrentLevelIdentifier()
  return l and tostring(l) or ""
end

-- changes when the open tier or the level changes; roads.lua keys its caches on it
function M.version() return version .. "@" .. level() end

-- the highest tier open on this map
function M.openTier(lv)
  lv = lv or level()
  return override[lv] or M.OPEN_TIER[lv] or 0
end

-- open (or close) tiers on a map until the game restarts: for tests
function M.setOpenTier(n, lv)
  lv = lv or level()
  override[lv] = tonumber(n) or 0
  version = version + 1
  log("I", logTag, string.format("tier %d is now the highest open on %s", override[lv], lv))
end

function M.tierName(n)
  for _, t in ipairs(M.TIERS) do
    if t.id == n then return t.name end
  end
  return "tier " .. tostring(n)
end

-- true when the level has a service area at all (a map without zones is not restricted)
function M.hasZones(lv) return M.ZONES[lv or level()] ~= nil end

local function inPoly(poly, x, y)
  local inside = false
  local n = #poly
  local j = n
  for i = 1, n do
    local xi, yi, xj, yj = poly[i][1], poly[i][2], poly[j][1], poly[j][2]
    if (yi > y) ~= (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi then inside = not inside end
    j = i
  end
  return inside
end

local function zoneHolds(z, x, y)
  if z.poly then return inPoly(z.poly, x, y) end
  for _, p in ipairs(z.polys or {}) do
    if inPoly(p, x, y) then return true end
  end
  return false
end

-- Where is this point? Returns the open zone holding it (or nil), then, when no open zone holds
-- it, the closed zone that does and the tier that would open it.
function M.where(x, y)
  local zs = M.ZONES[level()]
  if not zs then return nil end
  local open = M.openTier()
  local closed
  for _, z in ipairs(zs) do
    if zoneHolds(z, x, y) then
      if (z.tier or 1) <= open then return z end
      if not closed or (z.tier or 1) < (closed.tier or 1) then closed = z end
    end
  end
  return nil, closed, closed and (closed.tier or 1) or nil
end

function M.served(x, y)
  if not M.hasZones() then return true end
  return M.where(x, y) ~= nil
end

-- The lowest tier a road needs, from its class (the same split the 2D map draws):
--   highway (posted 50 mph and up)                  tier 3
--   main road (well surfaced and 3.5 m half-width)   tier 1
--   residential street, lane, lot, track             tier 2
function M.linkTier(link, radius)
  link = link or {}
  if (tonumber(link.speedLimit) or 0) >= M.FAST then return 3 end
  local drv = link.drivability or 1
  if link.type == "private" or drv < 0.55 or (radius or 3) < 2.4 then return 2 end
  if drv >= 0.9 and (radius or 3) >= 3.5 then return 1 end
  return 2
end

-- is this road's own class open? (no zone test: roads.lua uses it for the corridors that join pieces)
function M.classOK(link, radius)
  return M.linkTier(link, radius) <= M.openTier()
end

-- A road link may be used when both ends and its middle are in served zones and its own tier is
-- open. Called by roads.lua for every link it considers (cached there per version).
function M.linkOK(pa, pb, link, radius)
  if not M.hasZones() then return true end
  if M.linkTier(link, radius) > M.openTier() then return false end
  return M.served((pa.x + pb.x) * 0.5, (pa.y + pb.y) * 0.5) and M.served(pa.x, pa.y) and M.served(pb.x, pb.y)
end

-- ---------------------------------------------------------------------------------
-- export for the 2D map: zones (flat polygons) with whether each is open, and the tiers
-- ---------------------------------------------------------------------------------
function M.export()
  local lv = level()
  local open = M.openTier(lv)
  local zones, tiers = {}, {}
  for _, z in ipairs(M.ZONES[lv] or {}) do
    local polys = {}
    for _, p in ipairs(z.polys or {z.poly}) do
      local flat = {}
      for _, pt in ipairs(p) do flat[#flat + 1] = pt[1] flat[#flat + 1] = pt[2] end
      polys[#polys + 1] = flat
    end
    zones[#zones + 1] = {id = z.id, name = z.name, tier = z.tier or 1, open = (z.tier or 1) <= open, polys = polys}
  end
  for _, t in ipairs(M.TIERS) do
    tiers[#tiers + 1] = {id = t.id, name = t.name, open = t.id <= open, note = t.note}
  end
  return {level = lv, openTier = open, zones = zones, tiers = tiers}
end

-- where the player stands, for the app's status line
function M.status(x, y)
  if not M.hasZones() then return {restricted = false} end
  local act, closed, tier = M.where(x, y)
  if act then return {restricted = true, inside = true, zone = act.name, tier = act.tier or 1, openTier = M.openTier()} end
  return {restricted = true, inside = false, zone = closed and closed.name or nil, opensWith = tier and M.tierName(tier) or nil,
          openTier = M.openTier()}
end

return M
