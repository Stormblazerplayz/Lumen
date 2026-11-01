-- Lumen road helpers (Frontier Studios).
-- Which roads a robotaxi may use, routes that stop at a point next to you, travel-time
-- estimates, hidden spots for fleet cars to appear at, and the road network export that the
-- app draws as its 2D map. Plain module: require("lumen/roads").

local M = {}

-- Public-road filter. BeamNG traffic accepts drivability >= 0.5; driveways, yard lanes and
-- parking aisles are link type "private" and about 2 m half-width on West Coast USA.
-- 2026-10-31: a 2.2 m half-width cutoff with no private check let a dispatch start inside a
-- gated industrial yard, and the AI wedged the car on the gate post on its way out.
M.MIN_DRIVABILITY = 0.55
M.MIN_RADIUS = 2.4

local GRID = 80                   -- spatial index cell (m)
local STOP_CLEAR = 12             -- minimum distance from a junction node to a stop point
local INTERSECTION_DELAY = 2.5    -- s added per junction on a route
local SPEED_FACTOR = 0.82         -- the legal-mode AI averages about this share of the limit

local index = nil                 -- built lazily per road network, see buildIndex()

local function nodes()
  local m = map.getMap()
  return m and m.nodes or {}
end
M.nodes = nodes

-- The service area (lumen/geofence.lua) plugs in here: gate = {link = function(pa, pb, link, radius),
-- version = function(), classOK = function(link, radius)}, set by the service. A link outside the
-- served network (below: the links the gate admits, joined into one network by short corridors) is
-- never picked as a stop, a spawn spot, a cruise target or part of a route. nil allows every road.
M.gate = nil
local served = nil           -- the served network for the gate's current version (see buildServed)
local servedState            -- forward declaration: built below the heap helpers
local lastVerCheck = -1e9

local function allowed(L)
  local g = M.gate
  if not g then return true end
  return servedState(g).edge[L.key] == true
end
M.allowed = allowed

-- the gate, with the served network current for its version (nil when no service area is set)
local function refreshGate()
  local gate = M.gate
  if gate then servedState(gate) end
  return gate
end

-- may a route use the road between nodes a and b? Call refreshGate() first.
local function edgeAllowed(gate, a, b, na, nb, l)
  return served ~= nil and served.edge[M.linkKey(a, b)] == true
end

-- ---------------------------------------------------------------------------------
-- index of links (each road segment once) bucketed into grid cells
-- ---------------------------------------------------------------------------------
local function cellKey(ix, iy) return ix * 65536 + iy end

local function buildIndex()
  local m = map.getMap()
  local mn = m and m.nodes or {}
  local links, cells, count = {}, {}, 0
  for a, na in pairs(mn) do
    count = count + 1
    for b, l in pairs(na.links or {}) do
      local nb = mn[b]
      if nb and (a < b or not (nb.links and nb.links[a])) then
        local radius = math.min(na.radius or 0, nb.radius or 0)
        local rec = {
          a = a, b = b, pa = na.pos, pb = nb.pos, link = l, key = M.linkKey(a, b),
          radius = radius, width = (na.radius or 2) + (nb.radius or 2),
          len = na.pos:distance(nb.pos),
          public = l.type ~= "private" and (l.drivability or 1) >= M.MIN_DRIVABILITY and radius >= M.MIN_RADIUS,
        }
        links[#links + 1] = rec
        local i = #links
        local x0, x1 = math.floor(math.min(na.pos.x, nb.pos.x) / GRID), math.floor(math.max(na.pos.x, nb.pos.x) / GRID)
        local y0, y1 = math.floor(math.min(na.pos.y, nb.pos.y) / GRID), math.floor(math.max(na.pos.y, nb.pos.y) / GRID)
        for ix = x0, x1 do
          for iy = y0, y1 do
            local k = cellKey(ix, iy)
            local c = cells[k]
            if not c then c = {} cells[k] = c end
            c[#c + 1] = i
          end
        end
      end
    end
  end
  index = {map = m, links = links, cells = cells, nodeCount = count}
  return index
end

local function ensureIndex()
  local m = map.getMap()
  if not index or index.map ~= m then buildIndex() end
  return index
end

function M.invalidate() index = nil end

function M.nodeCount()
  return ensureIndex().nodeCount
end

-- legal direction of travel from node `from` to node `to` along link record L
function M.legal(L, from, to)
  local l = L.link
  if not l.oneWay then return true end
  return l.inNode == from and l.outNode == to
end

-- closest link to pos; opts.public limits it to public roads, opts.maxDist is the search
-- radius, opts.dir (a car's heading) pushes away one-way links that run against it, so a car
-- on a divided highway snaps to its own carriageway
function M.closestLink(pos, opts)
  opts = opts or {}
  local idx = ensureIndex()
  local r = opts.maxDist or 150
  local best, bestD2, bestPoint, bestT = nil, r * r, nil, nil
  local seen = {}
  for ix = math.floor((pos.x - r) / GRID), math.floor((pos.x + r) / GRID) do
    for iy = math.floor((pos.y - r) / GRID), math.floor((pos.y + r) / GRID) do
      local c = idx.cells[cellKey(ix, iy)]
      if c then
        for _, i in ipairs(c) do
          if not seen[i] then
            seen[i] = true
            local L = idx.links[i]
            if not opts.public or (L.public and allowed(L)) then
              local t = clamp(pos:xnormOnLine(L.pa, L.pb), 0, 1)
              local p = L.pa + (L.pb - L.pa) * t
              local d2 = p:squaredDistance(pos)
              if opts.dir and L.link.oneWay then
                local mn = idx.map.nodes
                local inN, outN = mn[L.link.inNode], mn[L.link.outNode]
                if inN and outN and (outN.pos - inN.pos):dot(opts.dir) < 0 then
                  d2 = (math.sqrt(d2) + 25) ^ 2
                end
              end
              if d2 < bestD2 then best, bestD2, bestPoint, bestT = L, d2, p, t end
            end
          end
        end
      end
    end
  end
  if not best then return nil end
  return best, bestPoint, math.sqrt(bestD2), bestT
end

-- the n closest links to pos within opts.maxDist (public roads only with opts.public), closest
-- first, as {L, t, d}. Callers use the first one a car can reach: road networks have islands
-- (road pieces, lots and trails that do not connect to the main roads).
function M.closestLinks(pos, opts, n)
  opts = opts or {}
  local idx = ensureIndex()
  local r = opts.maxDist or 300
  local found, seen = {}, {}
  for ix = math.floor((pos.x - r) / GRID), math.floor((pos.x + r) / GRID) do
    for iy = math.floor((pos.y - r) / GRID), math.floor((pos.y + r) / GRID) do
      local c = idx.cells[cellKey(ix, iy)]
      if c then
        for _, i in ipairs(c) do
          if not seen[i] then
            seen[i] = true
            local L = idx.links[i]
            if not opts.public or (L.public and allowed(L)) then
              local t = clamp(pos:xnormOnLine(L.pa, L.pb), 0, 1)
              local d = (L.pa + (L.pb - L.pa) * t):distance(pos)
              if d <= r then found[#found + 1] = {L = L, t = t, d = d} end
            end
          end
        end
      end
    end
  end
  table.sort(found, function(a, b) return a.d < b.d end)
  local out = {}
  for i = 1, math.min(n or 8, #found) do out[i] = found[i] end
  return out
end

-- a random public road between minDist and maxDist from centre (cruising targets); nil after
-- 40 misses
function M.randomLink(centre, minDist, maxDist, comp)
  local idx = ensureIndex()
  local n = #idx.links
  if n == 0 then return nil end
  local lo, hi = minDist * minDist, maxDist * maxDist
  for _ = 1, 40 do
    local L = idx.links[math.random(n)]
    if L.public and L.len > 20 and allowed(L) and (not comp or M.component(L) == comp) then
      local mid = (L.pa + L.pb) * 0.5
      local d2 = (mid.x - centre.x) ^ 2 + (mid.y - centre.y) ^ 2
      if d2 >= lo and d2 <= hi then return L end
    end
  end
  return nil
end

-- a point on link L near parameter t, kept clear of junctions (an end node with more than two
-- links): far enough that the car is straight in its lane, not mid-turn on a corner
local function endClearance(node)
  if map.getNodeLinkCount(node) <= 2 then return 3 end
  local n = nodes()[node]
  return math.max(STOP_CLEAR, (n and n.radius or 4) + 8)
end

function M.stopPoint(L, t)
  local ca, cb = endClearance(L.a), endClearance(L.b)
  if L.len > ca + cb + 2 then
    t = clamp(t, ca / L.len, 1 - cb / L.len)
  else
    t = 0.5
  end
  return L.pa + (L.pb - L.pa) * t
end

-- ---------------------------------------------------------------------------------
-- routes
-- ---------------------------------------------------------------------------------
function M.linkKey(a, b)
  if a < b then return a .. "|" .. b end
  return b .. "|" .. a
end

-- binary heap of {priority, node}
local function heapPush(h, pri, item)
  local i = #h + 1
  h[i] = {pri, item}
  while i > 1 do
    local p = math.floor(i / 2)
    if h[p][1] <= h[i][1] then break end
    h[p], h[i] = h[i], h[p]
    i = p
  end
end

local function heapPop(h)
  local top = h[1]
  local last = table.remove(h)
  if #h > 0 then
    h[1] = last
    local i, n = 1, #h
    while true do
      local l, r, m = 2 * i, 2 * i + 1, i
      if l <= n and h[l][1] < h[m][1] then m = l end
      if r <= n and h[r][1] < h[m][1] then m = r end
      if m == i then break end
      h[m], h[i] = h[i], h[m]
      i = m
    end
  end
  return top[2]
end

-- ---------------------------------------------------------------------------------
-- the served network: the gate's links, joined into one network, in labelled pieces
-- ---------------------------------------------------------------------------------
local CORRIDOR_MAX = 900     -- cost metres of closed road a corridor may cross to join a piece to the main one
local MIN_ISLAND = 250       -- m: a piece shorter than this that no corridor reaches is dropped

local function buildServed(gate, ver)
  local t0 = os.clock()
  local mn = nodes()
  local S = {ver = ver, edge = {}, comp = {}, len = {}, cx = {}, cy = {}, main = nil, count = 0,
             corridorLinks = 0, corridorLen = 0, dropped = 0, droppedLen = 0, links = 0, km = 0}
  -- every graph link once
  local links = {}
  for a, na in pairs(mn) do
    for b, l in pairs(na.links or {}) do
      local nb = mn[b]
      if nb and (a < b or not (nb.links and nb.links[a])) then
        links[#links + 1] = {a = a, b = b, key = M.linkKey(a, b), len = na.pos:distance(nb.pos), l = l,
                             na = na, nb = nb, rad = math.min(na.radius or 3, nb.radius or 3)}
      end
    end
  end
  -- 1. the links the zones and tiers admit
  for _, k in ipairs(links) do
    if gate.link(k.na.pos, k.nb.pos, k.l, k.rad) then S.edge[k.key] = true end
  end
  -- label the pieces over the current S.edge
  local adj = {}
  local function label()
    adj = {}
    for _, k in ipairs(links) do
      if S.edge[k.key] then
        local x = adj[k.a]
        if not x then x = {} adj[k.a] = x end
        x[#x + 1] = k.b
        local y = adj[k.b]
        if not y then y = {} adj[k.b] = y end
        y[#y + 1] = k.a
      end
    end
    local comp, lenOf, nodesOf, n = {}, {}, {}, 0
    for node in pairs(adj) do
      if not comp[node] then
        n = n + 1
        local stack, members = {node}, {}
        comp[node] = n
        while #stack > 0 do
          local cur = table.remove(stack)
          members[#members + 1] = cur
          for _, nb in ipairs(adj[cur]) do
            if not comp[nb] then
              comp[nb] = n
              stack[#stack + 1] = nb
            end
          end
        end
        nodesOf[n] = members
        lenOf[n] = 0
      end
    end
    for _, k in ipairs(links) do
      if S.edge[k.key] then lenOf[comp[k.a]] = lenOf[comp[k.a]] + k.len end
    end
    return comp, lenOf, nodesOf, n
  end
  local comp, lenOf, nodesOf, n = label()
  -- 2. corridors: join every piece to the main one over the shortest closed (but open-class) road
  if gate.classOK and n > 1 then
    local main, best = nil, -1
    for id, L in pairs(lenOf) do
      if L > best then main, best = id, L end
    end
    local cadj = {}
    for _, k in ipairs(links) do
      if not S.edge[k.key] and gate.classOK(k.l, k.rad) then
        -- public roads preferred: lanes and lots cost 30 times their length
        local w = k.len * ((k.l.type == "private" or k.rad < M.MIN_RADIUS or (k.l.drivability or 1) < M.MIN_DRIVABILITY) and 30 or 1)
        local x = cadj[k.a]
        if not x then x = {} cadj[k.a] = x end
        x[#x + 1] = {k.b, k, w}
        local y = cadj[k.b]
        if not y then y = {} cadj[k.b] = y end
        y[#y + 1] = {k.a, k, w}
      end
    end
    local joined = {[main] = true}
    local dist, prev, heap, done = {}, {}, {}, {}
    for _, node in ipairs(nodesOf[main]) do
      dist[node] = 0
      heapPush(heap, 0, node)
    end
    while #heap > 0 do
      local cur = heapPop(heap)
      if not done[cur] then
        done[cur] = true
        local c = comp[cur]
        if c and not joined[c] then
          -- reached another piece: keep the corridor back to a joined one, then treat the whole piece as joined
          local m = cur
          while prev[m] do
            local link = prev[m][2]
            if not S.edge[link.key] then
              S.edge[link.key] = true
              S.corridorLinks = S.corridorLinks + 1
              S.corridorLen = S.corridorLen + link.len
            end
            m = prev[m][1]
          end
          joined[c] = true
          for _, node in ipairs(nodesOf[c]) do
            dist[node] = 0
            prev[node] = nil
            done[node] = nil
            heapPush(heap, 0, node)
          end
        else
          local base = dist[cur]
          for _, e in ipairs(cadj[cur] or {}) do
            local nd = base + e[3]
            local to = e[1]
            if nd <= CORRIDOR_MAX and (dist[to] == nil or nd < dist[to]) then
              dist[to] = nd
              prev[to] = {cur, e[2]}
              heapPush(heap, nd, to)
            end
          end
        end
      end
    end
    comp, lenOf, nodesOf, n = label()
  end
  -- 3. drop what is still too small to matter
  for _, k in ipairs(links) do
    if S.edge[k.key] and lenOf[comp[k.a]] < MIN_ISLAND then
      S.edge[k.key] = nil
      S.dropped = S.dropped + 1
      S.droppedLen = S.droppedLen + k.len
    end
  end
  comp, lenOf, nodesOf, n = label()
  S.comp, S.len, S.count = comp, lenOf, n
  local best = -1
  for id, L in pairs(lenOf) do
    local sx, sy = 0, 0
    for _, node in ipairs(nodesOf[id]) do
      sx, sy = sx + mn[node].pos.x, sy + mn[node].pos.y
    end
    S.cx[id], S.cy[id] = sx / #nodesOf[id], sy / #nodesOf[id]
    S.km = S.km + L / 1000
    if L > best then S.main, best = id, L end
  end
  for _ in pairs(S.edge) do S.links = S.links + 1 end
  local mainKm = S.main and lenOf[S.main] / 1000 or 0
  log("I", "lumen_roads", string.format("service area network (%s): %d links, %.1f km in %d piece(s); %d corridor link(s) (%.0f m) joined pieces; %d small island link(s) (%.0f m) dropped; main piece %.1f km (%d%%); built in %d ms",
    tostring(ver), S.links, S.km, n, S.corridorLinks, S.corridorLen, S.dropped, S.droppedLen, mainKm, S.km > 0 and math.floor(100 * mainKm / S.km + 0.5) or 0,
    math.floor((os.clock() - t0) * 1000 + 0.5)))
  local ids = {}
  for id in pairs(lenOf) do ids[#ids + 1] = id end
  table.sort(ids, function(x, y) return lenOf[x] > lenOf[y] end)
  for i = 1, math.min(#ids, 6) do
    log("I", "lumen_roads", string.format("  piece %d: %.2f km around (%.0f, %.0f)%s", ids[i], lenOf[ids[i]] / 1000, S.cx[ids[i]], S.cy[ids[i]], ids[i] == S.main and " (main)" or ""))
  end
  return S
end

servedState = function(gate)
  local now = os.clock()
  if served and now - lastVerCheck < 0.25 then return served end
  lastVerCheck = now
  local v = gate.version()
  if not served or served.ver ~= v then served = buildServed(gate, v) end
  return served
end

-- the piece (component) of the served network a road belongs to; nil when the road is not served
function M.component(L)
  local g = M.gate
  if not g then return nil end
  local S = servedState(g)
  if not S.edge[L.key] then return nil end
  return S.comp[L.a]
end

-- the piece a car at pos is on: its own road when served, else the nearest served road within 80 m
function M.componentAt(pos, dir)
  local g = M.gate
  if not g then return nil end
  local S = servedState(g)
  local L = M.closestLink(pos, {maxDist = 60, dir = dir})
  local c = L and S.edge[L.key] and S.comp[L.a] or nil
  if not c then
    local P = M.closestLink(pos, {public = true, maxDist = 80})
    c = P and M.component(P) or nil
  end
  return c
end

-- the piece a rider belongs to: the nearest served road within 600 m, else the biggest piece
function M.areaComponent(pos)
  local g = M.gate
  if not g then return nil end
  local S = servedState(g)
  local L = M.closestLink(pos, {public = true, maxDist = 600})
  return (L and M.component(L)) or S.main
end

-- size and centre of a piece, for logs: km, x, y
function M.pieceInfo(id)
  local g = M.gate
  if not g or not id then return nil end
  local S = servedState(g)
  if not S.len[id] then return nil end
  return {km = S.len[id] / 1000, x = S.cx[id], y = S.cy[id], main = id == S.main}
end

-- A* over the road graph that never uses a link in avoid (keys from linkKey). Same costs as
-- the map.getPath call below: length, x8 at or below the public cutoff, x1000 against one-way.
local function pathAvoiding(from, to, avoid, soft)
  local mn = nodes()
  if not mn[from] or not mn[to] then return nil end
  local goal = mn[to].pos
  -- the service area: edges outside it are not used, except within 80 m of where the route starts
  -- (a car already standing on a closed road must be able to leave it)
  local gate = refreshGate()
  local startPos = mn[from].pos
  local g, came, closed, open = {[from] = 0}, {}, {}, {}
  heapPush(open, mn[from].pos:distance(goal), from)
  while #open > 0 do
    local cur = heapPop(open)
    if cur == to then
      local out = {cur}
      while came[cur] do
        cur = came[cur]
        table.insert(out, 1, cur)
      end
      return out
    end
    if not closed[cur] then
      closed[cur] = true
      local nc = mn[cur]
      for nb, l in pairs(nc.links or {}) do
        local nn = mn[nb]
        local blocked = false
        local closedCost = 1
        if gate and nn and nc.pos:distance(startPos) > 80 and not edgeAllowed(gate, cur, nb, nc, nn, l) then
          if soft then closedCost = 8 else blocked = true end      -- soft: a closed road costs 8x instead of being a wall
        end
        if nn and not blocked and not closed[nb] and not avoid[M.linkKey(cur, nb)] then
          local cost = nc.pos:distance(nn.pos) * closedCost
          if (l.drivability or 1) <= M.MIN_DRIVABILITY then cost = cost * 8 end
          -- public roads, please: driveways, alleys, lots and narrow lanes only where a trip
          -- starts or ends in one (2026-11-01: a route through a gap behind buildings ended in a wall)
          if l.type == "private" or math.min(nc.radius or 3, nn.radius or 3) < M.MIN_RADIUS then cost = cost * 30 end
          if l.oneWay and l.inNode ~= cur then cost = cost * 1000 end
          local ng = g[cur] + cost
          if not g[nb] or ng < g[nb] then
            g[nb], came[nb] = ng, cur
            heapPush(open, ng + nn.pos:distance(goal), nb)
          end
        end
      end
    end
  end
  return nil
end

-- avoid: optional {[linkKey] = true} of roads to keep off (blocked streets)
local lastSoftLog = -1e9
function M.path(from, to, avoid)
  if from == to then return {from} end
  local p = pathAvoiding(from, to, avoid or {})
  if p then return p end
  if M.gate then
    -- A route over a closed road beats no route (Storm, 2026-11-02: "I am clicking everywhere and I can't
    -- go anywhere"). The service area says where Lumens pick up, drop off and roam, not whether a road
    -- exists. Served roads are preferred (closed ones cost 8x). The log says where the gap was.
    p = pathAvoiding(from, to, avoid or {}, true)
    if p then
      local now = os.clock()
      if now - lastSoftLog > 5 then
        lastSoftLog = now
        local mn, gate = nodes(), refreshGate()
        local closedN, fx, fy, fAt, acc = 0, nil, nil, nil, 0
        for i = 1, #p - 1 do
          local na, nb = mn[p[i]], mn[p[i + 1]]
          if na and nb then
            acc = acc + na.pos:distance(nb.pos)
            if not edgeAllowed(gate, p[i], p[i + 1], na, nb, nil) and na.pos:distance(mn[p[1]].pos) > 80 then
              closedN = closedN + 1
              if not fx then fx, fy, fAt = na.pos.x, na.pos.y, acc end
            end
          end
        end
        log("I", "lumen_roads", string.format("route needed %d closed link(s): the served network alone has no route from %s to %s (first closed link at (%.0f, %.0f), %.0f m along a %d-node route)",
          closedN, tostring(from), tostring(to), fx or 0, fy or 0, fAt or 0, #p))
      end
      return p
    end
    return nil
  end
  -- With a service area set there is no unconstrained fallback: map.getPath ignores the gate, and
  -- a destination reachable only over a closed road (a highway, a closed zone) would be routed
  -- through it. The caller says "no route inside the service area" instead.
  if M.gate or (avoid and next(avoid)) then return nil end
  -- penalize roads at or below the public cutoff and heavily penalize driving against
  -- one-way streets, so the route matches what a legal driver would take
  local p = map.getPath(from, to, M.MIN_DRIVABILITY, 1000, 1, 8)
  if p and #p > 0 then return p end
  return nil
end

local function linkSpeed(a, b)
  local mn = nodes()
  local na = mn[a]
  local l = na and na.links and na.links[b]
  if not l then
    local nb = mn[b]
    l = nb and nb.links and nb.links[a]
  end
  local v = l and l.speedLimit or 13.9
  return math.max(4, math.min(v, 31)) * SPEED_FACTOR
end

-- trip = {nodes = AI path, points = {start, node1 .. node(n-1), stopPoint}, segV, junction, idx}
-- The last AI node lies beyond the stop point, so the car passes it; the service brakes the
-- car to a halt at the stop point.
local function finishTrip(nodeList, pos, stop)
  local mn = nodes()
  local n = #nodeList
  local pts, segV, junction = {vec3(pos)}, {}, {}
  for i = 1, n - 1 do
    local nd = mn[nodeList[i]]
    if not nd then return nil end
    pts[#pts + 1] = vec3(nd.pos)
  end
  pts[#pts + 1] = vec3(stop)
  -- segment i runs from pts[i] to pts[i + 1]
  for i = 1, #pts - 1 do
    if i == 1 then
      segV[i] = n >= 2 and linkSpeed(nodeList[1], nodeList[2]) or 11
    else
      segV[i] = linkSpeed(nodeList[i - 1], nodeList[i])
    end
  end
  for i = 2, #pts - 1 do
    junction[i] = map.getNodeLinkCount(nodeList[i - 1]) > 2
  end
  local trip = {nodes = nodeList, points = pts, segV = segV, junction = junction, idx = 1, stop = vec3(stop),
                stopDir = (pts[#pts] - pts[#pts - 1]):normalized()}
  M.measure(trip, pos)
  return trip
end

-- From node cur (reached from prev): the continuation closest to straight ahead, in the legal
-- direction, preferring public roads; nil at a dead end.
local function straightOn(prev, cur)
  local mn = nodes()
  local nc, np = mn[cur], mn[prev]
  if not nc or not np then return nil end
  local inDir = vec3(nc.pos.x - np.pos.x, nc.pos.y - np.pos.y, 0):normalized()
  local best, bestScore
  local gate = refreshGate()
  for nb, l in pairs(nc.links or {}) do
    local nn = mn[nb]
    if nb ~= prev and nn and (not l.oneWay or l.inNode == cur) and not (gate and not edgeAllowed(gate, cur, nb, nc, nn, l)) then
      local d = vec3(nn.pos.x - nc.pos.x, nn.pos.y - nc.pos.y, 0):normalized()
      local score = d:dot(inDir)
      if l.type == "private" or (l.drivability or 1) <= M.MIN_DRIVABILITY then score = score - 0.6 end
      if not bestScore or score > bestScore then best, bestScore = nb, score end
    end
  end
  return best
end

M.straightOn = straightOn
M.tripFromNodes = function(nodeList, pos, stop) return finishTrip(nodeList, pos, stop) end

-- true if the trip turns by more than maxDeg within dist metres of the car
local function sharpTurnWithin(trip, dir, dist, maxDeg)
  local pts = trip.points
  local prevDir = vec3(dir.x, dir.y, 0):normalized()
  local cosMax = math.cos(math.rad(maxDeg))
  local acc = 0
  for i = 1, #pts - 1 do
    local seg = pts[i + 1] - pts[i]
    local len = seg:length()
    if len > 0.5 then
      local d = vec3(seg.x, seg.y, 0):normalized()
      if prevDir:dot(d) < cosMax then return true end
      prevDir = d
      acc = acc + len
      if acc > dist then return false end
    end
  end
  return false
end

-- Plans a route for a car at pos facing dir that stops at point `stop` on link L.
-- opts.startNode forces the first node (e.g. the node behind the car, to turn around);
-- opts.avoid is a set of blocked links to route around; opts.speed (m/s) is how fast the car
-- is going: a moving car never takes a sharp turn inside its braking distance (Storm: "instead
-- of just going by and rerouting"), it carries straight on and the route starts beyond it.
local planTripRaw

function M.planTrip(pos, dir, L, stop, opts)
  local best, CL, ahead = planTripRaw(pos, dir, L, stop, opts)
  if not best and opts and opts.avoid and next(opts.avoid) then
    -- the blocked-road memory is a memory, not a wall: a route over a road that was blocked a minute ago
    -- beats no route at all
    local o2 = {}
    for k, v in pairs(opts) do o2[k] = v end
    o2.avoid = nil
    best, CL, ahead = planTripRaw(pos, dir, L, stop, o2)
  end
  local speed = opts and opts.speed or 0
  if best and CL and speed > 6 and not (opts and opts.startNode) then
    local dist = math.min(250, speed * speed / 5 + speed * 1.2)    -- about 2.5 m/s2 plus 1.2 s
    if sharpTurnWithin(best, dir, dist, 35) then
      local mn = nodes()
      local prev = (ahead == CL.a) and CL.b or CL.a
      local prefix, cur = {ahead}, ahead
      local acc = pos:distance(mn[ahead].pos)
      while acc < dist and cur ~= L.a and cur ~= L.b and #prefix < 40 do
        local nxt = straightOn(prev, cur)
        if not nxt then break end
        acc = acc + mn[cur].pos:distance(mn[nxt].pos)
        prefix[#prefix + 1] = nxt
        prev, cur = cur, nxt
      end
      -- from the end of the straight run, never straight back the way it came
      local noBack = {}
      if opts and opts.avoid then for k in pairs(opts.avoid) do noBack[k] = true end end
      noBack[M.linkKey(prev, cur)] = true
      local committed
      for _, o in ipairs({{L.a, L.b}, {L.b, L.a}}) do
        local first, last = o[1], o[2]
        if M.legal(L, first, last) then
          local p = M.path(cur, first, noBack) or M.path(cur, first, opts and opts.avoid)
          if p then
            local list = {}
            for i = 1, #prefix - 1 do list[#list + 1] = prefix[i] end
            for i = 1, #p do list[#list + 1] = p[i] end
            list[#list + 1] = last
            local trip = finishTrip(list, pos, stop)
            if trip and (not committed or trip.eta < committed.eta) then committed = trip end
          end
        end
      end
      if committed then
        committed.committed = true
        return committed
      end
    end
  end
  return best
end

-- a route that turns straight back on itself somewhere: A, B, A (or, from the car, back to the
-- node behind it). No car can make that hairpin; such routes are a last resort only.
local function hasReversal(list, behindNode)
  if behindNode and list[2] == behindNode and #list >= 2 then return true end
  for i = 1, #list - 2 do
    if list[i] == list[i + 2] then return true end
  end
  return false
end
M.hasReversal = hasReversal

planTripRaw = function(pos, dir, L, stop, opts)
  local CL = M.closestLink(pos, {maxDist = 60, dir = dir})
  if not CL then return nil end
  local ahead = (CL.pb - CL.pa):dot(dir) >= 0 and CL.b or CL.a
  if opts and opts.startNode then ahead = opts.startNode end
  -- already on the stop link with the stop point ahead: drive straight to it
  if CL == L and (stop - pos):dot(dir) > 0 then
    return finishTrip({ahead}, pos, stop), nil, nil
  end
  -- No U-turn at the first node: from the node ahead the route may not run straight back down the
  -- road the car is on (a car cannot make that hairpin; it carries on and loops round instead).
  -- Only when that leaves no route at all (a dead end) is the way back allowed. 2026-11-01: random
  -- cruise targets behind the car made routes that reversed at the next node.
  local behindNode = (ahead == CL.a) and CL.b or CL.a
  local noBack = {}
  if opts and opts.avoid then for k in pairs(opts.avoid) do noBack[k] = true end end
  if not (opts and opts.startNode) then noBack[M.linkKey(ahead, behindNode)] = true end
  local best, fallback
  for _, o in ipairs({{L.a, L.b}, {L.b, L.a}}) do
    local first, last = o[1], o[2]
    if M.legal(L, first, last) then
      local p = M.path(ahead, first, noBack)
      if not p and not (opts and opts.startNode) then p = M.path(ahead, first, opts and opts.avoid) end
      if p then
        local list = {}
        for i = 1, #p do list[i] = p[i] end
        list[#list + 1] = last
        local trip = finishTrip(list, pos, stop)
        if trip then
          if hasReversal(list, not (opts and opts.startNode) and behindNode or nil) then
            if not fallback or trip.eta < fallback.eta then fallback = trip end
          elseif not best or trip.eta < best.eta then
            best = trip
          end
        end
      end
    end
  end
  if not best and fallback and not (opts and opts.noReversal) then best = fallback end
  return best, CL, ahead
end

-- A node is inside the service area when some road leaving it may be used.
local function nodeInside(gate, n, mn)
  local nc = mn[n]
  if not nc then return false end
  for nb, l in pairs(nc.links or {}) do
    local nn = mn[nb]
    if nn and edgeAllowed(gate, n, nb, nc, nn, l) then return true end
  end
  return false
end

-- A car outside the service area: the shortest legal way back in. Dijkstra over every road with the
-- gate ignored (one-way streets respected, public roads preferred) up to the first node that has a
-- road the service may use, then out along that road. Returns a trip like planTrip, or nil.
function M.reenterTrip(pos, dir)
  local gate = refreshGate()
  if not gate then return nil end
  local mn = nodes()
  local CL = M.closestLink(pos, {maxDist = 60, dir = dir})
  if not CL then return nil end
  local ahead = (CL.pb - CL.pa):dot(dir) >= 0 and CL.b or CL.a
  local dist, came, closed, open = {[ahead] = 0}, {}, {}, {}
  heapPush(open, 0, ahead)
  local goal, expanded = nil, 0
  while #open > 0 and expanded < 8000 do
    local cur = heapPop(open)
    if not closed[cur] then
      closed[cur] = true
      expanded = expanded + 1
      if nodeInside(gate, cur, mn) then
        goal = cur
        break
      end
      local nc = mn[cur]
      for nb, l in pairs(nc.links or {}) do
        local nn = mn[nb]
        if nn and not closed[nb] and (not l.oneWay or l.inNode == cur) then
          local cost = nc.pos:distance(nn.pos)
          if l.type == "private" or math.min(nc.radius or 3, nn.radius or 3) < M.MIN_RADIUS then cost = cost * 30 end
          local nd = dist[cur] + cost
          if not dist[nb] or nd < dist[nb] then
            dist[nb], came[nb] = nd, cur
            heapPush(open, nd, nb)
          end
        end
      end
    end
  end
  if not goal then return nil end
  local list = {goal}
  local c = goal
  while came[c] do
    c = came[c]
    table.insert(list, 1, c)
  end
  -- out along an allowed road from the goal node
  local ng = mn[goal]
  local out
  for nb, l in pairs(ng.links or {}) do
    local nn = mn[nb]
    if nn and edgeAllowed(gate, goal, nb, ng, nn, l) and (not l.oneWay or l.inNode == goal) then
      out = nb
      break
    end
  end
  if not out then return nil end
  list[#list + 1] = out
  return finishTrip(list, pos, ng.pos + (mn[out].pos - ng.pos) * 0.5)
end

-- Updates trip.idx / remaining / eta / offRoute for the car's current position.
function M.measure(trip, pos)
  local pts = trip.points
  local nseg = #pts - 1
  local bestI, bestD2 = trip.idx, math.huge
  for i = math.max(1, trip.idx - 1), math.min(nseg, trip.idx + 8) do
    local a, b = pts[i], pts[i + 1]
    local t = clamp(pos:xnormOnLine(a, b), 0, 1)
    local d2 = (a + (b - a) * t):squaredDistance(pos)
    if d2 < bestD2 - 0.01 then bestI, bestD2 = i, d2 end
  end
  trip.idx = bestI
  local nextPt = pts[bestI + 1]
  local partial = pos:distance(nextPt)
  local remaining = partial
  local eta = partial / trip.segV[bestI]
  for i = bestI + 1, nseg do
    local seg = pts[i]:distance(pts[i + 1])
    remaining = remaining + seg
    eta = eta + seg / trip.segV[i]
    if trip.junction[i] then eta = eta + INTERSECTION_DELAY end
  end
  trip.remaining = remaining
  trip.eta = eta + 4            -- the final approach and stop
  trip.offRoute = math.sqrt(bestD2)
  trip.len = trip.len or remaining
  return remaining, trip.eta
end

-- Seconds to add for slow vehicles waiting on the next 250 m of the route (within 3.2 m of the
-- route line, so cars parked at the kerb do not count): 6 s each, at most 45 s.
function M.queueDelay(trip, pos, selfId)
  local pts, acc = {pos}, 0
  for i = trip.idx + 1, #trip.points do
    local p = trip.points[i]
    acc = acc + pts[#pts]:distance(p)
    pts[#pts + 1] = p
    if acc > 250 then break end
  end
  local n = 0
  for oid, o in pairs(map.objects) do
    if oid ~= selfId and o.pos and o.vel and o.vel:length() < 2.5 then
      for k = 1, #pts - 1 do
        local a, b = pts[k], pts[k + 1]
        local t = clamp(o.pos:xnormOnLine(a, b), 0, 1)
        if (a + (b - a) * t):distance(o.pos) < 3.2 then
          n = n + 1
          break
        end
      end
    end
  end
  return math.min(45, n * 6)
end

-- metres to the right of travel for each trip segment: half a road on two-way roads (the car
-- keeps to its lane), zero on one-way roads. Cached on the trip.
function M.tripLaneOffsets(trip)
  if trip.laneOffsets then return trip.laneOffsets end
  local mn = nodes()
  local side = map.getRoadRules().rightHandDrive and -1 or 1
  local nn, out = trip.nodes, {}
  for i = 1, #trip.points - 1 do
    local a, b
    if i == 1 then a, b = nn[1], nn[2] else a, b = nn[i - 1], nn[i] end
    local off = 0
    local na, nb = a and mn[a], b and mn[b]
    if na and nb then
      local l = (na.links and na.links[b]) or (nb.links and nb.links[a])
      if l and not l.oneWay then off = math.min(na.radius or 3, nb.radius or 3) * 0.5 * side end
    end
    out[i] = off
  end
  trip.laneOffsets = out
  return out
end

-- the rest of the route as a flat {x1, y1, x2, y2, ...} list, starting at pos
function M.routeXY(trip, pos)
  local out = {math.floor(pos.x * 2 + 0.5) / 2, math.floor(pos.y * 2 + 0.5) / 2}
  for i = trip.idx + 1, #trip.points do
    local p = trip.points[i]
    out[#out + 1] = math.floor(p.x * 2 + 0.5) / 2
    out[#out + 1] = math.floor(p.y * 2 + 0.5) / 2
  end
  return out
end

-- ---------------------------------------------------------------------------------
-- hidden spots for fleet cars
-- ---------------------------------------------------------------------------------
-- true if pos is out of the camera's sight: far, outside the view cone, or behind geometry
function M.isHidden(pos, farEnough)
  local cam = core_camera.getPosition()
  local d = pos:distance(cam)
  if d > (farEnough or 450) then return true end
  if d < 1 then return false end
  local to = (pos - cam) * (1 / d)
  if core_camera.getForward():dot(to) < 0.45 then return true end
  local target = pos + vec3(0, 0, 1.2)
  local dir = (target - cam)
  local len = dir:length()
  dir = dir * (1 / len)
  return castRayStatic(cam, dir, len) < len - 2
end

local function nearAnyVehicle(pos, r)
  for _, v in ipairs(getAllVehicles()) do
    if v:getPosition():squaredDistance(pos) < r * r then return true end
  end
  return false
end

-- Returns pos, dir, up for a spot on a public road between minDist and maxDist from
-- origin, out of sight when closer than opts.farEnough, at least opts.spacing from every
-- position in opts.avoid, facing the legal direction in the correct lane.
function M.findSpot(origin, minDist, maxDist, opts)
  opts = opts or {}
  local idx = ensureIndex()
  local cands = {}
  for i, L in ipairs(idx.links) do
    if L.public and L.len >= 24 and allowed(L) and (not opts.component or M.component(L) == opts.component) then
      local mid = (L.pa + L.pb) * 0.5
      local d = mid:distance(origin)
      if d >= minDist and d <= maxDist then cands[#cands + 1] = i end
    end
  end
  -- shuffle
  for i = #cands, 2, -1 do
    local j = math.random(i)
    cands[i], cands[j] = cands[j], cands[i]
  end
  local mn = nodes()
  local legalSide = map.getRoadRules().rightHandDrive and -1 or 1
  local spacing2 = (opts.spacing or 0) ^ 2
  for k = 1, math.min(#cands, 80) do
    local L = idx.links[cands[k]]
    local t = 0.3 + math.random() * 0.4
    local p = L.pa + (L.pb - L.pa) * t
    local ok = true
    for _, a in ipairs(opts.avoid or {}) do
      if a:squaredDistance(p) < spacing2 then ok = false break end
    end
    if ok and not nearAnyVehicle(p, 25) and M.isHidden(p, opts.farEnough) then
      -- legal direction and lane
      local from, to = L.a, L.b
      if L.link.oneWay then
        from, to = L.link.inNode, L.link.outNode
      elseif math.random() < 0.5 then
        from, to = L.b, L.a
      end
      local pf, pt = mn[from].pos, mn[to].pos
      local dir = (pt - pf):normalized()
      local up = ((mn[from].normal or vec3(0, 0, 1)) + (mn[to].normal or vec3(0, 0, 1))):normalized()
      local right = dir:cross(up):normalized()
      local offset = 0
      if not L.link.oneWay then
        offset = L.radius * 0.5 * legalSide
      elseif L.radius >= 4.5 then
        offset = (math.random() < 0.5 and -0.5 or 0.5) * L.radius * 0.5
      end
      p = p + right * offset
      local h = be:getSurfaceHeightBelow(p + vec3(0, 0, 3))
      if h and h > -1e6 then p.z = h end
      return p, dir, up, L
    end
  end
  return nil
end

-- ---------------------------------------------------------------------------------
-- 2D map export
-- ---------------------------------------------------------------------------------
-- {level, bounds = {x0, y0, x1, y1}, segs = {x1, y1, x2, y2, width, class, ...}}
-- class 0 highway, 1 major road, 2 minor road, 3 private or dirt
function M.exportRoads()
  local idx = ensureIndex()
  local segs = {}
  local x0, y0, x1, y1 = math.huge, math.huge, -math.huge, -math.huge
  local function r2(v) return math.floor(v * 2 + 0.5) / 2 end
  for _, L in ipairs(idx.links) do
    local l = L.link
    local cls
    if not L.public then cls = 3
    elseif (l.speedLimit or 0) >= 22 then cls = 0
    elseif (l.drivability or 1) >= 0.9 and L.radius >= 3.5 then cls = 1
    else cls = 2 end
    local n = #segs
    segs[n + 1], segs[n + 2], segs[n + 3], segs[n + 4] = r2(L.pa.x), r2(L.pa.y), r2(L.pb.x), r2(L.pb.y)
    segs[n + 5], segs[n + 6] = r2(L.width), cls
    x0, y0 = math.min(x0, L.pa.x, L.pb.x), math.min(y0, L.pa.y, L.pb.y)
    x1, y1 = math.max(x1, L.pa.x, L.pb.x), math.max(y1, L.pa.y, L.pb.y)
  end
  return {level = getCurrentLevelIdentifier(), segs = segs, bounds = {x0, y0, x1, y1}}
end

return M
