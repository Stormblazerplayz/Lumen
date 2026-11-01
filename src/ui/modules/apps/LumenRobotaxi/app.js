// Lumen Robotaxi UI app (Frontier Studios).
// Talks to the GE extension lumen_robotaxi: it pushes "LumenRobotaxiUpdate" four times a
// second (ride state, ETA, fare, positions of you, your Lumen and the fleet, the route) and
// "LumenRobotaxiRoads" once per level (every road segment, for the 2D map). Buttons call the
// extension through bngApi.engineLua. The map's Lidar tab gets "LumenRobotaxiLidar" from
// lumen_lidar five times a second while it is open (lumen/lidar.lua, setUiEnabled).

// ---------------------------------------------------------------- lidar view geometry
// Pure functions (no DOM, no Angular) shared by the directive below and the node test
// testenv/lidar_ui_test.js. Points arrive in decimetres in the car's own frame: x to its right,
// y along its nose, origin in the middle of the car. On the canvas the car sits in the middle with
// its nose up, and `range` metres reach the nearer canvas edge, less PAD pixels.
var lumenLidarGeom = (function () {
  const RANGES = [20, 40, 60];             // m; + and - step through these, 40 to start
  const PAD = 14;                           // px between the range circle and the nearer canvas edge
  const IDLE = 'Lidar starts when your Lumen is on its way.';
  const num = (n) => (typeof n === 'number' && isFinite(n)) ? n : NaN;
  function view(cw, ch, range) {
    const r = Math.max(8, Math.min(cw, ch) / 2 - PAD);
    return { cw: cw, ch: ch, cx: cw / 2, cy: ch / 2, r: r, s: r / range, range: range };
  }
  // car frame (decimetres) to canvas pixels
  function toCanvas(xdm, ydm, v) {
    return [v.cx + xdm * 0.1 * v.s, v.cy - ydm * 0.1 * v.s];
  }
  // a marker beyond the picture's safe area (insets in px: l, t, r, b) is pinned to its edge on the
  // line from the car towards it; null when it fits
  function edge(px, py, v, ins) {
    const x0 = ins.l, x1 = v.cw - ins.r, y0 = ins.t, y1 = v.ch - ins.b;
    if (px >= x0 && px <= x1 && py >= y0 && py <= y1) return null;
    const dx = px - v.cx, dy = py - v.cy;
    let k = Infinity;
    if (dx > 0) k = Math.min(k, (x1 - v.cx) / dx); else if (dx < 0) k = Math.min(k, (x0 - v.cx) / dx);
    if (dy > 0) k = Math.min(k, (y1 - v.cy) / dy); else if (dy < 0) k = Math.min(k, (y0 - v.cy) / dy);
    k = Math.max(0, Math.min(1, k));
    const L = Math.hypot(dx, dy) || 1;
    return { x: v.cx + dx * k, y: v.cy + dy * k, ux: dx / L, uy: dy / L };
  }
  function distText(m) {
    if (!(m >= 0)) return '';
    if (m >= 995) return (m / 1000).toFixed(1) + ' km';
    if (m >= 100) return Math.round(m / 10) * 10 + ' m';
    return Math.round(m) + ' m';
  }
  // ground dots fade with distance, as the world view fades its teal (60 m: the sensor's reach)
  function groundAlpha(m) { return 0.9 - 0.65 * Math.min(1, Math.max(0, m) / 60); }
  // a "LumenRobotaxiLidar" payload made safe to draw: an empty Lua table arrives as {}, not [],
  // and anything missing gets the Lumen's own defaults
  function snapshot(d) {
    if (!d || typeof d !== 'object') return { none: true, reason: IDLE };
    if (d.none) return { none: true, reason: (typeof d.reason === 'string' && d.reason) ? d.reason : IDLE };
    const car = d.car || {}, you = d.you || null;
    const hl = num(car.hl) > 0.3 ? car.hl : 2.3, hw = num(car.hw) > 0.2 ? car.hw : 0.95;
    const yx = you ? num(you.x) : NaN, yy = you ? num(you.y) : NaN;
    return {
      none: false, pts: Array.isArray(d.pts) ? d.pts : [], range: num(d.range) > 0 ? d.range : 60, car: { hl: hl, hw: hw },
      you: (isFinite(yx) && isFinite(yy)) ? { x: yx, y: yy, d: num(you.d) >= 0 ? you.d : Math.hypot(yx, yy) / 10 } : null,
      speed: num(d.speed) > 0 ? d.speed : 0,
    };
  }
  return { RANGES: RANGES, PAD: PAD, IDLE: IDLE, view: view, toCanvas: toCanvas, edge: edge, distText: distText,
           groundAlpha: groundAlpha, snapshot: snapshot };
})();
// node (the unit test) gets the geometry; in the game this line does nothing
if (typeof module === 'object' && module && module.exports && typeof window === 'undefined') module.exports = lumenLidarGeom;

angular.module('beamng.apps')
.directive('lumenRobotaxi', [function () {
  return {
    template: `
<div class="lrx">
  <style>
    .lrx { width:100%; height:100%; box-sizing:border-box; padding:12px 14px 10px; border-radius:16px;
           background: linear-gradient(165deg, rgba(13,15,19,0.94), rgba(20,24,30,0.92));
           border:1px solid rgba(24,214,196,0.20); color:#e9ecef; font-family:'Overpass','Noto Sans',sans-serif;
           display:flex; flex-direction:column; gap:9px; overflow:hidden; pointer-events:auto;
           box-shadow: 0 10px 30px rgba(0,0,0,0.38); }
    .lrx-head { display:flex; align-items:center; justify-content:space-between; }
    .lrx-mark { font-size:20px; font-weight:600; letter-spacing:0.2px; }
    .lrx-mark i { display:inline-block; width:7px; height:7px; border-radius:50%; background:#18d6c4; margin-left:2px; vertical-align:super; }
    .lrx-pill { font-size:10.5px; padding:3px 9px; border-radius:999px; background:rgba(255,255,255,0.07); color:#aab2bd;
                text-transform:uppercase; letter-spacing:0.7px; white-space:nowrap; }
    .lrx-pill.live { background:rgba(24,214,196,0.16); color:#5ff0e0; }
    .lrx-pill.warn { background:rgba(255,184,77,0.16); color:#ffcf85; }
    .lrx-mapwrap { position:relative; flex:1 1 auto; min-height:150px; border-radius:12px; overflow:hidden;
                   background:#0d1014; border:1px solid rgba(255,255,255,0.05); }
    .lrx-mapwrap canvas { position:absolute; left:0; top:0; width:100%; height:100%; display:block; }
    .lrx-zoom { position:absolute; right:8px; bottom:8px; display:flex; flex-direction:column; gap:4px; z-index:2; }
    .lrx-zoom div { width:26px; height:26px; line-height:25px; text-align:center; font-size:16px; color:#cfeeea; background:rgba(14,20,26,0.74);
                    border:1px solid rgba(24,214,196,0.35); border-radius:7px; cursor:pointer; user-select:none; }
    .lrx-zoom div:hover { background:rgba(24,214,196,0.24); }
    .lrx-mapnote { position:absolute; left:0; right:0; top:50%; transform:translateY(-50%); text-align:center;
                   font-size:12px; color:#6f7a86; pointer-events:none; }
    .lrx-tabs { position:absolute; left:8px; top:8px; z-index:3; display:flex; gap:2px; padding:2px; border-radius:9px;
                background:rgba(14,20,26,0.78); border:1px solid rgba(24,214,196,0.28); }
    .lrx-tabs div { padding:3px 10px; border-radius:7px; font-size:11.5px; line-height:16px; color:#8e97a3; cursor:pointer;
                    user-select:none; transition: background 0.15s ease, color 0.15s ease; }
    .lrx-tabs div:hover { color:#e9fbf8; }
    .lrx-tabs div.on { background:rgba(24,214,196,0.22); color:#bff8f1; }
    .lrx-zoom div.off { opacity:0.35; cursor:default; }
    .lrx-zoom div.off:hover { background:rgba(14,20,26,0.74); }
    .lrx-lnote { position:absolute; left:50%; top:50%; transform:translate(-50%, -50%); max-width:78%; box-sizing:border-box;
                 text-align:center; font-size:12px; line-height:1.35; color:#aab2bd; pointer-events:none; z-index:2;
                 background:rgba(11,14,18,0.84); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:7px 11px; }
    .lrx-main { font-size:15.5px; line-height:1.3; color:#f3f5f7; font-weight:600; }
    .lrx-sub { font-size:12px; color:#8e97a3; line-height:1.35; margin-top:2px; }
    .lrx-meter { display:flex; gap:8px; }
    .lrx-meter div { flex:1; background:rgba(255,255,255,0.045); border-radius:10px; padding:7px 9px; min-width:0; }
    .lrx-meter b { display:block; font-size:16px; color:#fff; font-weight:600; white-space:nowrap; }
    .lrx-meter span { font-size:10.5px; color:#8e97a3; text-transform:uppercase; letter-spacing:0.5px; }
    .lrx-btn { border:none; border-radius:11px; padding:9px 12px; font-size:13.5px; font-weight:600; cursor:pointer;
               background:#18d6c4; color:#06201d; transition: transform 0.08s ease, filter 0.15s ease; }
    .lrx-btn:hover { filter:brightness(1.08); }
    .lrx-btn:active { transform:scale(0.98); }
    .lrx-btn[disabled] { background:rgba(255,255,255,0.08); color:#6f7a86; cursor:default; filter:none; transform:none; }
    .lrx-btn.ghost { background:rgba(255,255,255,0.07); color:#d7dce2; font-weight:500; }
    .lrx-row { display:flex; gap:8px; }
    .lrx-row .lrx-btn { flex:1; }
    .lrx-list { display:flex; flex-direction:column; gap:5px; overflow-y:auto; max-height:118px; padding-right:2px; }
    .lrx-dest { display:flex; justify-content:space-between; align-items:center; padding:7px 10px; border-radius:9px;
                background:rgba(255,255,255,0.05); cursor:pointer; font-size:13px; }
    .lrx-dest:hover { background:rgba(24,214,196,0.14); }
    .lrx-dest.on { background:rgba(24,214,196,0.18); color:#bff8f1; }
    .lrx-dest.unserved { opacity:0.45; }
    .lrx-dest em { font-style:normal; font-size:11px; color:#8e97a3; margin-left:8px; white-space:nowrap; }
    .lrx-think { background:rgba(24,214,196,0.06); border:1px solid rgba(24,214,196,0.16); border-radius:10px; padding:7px 9px; }
    .lrx-think-head { font-size:10.5px; color:#5ff0e0; text-transform:uppercase; letter-spacing:0.7px; display:flex; align-items:center; gap:6px; }
    .lrx-think-dot { width:6px; height:6px; border-radius:50%; background:#18d6c4; animation: lrxPulse 1.4s ease-in-out infinite; }
    @keyframes lrxPulse { 0%,100% { opacity:0.35; transform:scale(0.85); } 50% { opacity:1; transform:scale(1.15); } }
    .lrx-think-now { font-size:12.5px; color:#e9fbf8; line-height:1.35; margin-top:4px; }
    .lrx-think-now.wait { color:#ffd9a0; }
    .lrx-think-now.warn { color:#ffb3a8; }
    .lrx-think-old { font-size:11px; color:#7f8a95; line-height:1.3; margin-top:3px; }
    .lrx-seats { display:flex; gap:5px; }
    .lrx-seat { flex:1; text-align:center; padding:6px 4px; border-radius:9px; font-size:11.5px; cursor:pointer;
                background:rgba(255,255,255,0.05); color:#c9d0d8; white-space:nowrap; }
    .lrx-seat:hover { background:rgba(24,214,196,0.14); }
    .lrx-seat.on { background:rgba(24,214,196,0.22); color:#bff8f1; }
    .lrx-foot { display:flex; justify-content:space-between; align-items:center; }
    .lrx-link { font-size:12px; color:#8e97a3; cursor:pointer; }
    .lrx-link.on { color:#5ff0e0; }
    .lrx-brain { display:flex; align-items:center; gap:10px; }
    .lrx-brain .lrx-seats { flex:1; }
    .lrx-seat.off { opacity:0.45; }
  </style>
  <div class="lrx-head">
    <div class="lrx-mark">robotaxi<i></i></div>
    <div class="lrx-pill" ng-class="pillClass()">{{ label() }}</div>
  </div>

  <div class="lrx-mapwrap">
    <canvas ng-show="tab !== 'lidar'"></canvas>
    <canvas class="lrx-lidar" ng-show="tab === 'lidar'"></canvas>
    <div class="lrx-tabs"><div ng-class="{on: tab !== 'lidar'}" ng-click="setTab('map')" title="The 2D map">Map</div><div ng-class="{on: tab === 'lidar'}" ng-click="setTab('lidar')" title="What your Lumen's lidar sees, live">Lidar</div></div>
    <div class="lrx-zoom" ng-show="tab !== 'lidar'"><div ng-click="zoomBy(1.6)" title="Zoom in">+</div><div ng-click="zoomBy(1 / 1.6)" title="Zoom out">&minus;</div><div ng-click="zoomFit()" title="Fit the route">&#10530;</div></div>
    <div class="lrx-zoom" ng-show="tab === 'lidar'"><div ng-click="lidarZoom(-1)" ng-class="{off: !lidarCan(-1)}" title="Closer">+</div><div ng-click="lidarZoom(1)" ng-class="{off: !lidarCan(1)}" title="Farther">&minus;</div></div>
    <div class="lrx-mapnote" ng-if="tab !== 'lidar' && mapNote()">{{ mapNote() }}</div>
    <div class="lrx-lnote" ng-if="tab === 'lidar' && lidarNote">{{ lidarNote }}</div>
  </div>

  <div>
    <div class="lrx-main">{{ headline() }}</div>
    <div class="lrx-sub" ng-if="subline()">{{ subline() }}</div>
  </div>

  <div class="lrx-meter" ng-if="s.state === 'arriving' || s.state === 'riding' || s.state === 'arrived' || s.state === 'departing' || s.state === 'pulledOver'">
    <div ng-if="s.state === 'arriving' || s.state === 'riding'"><b>{{ etaText() }}</b><span>eta</span></div>
    <div ng-if="s.state === 'arriving' || s.state === 'riding'"><b>{{ distText() }}</b><span>{{ s.state === 'arriving' ? 'away' : 'to go' }}</span></div>
    <div ng-if="s.state === 'arrived' || s.state === 'departing' || s.state === 'pulledOver'"><b>{{ (s.rideKm || 0).toFixed(1) }} km</b><span>trip</span></div>
    <div ng-if="s.state !== 'arriving'"><b>{{ '$' + (s.fare || 0).toFixed(2) }}</b><span>fare</span></div>
  </div>

  <div class="lrx-list" ng-if="(s.state === 'boarded' || s.state === 'riding' || s.state === 'arrived' || s.state === 'pulledOver') && s.destinations">
    <div class="lrx-dest" ng-repeat="d in s.destinations" ng-click="go(d.id)" ng-class="{on: d.name === s.destName, unserved: d.served === false}">
      <span>{{ d.name }}</span><em ng-if="d.km">{{ d.km.toFixed(1) }} km</em>
    </div>
  </div>

  <div class="lrx-sub" ng-if="(s.state === 'boarded' || s.state === 'riding' || s.state === 'arrived' || s.state === 'pulledOver') && s.destinations">Or tap the map to go anywhere in the service area.</div>

  <div class="lrx-think" ng-if="s.thoughts && s.thoughts.length && s.state !== 'idle' && s.state !== 'departing'">
    <div class="lrx-think-head"><span class="lrx-think-dot"></span>Lumen is thinking</div>
    <div class="lrx-think-now" ng-class="s.thoughts[s.thoughts.length - 1].kind">{{ s.thoughts[s.thoughts.length - 1].text }}</div>
    <div class="lrx-think-old" ng-repeat="th in s.thoughts.slice(0, -1).slice(-2).reverse()">{{ th.text }}</div>
  </div>

  <div class="lrx-seats" ng-if="s.seats">
    <div class="lrx-seat" ng-repeat="st in s.seats" ng-class="{on: st.id === s.seat}" ng-click="sit(st.id)">{{ seatLabel(st) }}</div>
  </div>

  <div class="lrx-row" ng-if="s.state !== 'departing'">
    <button class="lrx-btn" ng-if="s.state === 'idle'" ng-disabled="!s.enabled || !s.active" ng-click="call()">Call a Lumen</button>
    <button class="lrx-btn" ng-if="s.state === 'waiting'" ng-click="board()">Board</button>
    <button class="lrx-btn" ng-if="s.state === 'arrived'" ng-click="exit()">Get out</button>
    <button class="lrx-btn ghost" ng-if="s.state === 'boarded'" ng-click="exit()">Get out</button>
    <button class="lrx-btn" ng-if="s.state === 'riding'" ng-click="pullOver()">Pull over</button>
    <button class="lrx-btn" ng-if="s.state === 'pulledOver'" ng-click="exit()">Get out</button>
    <button class="lrx-btn ghost" ng-if="s.state === 'pulledOver'" ng-click="continueTrip()">Continue trip</button>
    <button class="lrx-btn ghost" ng-if="s.state === 'dispatching' || s.state === 'arriving' || s.state === 'waiting'" ng-click="cancel()">Cancel</button>
  </div>

  <div class="lrx-brain">
    <span class="lrx-sub">Driver</span>
    <div class="lrx-seats">
      <div class="lrx-seat" ng-class="{on: s.brain !== 'nn'}" ng-click="setBrain('drive')" title="Lumen Drive: the rule-based driver">Drive</div>
      <div class="lrx-seat" ng-class="{on: s.brain === 'nn', off: s.nnReady === false}" ng-click="setBrain('nn')" title="Lumen NN: the neural-network driver">NN</div>
    </div>
  </div>

  <div class="lrx-foot">
    <span class="lrx-sub">{{ fleetText() }}</span>
    <span>
      <span class="lrx-link" ng-class="{on: s.arpath}" ng-click="arpath()">AR path</span>
      <span class="lrx-sub"> · </span>
      <span class="lrx-link" ng-class="{on: s.lidar}" ng-click="lidar()">{{ s.lidar ? 'Lidar on' : 'Lidar view' }}</span>
    </span>
  </div>
</div>`,
    replace: true,
    restrict: 'EA',
    link: function (scope, element) {
      scope.s = { state: 'idle', fleet: [] };
      scope.tab = 'map';                       // the segmented control on the map: 'map' or 'lidar'
      const lua = (code) => bngApi.engineLua('if extensions.lumen_robotaxi then ' + code + ' end');
      scope.call = () => lua('extensions.lumen_robotaxi.callTaxi()');
      scope.board = () => lua('extensions.lumen_robotaxi.board()');
      scope.exit = () => lua('extensions.lumen_robotaxi.exitTaxi()');
      scope.cancel = () => lua('extensions.lumen_robotaxi.cancel()');
      scope.lidar = () => lua('extensions.lumen_robotaxi.toggleLidar()');
      scope.arpath = () => lua('extensions.lumen_robotaxi.toggleArPath()');
      scope.setBrain = (name) => lua('extensions.lumen_robotaxi.setBrain(' + JSON.stringify(name) + ')');
      scope.go = (id) => lua('extensions.lumen_robotaxi.setDestination(' + JSON.stringify(id) + ')');
      scope.sit = (id) => lua('extensions.lumen_robotaxi.setSeat(' + JSON.stringify(id) + ')');
      scope.pullOver = () => lua('extensions.lumen_robotaxi.pullOver()');
      scope.continueTrip = () => lua('extensions.lumen_robotaxi.continueTrip()');
      const SEAT_SHORT = { rear_right: 'Rear R', rear_left: 'Rear L', front_passenger: 'Front R', driver: 'Driver' };
      scope.seatLabel = (st) => SEAT_SHORT[st.id] || st.name;

      // ---------------------------------------------------------------- text
      const LABELS = { idle: 'ready', dispatching: 'finding a car', arriving: 'on the way', waiting: 'arrived',
                       boarded: 'aboard', riding: 'driving itself', arrived: 'destination', departing: 'thank you',
                       pulling: 'pulling over', pulledOver: 'pulled over' };
      scope.label = () => (!scope.s.enabled && scope.s.state === 'idle') ? 'paused' : (LABELS[scope.s.state] || scope.s.state);
      scope.pillClass = () => ({ live: scope.s.state !== 'idle' && scope.s.state !== 'dispatching', warn: scope.s.state === 'dispatching' || !scope.s.enabled });
      const mins = (t) => t < 50 ? '<1 min' : Math.max(1, Math.round(t / 60)) + ' min';
      scope.etaText = () => mins(scope.s.eta || 0);
      scope.distText = () => {
        const d = scope.s.remaining || 0;
        return d >= 1000 ? (d / 1000).toFixed(1) + ' km' : Math.round(d / 10) * 10 + ' m';
      };
      scope.headline = () => {
        const s = scope.s;
        switch (s.state) {
          case 'idle':
            if (!s.enabled) return 'Robotaxi service is paused';
            if (!s.active) return 'No robotaxis on this map';
            return 'Need a ride?';
          case 'dispatching': return 'Finding you a Lumen…';
          case 'arriving': return s.resume ? 'A new Lumen is on the way' : 'Your Lumen is on the way';
          case 'waiting': return 'Your Lumen is here';
          case 'boarded': return 'Where to?';
          case 'riding': return 'Heading to ' + (s.destName || 'your destination');
          case 'arrived': return "You've arrived" + (s.destName ? ' at ' + s.destName : '');
          case 'pulling': return 'Pulling over';
          case 'pulledOver': return 'Pulled over';
          case 'departing': return 'Thanks for riding';
          default: return '';
        }
      };
      scope.subline = () => {
        const s = scope.s;
        switch (s.state) {
          case 'idle':
            if (!s.enabled) return 'Turn it on from the Robotaxi radial menu.';
            if (!s.active) return 'Robotaxis run in Freeroam on maps with roads.';
            if (s.geo && s.geo.restricted && !s.geo.inside)
              return 'You are outside the Lumen service area' + (s.geo.zone ? ' (' + s.geo.zone + ' opens with ' + s.geo.opensWith + ').' : '.') + ' Driving a Lumen yourself works anywhere.';
            if (!s.available) return 'Lumens are heading into the area.';
            return s.available + (s.available === 1 ? ' Lumen' : ' Lumens') + ' nearby' +
                   (s.nearestEta ? ', the closest about ' + mins(s.nearestEta) + ' away.' : '.');
          case 'dispatching': return 'Every Lumen is busy or being serviced. The next free one is yours.';
          case 'arriving':
            if (s.pickup && s.pickup.remote) return 'You are outside the service area. Walk to the pin: your Lumen will meet you there.';
            if (s.traffic) return 'Slow traffic on the way. The ETA includes it.';
            return s.resume && s.destName ? 'It will finish your trip to ' + s.destName + '.' : 'It will pull up right next to you.';
          case 'waiting': return 'Walk up and get in, or press Board.';
          case 'boarded': return s.resume ? 'Continuing your trip…' : 'Pick a destination below. Tap a seat to move.';
          case 'riding': return s.traffic ? 'Slow traffic ahead. The ETA includes it.' : '';
          case 'arrived': return 'Get out, or pick another destination below.';
          case 'pulling': return 'Finding a safe spot at the curb.';
          case 'pulledOver': return (s.destName ? 'Get out here, continue to ' + s.destName : 'Get out here, continue your trip') + ', or pick another destination.';
          case 'departing': return 'Your Lumen is heading back out.';
          default: return '';
        }
      };
      scope.fleetText = () => {
        const s = scope.s;
        if (!s.active) return 'Lumen · Frontier';
        return (s.fleetSize || 4) + ' Lumens · ' + (s.available || 0) + ' free';
      };
      scope.mapNote = () => {
        if (!roads) return scope.s.active ? 'Loading map…' : '';
        if (!roads.count) return 'No road map for this level';
        return '';
      };

      // ---------------------------------------------------------------- map data
      let roads = null;            // {level, count, cls: [Float32Array x4 per class], width: [m], bounds}
      let zones = null;            // the service area: {openTier, zones: [{name, tier, open, polys: [[x, y, ...]]}], tiers}
      const CLASS_STYLE = [         // highway, major, minor, private/dirt
        { color: '#3d4a58', min: 1.8, max: 13 },
        { color: '#323c48', min: 1.4, max: 10 },
        { color: '#272f39', min: 1.1, max: 7 },
        { color: '#1d232a', min: 0.8, max: 4 },
      ];
      function loadRoads(data) {
        if (!data || !data.segs) { roads = { count: 0 }; return; }
        const n = data.segs.length / 6;
        const buckets = [[], [], [], []], widths = [0, 0, 0, 0], counts = [0, 0, 0, 0];
        for (let i = 0; i < n; i++) {
          const o = i * 6, c = Math.max(0, Math.min(3, data.segs[o + 5] | 0));
          buckets[c].push(data.segs[o], data.segs[o + 1], data.segs[o + 2], data.segs[o + 3]);
          widths[c] += data.segs[o + 4]; counts[c]++;
        }
        roads = { level: data.level, count: n, cls: buckets.map((b) => new Float32Array(b)),
                  width: widths.map((w, i) => counts[i] ? w / counts[i] : 6), bounds: data.bounds };
      }

      // smoothed marker positions: each update becomes a short glide
      const markers = {};
      function track(key, x, y, dx, dy, now) {
        const m = markers[key];
        if (!m) { markers[key] = { x0: x, y0: y, x1: x, y1: y, dx: dx, dy: dy, t: now }; return; }
        const cur = pos(m, now);
        m.x0 = cur.x; m.y0 = cur.y; m.x1 = x; m.y1 = y; m.dx = dx; m.dy = dy; m.t = now;
      }
      function pos(m, now) {
        const k = Math.min(1, (now - m.t) / 260);
        return { x: m.x0 + (m.x1 - m.x0) * k, y: m.y0 + (m.y1 - m.y0) * k };
      }

      scope.$on('LumenRobotaxiUpdate', function (event, data) {
        const now = performance.now();
        data = data || { state: 'idle' };
        if (data.player) track('player', data.player.x, data.player.y, data.player.dx, data.player.dy, now);
        const seen = {};
        (Array.isArray(data.fleet) ? data.fleet : Object.values(data.fleet || {})).forEach(function (c, i) {
          const key = c.mine ? 'mine' : 'car' + i;
          seen[key] = true;
          track(key, c.x, c.y, c.dx, c.dy, now);
          markers[key].mine = c.mine;
        });
        Object.keys(markers).forEach(function (k) { if (k !== 'player' && !seen[k]) delete markers[k]; });
        scope.$evalAsync(function () { scope.s = data; });
      });
      scope.$on('LumenRobotaxiRoads', function (event, data) { loadRoads(data); });
      scope.$on('LumenRobotaxiZones', function (event, data) { zones = (data && Array.isArray(data.zones) && data.zones.length) ? data : null; });

      // ---------------------------------------------------------------- map drawing
      const canvas = element[0].querySelector('canvas');
      const ctx = canvas.getContext('2d');
      const view = { cx: 0, cy: 0, scale: 0.25, ready: false };
      // what the rider did to the automatic fit: zoom (wheel or buttons) and pan (drag). Storm, 2026-11-01:
      // "I can't zoom out in the tiny map". The fit still follows the route; this is an offset on top of it.
      const user = { zoom: 1, panX: 0, panY: 0, moved: false };
      let cw = 0, ch = 0, dpr = 1, raf = null, lastDraw = 0;

      function autoView() {
        const s = scope.s, pts = [];
        const add = (x, y) => { if (isFinite(x) && isFinite(y)) pts.push([x, y]); };
        const p = markers.player && pos(markers.player, performance.now());
        const mine = markers.mine && pos(markers.mine, performance.now());
        if (p) add(p.x, p.y);
        if (mine) add(mine.x, mine.y);
        if (s.pickup && (s.state === 'arriving' || s.state === 'dispatching')) add(s.pickup.x, s.pickup.y);
        if (s.dest && (s.state === 'riding' || s.state === 'boarded' || s.state === 'arrived')) add(s.dest.x, s.dest.y);
        if (s.route && s.state !== 'waiting') for (let i = 0; i < s.route.length; i += 2) add(s.route[i], s.route[i + 1]);
        if (s.state === 'idle' && p) {
          Object.keys(markers).forEach(function (k) {
            if (k.indexOf('car') === 0) {
              const c = pos(markers[k], performance.now());
              if (Math.hypot(c.x - p.x, c.y - p.y) < 1300) add(c.x, c.y);
            }
          });
        }
        if (!pts.length) return null;
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        pts.forEach(function (q) { x0 = Math.min(x0, q[0]); y0 = Math.min(y0, q[1]); x1 = Math.max(x1, q[0]); y1 = Math.max(y1, q[1]); });
        const minSpan = (s.state === 'waiting') ? 140 : 220;
        const spanX = Math.max(minSpan, (x1 - x0) * 1.25), spanY = Math.max(minSpan, (y1 - y0) * 1.25);
        const scale = Math.min(cw / spanX, ch / spanY, 4);
        return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, scale: Math.max(scale, cw / 8000) };
      }

      function targetView() {
        const a = autoView();
        if (!a) return null;
        return { cx: a.cx + user.panX, cy: a.cy + user.panY, scale: Math.max(a.scale * user.zoom, cw / 30000) };
      }

      // zoom by `factor` keeping the world point under the screen point (mx, my) where it is
      function zoomAt(factor, mx, my) {
        const a = autoView();
        if (!a || !view.ready || !view.sc) return;
        const sc2 = Math.max(a.scale * Math.min(40, Math.max(0.03, user.zoom * factor)), cw / 30000);
        const wx = (mx - view.ox) / view.sc, wy = (view.oy - my) / view.sc;
        user.zoom = sc2 / a.scale;
        user.panX = (wx - (mx - cw / 2) / sc2) - a.cx;
        user.panY = (wy + (my - ch / 2) / sc2) - a.cy;
      }
      scope.zoomBy = function (f) { zoomAt(f, cw / 2, ch / 2); };
      scope.zoomFit = function () { user.zoom = 1; user.panX = 0; user.panY = 0; };

      function niceScale(pxPerM) {
        const steps = [20, 50, 100, 200, 500, 1000, 2000, 5000];
        for (let i = 0; i < steps.length; i++) if (steps[i] * pxPerM >= 56) return steps[i];
        return steps[steps.length - 1];
      }

      function chevron(x, y, dx, dy, size, fill, stroke, glow) {
        const len = Math.hypot(dx, dy) || 1;
        const ux = dx / len, uy = -dy / len;            // screen space: y down
        const px = -uy, py = ux;
        ctx.save();
        if (glow) { ctx.shadowColor = glow; ctx.shadowBlur = 12; }
        ctx.beginPath();
        ctx.moveTo(x + ux * size, y + uy * size);
        ctx.lineTo(x - ux * size * 0.75 + px * size * 0.72, y - uy * size * 0.75 + py * size * 0.72);
        ctx.lineTo(x - ux * size * 0.35, y - uy * size * 0.35);
        ctx.lineTo(x - ux * size * 0.75 - px * size * 0.72, y - uy * size * 0.75 - py * size * 0.72);
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
        if (stroke) { ctx.shadowBlur = 0; ctx.lineWidth = 1.4; ctx.strokeStyle = stroke; ctx.stroke(); }
        ctx.restore();
      }

      function pin(x, y, color) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.bezierCurveTo(x - 7, y - 9, x - 7, y - 17, x, y - 17);
        ctx.bezierCurveTo(x + 7, y - 17, x + 7, y - 9, x, y);
        ctx.fillStyle = '#f4f7f9';
        ctx.shadowColor = 'rgba(0,0,0,0.5)'; ctx.shadowBlur = 6;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.beginPath(); ctx.arc(x, y - 11.5, 3.2, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
        ctx.restore();
      }

      function draw(now) {
        raf = requestAnimationFrame(draw);
        if (scope.tab === 'lidar') return;             // the map is hidden behind the Lidar tab: no map work
        if (now - lastDraw < 30) return;               // about 30 fps is plenty for a map
        lastDraw = now;
        const wrap = canvas.parentNode;
        const w = wrap.clientWidth, h = wrap.clientHeight;
        if (!w || !h) return;
        dpr = window.devicePixelRatio || 1;
        if (w !== cw || h !== ch || canvas.width !== Math.round(w * dpr)) {
          cw = w; ch = h;
          canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const t = targetView();
        if (t) {
          if (!view.ready) { view.cx = t.cx; view.cy = t.cy; view.scale = t.scale; view.ready = true; }
          const k = 0.12;
          view.cx += (t.cx - view.cx) * k;
          view.cy += (t.cy - view.cy) * k;
          view.scale = Math.exp(Math.log(view.scale) + (Math.log(t.scale) - Math.log(view.scale)) * k);
        }
        const sc = view.scale, ox = cw / 2 - view.cx * sc, oy = ch / 2 + view.cy * sc;
        const SX = (x) => x * sc + ox, SY = (y) => -y * sc + oy;
        view.ox = ox; view.oy = oy; view.sc = sc;

        // background
        const g = ctx.createRadialGradient(cw / 2, ch / 2, 10, cw / 2, ch / 2, Math.max(cw, ch) * 0.75);
        g.addColorStop(0, '#11161b'); g.addColorStop(1, '#0b0e12');
        ctx.fillStyle = g; ctx.fillRect(0, 0, cw, ch);

        // roads, back to front: private, minor, major, highway
        if (roads && roads.count && view.ready) {
          const wx0 = (0 - ox) / sc - 30, wx1 = (cw - ox) / sc + 30;
          const wy0 = (oy - ch) / sc - 30, wy1 = oy / sc + 30;
          ctx.lineCap = 'round'; ctx.lineJoin = 'round';
          for (let c = 3; c >= 0; c--) {
            const a = roads.cls[c], st = CLASS_STYLE[c];
            ctx.beginPath();
            for (let i = 0; i < a.length; i += 4) {
              const x1 = a[i], y1 = a[i + 1], x2 = a[i + 2], y2 = a[i + 3];
              if ((x1 < wx0 && x2 < wx0) || (x1 > wx1 && x2 > wx1) || (y1 < wy0 && y2 < wy0) || (y1 > wy1 && y2 > wy1)) continue;
              ctx.moveTo(SX(x1), SY(y1)); ctx.lineTo(SX(x2), SY(y2));
            }
            ctx.lineWidth = Math.max(st.min, Math.min(st.max, roads.width[c] * sc));
            ctx.strokeStyle = st.color;
            ctx.stroke();
          }
        }

        // the service area: open zones glow softly, closed ones are dashed and say which tier opens them
        if (zones && Array.isArray(zones.zones) && view.ready) {
          zones.zones.forEach(function (z) {
            (Array.isArray(z.polys) ? z.polys : []).forEach(function (flat) {
              ctx.beginPath();
              for (let i = 0; i < flat.length; i += 2) {
                if (i === 0) ctx.moveTo(SX(flat[i]), SY(flat[i + 1])); else ctx.lineTo(SX(flat[i]), SY(flat[i + 1]));
              }
              ctx.closePath();
              if (z.open) {
                ctx.fillStyle = 'rgba(24,214,196,0.07)'; ctx.fill();
                ctx.setLineDash([]); ctx.lineWidth = 1.4; ctx.strokeStyle = 'rgba(24,214,196,0.55)'; ctx.stroke();
              } else {
                ctx.setLineDash([5, 5]); ctx.lineWidth = 1.1; ctx.strokeStyle = 'rgba(142,151,163,0.55)'; ctx.stroke();
                ctx.setLineDash([]);
              }
            });
          });
          ctx.font = '10px Overpass, Noto Sans, sans-serif'; ctx.textAlign = 'center';
          zones.zones.forEach(function (z) {
            const f = Array.isArray(z.polys) ? z.polys[0] : null;
            if (!f || f.length < 6) return;
            let cx = 0, cy = 0, x0 = Infinity, x1 = -Infinity;
            for (let i = 0; i < f.length; i += 2) { cx += f[i]; cy += f[i + 1]; x0 = Math.min(x0, f[i]); x1 = Math.max(x1, f[i]); }
            cx /= f.length / 2; cy /= f.length / 2;
            if ((x1 - x0) * sc < 90) return;
            ctx.fillStyle = z.open ? 'rgba(95,240,224,0.75)' : 'rgba(142,151,163,0.7)';
            ctx.fillText(z.open ? z.name : z.name + ' · tier ' + z.tier, SX(cx), SY(cy));
          });
          ctx.textAlign = 'start';
        }

        const s = scope.s;
        canvas.style.cursor = (s.state === 'boarded' || s.state === 'riding' || s.state === 'arrived' || s.state === 'pulledOver') ? 'crosshair' : 'default';
        // route: soft glow under a crisp line
        if (s.route && s.route.length >= 4 && s.state !== 'waiting') {
          ctx.lineCap = 'round'; ctx.lineJoin = 'round';
          ctx.beginPath();
          const mine = markers.mine ? pos(markers.mine, now) : null;
          ctx.moveTo(SX(mine ? mine.x : s.route[0]), SY(mine ? mine.y : s.route[1]));
          for (let i = 2; i < s.route.length; i += 2) ctx.lineTo(SX(s.route[i]), SY(s.route[i + 1]));
          ctx.strokeStyle = 'rgba(24,214,196,0.22)'; ctx.lineWidth = 9; ctx.stroke();
          ctx.strokeStyle = '#18d6c4'; ctx.lineWidth = 3.2; ctx.stroke();
        }

        // pickup and destination pins
        if (s.pickup && (s.state === 'arriving' || s.state === 'dispatching' || s.state === 'waiting')) pin(SX(s.pickup.x), SY(s.pickup.y), '#18d6c4');
        if (s.dest && (s.state === 'boarded' || s.state === 'riding' || s.state === 'arrived' || s.resume)) pin(SX(s.dest.x), SY(s.dest.y), '#ff8a5c');

        // other fleet cars, then yours
        Object.keys(markers).forEach(function (k) {
          if (k.indexOf('car') !== 0) return;
          const m = markers[k], q = pos(m, now);
          chevron(SX(q.x), SY(q.y), m.dx, m.dy, 6.5, 'rgba(24,214,196,0.55)', null, null);
        });
        const riding = s.state === 'boarded' || s.state === 'riding' || s.state === 'arrived';
        if (markers.mine) {
          const m = markers.mine, q = pos(m, now);
          chevron(SX(q.x), SY(q.y), m.dx, m.dy, 10, '#18d6c4', '#e9fffd', 'rgba(24,214,196,0.85)');
        }

        // you (hidden while you ride, your Lumen is you)
        if (markers.player && !riding) {
          const q = pos(markers.player, now), x = SX(q.x), y = SY(q.y);
          const ph = (now % 1700) / 1700;
          ctx.beginPath(); ctx.arc(x, y, 5 + ph * 11, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(255,255,255,' + (0.30 * (1 - ph)).toFixed(3) + ')'; ctx.fill();
          ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2);
          ctx.fillStyle = '#ffffff'; ctx.fill();
          ctx.lineWidth = 2; ctx.strokeStyle = '#0b0e12'; ctx.stroke();
        }

        // scale bar and north
        if (view.ready) {
          const m = niceScale(sc), px = m * sc, bx = 10, by = ch - 12;
          ctx.strokeStyle = 'rgba(233,236,239,0.55)'; ctx.lineWidth = 1.5; ctx.lineCap = 'butt';
          ctx.beginPath(); ctx.moveTo(bx, by - 4); ctx.lineTo(bx, by); ctx.lineTo(bx + px, by); ctx.lineTo(bx + px, by - 4); ctx.stroke();
          ctx.fillStyle = 'rgba(233,236,239,0.7)'; ctx.font = '10px Overpass, Noto Sans, sans-serif';
          ctx.fillText(m >= 1000 ? (m / 1000) + ' km' : m + ' m', bx + px + 6, by + 1);
          ctx.fillText('N', cw - 14, 16);
          ctx.beginPath(); ctx.moveTo(cw - 11, 20); ctx.lineTo(cw - 14.5, 27); ctx.lineTo(cw - 7.5, 27); ctx.closePath(); ctx.fill();
        }
      }
      raf = requestAnimationFrame(draw);
      scope.$on('$destroy', function () { if (raf) cancelAnimationFrame(raf); });

      // tap the map to go there (Storm, 2026-10-31: "Why can I not put in a custom destination?"):
      // any point; the service snaps it to a road it can reach inside the service area
      canvas.addEventListener('wheel', function (ev) {
        ev.preventDefault();
        const r = canvas.getBoundingClientRect();
        zoomAt(ev.deltaY < 0 ? 1.25 : 1 / 1.25, ev.clientX - r.left, ev.clientY - r.top);
      }, { passive: false });
      let drag = null;
      canvas.addEventListener('mousedown', function (ev) { drag = { x: ev.clientX, y: ev.clientY }; });
      const onMove = function (ev) {
        if (!drag) return;
        const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
        if (!user.moved && Math.hypot(dx, dy) < 5) return;
        user.moved = true;                                   // a drag, not a tap: the click that follows is ignored
        user.panX -= dx / view.sc; user.panY += dy / view.sc;   // dragging the map right moves the centre left
        view.cx -= dx / view.sc; view.cy += dy / view.sc;        // follow the finger at once, no easing
        drag.x = ev.clientX; drag.y = ev.clientY;
      };
      const onUp = function () { drag = null; setTimeout(function () { user.moved = false; }, 0); };
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
      scope.$on('$destroy', function () { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); });

      canvas.addEventListener('click', function (ev) {
        if (user.moved) return;
        const st = scope.s && scope.s.state;
        if (!(st === 'boarded' || st === 'riding' || st === 'arrived' || st === 'pulledOver') || !view.ready || !view.sc) return;
        const r = canvas.getBoundingClientRect();
        const x = ((ev.clientX - r.left) - view.ox) / view.sc;
        const y = (view.oy - (ev.clientY - r.top)) / view.sc;
        lua('extensions.lumen_robotaxi.setDestinationXY(' + x.toFixed(1) + ',' + y.toFixed(1) + ')');
      });

      // ---------------------------------------------------------------- lidar tab
      // Storm, 2026-11-01: "Make the app able to view the front camera so you can see your robotaxi
      // coming", then "Actually just a live lidar view" (a HUD app cannot show a camera feed): your
      // Lumen from above as its lidar sees it, live. lumen/lidar.lua scans for the app only while this
      // tab asks (a 6 s request, renewed every 2 s while the tab is open) and sends a picture five
      // times a second. The canvas is drawn when one arrives or the view changes, at most every
      // 100 ms, never on a free-running loop.
      const G = lumenLidarGeom;
      const lcanvas = element[0].querySelector('canvas.lrx-lidar');
      const lctx = lcanvas.getContext('2d');
      const STALE_MS = 3000;                   // no word from lidar.lua for this long: it has gone (a Lua reload)
      const SAFE = { l: 14, t: 38, r: 14, b: 26 };   // px kept clear for the tabs (top) and the legend (bottom)
      const lidar = { snap: null, reason: null, at: 0, opened: 0, range: 1, timer: null, lease: null, last: -1e9 };
      scope.lidarNote = '';
      const lidarAsk = (on) => bngApi.engineLua('if extensions.lumen_lidar then extensions.lumen_lidar.setUiEnabled(' + (on ? 'true' : 'false') + ') end');

      scope.setTab = function (t) {
        t = t === 'lidar' ? 'lidar' : 'map';
        if (t === scope.tab) return;
        scope.tab = t;
        clearInterval(lidar.lease); lidar.lease = null;
        if (t === 'lidar') {
          // a fresh start: an old picture from the last visit is not shown as if it were live
          lidar.snap = null; lidar.reason = null; lidar.at = 0; lidar.opened = performance.now();
          lidarAsk(true);
          lidar.lease = setInterval(lidarTick, 2000);
        } else {
          lidarAsk(false);
        }
        lidarNote();
        lidarDraw();
      };
      scope.lidarCan = (step) => lidar.range + step >= 0 && lidar.range + step < G.RANGES.length;
      scope.lidarZoom = function (step) {
        if (!scope.lidarCan(step)) return;
        lidar.range += step;
        lidarDraw();
      };

      // every 2 s while the tab is open: renew the request, and notice when lidar.lua has gone quiet
      function lidarTick() {
        if (scope.tab !== 'lidar') return;
        lidarAsk(true);
        lidarNote();
        lidarDraw();
      }

      function lidarNote() {
        const now = performance.now();
        let note = '';
        if (scope.tab === 'lidar') {
          if (lidar.reason) note = lidar.reason;
          else if (!lidar.snap) note = now - lidar.opened < STALE_MS ? 'Starting the lidar…' : G.IDLE;
          else if (now - lidar.at > STALE_MS) note = 'Waiting for the lidar…';
        }
        if (note !== scope.lidarNote) scope.$evalAsync(function () { scope.lidarNote = note; });
      }

      scope.$on('LumenRobotaxiLidar', function (event, data) {
        if (scope.tab !== 'lidar') return;                 // a late picture after the tab closed
        const d = G.snapshot(data);
        lidar.at = performance.now();
        if (d.none) { lidar.snap = null; lidar.reason = d.reason; } else { lidar.snap = d; lidar.reason = null; }
        lidarNote();
        lidarDraw();
      });

      function lidarDraw() {
        if (lidar.timer || scope.tab !== 'lidar') return;
        lidar.timer = setTimeout(function () {
          lidar.timer = null;
          lidar.last = performance.now();
          paintLidar();
        }, Math.max(0, 100 - (performance.now() - lidar.last)));
      }

      // a rounded rectangle path with its own radius for the top (nose) and bottom (tail) corners
      function roundRect(c, x, y, w, h, rTop, rBottom) {
        rTop = Math.min(rTop, w / 2, h / 2); rBottom = Math.min(rBottom, w / 2, h / 2);
        c.beginPath();
        c.moveTo(x + rTop, y);
        c.lineTo(x + w - rTop, y);
        c.arcTo(x + w, y, x + w, y + rTop, rTop);
        c.lineTo(x + w, y + h - rBottom);
        c.arcTo(x + w, y + h, x + w - rBottom, y + h, rBottom);
        c.lineTo(x + rBottom, y + h);
        c.arcTo(x, y + h, x, y + h - rBottom, rBottom);
        c.lineTo(x, y + rTop);
        c.arcTo(x, y, x + rTop, y, rTop);
        c.closePath();
      }

      // text with a dark rim, readable over the points
      function rimText(c, text, x, y, fill) {
        c.lineWidth = 3; c.lineJoin = 'round'; c.strokeStyle = 'rgba(6,8,11,0.85)';
        c.strokeText(text, x, y);
        c.fillStyle = fill; c.fillText(text, x, y);
      }

      const overlaps = (a, b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

      // range rings at 10, 20 and 40 m; each label goes to the first free 45 degree spot (up-right, up-left,
      // down-right, down-left) clear of the tabs, the readouts, the buttons and the "You" label (`keep`);
      // the sensor's own reach is dashed
      function lidarRings(c, v, reach, keep) {
        const far = Math.hypot(v.cw, v.ch) / 2;
        c.save();
        c.font = '9.5px Overpass, Noto Sans, sans-serif';
        [10, 20, 40].forEach(function (m) {
          const r = m * v.s;
          if (r < 18 || r > far) return;                   // too small to read beside the car, or off the picture
          c.beginPath(); c.arc(v.cx, v.cy, r, 0, Math.PI * 2);
          c.lineWidth = 1; c.strokeStyle = 'rgba(24,214,196,0.16)'; c.stroke();
          const text = m + ' m', tw = c.measureText(text).width, d = r * Math.SQRT1_2;
          const spots = [[v.cx + d + 3, v.cy - d - 3], [v.cx - d - 3 - tw, v.cy - d - 3], [v.cx + d + 3, v.cy + d + 11], [v.cx - d - 3 - tw, v.cy + d + 11]];
          for (let i = 0; i < spots.length; i++) {
            const lx = spots[i][0], ly = spots[i][1];
            const b = { x0: lx - 2, y0: ly - 10, x1: lx + tw + 2, y1: ly + 3 };
            if (b.x0 < 0 || b.y0 < 0 || b.x1 > v.cw || b.y1 > v.ch || keep.some((k) => overlaps(b, k))) continue;
            rimText(c, text, lx, ly, 'rgba(142,151,163,0.8)');
            break;
          }
        });
        if (reach * v.s <= far) {
          c.setLineDash([3, 5]);
          c.beginPath(); c.arc(v.cx, v.cy, reach * v.s, 0, Math.PI * 2);
          c.lineWidth = 1; c.strokeStyle = 'rgba(24,214,196,0.10)'; c.stroke();
          c.setLineDash([]);
        }
        c.restore();
      }

      // a batch of dots in one colour: one path, one fill
      function dots(c, xy, r, color, round) {
        if (!xy.length) return;
        c.beginPath();
        for (let i = 0; i < xy.length; i += 2) {
          if (round) { c.moveTo(xy[i] + r, xy[i + 1]); c.arc(xy[i], xy[i + 1], r, 0, Math.PI * 2); }
          else c.rect(xy[i] - r, xy[i + 1] - r, r * 2, r * 2);
        }
        c.fillStyle = color; c.fill();
      }

      // ground teal in four 15 m bands, fading with distance; static amber; vehicles magenta (the world
      // view's colours); people white with a soft glow
      function lidarPoints(c, v, p) {
        const groups = [[], [], [], [], [], [], []];
        for (let i = 0; i + 2 < p.length; i += 3) {
          const x = p[i], y = p[i + 1], k = p[i + 2];
          if (typeof x !== 'number' || typeof y !== 'number') continue;
          let g;
          if (k === 1) g = Math.min(3, Math.floor(Math.hypot(x, y) / 150));
          else if (k === 2) g = 4; else if (k === 3) g = 5; else if (k === 4) g = 6; else continue;
          const q = G.toCanvas(x, y, v);
          if (q[0] < -4 || q[1] < -4 || q[0] > v.cw + 4 || q[1] > v.ch + 4) continue;
          groups[g].push(q[0], q[1]);
        }
        c.save();
        for (let b = 0; b < 4; b++) dots(c, groups[b], 0.8, 'rgba(24,214,196,' + G.groundAlpha(b * 15 + 7.5).toFixed(3) + ')', false);
        dots(c, groups[4], 1.0, '#ffb84d', false);
        dots(c, groups[5], 1.15, '#ff4fd8', false);
        c.shadowColor = 'rgba(255,255,255,0.9)'; c.shadowBlur = 7;
        dots(c, groups[6], 1.5, '#ffffff', true);
        c.restore();
      }

      // your Lumen in the middle, nose up: white body, black glass canopy ringed by the turquoise halo, red tail
      // lights at the back and a heading tick ahead of the nose; to scale, but never smaller than 16 x 8 px so it
      // stays readable at 60 m
      function lidarCar(c, v, car) {
        const L = Math.max(16, car.hl * 2 * v.s), W = Math.max(8, car.hw * 2 * v.s);
        const x = v.cx - W / 2, y = v.cy - L / 2;
        c.save();
        c.shadowColor = 'rgba(24,214,196,0.75)'; c.shadowBlur = 12;
        roundRect(c, x, y, W, L, W * 0.45, W * 0.3);
        c.fillStyle = '#eef7f6'; c.fill();
        c.shadowBlur = 0;
        roundRect(c, x + W * 0.17, y + L * 0.24, W * 0.66, L * 0.5, W * 0.28, W * 0.2);
        c.fillStyle = '#0d1418'; c.fill();
        c.lineWidth = Math.max(1, W * 0.08); c.strokeStyle = '#18d6c4'; c.stroke();
        const th = Math.max(1.4, L * 0.05);
        c.fillStyle = '#ff5a6a';
        c.fillRect(x + W * 0.1, y + L - th - 0.6, W * 0.26, th);
        c.fillRect(x + W * 0.64, y + L - th - 0.6, W * 0.26, th);
        c.beginPath(); c.moveTo(v.cx, y - 8); c.lineTo(v.cx - 3.6, y - 3.5); c.lineTo(v.cx + 3.6, y - 3.5); c.closePath();
        c.fillStyle = '#18d6c4'; c.fill();
        c.restore();
      }

      // where "You" goes: the dot (or, beyond the picture, an arrow on its edge) and its label, worked out before
      // anything is drawn so the ring labels can keep clear of it
      function youLayout(c, v, you) {
        if (!you) return null;
        const q = G.toCanvas(you.x, you.y, v);
        const e = G.edge(q[0], q[1], v, SAFE);
        const label = 'You · ' + G.distText(you.d);
        c.save();
        c.font = '600 10.5px Overpass, Noto Sans, sans-serif';
        const tw = c.measureText(label).width;
        c.restore();
        let lx, ly;
        if (!e) {
          lx = q[0] + 10; ly = q[1];
          if (lx + tw > v.cw - 6) lx = q[0] - 10 - tw;
        } else {
          lx = e.x - e.ux * 16 - tw / 2 - e.ux * tw / 2; ly = e.y - e.uy * 16;    // inward from the arrow
        }
        lx = Math.max(6, Math.min(v.cw - 6 - tw, lx)); ly = Math.max(SAFE.t - 4, Math.min(v.ch - SAFE.b, ly));
        const mark = e ? { x: e.x, y: e.y } : { x: q[0], y: q[1] };
        const box = { x0: Math.min(lx, mark.x - 9) - 3, y0: Math.min(ly - 8, mark.y - 9), x1: Math.max(lx + tw, mark.x + 9) + 3, y1: Math.max(ly + 8, mark.y + 9) };
        return { q: q, e: e, label: label, lx: lx, ly: ly, box: box };
      }

      // you: a white dot with a soft halo and how far away you are; beyond the picture, an arrow on its edge
      // pointing at you
      function lidarYou(c, lay) {
        if (!lay) return;
        const q = lay.q, e = lay.e;
        c.save();
        if (!e) {
          const halo = c.createRadialGradient(q[0], q[1], 0, q[0], q[1], 14);
          halo.addColorStop(0, 'rgba(255,255,255,0.40)'); halo.addColorStop(1, 'rgba(255,255,255,0)');
          c.fillStyle = halo; c.beginPath(); c.arc(q[0], q[1], 14, 0, Math.PI * 2); c.fill();
          c.beginPath(); c.arc(q[0], q[1], 4.5, 0, Math.PI * 2);
          c.fillStyle = '#ffffff'; c.fill();
          c.lineWidth = 2; c.strokeStyle = '#0b0e12'; c.stroke();
        } else {
          const ux = e.ux, uy = e.uy, px = -uy, py = ux;
          c.beginPath();
          c.moveTo(e.x + ux * 7, e.y + uy * 7);
          c.lineTo(e.x - ux * 4 + px * 5.5, e.y - uy * 4 + py * 5.5);
          c.lineTo(e.x - ux * 1.5, e.y - uy * 1.5);
          c.lineTo(e.x - ux * 4 - px * 5.5, e.y - uy * 4 - py * 5.5);
          c.closePath();
          c.shadowColor = 'rgba(255,255,255,0.7)'; c.shadowBlur = 8;
          c.fillStyle = '#ffffff'; c.fill();
          c.shadowBlur = 0;
        }
        c.font = '600 10.5px Overpass, Noto Sans, sans-serif';
        c.textBaseline = 'middle';
        rimText(c, lay.label, lay.lx, lay.ly, '#f3f5f7');
        c.restore();
      }

      // the speed readout's box, top right: the ring labels keep clear of it
      function speedBox(c, v, snap) {
        c.save();
        c.font = '600 11px Overpass, Noto Sans, sans-serif';
        const tw = c.measureText(Math.round(snap.speed * 3.6) + ' km/h').width;
        c.restore();
        return { x0: v.cw - 12 - tw - 14, y0: 4, x1: v.cw, y1: 26 };
      }

      // speed with a live dot (turquoise while pictures arrive) at the top right; the legend on its own dark chip at
      // the bottom left, so points underneath cannot pass for legend marks
      function lidarReadouts(c, v, snap, stale) {
        c.save();
        c.font = '600 11px Overpass, Noto Sans, sans-serif';
        c.textAlign = 'right';
        const sp = Math.round(snap.speed * 3.6) + ' km/h';
        rimText(c, sp, v.cw - 12, 20, 'rgba(233,236,239,0.88)');
        c.beginPath(); c.arc(v.cw - 20 - c.measureText(sp).width, 16, 3, 0, Math.PI * 2);
        c.fillStyle = stale ? '#6f7a86' : '#18d6c4'; c.fill();
        c.textAlign = 'left';
        c.font = '9.5px Overpass, Noto Sans, sans-serif';
        const items = [['#18d6c4', 'road'], ['#ffb84d', 'world'], ['#ff4fd8', 'cars'], ['#ffffff', 'people']];
        const width = items.reduce((sum, it) => sum + 9 + c.measureText(it[1]).width + 10, 0) - 10;
        const y = v.ch - 10;
        roundRect(c, 4, y - 12, width + 12, 17, 8, 8);
        c.fillStyle = 'rgba(6,8,11,0.9)'; c.fill();
        let x = 10;
        items.forEach(function (it) {
          c.beginPath(); c.arc(x + 3, y - 3.5, 2.6, 0, Math.PI * 2); c.fillStyle = it[0]; c.fill();
          c.fillStyle = 'rgba(142,151,163,0.95)'; c.fillText(it[1], x + 9, y);
          x += 9 + c.measureText(it[1]).width + 10;
        });
        c.restore();
      }

      function paintLidar() {
        if (scope.tab !== 'lidar') return;
        const wrap = lcanvas.parentNode;
        const w = wrap.clientWidth, h = wrap.clientHeight;
        if (!w || !h) return;
        const ratio = window.devicePixelRatio || 1;
        if (lcanvas.width !== Math.round(w * ratio) || lcanvas.height !== Math.round(h * ratio)) {
          lcanvas.width = Math.round(w * ratio); lcanvas.height = Math.round(h * ratio);
        }
        const c = lctx, snap = lidar.snap;
        c.setTransform(ratio, 0, 0, ratio, 0, 0);
        const v = G.view(w, h, G.RANGES[lidar.range]);
        // background: darker than the map, a faint glow around the car
        const bg = c.createRadialGradient(v.cx, v.cy, 4, v.cx, v.cy, Math.max(w, h) * 0.75);
        bg.addColorStop(0, '#0f1519'); bg.addColorStop(1, '#06080b');
        c.fillStyle = bg; c.fillRect(0, 0, w, h);
        // what the ring labels keep clear of: the tabs, the +/- buttons, the legend, the readout and "You"
        const keep = [{ x0: 0, y0: 0, x1: 118, y1: 38 }, { x0: w - 42, y0: h - 72, x1: w, y1: h }, { x0: 0, y0: h - 26, x1: 190, y1: h }];
        const you = snap ? youLayout(c, v, snap.you) : null;
        if (snap) keep.push(speedBox(c, v, snap));
        if (you) keep.push(you.box);
        lidarRings(c, v, snap ? snap.range : 60, keep);
        if (!snap) return;                                 // the note in the middle says why
        const stale = performance.now() - lidar.at > STALE_MS;
        c.globalAlpha = stale ? 0.35 : 1;                  // lidar.lua went quiet: dim the last picture
        lidarPoints(c, v, snap.pts);
        lidarCar(c, v, snap.car);
        lidarYou(c, you);
        c.globalAlpha = 1;
        lidarReadouts(c, v, snap, stale);
      }

      // the app can be resized: redraw at the new size (pictures keep coming anyway; this covers the empty states)
      const resize = typeof ResizeObserver === 'function' ? new ResizeObserver(function () { lidarDraw(); }) : null;
      if (resize) resize.observe(lcanvas.parentNode);
      scope.$on('$destroy', function () {
        clearTimeout(lidar.timer); clearInterval(lidar.lease);
        if (resize) resize.disconnect();
        if (scope.tab === 'lidar') lidarAsk(false);
      });

      lua('extensions.lumen_robotaxi.requestRoads()');
      lua('extensions.lumen_robotaxi.requestUi()');
    }
  };
}]);
