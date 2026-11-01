// Offline checks of the Lidar tab in src/ui/modules/apps/LumenRobotaxi/app.js, under node (no game, no browser).
// usage: node testenv/lidar_ui_test.js [other/app.js]        exit code 1 when anything fails
//        (another app.js: a candidate, or a deliberately broken copy to prove a check can fail)
//
// Part 1 checks the pure car-frame-to-canvas geometry (lumenLidarGeom, exported to node only).
// Part 2 runs the real directive: its template and link function with a fake Angular scope, a recording 2D canvas
// context, a fake bngApi and a virtual clock, so the wiring is proved, not just the maths: the exact engineLua calls,
// where each kind of point is painted, the 100 ms redraw limit, the request renewals, the empty states, and that the
// map stops drawing while it is hidden. Written 2026-11-01 with the tab (Storm: "Actually just a live lidar view").
'use strict';
const assert = require('assert');
const path = require('path');

const APP = process.argv[2] ? path.resolve(process.argv[2]) : path.join(__dirname, '..', 'src', 'ui', 'modules', 'apps', 'LumenRobotaxi', 'app.js');
const ON = 'if extensions.lumen_lidar then extensions.lumen_lidar.setUiEnabled(true) end';
const OFF = 'if extensions.lumen_lidar then extensions.lumen_lidar.setUiEnabled(false) end';

// the directive registers itself with angular when the file loads: catch it
let DIRECTIVE = null;
globalThis.angular = { module: () => ({ directive: (name, arr) => { DIRECTIVE = { name, factory: arr[arr.length - 1] }; } }) };
const G = require(APP);

const tests = [];
const test = (name, fn) => tests.push([name, fn]);
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} is not ${b} (+-${tol})`);

// ------------------------------------------------------------------------------------------------ part 1: geometry
test('the module loads under node and exports the geometry', () => {
  assert.ok(G && typeof G.toCanvas === 'function' && typeof G.view === 'function', 'no geometry exported');
  assert.deepStrictEqual(G.RANGES, [20, 40, 60]);
  assert.ok(DIRECTIVE && DIRECTIVE.name === 'lumenRobotaxi', 'the directive did not register');
});

test('car frame to canvas: ahead is up, right is right, decimetres to metres', () => {
  for (const [w, h] of [[300, 200], [302, 188], [260, 420]]) {
    for (const range of G.RANGES) {
      const v = G.view(w, h, range);
      const r = Math.min(w, h) / 2 - G.PAD;
      near(v.s, r / range, 1e-9, 'pixels per metre');
      const c0 = G.toCanvas(0, 0, v);
      assert.deepStrictEqual(c0, [w / 2, h / 2], 'the car is not in the middle');
      const ahead = G.toCanvas(0, 100, v);                    // 10 m straight ahead
      near(ahead[0], w / 2, 1e-9, 'ahead x'); near(ahead[1], h / 2 - 10 * v.s, 1e-9, 'ahead y');
      const right = G.toCanvas(50, 0, v);                     // 5 m to the right
      near(right[0], w / 2 + 5 * v.s, 1e-9, 'right x'); near(right[1], h / 2, 1e-9, 'right y');
      const behindLeft = G.toCanvas(-30, -40, v);
      assert.ok(behindLeft[0] < w / 2 && behindLeft[1] > h / 2, 'behind-left is not down-left');
      const edge = G.toCanvas(0, range * 10, v);             // `range` metres ahead: the range circle's top
      near(edge[1], G.PAD + (h - Math.min(w, h)) / 2, 1e-9, 'range circle top');
    }
  }
});

test('markers beyond the picture are pinned to its edge, on the line to them', () => {
  const v = G.view(300, 200, 40), ins = { l: 14, t: 38, r: 14, b: 26 };
  assert.strictEqual(G.edge(150, 100, v, ins), null);
  assert.strictEqual(G.edge(150 + 120, 100 + 60, v, ins), null);
  const up = G.edge(150, -900, v, ins);                        // far ahead
  near(up.x, 150, 1e-9, 'x'); near(up.y, 38, 1e-9, 'y'); near(up.uy, -1, 1e-9, 'direction');
  const back = G.edge(150, 5000, v, ins);
  near(back.y, 200 - 26, 1e-9, 'behind y');
  const diag = G.edge(150 + 400, 100 - 400, v, ins);           // ahead-right, 45 degrees
  near(diag.x - 150, 100 - diag.y, 1e-9, 'stays on the 45 degree line');
  assert.ok(diag.y >= 38 - 1e-9 && diag.x <= 300 - 14 + 1e-9, 'pinned outside the safe area');
});

test('distance text, ground fade and payload clean-up', () => {
  assert.strictEqual(G.distText(8.5), '9 m');
  assert.strictEqual(G.distText(0), '0 m');
  assert.strictEqual(G.distText(142), '140 m');
  assert.strictEqual(G.distText(996), '1.0 km');
  assert.strictEqual(G.distText(2349), '2.3 km');
  assert.strictEqual(G.distText(NaN), '');
  assert.ok(G.groundAlpha(0) > G.groundAlpha(20) && G.groundAlpha(20) > G.groundAlpha(59), 'ground does not fade');
  near(G.groundAlpha(60), G.groundAlpha(500), 1e-12, 'fade stops at the sensor reach');
  const s = G.snapshot({ range: 60, pts: {}, car: { hl: 2.45, hw: 0.98 }, speed: 3 });    // empty Lua table: {}
  assert.deepStrictEqual(s.pts, []);
  assert.strictEqual(s.you, null);
  assert.deepStrictEqual(G.snapshot({ pts: [1, 2, 3], you: { x: null, y: 5 } }).you, null);
  assert.deepStrictEqual(G.snapshot({ pts: [], you: { x: 30, y: 40 } }).you, { x: 30, y: 40, d: 5 });
  assert.deepStrictEqual(G.snapshot({ pts: [] }).car, { hl: 2.3, hw: 0.95 });
  assert.deepStrictEqual(G.snapshot({ none: true }), { none: true, reason: G.IDLE });
  assert.deepStrictEqual(G.snapshot({ none: true, reason: 'x' }), { none: true, reason: 'x' });
  assert.deepStrictEqual(G.snapshot(null), { none: true, reason: G.IDLE });
});

// ------------------------------------------------------------------------------------------------ part 2: the directive
function recorder() {
  const ops = [];
  let st = { fillStyle: '#000', strokeStyle: '#000', lineWidth: 1, shadowBlur: 0, shadowColor: 'rgba(0,0,0,0)', globalAlpha: 1,
             font: '10px sans-serif', textAlign: 'start', textBaseline: 'alphabetic', lineJoin: 'miter', lineCap: 'butt', dash: [] };
  const stack = [];
  let pathOps = [];
  const ctx = {
    ops,
    save() { stack.push(Object.assign({}, st)); },
    restore() { if (stack.length) st = stack.pop(); },
    setTransform() {}, beginPath() { pathOps = []; }, closePath() { pathOps.push({ op: 'close' }); },
    moveTo(x, y) { pathOps.push({ op: 'moveTo', x, y }); }, lineTo(x, y) { pathOps.push({ op: 'lineTo', x, y }); },
    arcTo(x1, y1, x2, y2, r) { pathOps.push({ op: 'arcTo', x: x1, y: y1, r }); },
    bezierCurveTo(a, b, c, d, x, y) { pathOps.push({ op: 'bezier', x, y }); },
    arc(x, y, r) { pathOps.push({ op: 'arc', x, y, r }); },
    rect(x, y, w, h) { pathOps.push({ op: 'rect', x, y, w, h }); },
    fill() { ops.push({ op: 'fill', style: st.fillStyle, blur: st.shadowBlur, alpha: st.globalAlpha, path: pathOps.slice() }); },
    stroke() { ops.push({ op: 'stroke', style: st.strokeStyle, width: st.lineWidth, dash: st.dash.slice(), path: pathOps.slice() }); },
    fillRect(x, y, w, h) { ops.push({ op: 'fillRect', x, y, w, h, style: st.fillStyle }); },
    fillText(text, x, y) { ops.push({ op: 'fillText', text, x, y, style: st.fillStyle, font: st.font, align: st.textAlign }); },
    strokeText(text, x, y) { ops.push({ op: 'strokeText', text, x, y }); },
    measureText(text) { return { width: String(text).length * 5.5 }; },
    createRadialGradient() { return { addColorStop() {} }; },
    setLineDash(d) { st.dash = d.slice(); }, getLineDash() { return st.dash.slice(); },
  };
  for (const k of Object.keys(st)) {
    if (k === 'dash') continue;
    Object.defineProperty(ctx, k, { get: () => st[k], set: (val) => { st[k] = val; } });
  }
  return ctx;
}

function rig() {
  // a virtual clock and timers
  const clock = { now: 1000, timers: [], seq: 0 };
  Object.defineProperty(globalThis, 'performance', { value: { now: () => clock.now }, configurable: true, writable: true });
  globalThis.setTimeout = (fn, ms) => { const id = ++clock.seq; clock.timers.push({ id, at: clock.now + Math.max(0, ms || 0), fn }); return id; };
  globalThis.setInterval = (fn, ms) => { const id = ++clock.seq; clock.timers.push({ id, at: clock.now + ms, fn, every: ms }); return id; };
  globalThis.clearTimeout = globalThis.clearInterval = (id) => { clock.timers = clock.timers.filter((t) => t.id !== id); };
  clock.advance = (ms) => {
    const end = clock.now + ms;
    for (;;) {
      clock.timers.sort((a, b) => a.at - b.at || a.id - b.id);
      const t = clock.timers[0];
      if (!t || t.at > end) break;
      clock.now = t.at;
      if (t.every) t.at += t.every; else clock.timers.shift();
      t.fn();
    }
    clock.now = end;
  };
  const raf = [];
  globalThis.requestAnimationFrame = (fn) => { raf.push(fn); return raf.length; };
  globalThis.cancelAnimationFrame = () => {};
  globalThis.window = { devicePixelRatio: 1, addEventListener() {}, removeEventListener() {} };
  const lua = [];
  globalThis.bngApi = { engineLua: (code) => lua.push(code) };

  const wrap = { clientWidth: 300, clientHeight: 200 };
  const canvas = (cls) => ({ cls, parentNode: wrap, width: 0, height: 0, style: {}, listeners: {}, ctx: recorder(),
    getContext() { return this.ctx; }, addEventListener(type, fn) { this.listeners[type] = fn; },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 300, height: 200 }) });
  const map = canvas(''), lidar = canvas('lrx-lidar');
  const element = [{ querySelector: (sel) => sel === 'canvas' ? map : sel === 'canvas.lrx-lidar' ? lidar : null }];
  const handlers = {};
  const scope = { $on(name, fn) { (handlers[name] = handlers[name] || []).push(fn); }, $evalAsync(fn) { fn(); } };
  const emit = (name, data) => (handlers[name] || []).forEach((fn) => fn({ name }, data));
  const def = DIRECTIVE.factory();
  def.link(scope, element);
  // one map frame through requestAnimationFrame (the map loop re-queues itself each time)
  const mapFrame = () => { const fn = raf.shift(); clock.now += 40; fn(clock.now); };
  const paints = () => lidar.ctx.ops.filter((o) => o.op === 'fillRect' && o.x === 0 && o.y === 0).length;
  return { clock, lua, map, lidar, scope, emit, def, mapFrame, paints, handlers };
}

const SNAP = {
  range: 60, speed: 12.5, car: { hl: 2.45, hw: 0.98 }, you: { x: -30, y: 80, d: 8.5 },
  //       static ahead   vehicle right  person          ground 30 m   ground 10 m   ground 50 m behind (off the picture at 40 m)
  pts: [0, 100, 2,  50, 0, 3,  -30, 80, 4,  0, 300, 1,  100, 0, 1,  0, -500, 1],
};
const fills = (r, style) => r.lidar.ctx.ops.filter((o) => o.op === 'fill' && (typeof style === 'string' ? o.style === style : style.test(o.style)));
const hasDot = (ops, x, y, tol) => ops.some((o) => o.path.some((p) => (p.op === 'rect' && Math.abs(p.x + p.w / 2 - x) <= tol && Math.abs(p.y + p.h / 2 - y) <= tol)
  || (p.op === 'arc' && Math.abs(p.x - x) <= tol && Math.abs(p.y - y) <= tol)));
const box = (path) => {
  const xs = path.filter((p) => 'x' in p).map((p) => p.x), ys = path.filter((p) => 'y' in p).map((p) => p.y);
  return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
};

test('the template has the Map | Lidar control, a second canvas and its own buttons', () => {
  const t = DIRECTIVE.factory().template;
  assert.ok(/<div class="lrx-tabs">.*>Map<\/div>.*>Lidar<\/div><\/div>/.test(t), 'no Map | Lidar control');
  assert.ok(t.indexOf('<canvas ng-show="tab !== \'lidar\'">') < t.indexOf('<canvas class="lrx-lidar"'), 'the map canvas must stay the first canvas');
  assert.ok(t.includes('ng-click="lidarZoom(-1)"') && t.includes('ng-click="lidarZoom(1)"'), 'no range buttons');
  assert.ok(t.includes('ng-click="zoomBy(1.6)"') && t.includes('ng-click="zoomFit()"'), 'the map buttons are gone');
  assert.ok(t.includes("ng-click=\"lidar()\""), 'the world-view "Lidar view" link is gone');
});

test('the map tab works as before and stops drawing behind the Lidar tab', () => {
  const r = rig();
  assert.strictEqual(r.scope.tab, 'map');
  assert.ok(r.lua.some((c) => c.includes('requestRoads()')) && r.lua.some((c) => c.includes('requestUi()')), 'the map no longer asks for its data');
  r.mapFrame();
  const n = r.map.ctx.ops.length;
  assert.ok(n > 0, 'the map did not draw');
  r.scope.setTab('lidar');
  r.mapFrame(); r.mapFrame();
  assert.strictEqual(r.map.ctx.ops.length, n, 'the hidden map kept drawing');
  r.scope.setTab('map');
  r.mapFrame();
  assert.ok(r.map.ctx.ops.length > n, 'the map did not come back');
});

test('opening the tab asks lidar.lua, leaving it and $destroy say off', () => {
  const r = rig();
  r.scope.setTab('lidar');
  assert.strictEqual(r.lua[r.lua.length - 1], ON);
  assert.strictEqual(r.scope.lidarNote, 'Starting the lidar…');
  r.scope.setTab('lidar');                                        // a second click changes nothing
  assert.strictEqual(r.lua.filter((c) => c === ON).length, 1);
  r.scope.setTab('map');
  assert.strictEqual(r.lua[r.lua.length - 1], OFF);
  assert.strictEqual(r.scope.lidarNote, '');
  r.clock.advance(10000);
  assert.strictEqual(r.lua.filter((c) => c === ON).length, 1, 'still renewing after the tab closed');
  const p = r.paints();
  r.emit('LumenRobotaxiLidar', SNAP);                            // a late picture
  r.clock.advance(500);
  assert.strictEqual(r.paints(), p, 'painted a picture that came after the tab closed');
  r.scope.setTab('lidar');
  r.handlers.$destroy.forEach((fn) => fn());
  assert.strictEqual(r.lua[r.lua.length - 1], OFF, '$destroy did not turn the feed off');
  const ons = r.lua.filter((c) => c === ON).length;
  r.clock.advance(10000);
  assert.strictEqual(r.lua.filter((c) => c === ON).length, ons, 'renewing after $destroy');
});

test('reopening the tab never shows the last visit\'s picture as live', () => {
  const r = rig();
  r.scope.setTab('lidar');
  r.emit('LumenRobotaxiLidar', SNAP);
  r.clock.advance(100);
  assert.strictEqual(fills(r, '#eef7f6').length, 1);
  r.scope.setTab('map');
  r.clock.advance(1000);
  r.lidar.ctx.ops.length = 0;
  r.scope.setTab('lidar');
  r.clock.advance(100);
  assert.strictEqual(fills(r, '#eef7f6').length, 0, 'the old picture came back on reopening');
  assert.strictEqual(r.scope.lidarNote, 'Starting the lidar…');
});

test('the request is renewed every 2 s while the tab is open', () => {
  const r = rig();
  r.scope.setTab('lidar');
  r.clock.advance(10050);
  assert.strictEqual(r.lua.filter((c) => c === ON).length, 6, 'expected the opening call and 5 renewals in 10 s');
});

test('a picture lands where the geometry says, in the right colours', () => {
  const r = rig();
  r.scope.setTab('lidar');
  r.emit('LumenRobotaxiLidar', SNAP);
  r.clock.advance(100);
  assert.strictEqual(r.paints(), 1);
  assert.strictEqual(r.scope.lidarNote, '');
  const v = G.view(300, 200, 40);                                   // 40 m to start
  const at = (x, y) => G.toCanvas(x, y, v);
  assert.ok(hasDot(fills(r, '#ffb84d'), ...at(0, 100), 0.01), 'the static point 10 m ahead is not amber at its place');
  assert.ok(hasDot(fills(r, '#ff4fd8'), ...at(50, 0), 0.01), 'the vehicle point 5 m right is not magenta at its place');
  const people = fills(r, '#ffffff').filter((o) => o.blur > 0 && o.path.some((p) => p.op === 'arc' && p.r < 3));
  assert.ok(hasDot(people, ...at(-30, 80), 0.01), 'the person is not a glowing white dot at its place');
  const ground = fills(r, /^rgba\(24,214,196,/);
  const g10 = ground.find((o) => hasDot([o], ...at(100, 0), 0.01)), g30 = ground.find((o) => hasDot([o], ...at(0, 300), 0.01));
  assert.ok(g10 && g30, 'ground points missing');
  const alpha = (o) => +o.style.match(/,([\d.]+)\)$/)[1];
  assert.ok(alpha(g10) > alpha(g30), 'ground does not fade with distance');
  assert.ok(!ground.some((o) => hasDot([o], ...at(0, -500), 0.5)), 'a point beyond the canvas was drawn');
  const car = fills(r, '#eef7f6');
  assert.strictEqual(car.length, 1, 'no car body');
  const b = box(car[0].path);
  near((b.x0 + b.x1) / 2, 150, 0.01, 'car centre x'); near((b.y0 + b.y1) / 2, 100, 0.01, 'car centre y');
  assert.ok(b.y1 - b.y0 > b.x1 - b.x0, 'the car is not nose up (taller than wide)');
  near(b.y1 - b.y0, Math.max(16, 4.9 * v.s), 0.01, 'car length'); near(b.x1 - b.x0, Math.max(8, 1.96 * v.s), 0.01, 'car width');
  const texts = r.lidar.ctx.ops.filter((o) => o.op === 'fillText').map((o) => o.text);
  assert.ok(texts.includes('You · 9 m'), 'no "You" label: ' + texts.join(' | '));
  assert.ok(texts.includes('45 km/h'), 'no speed readout');
  for (const m of ['10 m', '20 m', '40 m']) assert.ok(texts.includes(m), 'no ring label ' + m);
  for (const k of ['road', 'world', 'cars', 'people']) assert.ok(texts.includes(k), 'no legend entry ' + k);
  const rings = r.lidar.ctx.ops.filter((o) => o.op === 'stroke' && o.path.length === 1 && o.path[0].op === 'arc' && near1(o.path[0].x, 150) && near1(o.path[0].y, 100));
  const radii = rings.map((o) => Math.round(o.path[0].r * 100) / 100).sort((a, b) => a - b);
  assert.deepStrictEqual(radii, [10, 20, 40, 60].map((m) => Math.round(m * v.s * 100) / 100), 'rings at the wrong radii');
  assert.ok(rings.every((o) => o.width === 1), 'a ring is not 1 px wide');
  const you = fills(r, '#ffffff').find((o) => o.path.some((p) => p.op === 'arc' && Math.abs(p.r - 4.5) < 1e-9));
  assert.ok(you && hasDot([you], ...at(-30, 80), 0.01), 'no "You" dot at the rider');
});
const near1 = (a, b) => Math.abs(a - b) < 1e-6;

test('redraws only when a picture comes, at most every 100 ms', () => {
  const r = rig();
  r.scope.setTab('lidar');
  r.clock.advance(100);
  const p0 = r.paints();
  for (let i = 0; i < 5; i++) r.emit('LumenRobotaxiLidar', SNAP);     // a burst
  r.clock.advance(99);
  r.clock.advance(1);
  assert.strictEqual(r.paints() - p0, 1, 'a burst of pictures painted more than once');
  const p1 = r.paints();
  for (let i = 0; i < 10; i++) { r.emit('LumenRobotaxiLidar', SNAP); r.clock.advance(200); }    // 5 Hz for 2 s
  assert.ok(Math.abs(r.paints() - p1 - 10) <= 1, `5 Hz pictures: ${r.paints() - p1} paints in 2 s`);
  const p2 = r.paints();
  r.clock.advance(1900);                                              // silence, no renewal tick inside
  assert.ok(r.paints() - p2 <= 1, 'painting without pictures (a free-running loop?)');
  for (let i = 0; i < 20; i++) { r.emit('LumenRobotaxiLidar', SNAP); r.clock.advance(20); }   // 50 Hz flood
  assert.ok(r.paints() - p2 <= 6, `a 50 Hz flood painted ${r.paints() - p2} times in 400 ms`);
});

test('range buttons step 40 -> 20 -> 40 -> 60 and stop at the ends', () => {
  const r = rig();
  r.scope.setTab('lidar');
  r.emit('LumenRobotaxiLidar', SNAP);
  r.clock.advance(100);
  assert.ok(r.scope.lidarCan(-1) && r.scope.lidarCan(1));
  for (const [step, range] of [[-1, 20], [-1, 20], [1, 40], [1, 60], [1, 60]]) {
    r.lidar.ctx.ops.length = 0;
    r.scope.lidarZoom(step);
    r.emit('LumenRobotaxiLidar', SNAP);              // a click at the end changes nothing, so the next picture paints
    r.clock.advance(100);
    const v = G.view(300, 200, range);
    assert.ok(hasDot(fills(r, '#ffb84d'), ...G.toCanvas(0, 100, v), 0.01), `at ${range} m the point 10 m ahead is not where it belongs`);
  }
  assert.ok(!r.scope.lidarCan(1) && r.scope.lidarCan(-1), 'the farther button is not off at 60 m');
});

test('empty states: starting, a reason, nothing at all, a quiet lidar', () => {
  const r = rig();
  r.scope.setTab('lidar');
  assert.strictEqual(r.scope.lidarNote, 'Starting the lidar…');
  r.clock.advance(4100);                                              // nothing ever arrives (no lumen_lidar)
  assert.strictEqual(r.scope.lidarNote, G.IDLE);
  assert.strictEqual(fills(r, '#eef7f6').length, 0, 'a car was drawn without a picture');
  r.emit('LumenRobotaxiLidar', { none: true, reason: 'Finding you a Lumen. Its lidar shows here once it is on the way.' });
  assert.strictEqual(r.scope.lidarNote, 'Finding you a Lumen. Its lidar shows here once it is on the way.');
  r.emit('LumenRobotaxiLidar', { none: true });
  assert.strictEqual(r.scope.lidarNote, G.IDLE);
  r.emit('LumenRobotaxiLidar', { range: 60, pts: {}, car: { hl: 2.45, hw: 0.98 }, speed: 0 });   // empty picture: {}
  r.clock.advance(100);
  assert.strictEqual(r.scope.lidarNote, '');
  assert.strictEqual(fills(r, '#eef7f6').length, 1, 'an empty picture should still show the car');
  r.lidar.ctx.ops.length = 0;
  r.emit('LumenRobotaxiLidar', SNAP);
  r.clock.advance(100);
  r.lidar.ctx.ops.length = 0;
  r.clock.advance(4000);                                              // lidar.lua goes quiet (a Lua reload)
  assert.strictEqual(r.scope.lidarNote, 'Waiting for the lidar…');
  const car = fills(r, '#eef7f6');
  assert.ok(car.length && car[car.length - 1].alpha < 0.5, 'the last picture is not dimmed when stale');
  assert.ok(car[0].alpha === 1, 'the picture was dimmed while it was still fresh');
});

test('ring labels keep clear of the "You" label and the speed, wherever the rider stands', () => {
  const r = rig();
  r.scope.setTab('lidar');
  const W = 5.5;                                                      // the fake measureText: 5.5 px a character
  let paints = 0, labels = 0;
  for (const step of [-1, 1, 1]) {                                    // 20, 40, 60 m
    r.scope.lidarZoom(step);
    for (let x = -600; x <= 600; x += 75) {
      for (let y = -500; y <= 700; y += 75) {
        r.lidar.ctx.ops.length = 0;
        r.emit('LumenRobotaxiLidar', Object.assign({}, SNAP, { you: { x, y, d: Math.hypot(x, y) / 10 } }));
        r.clock.advance(100);
        const texts = r.lidar.ctx.ops.filter((o) => o.op === 'fillText');
        const you = texts.find((o) => o.text.startsWith('You · '));
        const speed = texts.find((o) => o.text.endsWith('km/h'));
        assert.ok(you && speed, 'a paint without the "You" label or the speed');
        const youBox = { x0: you.x, y0: you.y - 8, x1: you.x + you.text.length * W, y1: you.y + 8 };
        const speedBox = { x0: speed.x - speed.text.length * W, y0: speed.y - 11, x1: speed.x, y1: speed.y + 3 };
        for (const o of texts.filter((t) => /^(10|20|40) m$/.test(t.text))) {
          const b = { x0: o.x - 2, y0: o.y - 10, x1: o.x + o.text.length * W + 2, y1: o.y + 3 };
          const hit = (k) => b.x0 < k.x1 && k.x0 < b.x1 && b.y0 < k.y1 && k.y0 < b.y1;
          assert.ok(!hit(youBox), `"${o.text}" at (${o.x.toFixed(0)}, ${o.y.toFixed(0)}) overlaps "${you.text}" (rider at ${x}, ${y})`);
          assert.ok(!hit(speedBox), `"${o.text}" overlaps the speed readout`);
          labels++;
        }
        paints++;
      }
    }
  }
  assert.ok(labels >= paints * 1.5, `only ${labels} ring labels in ${paints} pictures: the rule hides too many`);
});

test('a rider beyond the picture gets an arrow on its edge', () => {
  const r = rig();
  r.scope.setTab('lidar');
  r.emit('LumenRobotaxiLidar', Object.assign({}, SNAP, { you: { x: 0, y: 3000, d: 300 } }));      // 300 m ahead
  r.clock.advance(100);
  const texts = r.lidar.ctx.ops.filter((o) => o.op === 'fillText');
  const label = texts.find((o) => o.text === 'You · 300 m');
  assert.ok(label, 'no label for the far rider');
  const arrow = fills(r, '#ffffff').find((o) => o.path.length === 5 && o.path[0].op === 'moveTo');
  assert.ok(arrow, 'no edge arrow');
  near(arrow.path[0].x, 150, 1e-6, 'arrow tip x'); near(arrow.path[0].y, 38 - 7, 1e-6, 'arrow tip y (pointing up at the top edge)');
  assert.ok(label.y > arrow.path[0].y, 'the label is not inward of the arrow');
  r.emit('LumenRobotaxiLidar', Object.assign({}, SNAP, { you: { x: -2500, y: -2500, d: 353.6 } }));   // far behind-left
  r.lidar.ctx.ops.length = 0;
  r.clock.advance(100);
  const a2 = fills(r, '#ffffff').find((o) => o.path.length === 5 && o.path[0].op === 'moveTo');
  assert.ok(a2 && a2.path[0].x < 150 && a2.path[0].y > 100, 'the arrow does not point down-left');
  assert.ok(r.lidar.ctx.ops.some((o) => o.op === 'fillText' && o.text === 'You · 350 m'));
});

let failed = 0;
for (const [name, fn] of tests) {
  try {
    fn();
    console.log('PASS  ' + name);
  } catch (e) {
    failed++;
    console.log('FAIL  ' + name + ': ' + (e && e.message ? e.message : e));
  }
}
console.log(`\n${tests.length - failed}/${tests.length} passed`);
process.exitCode = failed ? 1 : 0;
