"use strict";
// PDN design assistant page. All answers come from the server (assistant_web/engine.py).

const COL = { 1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a" };
const SVGNS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);

const S = {
  setup: null,
  codes: new Array(52).fill(0),
  target: "0.8",
  part: 1,                // 0 = eraser
  mode: "safe",
  history: [],
  check: null,
  prevSpectrum: null,
  mapIndex: null,         // frequency point of the map; null = no map until the curve is clicked
  map: null,
  scaleMode: "fixed",
  preview: null,
  exampleId: null,
  seq: 0,
  fixSeq: 0,
  fixCache: new Map(),
};
let el = {};               // svg groups and elements

// ---------------------------------------------------------------- helpers
async function api(path, body) {
  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  let r;
  try { r = await fetch(path, opt); } catch (e) {
    banner("Server not reachable. Start it with: python -m assistant_web.server", "error", 0);
    throw e;
  }
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* keep statusText */ }
    throw new Error(msg);
  }
  return r.json();
}
const kOf = (c) => c.reduce((a, v) => a + (v ? 1 : 0), 0);
const kMax = () => S.setup.model.k_range[1];
const tgtObj = () => S.setup.targets.find((t) => t.name === S.target);
const fmtMHz = (f) => (f < 10 ? f.toFixed(1) : f.toFixed(0));
const fmtOhm = (v) => (v < 0.1 ? v.toFixed(3) : v.toPrecision(3));
const fmtInt = (n) => n.toLocaleString("en-US");
const pct = (m) => `${Math.round(m * 100)} %`;
const partName = (t) => S.setup.parts[String(t)].name;
const sameCodes = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
function svg(tag, attrs = {}, parent = null) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
let bannerTimer = null;
function banner(msg, kind = "info", ms = 3500) {
  const b = $("banner");
  b.textContent = msg;
  b.className = "banner" + (kind === "error" ? " error" : "");
  clearTimeout(bannerTimer);
  if (ms) bannerTimer = setTimeout(() => b.classList.add("hidden"), ms);
}
function jet(t) {
  const c = (x) => Math.max(0, Math.min(1, x));
  return [c(1.5 - Math.abs(4 * t - 3)), c(1.5 - Math.abs(4 * t - 2)), c(1.5 - Math.abs(4 * t - 1))].map((v) => Math.round(v * 255));
}

// ---------------------------------------------------------------- board
function buildBoard() {
  const b = S.setup.board;
  const [W, H] = b.board_mm;
  const Y = (y) => H - y;
  const root = $("board");
  root.setAttribute("viewBox", `-14 -14 ${W + 28} ${H + 28}`);
  const defs = svg("defs", {}, root);
  const pts = b.outline.map(([x, y]) => `${x},${Y(y)}`).join(" ");
  const clip = svg("clipPath", { id: "boardclip" }, defs);
  svg("polygon", { points: pts }, clip);
  el.base = svg("polygon", { points: pts, fill: "#0f4a24" }, root);
  const [x0, x1, y0, y1] = b.map_extent;
  el.img = svg("image", { x: x0, y: Y(y1), width: x1 - x0, height: y1 - y0, preserveAspectRatio: "none",
    "clip-path": "url(#boardclip)", visibility: "hidden" }, root);
  svg("polygon", { points: pts, fill: "none", stroke: "#1b1b1b", "stroke-width": 1.6 }, root);
  el.hot = svg("g", {}, root);
  el.prev = svg("g", { "pointer-events": "none" }, root);
  el.slots = svg("g", {}, root);
  el.over = svg("g", { "pointer-events": "none" }, root);
  // IC port
  const [ix, iy] = b.ic_port;
  const ic = svg("g", {}, root);
  svg("rect", { x: ix - 5, y: Y(iy) - 5, width: 10, height: 10, fill: "#fff", stroke: "#111", "stroke-width": 1.4 }, ic);
  const t = svg("text", { x: ix, y: Y(iy) + 14, "text-anchor": "middle", "font-size": 8, "font-weight": 700,
    fill: "#fff", stroke: "#111", "stroke-width": 2.2, "paint-order": "stroke" }, ic);
  t.textContent = "IC port";
  // slots
  el.slotEls = b.slots.map(([x, y], i) => {
    const g = svg("g", { class: "slot", "data-i": i, transform: `translate(${x},${Y(y)})` }, el.slots);
    const title = svg("title", {}, g);
    const ring = svg("circle", { class: "ring", r: 8.5 }, g);
    const lab = svg("text", { "text-anchor": "middle", "dominant-baseline": "central", "paint-order": "stroke" }, g);
    const sub = svg("text", { y: 14.5, "text-anchor": "middle", "font-size": 5, fill: "#fff", stroke: "#111",
      "stroke-width": 1.6, "paint-order": "stroke" }, g);
    g.addEventListener("click", () => onSlot(i));
    return { g, title, ring, lab, sub };
  });
}

function renderSlots() {
  S.codes.forEach((t, i) => {
    const s = el.slotEls[i];
    const name = `C${i + 1}`;
    if (t) {
      s.ring.setAttribute("fill", "#fff");
      s.ring.setAttribute("stroke", COL[t]);
      s.ring.setAttribute("stroke-width", 2.8);
      s.lab.textContent = partName(t);
      s.lab.setAttribute("font-size", 7.2);
      s.lab.setAttribute("font-weight", 800);
      s.lab.setAttribute("fill", COL[t]);
      s.lab.setAttribute("stroke", "none");
      s.sub.textContent = name;
      const p = S.setup.parts[String(t)];
      s.title.textContent = `${name}: ${p.name} (${p.value}, ${p.esl}, ${p.esr}). Click to ${S.part && S.part !== t ? "change to " + partName(S.part) : "remove"}.`;
    } else {
      s.ring.setAttribute("fill", "rgba(255,255,255,0.16)");
      s.ring.setAttribute("stroke", "rgba(255,255,255,0.8)");
      s.ring.setAttribute("stroke-width", 1.1);
      s.lab.textContent = name;
      s.lab.setAttribute("font-size", 5.4);
      s.lab.setAttribute("font-weight", 600);
      s.lab.setAttribute("fill", "#fff");
      s.lab.setAttribute("stroke", "#111");
      s.lab.setAttribute("stroke-width", 1.4);
      s.sub.textContent = "";
      s.title.textContent = `${name}: empty. ${S.part ? "Click to place " + partName(S.part) : ""}`;
    }
  });
}

function slotXY(slot) {               // 1-based slot -> svg coordinates
  const [x, y] = S.setup.board.slots[slot - 1];
  return [x, S.setup.board.board_mm[1] - y];
}

function renderPreview() {
  el.over.replaceChildren();
  const p = S.preview;
  if (!p) return;
  const ghost = (slot, t) => {
    const [x, y] = slotXY(slot);
    svg("circle", { cx: x, cy: y, r: 12.5, fill: "rgba(255,255,255,0.92)", stroke: COL[t], "stroke-width": 2.6,
      "stroke-dasharray": "3 2" }, el.over);
    const tx = svg("text", { x, y, "text-anchor": "middle", "dominant-baseline": "central", "font-size": 8,
      "font-weight": 800, fill: COL[t] }, el.over);
    tx.textContent = partName(t);
  };
  const cross = (slot) => {
    const [x, y] = slotXY(slot);
    svg("circle", { cx: x, cy: y, r: 10.5, fill: "rgba(0,0,0,0.35)", stroke: "#fff", "stroke-width": 1.2 }, el.over);
    svg("path", { d: `M${x - 6},${y - 6} L${x + 6},${y + 6} M${x + 6},${y - 6} L${x - 6},${y + 6}`, stroke: "#fff",
      "stroke-width": 2.6, "stroke-linecap": "round" }, el.over);
  };
  for (const ch of p.changes) {
    if (ch.kind === "retype" || ch.kind === "add") ghost(ch.slot, ch.to);
    else if (ch.kind === "remove") cross(ch.slot);
    else if (ch.kind === "move") {
      const [ax, ay] = slotXY(ch.from_slot), [bx, by] = slotXY(ch.to_slot);
      svg("line", { x1: ax, y1: ay, x2: bx, y2: by, stroke: "#fff", "stroke-width": 2.2, "stroke-dasharray": "4 3" }, el.over);
      cross(ch.from_slot);
      ghost(ch.to_slot, ch.to);
    }
  }
}

// ---------------------------------------------------------------- map
function drawMap() {
  const m = S.map;
  if (S.mapIndex !== null) $("freqlabel").textContent = kOf(S.codes) > 0
    ? `map at ${fmtMHz(S.setup.freqs_mhz[S.mapIndex])} MHz` : "no map without decaps";
  const on = S.mapIndex !== null && m && kOf(S.codes) > 0;
  el.img.setAttribute("visibility", on ? "visible" : "hidden");
  el.base.setAttribute("fill", on ? "#222" : "#0f4a24");
  el.hot.replaceChildren();
  $("scalemode").classList.toggle("hidden", !on);
  document.querySelector(".cbar").classList.toggle("hidden", !on);
  if (!on) { drawColorbar(null, null); $("hotlabel").textContent = ""; return; }
  const [lo, hi] = S.scaleMode === "fixed" ? [m.lo, m.hi] : [m.board_min, m.board_max];
  const L0 = Math.log(lo), L1 = Math.log(hi);
  const cv = document.createElement("canvas");
  cv.width = 64; cv.height = 64;
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(64, 64);
  for (let r = 0; r < 64; r++) {
    for (let q = 0; q < 64; q++) {
      const v = m.values[r][q];
      const o = ((63 - r) * 64 + q) * 4;          // row 0 = bottom of the board
      if (v === null || v <= 0) { img.data[o + 3] = 0; continue; }
      const t = Math.max(0, Math.min(1, (Math.log(v) - L0) / (L1 - L0 || 1)));
      const [R, G, B] = jet(t);
      img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  el.img.setAttribute("href", cv.toDataURL());
  // hot spot
  const [W, H] = S.setup.board.board_mm;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const hx = clamp(m.hot.x, 9, W - 9), hy = H - clamp(m.hot.y, 9, H - 9);   // edge pixels sit on the outline
  svg("circle", { cx: hx, cy: hy, r: 10, fill: "none", stroke: "#fff", "stroke-width": 3.2 }, el.hot);
  svg("circle", { cx: hx, cy: hy, r: 10, fill: "none", stroke: "#111", "stroke-width": 1.4, "stroke-dasharray": "3 2" }, el.hot);
  $("hotlabel").textContent = `hot spot ${fmtOhm(m.hot.ohm)} Ω`;
  drawColorbar(lo, hi);
}

function drawColorbar(lo, hi) {
  const cv = $("cbar"), ctx = cv.getContext("2d");
  for (let x = 0; x < cv.width; x++) {
    const [R, G, B] = jet(x / (cv.width - 1));
    ctx.fillStyle = lo === null ? "#eee" : `rgb(${R},${G},${B})`;
    ctx.fillRect(x, 0, 1, cv.height);
  }
  $("cb-lo").textContent = lo === null ? "" : `${fmtOhm(lo)} Ω`;
  $("cb-mid").textContent = lo === null ? "" : fmtOhm(Math.sqrt(lo * hi));
  $("cb-hi").textContent = lo === null ? "" : `${fmtOhm(hi)} Ω`;
}

let mapBusy = false, mapWanted = null;
function requestMap() {
  if (kOf(S.codes) === 0 || S.mapIndex === null) { S.map = null; drawMap(); return; }
  mapWanted = { codes: S.codes.slice(), index: S.mapIndex };
  if (!mapBusy) pumpMap();
}
async function pumpMap() {
  mapBusy = true;
  while (mapWanted) {
    const w = mapWanted;
    mapWanted = null;
    try {
      const m = await api("/api/map", w);
      if (sameCodes(w.codes, S.codes) && w.index === S.mapIndex) { S.map = m; drawMap(); }
    } catch (e) { banner(`Map: ${e.message}`, "error"); }
  }
  mapBusy = false;
}
function setMapIndex(i) {
  // i = a frequency point (0..230) picked on the spectrum curve, or null for no map
  S.mapIndex = i === null ? null : Math.max(0, Math.min(S.setup.freqs_mhz.length - 1, i));
  const lab = $("freqlabel");
  if (S.mapIndex === null) {
    lab.textContent = "Click the curve to show a map";
    lab.classList.add("none");
  } else {
    lab.textContent = `map at ${fmtMHz(S.setup.freqs_mhz[S.mapIndex])} MHz`;
    lab.classList.remove("none");
  }
  $("clearmap").classList.toggle("hidden", S.mapIndex === null);
  drawPlot();
  requestMap();
}

function onPlotClick(ev) {
  const pt = (ev.points || []).find((p) => p.data.x && p.data.x.length === S.setup.freqs_mhz.length);
  if (!pt) return;
  if (kOf(S.codes) === 0) {
    banner(`Place a decap first: maps are predicted for K = ${S.setup.model.k_range[0]}–${kMax()}.`, "error");
    return;
  }
  setMapIndex(pt.pointIndex);
}

// ---------------------------------------------------------------- spectrum plot
function mapLine(f) {
  return { type: "line", xref: "x", yref: "paper", x0: f, x1: f, y0: 0, y1: 1,
    line: { color: "rgba(0,0,0,0.35)", width: 1, dash: "dot" } };
}
function yRange(traces) {
  // log10 range that shows every curve in full: at least 0.002-70 ohm (the dataset spans about
  // 0.003-50 ohm), widened when a curve goes beyond, with a small margin
  let lo = Infinity, hi = -Infinity;
  for (const t of traces) for (const v of t.y) if (v > 0) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  return [Math.min(Math.log10(0.002), Math.log10(lo) - 0.08), Math.max(Math.log10(70), Math.log10(hi) + 0.08)];
}
function drawPlot() {
  const f = S.setup.freqs_mhz, T = tgtObj(), c = S.check;
  const tr = [
    { x: f, y: T.curve, name: "target", line: { color: "#0b0b0b", width: 1.6 }, hoverinfo: "none" },
    { x: f, y: T.curve.map((v) => v * (1 - T.margin)), name: `safe bar (${(1 - T.margin).toFixed(2)}×)`,
      line: { color: "#0b0b0b", width: 1.2, dash: "dash" }, hoverinfo: "none" },
  ];
  if (S.prevSpectrum) tr.push({ x: f, y: S.prevSpectrum, name: "previous",
    line: { color: "#a19f97", width: 1.6, dash: "dot" }, hovertemplate: "previous %{y:.3f} Ω<extra></extra>" });
  if (c) {
    const bare = c.kind === "bare";
    tr.push({ x: f, y: c.spectrum, name: bare ? "bare board (sim.)" : "predicted",
      line: { color: bare ? "#8a5a2b" : "#2a78d6", width: 2.6 },
      hovertemplate: (bare ? "bare board" : "predicted") + " %{y:.3f} Ω<extra></extra>" });
    const wi = c.worst_index;
    tr.push({ x: [f[wi]], y: [c.spectrum[wi]], mode: "markers", name: "worst",
      marker: { color: "#c8312b", size: 11, symbol: "x-thin", line: { width: 3, color: "#c8312b" } }, hoverinfo: "skip" });
  }
  if (c && c.kind !== "bare" && S.mapIndex !== null) tr.push({ x: [f[S.mapIndex]], y: [c.spectrum[S.mapIndex]], mode: "markers",
    name: `map ${fmtMHz(f[S.mapIndex])} MHz`, hoverinfo: "skip",
    marker: { color: "#fff", size: 11, line: { color: "#0b0b0b", width: 2.2 } } });
  if (S.preview && S.preview.spectrum) tr.push({ x: f, y: S.preview.spectrum, name: "suggestion",
    line: { color: "#eb6834", width: 2.4, dash: "dot" }, hovertemplate: "suggestion %{y:.3f} Ω<extra></extra>" });
  const layout = {
    margin: { l: 58, r: 14, t: 6, b: 42 },
    xaxis: { type: "log", title: { text: "frequency (MHz)", standoff: 6 }, range: [Math.log10(0.93), Math.log10(640)], fixedrange: true,
      tickvals: [1, 2, 5, 10, 20, 50, 100, 200, 500], gridcolor: "#eceae4", zeroline: false },
    yaxis: { type: "log", title: { text: "|Z| (Ω)", standoff: 4 }, range: yRange(tr), gridcolor: "#eceae4", fixedrange: true,
      tickvals: [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
      ticktext: ["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1", "3", "10", "30", "100"] },
    legend: { orientation: "h", x: 0, y: 1.0, yanchor: "bottom", font: { size: 11 } },
    hovermode: "x unified", plot_bgcolor: "#fff", paper_bgcolor: "#fff",
    font: { family: "Segoe UI, system-ui, sans-serif", size: 12, color: "#0b0b0b" },
    shapes: S.mapIndex === null ? [] : [mapLine(f[S.mapIndex])],
    dragmode: false,
  };
  const div = $("spectrum");
  Plotly.react(div, tr, layout, { displayModeBar: false, responsive: true });
  if (!div.dataset.clickBound) { div.on("plotly_click", onPlotClick); div.dataset.clickBound = "1"; }
}

// ---------------------------------------------------------------- verdict
function renderVerdict() {
  const c = S.check, badge = $("badge"), d = $("direct");
  if (!c) return;
  const T = tgtObj();
  if (c.kind === "bare") {
    badge.className = "badge neutral";
    badge.innerHTML = "<small>NO DECAPS</small>BARE";
    $("vline1").textContent = "Bare board (simulated)";
    $("vline2").textContent = "Place a decap to get a prediction.";
    d.className = "chip neutral"; d.textContent = "direct –";
    $("rho").textContent = `ρ = ${c.direct.worst_ratio.toFixed(2)} at ${fmtMHz(c.direct.worst_mhz)} MHz`;
    return;
  }
  const s = c.safe;
  badge.className = "badge " + (s.passes ? "pass" : "fail");
  badge.innerHTML = `<small>SAFE</small>${s.passes ? "PASS" : "FAIL"}`;
  $("vline1").textContent = `${s.passes ? "Passes" : "Fails"} ${T.label} with a ${pct(c.margin)} margin`;
  $("vline2").textContent = s.passes ? "" : `needs ρ̂ ≤ ${(1 - c.margin).toFixed(2)}`;
  d.className = "chip " + (c.direct.passes ? "pass" : "fail");
  d.textContent = `direct ${c.direct.passes ? "PASS" : "FAIL"}`;
  $("rho").innerHTML = `ρ̂ = <b>${s.worst_ratio.toFixed(2)}</b> at ${fmtMHz(s.worst_mhz)} MHz`;
}

// ---------------------------------------------------------------- suggestions
function listItem(n, item, valHtml, sub) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="n">${n}</span><span class="what">${item.text}${sub ? `<small>${sub}</small>` : ""}</span>`
    + `<span class="val">${valHtml}</span>`;
  li.addEventListener("mouseenter", () => { S.preview = item; renderPreview(); drawPlot(); });
  li.addEventListener("mouseleave", () => { S.preview = null; renderPreview(); drawPlot(); });
  li.addEventListener("click", () => apply(item.codes));
  li.title = "Hover to preview, click to apply";
  return li;
}
const mini = (ok, label) => `<span class="mini ${ok ? "pass" : "fail"}">${label} ${ok ? "✓" : "✗"}</span>`;

function renderAdds() {
  const list = $("addlist"), st = $("addstatus");
  list.replaceChildren();
  const c = S.check;
  if (!c) return;
  const K = c.K;
  if (K >= kMax()) { st.textContent = `K = ${kMax()} is the maximum.`; return; }
  st.textContent = "";
  c.additions.forEach((a, i) => list.appendChild(listItem(i + 1, a,
    `ρ̂ ${a.ratio.toFixed(2)} ${mini(a.safe_pass, "safe")}${mini(a.direct_pass, "direct")}`, "")));
}

function setFixStatus(html) { $("fixstatus").innerHTML = html; }
async function runFixes() {
  const my = ++S.fixSeq;
  const list = $("fixlist");
  list.replaceChildren();
  const K = kOf(S.codes);
  if (K === 0) { setFixStatus("Place a decap first."); return; }
  const key = `${S.codes.join("")}|${S.target}|${S.mode}`;
  if (S.fixCache.has(key)) { renderFixes(S.fixCache.get(key)); return; }
  const body = { codes: S.codes.slice(), target: S.target, mode: S.mode };
  setFixStatus('<span class="spin"></span>searching…');
  let timer = null;
  try {
    let r = await api("/api/fixes", { ...body, max_radius: 2 });
    if (my !== S.fixSeq) return;
    if (r.status === "none") {
      const t0 = performance.now();
      const tick = () => setFixStatus(`<span class="spin"></span>searching 3 changes… `
        + `${Math.round((performance.now() - t0) / 1000)} s`);
      tick();
      timer = setInterval(tick, 500);
      r = await api("/api/fixes", { ...body, max_radius: 3 });
      clearInterval(timer);
      if (my !== S.fixSeq) return;
    }
    if (r.status === "cancelled") return;
    if (S.fixCache.size > 300) S.fixCache.clear();
    S.fixCache.set(key, r);
    renderFixes(r);
  } catch (e) {
    clearInterval(timer);
    if (my === S.fixSeq) setFixStatus(`Error: ${e.message}`);
  }
}

function renderFixes(r) {
  const list = $("fixlist");
  list.replaceChildren();
  if (r.status === "passes") { setFixStatus("Already passes."); return; }
  if (r.status === "no_model") { setFixStatus(`Place 1–${kMax()} decaps.`); return; }
  if (r.status === "none") { setFixStatus(`None within ${r.max_radius} changes. Try adding a decap.`); return; }
  const e = r.suggestions[0].edits;
  const st = $("fixstatus");
  setFixStatus(`${fmtInt(r.n_found)} found · ${e} change${e > 1 ? "s" : ""} · ${r.seconds.toFixed(1)} s`);
  st.title = `${fmtInt(r.n_scored)} placements judged; same K; ${r.mode === "safe" ? `safe: ρ̂ ≤ ${(1 - r.margin).toFixed(2)}` : "direct: ρ̂ ≤ 1"}`;
  r.suggestions.forEach((s, i) => list.appendChild(listItem(i + 1, s, `ρ̂ ${s.ratio.toFixed(2)}`, "")));
}

// ---------------------------------------------------------------- actions
function pushHistory() {
  S.history.push({ codes: S.codes.slice(), target: S.target });
  if (S.history.length > 200) S.history.shift();
  $("undo").disabled = false;
}
function changed(newCodes) {
  pushHistory();
  S.prevSpectrum = S.check ? S.check.spectrum : null;
  S.codes = newCodes;
  if (S.exampleId) {
    const ex = S.setup.examples.find((x) => x.id === S.exampleId);
    if (!ex || !sameCodes(ex.codes, S.codes)) { S.exampleId = null; $("example-note").classList.add("hidden"); $("examples").value = ""; }
  }
  refresh();
}
function onSlot(i) {
  const cur = S.codes[i];
  const next = S.part === 0 ? 0 : (cur === S.part ? 0 : S.part);
  if (next === cur) return;
  if (cur === 0 && kOf(S.codes) >= kMax()) { banner(`The model was trained for up to ${kMax()} decaps.`, "error"); return; }
  const c = S.codes.slice();
  c[i] = next;
  changed(c);
}
function apply(codes) { S.preview = null; renderPreview(); changed(codes.slice()); }

async function refresh() {
  renderSlots();
  $("kval").textContent = `K = ${kOf(S.codes)}`;
  const my = ++S.seq;
  runFixes();
  try {
    const c = await api("/api/check", { codes: S.codes, target: S.target });
    if (my !== S.seq) return;
    S.check = c;
    renderVerdict();
    renderAdds();
    drawPlot();
    requestMap();
  } catch (e) { banner(e.message, "error"); }
}

// ---------------------------------------------------------------- setup
function buildControls() {
  const sel = $("target");
  S.setup.targets.forEach((t) => {
    const o = document.createElement("option");
    o.value = t.name; o.textContent = t.label;
    sel.appendChild(o);
  });
  sel.value = S.target;
  sel.addEventListener("change", () => { S.target = sel.value; refresh(); });

  const pal = $("palette");
  const btn = (code, html, title) => {
    const b = document.createElement("button");
    b.dataset.code = code; b.innerHTML = html; b.title = title;
    b.addEventListener("click", () => setPart(code));
    pal.appendChild(b);
  };
  for (const [code, p] of Object.entries(S.setup.parts))
    btn(Number(code), `<span class="dot" style="background:${COL[code]}"></span>${p.name} <small>${p.value}</small>`,
      `${p.name}: ${p.value}, ESL ${p.esl}, ESR ${p.esr} (key ${code})`);
  btn(0, "✕", "Remove: click a decap to remove it (key 0)");
  setPart(1);

  const ex = $("examples");
  S.setup.examples.forEach((e) => {
    const o = document.createElement("option");
    o.value = e.id; o.textContent = e.name;
    ex.appendChild(o);
  });
  ex.addEventListener("change", () => {
    const e = S.setup.examples.find((x) => x.id === ex.value);
    if (!e) return;
    pushHistory();
    S.prevSpectrum = null;
    S.codes = e.codes.slice();
    S.target = e.target; $("target").value = e.target;
    S.exampleId = e.id;
    S.mapIndex = null; setMapIndex(null);
    $("example-note").textContent = e.note;
    $("example-note").classList.remove("hidden");
    refresh();
  });

  $("undo").disabled = true;
  $("undo").addEventListener("click", undo);
  $("clear").addEventListener("click", () => { if (kOf(S.codes)) changed(new Array(52).fill(0)); });

  $("clearmap").addEventListener("click", () => setMapIndex(null));
  for (const b of $("scalemode").querySelectorAll("button")) b.addEventListener("click", () => {
    S.scaleMode = b.dataset.v;
    $("scalemode").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    drawMap();
  });
  for (const b of $("mode").querySelectorAll("button")) b.addEventListener("click", () => {
    S.mode = b.dataset.v;
    $("mode").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    runFixes();
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT" && e.target.type !== "range") return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); return; }
    if (["1", "2", "3"].includes(e.key)) setPart(Number(e.key));
    if (e.key === "0" || e.key.toLowerCase() === "e") setPart(0);
  });

  const m = S.setup.model;
  $("foot").textContent = `Surrogate predictions (${m.checkpoint}, ${m.head}), not eCADSTAR.`;
}
function setPart(code) {
  S.part = code;
  $("palette").querySelectorAll("button").forEach((b) => b.classList.toggle("on", Number(b.dataset.code) === code));
  renderSlots();
}
function undo() {
  const h = S.history.pop();
  if (!h) return;
  S.prevSpectrum = S.check ? S.check.spectrum : null;
  S.codes = h.codes; S.target = h.target; $("target").value = h.target;
  $("undo").disabled = S.history.length === 0;
  refresh();
}

async function init() {
  try {
    S.setup = await api("/api/setup");
  } catch (e) { return; }
  buildBoard();
  buildControls();
  drawPlot();
  refresh();
  window.addEventListener("resize", () => Plotly.Plots.resize("spectrum"));
}
init();
