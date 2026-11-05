// VRCWin interface. Plain JS, no build step. Talks to ui/server.py.
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const NS = "http://www.w3.org/2000/svg";
const S = { meta: null, scen: null, scenName: "", menu: [], picks: [], selGoal: null, replay: null,
            playT: 0, playing: false, result: null, resultName: "", jobSel: null, lastSolveName: "" };

// ---------- helpers ----------
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("on");
  clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.remove("on"), 3500);
}
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined) e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
}
function svg(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmtP = p => (p * 100).toFixed(1) + "%";
const fmtM = m => (m >= 0 ? "+" : "") + Number(m).toFixed(1);
const clone = o => JSON.parse(JSON.stringify(o));
let debounceH;
function changed() {
  clearTimeout(debounceH);
  debounceH = setTimeout(async () => {
    $("#scen-json").value = JSON.stringify(S.scen, null, 2);
    try {
      const ev = await api("/api/evaluate", { scenario: S.scen });
      $("#score-now").innerHTML = `If the match ended now: <b class="red">red ${ev.score_now[0]}</b>
        <b class="blue">blue ${ev.score_now[1]}</b>` + (ev.model ? ` &nbsp;·&nbsp; value model P(red win) <b>${fmtP(ev.model.p_red_win)}</b>` : "");
      await refreshMenu();
    } catch (e) { $("#score-now").textContent = "Scenario error: " + e.message; }
  }, 250);
}

// ---------- tabs ----------
$$("#tabs button").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(name) {
  $$("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  $$(".tab").forEach(t => t.classList.toggle("on", t.id === "tab-" + name));
  if (name === "test") drawReplay();
  if (name === "solve") pollJobs();
  if (name === "results") refreshResults();
  if (name === "train") refreshTrain();
  if (name === "examine") { fillHeatRobots(); drawField($("#field-heat"), {}); runEval(false); }
  location.hash = name;
}

// ---------- field drawing ----------
const HALF = 1.83;
function drawField(el, o) {
  const m = S.meta, sc = S.scen;
  el.innerHTML = "";
  const g = svg("g");
  el.append(g);
  g.append(svg("rect", { x: -HALF, y: -HALF, width: 2 * HALF, height: 2 * HALF, fill: css("--field"), stroke: css("--field-line"), "stroke-width": 0.02 }));
  // heat underlay
  if (o.heat) {
    const st = o.heat.step;
    for (const [x, y, p] of o.heat.cells) {
      g.append(svg("rect", { x: x - st / 2, y: -y - st / 2, width: st, height: st, fill: heatColor(p - o.heat.ref), opacity: 0.85 }));
    }
  }
  // quadrant diagonals and alliance stations
  g.append(svg("line", { x1: -HALF, y1: -HALF, x2: HALF, y2: HALF, stroke: css("--field-line"), "stroke-width": 0.012, "stroke-dasharray": "0.05 0.04" }));
  g.append(svg("line", { x1: -HALF, y1: HALF, x2: HALF, y2: -HALF, stroke: css("--field-line"), "stroke-width": 0.012, "stroke-dasharray": "0.05 0.04" }));
  g.append(svg("rect", { x: -HALF - 0.06, y: -HALF, width: 0.05, height: 2 * HALF, fill: css("--red") }));
  g.append(svg("rect", { x: HALF + 0.01, y: -HALF, width: 0.05, height: 2 * HALF, fill: css("--blue") }));
  g.append(svg("circle", { cx: 0, cy: 0, r: m.midfield_radius, fill: "none", stroke: css("--muted"), "stroke-width": 0.015, "stroke-dasharray": "0.04 0.03" }));
  const toggles = o.toggles || sc.toggles;
  const goals = o.goals || sc.goals;
  // toggles
  for (const [q, t] of Object.entries(m.quadrants)) {
    const col = toggles[q] === "R" ? css("--red") : toggles[q] === "B" ? css("--blue") : css("--muted");
    const horiz = Math.abs(t.y) > Math.abs(t.x);
    const w = horiz ? 0.32 : 0.09, hh = horiz ? 0.09 : 0.32;
    const r = svg("rect", { x: t.x - w / 2, y: -t.y - hh / 2, width: w, height: hh, rx: 0.02, fill: col, class: o.edit ? "clickable" : "" });
    if (o.edit) r.addEventListener("click", () => { const c = ["R", "B", "N"]; sc.toggles[q] = c[(c.indexOf(sc.toggles[q]) + 1) % 3]; renderEditor(); });
    g.append(r);
    const lx = t.x * (horiz ? 0.88 : 0.78), ly = -t.y * (horiz ? 0.88 : 0.78) + (horiz ? 0 : 0.03);
    g.append(svg("text", { x: lx, y: ly, "font-size": 0.085, "text-anchor": "middle", fill: css("--muted") }, "Toggle " + q));
  }
  // loaders
  for (const [k, l] of Object.entries(m.loaders)) {
    g.append(svg("rect", { x: l.x - 0.08, y: -l.y - 0.08, width: 0.16, height: 0.16, fill: "none", stroke: l.alliance === "red" ? css("--red") : css("--blue"), "stroke-width": 0.02 }));
    const side = l.alliance === "red" ? "red" : "blue";
    const pins = (o.loader || sc.loader)[side] || {};
    const n = Object.values(pins).reduce((a, b) => a + b, 0);
    g.append(svg("text", { x: l.x, y: -l.y + 0.03, "font-size": 0.08, "text-anchor": "middle", fill: css("--ink") }, n));
  }
  // goals
  for (const [k, gl] of Object.entries(m.goals)) {
    const v = goals[k] || [0, 0, 0];
    const ring = gl.kind === "red" ? css("--red") : gl.kind === "blue" ? css("--blue") : css("--ink");
    const r = gl.kind === "tall" ? 0.13 : 0.1;
    const c = svg("circle", { cx: gl.x, cy: -gl.y, r, fill: css("--panel"), stroke: ring, "stroke-width": S.selGoal === k && o.edit ? 0.04 : 0.02, class: o.edit ? "clickable" : "" });
    if (o.edit) c.addEventListener("click", () => { S.selGoal = k; renderEditor(); const cell = $(`.goal-cell[data-g="${k}"] input`); if (cell) cell.focus(); });
    g.append(c);
    const parts = [[v[0], css("--red")], [v[1], css("--blue")], [v[2], css("--yellow")]];
    parts.forEach(([n, col], i) => g.append(svg("text", { x: gl.x - 0.06 + i * 0.06, y: -gl.y + 0.03, "font-size": 0.075, "text-anchor": "middle", fill: col, "font-weight": 700 }, n)));
    g.append(svg("text", { x: gl.x, y: -gl.y - r - 0.03, "font-size": 0.065, "text-anchor": "middle", fill: css("--muted") }, k));
  }
  // floor stacks
  for (const [k, st] of Object.entries(sc.floor || {})) {
    const n = o.floor ? o.floor[k] : Object.values(st.pins).reduce((a, b) => a + b, 0);
    const e = svg("g", { class: o.edit ? "drag" : "" });
    e.append(svg("rect", { x: st.x - 0.07, y: -st.y - 0.07, width: 0.14, height: 0.14, rx: 0.03, fill: css("--yellow"), opacity: 0.85 }));
    e.append(svg("text", { x: st.x, y: -st.y + 0.03, "font-size": 0.08, "text-anchor": "middle", fill: "#222", "font-weight": 700 }, n));
    e.append(svg("text", { x: st.x, y: -st.y + 0.15, "font-size": 0.06, "text-anchor": "middle", fill: css("--muted") }, k));
    if (o.edit) makeDraggable(el, e, p => { st.x = p.x; st.y = p.y; });
    g.append(e);
  }
  // robots
  const pos = o.robots || sc.robots.map(r => [r.x, r.y]);
  const holding = o.holding || sc.robots.map(r => r.holding || null);
  sc.robots.forEach((rb, i) => {
    const [x, y] = pos[i];
    const col = rb.alliance === "red" ? css("--red") : css("--blue");
    const e = svg("g", { class: o.edit ? "drag" : "" });
    e.append(svg("rect", { x: x - 0.19, y: -y - 0.19, width: 0.38, height: 0.38, rx: 0.05, fill: col, opacity: 0.9, stroke: o.heatRobot === i ? css("--ink") : "none", "stroke-width": 0.03 }));
    e.append(svg("text", { x, y: -y + 0.035, "font-size": 0.1, "text-anchor": "middle", fill: "#fff", "font-weight": 700 }, rb.name.replace(/^(red|blue) ?/, (m0, a) => a[0].toUpperCase())));
    if (holding[i]) e.append(svg("text", { x, y: -y + 0.15, "font-size": 0.07, "text-anchor": "middle", fill: "#fff" }, holding[i]));
    if (o.edit) makeDraggable(el, e, p => { rb.x = p.x; rb.y = p.y; });
    g.append(e);
  });
}
function heatColor(d) {
  // d = change in P(win) for that robot's alliance. Green better, purple worse.
  const a = Math.max(-1, Math.min(1, d / 0.15));
  return a >= 0 ? `rgba(31,160,90,${0.15 + 0.75 * a})` : `rgba(140,70,200,${0.15 + 0.75 * -a})`;
}
function makeDraggable(svgEl, node, set) {
  node.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    const move = e => {
      const pt = svgEl.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      const p = pt.matrixTransform(svgEl.getScreenCTM().inverse());
      const cl = v => Math.round(Math.max(-1.7, Math.min(1.7, v)) * 100) / 100;
      set({ x: cl(p.x), y: cl(-p.y) });
      drawField(svgEl, { edit: true });
    };
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); renderEditor(); };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });
}

// ---------- scenario editor ----------
function num(v, set, attrs = {}) {
  return h("input", { type: "number", value: v, step: attrs.step || 1, min: attrs.min, max: attrs.max, class: attrs.cls,
    onchange: e => { set(e.target.value === "" ? 0 : Number(e.target.value)); renderEditor(); } });
}
function renderEditor() {
  const sc = S.scen, m = S.meta;
  drawField($("#field-edit"), { edit: true });
  $("#scenario-name").textContent = S.scenName ? "Scenario: " + S.scenName : "Unsaved scenario";
  $("#f-desc").value = sc.description || "";
  $("#f-time").value = sc.time_remaining;
  $("#f-base-r").value = (sc.base || [0, 0])[0];
  $("#f-base-b").value = (sc.base || [0, 0])[1];
  // robots
  const R = $("#robots"); R.innerHTML = "";
  sc.robots.forEach((rb, i) => {
    const removed = new Set(rb.remove_capabilities || []);
    const scale = rb.scale || {};
    const setScale = (keys, v) => { rb.scale = rb.scale || {}; keys.forEach(k => { if (v === 1) delete rb.scale[k]; else rb.scale[k] = v; }); if (!Object.keys(rb.scale).length) delete rb.scale; };
    R.append(h("div", { class: "robot-card " + rb.alliance },
      h("div", { class: "row" },
        h("label", {}, "Name", h("input", { value: rb.name, onchange: e => { rb.name = e.target.value; renderEditor(); } })),
        h("label", {}, "x (m)", num(rb.x, v => rb.x = v, { step: 0.05 })),
        h("label", {}, "y (m)", num(rb.y, v => rb.y = v, { step: 0.05 })),
        h("label", {}, "Holding", (() => {
          const s = h("select", { onchange: e => { if (e.target.value) rb.holding = e.target.value; else delete rb.holding; renderEditor(); } },
            h("option", { value: "" }, "nothing"), ...m.pin_types.map(t => h("option", { value: t, selected: rb.holding === t }, t)));
          return s;
        })())),
      h("div", { class: "row" },
        h("label", {}, "Speed ×", num(scale.max_speed ?? 1, v => setScale(["max_speed"], v), { step: 0.05, cls: "tiny", min: 0.3, max: 2 })),
        h("label", {}, "Cycle time ×", num(scale.intake_time ?? 1, v => setScale(["intake_time", "place_time"], v), { step: 0.05, cls: "tiny", min: 0.3, max: 3 })),
        h("label", {}, "Toggle time ×", num(scale.toggle_time ?? 1, v => setScale(["toggle_time"], v), { step: 0.05, cls: "tiny", min: 0.3, max: 3 })),
        h("label", {}, "Push success ×", num(scale.p_push ?? 1, v => setScale(["p_push"], v), { step: 0.1, cls: "tiny", min: 0, max: 2.8 }))),
      h("div", { class: "caps" }, "Can:", ...m.capabilities.map(c => h("label", { class: "inline" },
        h("input", { type: "checkbox", checked: !removed.has(c), onchange: e => {
          const s = new Set(rb.remove_capabilities || []); e.target.checked ? s.delete(c) : s.add(c);
          if (s.size) rb.remove_capabilities = [...s]; else delete rb.remove_capabilities; renderEditor(); } }),
        { tall_goal: "Midfield Goal", toggle: "Toggles", push: "push" }[c] || c)))));
  });
  // goals
  const G = $("#goals"); G.innerHTML = "";
  for (const [k, gl] of Object.entries(m.goals)) {
    sc.goals[k] = sc.goals[k] || [0, 0, 0];
    const v = sc.goals[k];
    const kind = { red: "red Alliance", blue: "blue Alliance", short: "neutral", tall: "Midfield" }[gl.kind];
    G.append(h("div", { class: "goal-cell" + (S.selGoal === k ? " sel" : ""), "data-g": k },
      h("div", {}, h("b", {}, k), " ", h("span", { class: "hint" }, kind)),
      ...[0, 1, 2].map(j => num(v[j], x => v[j] = Math.max(0, x), { min: 0 }))));
  }
  // toggles
  const T = $("#toggles"); T.innerHTML = "";
  for (const q of Object.keys(m.quadrants)) {
    T.append(h("label", {}, "Toggle " + q, h("select", { onchange: e => { sc.toggles[q] = e.target.value; renderEditor(); } },
      ...[["R", "red"], ["B", "blue"], ["N", "neutral"]].map(([v, t]) => h("option", { value: v, selected: sc.toggles[q] === v }, t)))));
  }
  // floor stacks
  const F = $("#stacks"); F.innerHTML = "";
  for (const [k, st] of Object.entries(sc.floor || {})) {
    F.append(h("div", { class: "stack-row" }, h("b", {}, k),
      h("label", {}, "x", num(st.x, v => st.x = v, { step: 0.05, cls: "tiny" })),
      h("label", {}, "y", num(st.y, v => st.y = v, { step: 0.05, cls: "tiny" })),
      ...m.pin_types.map(t => h("label", {}, t, num(st.pins[t] || 0, v => { if (v > 0) st.pins[t] = v; else delete st.pins[t]; }, { min: 0, cls: "tiny" }))),
      h("button", { class: "ghost small", onclick: () => { delete sc.floor[k]; renderEditor(); } }, "remove")));
  }
  // loaders
  const L = $("#loaders"); L.innerHTML = "";
  for (const side of ["red", "blue"]) {
    sc.loader[side] = sc.loader[side] || {};
    L.append(h("div", { class: "stack-row" }, h("b", {}, side),
      ...m.pin_types.map(t => h("label", {}, t, num(sc.loader[side][t] || 0, v => { if (v > 0) sc.loader[side][t] = v; else delete sc.loader[side][t]; }, { min: 0, cls: "tiny" })))));
  }
  changed();
}
$("#f-desc").addEventListener("change", e => { S.scen.description = e.target.value; changed(); });
$("#f-time").addEventListener("change", e => { S.scen.time_remaining = Number(e.target.value); changed(); });
$("#f-base-r").addEventListener("change", e => { S.scen.base = [Number(e.target.value), (S.scen.base || [0, 0])[1]]; changed(); });
$("#f-base-b").addEventListener("change", e => { S.scen.base = [(S.scen.base || [0, 0])[0], Number(e.target.value)]; changed(); });
$("#add-stack").addEventListener("click", () => {
  S.scen.floor = S.scen.floor || {};
  let k = 1; while (S.scen.floor["S" + k]) k++;
  S.scen.floor["S" + k] = { x: 0, y: 1.0, pins: { YY: 1 } }; renderEditor();
});
$("#scen-json-apply").addEventListener("click", () => {
  try { S.scen = JSON.parse($("#scen-json").value); renderEditor(); toast("Applied"); } catch (e) { toast("Bad JSON: " + e.message); }
});
async function refreshScenList(sel) {
  const list = await api("/api/scenarios");
  $("#scen-list").innerHTML = "";
  list.forEach(f => $("#scen-list").append(h("option", { value: f.name, selected: f.name === sel }, f.name)));
}
async function loadScenario(name) {
  S.scen = await api("/api/scenario?name=" + encodeURIComponent(name));
  S.scenName = name;
  $("#scen-save-name").value = name.replace(/\.json$/, "");
  $("#solve-name").value = name.replace(/\.json$/, "");
  S.picks = [];
  renderEditor();
}
$("#scen-load").addEventListener("click", () => loadScenario($("#scen-list").value));
$("#scen-new").addEventListener("click", () => {
  S.scen = {
    description: "", time_remaining: 25, base: [0, 0], goals: {}, toggles: { L: "R", B: "R", R: "B", T: "B" },
    floor: { S_N: { x: 0, y: 0.6, pins: { YY: 1 } }, S_S: { x: 0, y: -0.6, pins: { YY: 1 } } },
    loader: { red: { RY: 3 }, blue: { BY: 3 } },
    robots: [{ name: "red 1", alliance: "red", x: -1.2, y: 0.3 }, { name: "red 2", alliance: "red", x: -1.2, y: -0.3 },
             { name: "blue 1", alliance: "blue", x: 1.2, y: 0.3 }, { name: "blue 2", alliance: "blue", x: 1.2, y: -0.3 }] };
  S.scenName = ""; S.picks = []; renderEditor();
});
$("#scen-save").addEventListener("click", async () => {
  const name = $("#scen-save-name").value.trim();
  if (!name) return toast("Give the scenario a name");
  try {
    const r = await api("/api/scenario", { name, scenario: S.scen });
    S.scenName = r.saved; $("#solve-name").value = r.saved.replace(/\.json$/, "");
    await refreshScenList(r.saved); renderEditor(); toast("Saved scenarios/" + r.saved);
  } catch (e) { toast(e.message); }
});

// ---------- test: fixed plans ----------
async function refreshMenu() {
  S.menu = await api("/api/plans", { scenario: S.scen });
  const P = $("#plan-pickers"); P.innerHTML = "";
  S.scen.robots.forEach((rb, i) => {
    const ids = S.menu[i].map(p => p.id);
    if (!ids.includes(S.picks[i])) S.picks[i] = ids.includes("score|mid@late") ? "score|mid@late" : ids[0];
    P.append(h("label", { class: "block" }, rb.name + " (" + rb.alliance + ")",
      h("select", { onchange: e => S.picks[i] = e.target.value },
        ...S.menu[i].map(p => h("option", { value: p.id, selected: p.id === S.picks[i] }, p.text)))));
  });
}
$("#sim-run").addEventListener("click", async () => {
  const b = $("#sim-run"); b.disabled = true; b.textContent = "Simulating...";
  try {
    const r = await api("/api/simulate", { scenario: S.scen, plans: S.picks, episodes: Number($("#sim-n").value), seed: Number($("#sim-seed").value) });
    const hm = new Map(r.histogram); const lo = r.histogram[0][0], hi = r.histogram[r.histogram.length - 1][0];
    r.histogram = []; for (let b0 = lo; b0 <= hi; b0 += 10) r.histogram.push([b0, hm.get(b0) || 0]);
    const maxc = Math.max(...r.histogram.map(x => x[1]));
    $("#sim-out").innerHTML = "";
    $("#sim-out").append(
      h("div", { class: "kpis" },
        kpi(fmtP(r.p_red_win), "P(red win)"), kpi(fmtM(r.e_margin_red), "E[red margin]"),
        kpi(`${r.margin_p10} to ${r.margin_p90}`, "red margin, 10th-90th pct")),
      h("table", {}, h("tr", {}, h("th", {}, "red margin"), h("th", {}, "share of " + r.episodes)),
        ...r.histogram.map(([b0, c]) => h("tr", {}, h("td", { class: "num" }, `${b0} to ${b0 + 9}`),
          h("td", {}, h("span", { class: "bar " + (b0 >= 0 ? "pos" : "neg"), style: `width:${(c / maxc) * 220}px` }), " " + fmtP(c / r.episodes))))));
    S.replay = r.replay; S.playT = r.replay.start_t; drawReplay();
  } catch (e) { toast(e.message); }
  b.disabled = false; b.textContent = "Simulate";
});
function kpi(v, l) { return h("div", { class: "kpi" }, h("div", { class: "v" }, v), h("div", { class: "l" }, l)); }
function replayAt(T) {
  const rp = S.replay, n = S.scen.robots.length;
  const pos = S.scen.robots.map(r => [r.x, r.y]);
  for (let i = 0; i < n; i++) {
    const segs = rp.segments.filter(s => s.robot === i && s.t0 >= T - 1e-9);
    if (!segs.length) continue;
    const s = segs.reduce((a, b) => (b.t0 <= a.t0 ? b : a));
    if (T <= s.arrive || s.t0 === s.arrive) pos[i] = s.to;
    else { const f = (s.t0 - T) / (s.t0 - s.arrive); pos[i] = [s.from[0] + (s.to[0] - s.from[0]) * f, s.from[1] + (s.to[1] - s.from[1]) * f]; }
  }
  let fr = rp.frames[0];
  for (const f of rp.frames) if (f.t >= T - 1e-9) fr = f;
  return { pos, fr };
}
function drawReplay() {
  if (!S.scen) return;
  if (!S.replay) { drawField($("#field-replay"), {}); return; }
  const T = S.playT, { pos, fr } = replayAt(T);
  drawField($("#field-replay"), { robots: pos, toggles: Object.fromEntries(Object.keys(S.meta.quadrants).map((q, k) => [q, fr.toggles[k]])), goals: fr.goals, holding: fr.holding, floor: fr.floor, loader: fr.loader });
  $("#clock").textContent = T.toFixed(1) + " s left";
  $("#scrub").value = Math.round((1 - T / S.replay.start_t) * 1000);
  $("#replay-score").innerHTML = `<b class="red">red ${fr.red}</b> <b class="blue">blue ${fr.blue}</b> <span class="hint">score if it ended at this moment · final ${S.replay.final[0]}-${S.replay.final[1]}</span>`;
  const L = $("#replay-log");
  if (L.dataset.for !== String(S.replay.start_t) + S.replay.segments.length) {
    L.innerHTML = ""; L.dataset.for = String(S.replay.start_t) + S.replay.segments.length;
    S.replay.segments.forEach((s, k) => L.append(h("li", { "data-t": s.t0 }, `${s.t0.toFixed(1)} s · ${s.name}: ${s.action}`)));
  }
  let last = null;
  $$("li", L).forEach(li => { const on = Number(li.dataset.t) >= T - 1e-9; li.classList.toggle("now", false); if (on) last = li; });
  if (last) { last.classList.add("now"); if (last.offsetTop < L.scrollTop || last.offsetTop > L.scrollTop + L.clientHeight - 20) L.scrollTop = last.offsetTop - L.clientHeight / 2; }
}
$("#scrub").addEventListener("input", e => { if (!S.replay) return; S.playT = S.replay.start_t * (1 - e.target.value / 1000); drawReplay(); });
$("#play").addEventListener("click", () => {
  if (!S.replay) return toast("Simulate first");
  S.playing = !S.playing; $("#play").textContent = S.playing ? "Pause" : "Play";
  if (S.playT <= 0.01) S.playT = S.replay.start_t;
  let prev = performance.now();
  const step = now => {
    if (!S.playing) return;
    S.playT = Math.max(0, S.playT - (now - prev) / 1000 * 2); prev = now;   // 2x real time
    drawReplay();
    if (S.playT <= 0) { S.playing = false; $("#play").textContent = "Play"; return; }
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
});
$("#tests-run").addEventListener("click", async () => {
  $("#tests-out").textContent = "Running...";
  const j = await api("/api/jobs", { kind: "tests", label: "rule tests" });
  const poll = async () => {
    const r = await api("/api/job?id=" + j.id);
    $("#tests-out").textContent = r.log;
    if (r.status === "running") setTimeout(poll, 800);
  };
  poll();
});

// ---------- solve / jobs ----------
async function startJob(spec) {
  try { const j = await api("/api/jobs", spec); S.jobSel = j.id; toast("Started: " + j.label); showTab("solve"); }
  catch (e) { toast(e.message); }
}
$("#solve-run").addEventListener("click", () => {
  const name = $("#solve-name").value.trim() || "ui_result";
  S.lastSolveName = name;
  startJob({ kind: "solve", scenario: S.scen, preset: $("#solve-preset").value, name, seed: Number($("#solve-seed").value), label: `solve ${name} (${$("#solve-preset").value})` });
});
$("#sens-run").addEventListener("click", () => {
  const name = $("#solve-name").value.trim() || "ui_result";
  startJob({ kind: "sensitivity", scenario: S.scen, preset: $("#solve-preset").value, name, label: `sensitivity ${name}` });
});
let jobTimer, seenDone = new Set();
async function pollJobs() {
  clearTimeout(jobTimer);
  let jobs = [];
  try { jobs = await api("/api/jobs"); } catch (e) { return; }
  const J = $("#jobs"); J.innerHTML = jobs.length ? "" : '<p class="hint">No jobs yet in this session.</p>';
  if (!S.jobSel && jobs.length) S.jobSel = jobs[0].id;
  for (const j of jobs) {
    J.append(h("div", { class: "jobrow" + (j.id === S.jobSel ? " sel" : ""), onclick: () => { S.jobSel = j.id; pollJobs(); } },
      h("span", { class: "st " + j.status }, j.status), h("b", {}, j.label), h("span", { class: "hint" }, Math.round(j.elapsed) + " s"),
      h("span", { class: "last" }, j.last),
      j.status === "running" ? h("button", { class: "small ghost", onclick: async e => { e.stopPropagation(); await api("/api/cancel", { id: j.id }); pollJobs(); } }, "Cancel") : null,
      j.status === "done" && j.result && j.result.result ? h("button", { class: "small", onclick: e => { e.stopPropagation(); openResult(j.result.result); } }, "Open result") : null));
    if (j.status !== "running" && !seenDone.has(j.id)) {
      seenDone.add(j.id);
      if (j.status === "done" && (j.kind === "dataset" || j.kind === "fit")) refreshTrain();
      if (performance.now() > 5000) toast(`${j.label}: ${j.status}`);
    }
  }
  if (S.jobSel) {
    try { const d = await api("/api/job?id=" + S.jobSel); $("#job-title").textContent = "Log: " + d.label; const L = $("#job-log"); const atEnd = L.scrollTop + L.clientHeight >= L.scrollHeight - 20; L.textContent = d.log; if (atEnd) L.scrollTop = L.scrollHeight; } catch (e) { }
  }
  if (jobs.some(j => j.status === "running")) jobTimer = setTimeout(pollJobs, 1500);
}

// ---------- results ----------
async function refreshResults(sel) {
  const list = await api("/api/results");
  const cur = sel || $("#res-list").value || S.resultName;
  $("#res-list").innerHTML = "";
  list.forEach(f => $("#res-list").append(h("option", { value: f.name, selected: f.name === cur }, f.name)));
  if (!S.result && list.length) openResult(cur || list[0].name, false);
}
async function openResult(name, jump = true) {
  S.result = await api("/api/result?name=" + encodeURIComponent(name));
  S.resultName = name;
  if (jump) { showTab("results"); await refreshResults(name); }
  renderResult();
}
$("#res-load").addEventListener("click", () => openResult($("#res-list").value));
function renderResult() {
  const r = S.result, V = $("#res-view"); V.innerHTML = "";
  if (!r) return;
  if (!r.overall && r.baseline) return renderSensitivity(r, V);
  if (!r.overall) { V.append(h("pre", { class: "log" }, JSON.stringify(r, null, 1))); return; }
  const o = r.overall;
  V.append(h("p", {}, r.scenario || ""),
    h("div", { class: "kpis" },
      kpi(fmtP(o.p_red_win), "P(red win)"), kpi(fmtP(1 - o.p_red_win), "P(blue win)"),
      kpi(fmtM(o.e_margin_red), "E[red margin]"), kpi(`${r.time_remaining} s`, "window solved"),
      kpi(`${fmtM(o.red_could_gain * 100)} / ${fmtM(o.blue_could_gain * 100)}`, "pts of P(win) red / blue could still gain")),
    h("p", { class: "hint" }, `Settings: ${r.preset || ""} ${JSON.stringify(r.settings || {})}${r.seconds ? ` · ${r.seconds} s` : ""}. "Could still gain" near 0 means the equilibrium is tight.`));
  for (const side of ["red", "blue"]) {
    V.append(h("div", { class: "card " + side }, h("h2", {}, `${side[0].toUpperCase() + side.slice(1)} equilibrium plan mix`),
      ...(r.equilibrium[side] || []).map(m => h("div", { class: "mix" }, h("span", { class: "w" }, fmtP(m.p)),
        Object.entries(m.plan).map(([k, v]) => h("span", {}, h("b", {}, k + ": "), v + "  ")))),
      ...Object.entries(r.lines).filter(([, d]) => d.alliance === side).map(([robot, d]) => lineTable(robot, d))));
  }
  if (r.sample_trace) {
    V.append(h("div", { class: "card" }, h("h2", {}, "Sample window under the most likely plans"),
      h("ol", { class: "log" }, ...r.sample_trace.map(e => h("li", {}, `${e.t.toFixed(1)} s · ${e.robot}: ${e.action}`))),
      h("p", {}, `Final ${r.sample_final[0]}-${r.sample_final[1]}`)));
  }
}
function lineTable(robot, d) {
  const best = d.lines[0].p_win;
  return h("div", {}, h("h3", {}, `${robot}: every line, scored as ${d.alliance}'s P(win) against the opponents' equilibrium mix`),
    h("table", {}, h("tr", {}, h("th", {}, "line"), h("th", { class: "num" }, "P(win)"), h("th", {}, ""), h("th", { class: "num" }, "margin"),
      h("th", { class: "num" }, "Δ vs best"), h("th", { class: "num" }, "eq. weight"), h("th", {}, "best partner line")),
      ...d.lines.map((l, k) => h("tr", { class: k === 0 ? "best" : "" }, h("td", {}, l.line), h("td", { class: "num" }, fmtP(l.p_win)),
        h("td", {}, h("span", { class: "bar", style: `width:${Math.max(1, l.p_win * 90)}px` })),
        h("td", { class: "num" }, fmtM(l.e_margin)), h("td", { class: "num" }, (l.p_win - best).toFixed(3)),
        h("td", { class: "num" }, l.eq_weight ? fmtP(l.eq_weight) : ""), h("td", { class: "hint" }, l.best_partner || "")))));
}
function renderSensitivity(r, V) {
  const robots = Object.keys(r.baseline.top);
  V.append(h("h2", {}, "Sensitivity: does each robot's best line survive a stat change?"),
    h("table", {}, h("tr", {}, h("th", {}, "run"), h("th", { class: "num" }, "P(red win)"), h("th", { class: "num" }, "E[red margin]"), h("th", {}, "top line changed for"), h("th", {}, "equilibrium changed for")),
      ...Object.entries(r).map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", { class: "num" }, fmtP(v.p_red_win)), h("td", { class: "num" }, fmtM(v.e_margin_red)),
        h("td", {}, k === "baseline" ? "" : (v.changed || []).join(", ") || "nobody"), h("td", {}, k === "baseline" ? "" : (v.eq_changed || []).join(", ") || "nobody")))),
    h("h3", {}, "Baseline top lines"),
    h("table", {}, ...robots.map(rb => h("tr", {}, h("td", {}, rb), h("td", {}, r.baseline.top[rb]), h("td", { class: "num" }, fmtP(r.baseline.top_p[rb]))))));
}
$("#res-to-scen").addEventListener("click", async () => {
  const r = S.result; if (!r) return;
  if (r.scenario_input) { S.scen = clone(r.scenario_input); S.scenName = ""; }
  else {
    const guess = S.resultName.replace(/(\.sensitivity)?\.json$/, ".json");
    try { await loadScenario(guess); } catch (e) { return toast("This result has no stored scenario and scenarios/" + guess + " is missing"); }
  }
  S.picks = []; renderEditor(); showTab("scenario"); toast("Scenario loaded");
});
$("#res-replay").addEventListener("click", async () => {
  const r = S.result; if (!r || !r.equilibrium) return toast("Open a solve result first");
  if (r.scenario_input) S.scen = clone(r.scenario_input);
  else { try { await loadScenario(S.resultName.replace(/\.json$/, ".json")); } catch (e) { } }
  renderEditor();
  await refreshMenu();
  const plan = Object.assign({}, r.equilibrium.red[0].plan, r.equilibrium.blue[0].plan);
  S.scen.robots.forEach((rb, i) => { const hit = S.menu[i].find(p => p.text === plan[rb.name]); if (hit) S.picks[i] = hit.id; });
  await refreshMenu();
  showTab("test"); $("#sim-run").click();
});

// ---------- train ----------
async function refreshTrain() {
  const info = await api("/api/model");
  const T = $("#train-info"); T.innerHTML = "";
  T.append(h("div", { class: "kpis" }, kpi(info.samples, "labelled snapshots")));
  const st = info.model;
  if (!st) {
    T.append(h("p", {}, info.samples >= 8 ? "No model fitted yet. Click Refit only." : "Need at least 8 labelled snapshots before a model can be fitted."));
  } else {
    const better = st.model.brier < st.baseline.brier;
    T.append(h("table", {}, h("tr", {}, h("th", {}, "out-of-fold, " + st.n + " snapshots"), h("th", { class: "num" }, "Brier"), h("th", { class: "num" }, "log loss"), h("th", { class: "num" }, "mean abs error")),
      h("tr", { class: "best" }, h("td", {}, "value model"), h("td", { class: "num" }, st.model.brier.toFixed(4)), h("td", { class: "num" }, st.model.log_loss.toFixed(3)), h("td", { class: "num" }, st.model.mae.toFixed(3))),
      h("tr", {}, h("td", {}, "margin + time baseline"), h("td", { class: "num" }, st.baseline.brier.toFixed(4)), h("td", { class: "num" }, st.baseline.log_loss.toFixed(3)), h("td", { class: "num" }, st.baseline.mae.toFixed(3)))),
      h("p", { class: better ? "" : "hint" }, better ? `The model beats the baseline (Brier ${st.model.brier.toFixed(4)} vs ${st.baseline.brier.toFixed(4)}).` : "The model does not beat the baseline yet; add more labels."),
      h("p", { class: "hint" }, "Fitted " + (st.fitted_at || "")));
  }
  if (info.recent.length) {
    T.append(h("h3", {}, "Latest labels"), h("table", {}, h("tr", {}, h("th", { class: "num" }, "s left"), h("th", { class: "num" }, "P(red win)"), h("th", { class: "num" }, "margin"), h("th", { class: "num" }, "solve s")),
      ...info.recent.slice().reverse().map(r => h("tr", {}, h("td", { class: "num" }, r.t), h("td", { class: "num" }, fmtP(r.p_red_win)), h("td", { class: "num" }, fmtM(r.e_margin_red)), h("td", { class: "num" }, r.seconds ?? "")))));
  }
  drawCalib(st);
  const W = $("#weights"); W.innerHTML = "";
  if (info.weights) {
    const mx = Math.max(...Object.values(info.weights).map(Math.abs)) || 1;
    W.append(h("p", { class: "hint" }, "Logit weight per standard deviation of each feature; positive pushes toward red."),
      h("table", {}, ...Object.entries(info.weights).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).map(([k, w]) =>
        h("tr", {}, h("td", {}, k), h("td", { class: "num" }, w.toFixed(2)), h("td", {}, h("span", { class: "bar " + (w >= 0 ? "pos" : "neg"), style: `width:${Math.abs(w) / mx * 160}px` }))))));
  }
}
function drawCalib(st) {
  const C = $("#calib"); C.innerHTML = "";
  const P = 30, W = 300 - 2 * P;
  C.append(svg("rect", { x: P, y: P, width: W, height: W, fill: "none", stroke: css("--line") }));
  C.append(svg("line", { x1: P, y1: P + W, x2: P + W, y2: P, stroke: css("--muted"), "stroke-dasharray": "3 3" }));
  C.append(svg("text", { x: 150, y: 296, "text-anchor": "middle" }, "predicted P(red win)"));
  const yl = svg("text", { x: 10, y: 150, "text-anchor": "middle", transform: "rotate(-90 10 150)" }, "solver P(red win)"); C.append(yl);
  if (!st || !st.calibration) return;
  for (const [pts, col] of [[st.calibration_baseline, css("--muted")], [st.calibration, css("--accent")]]) {
    for (const [p, y] of pts) C.append(svg("circle", { cx: P + p * W, cy: P + (1 - y) * W, r: 3, fill: col, opacity: 0.75 }));
  }
  C.append(svg("circle", { cx: P + 10, cy: P + 10, r: 3, fill: css("--accent") })); C.append(svg("text", { x: P + 18, y: P + 13 }, "model"));
  C.append(svg("circle", { cx: P + 10, cy: P + 24, r: 3, fill: css("--muted") })); C.append(svg("text", { x: P + 18, y: P + 27 }, "baseline"));
}
$("#train-gen").addEventListener("click", () => {
  const seed = $("#train-seed").value;
  startJob({ kind: "dataset", scenario: S.scen, samples: Number($("#train-n").value), seed: seed === "" ? undefined : Number(seed), label: `label ${$("#train-n").value} snapshots + fit` });
});
$("#train-fit").addEventListener("click", () => startJob({ kind: "fit", label: "fit value model" }));

// ---------- examine ----------
function fillHeatRobots() {
  const s = $("#heat-robot"), cur = s.value;
  s.innerHTML = "";
  S.scen.robots.forEach((r, i) => s.append(h("option", { value: i, selected: String(i) === cur }, r.name)));
}
async function runEval(heat) {
  const i = Number($("#heat-robot").value || 0);
  let ev;
  try { ev = await api("/api/evaluate", { scenario: S.scen, heat_robot: heat ? i : null }); } catch (e) { return toast(e.message); }
  const O = $("#eval-out"); O.innerHTML = "";
  O.append(h("div", { class: "kpis" }, kpi(`${ev.score_now[0]}-${ev.score_now[1]}`, "score if it ended now"),
    ev.model ? kpi(fmtP(ev.model.p_red_win), "value model P(red win)") : null));
  if (!ev.model) O.append(h("p", {}, "No value model yet. Train one in the Train tab; until then the features below are what it will use."));
  else {
    O.append(h("p", { class: "hint" }, `Trained on ${ev.model.stats.n} solver labels; out-of-fold Brier ${ev.model.stats.model.brier.toFixed(4)} (baseline ${ev.model.stats.baseline.brier.toFixed(4)}). For a decision, run a solve; the model is the fast first read.`));
    const c = ev.model.contributions, mx = Math.max(...Object.values(c).map(Math.abs)) || 1;
    O.append(h("h3", {}, "Why: contribution to red's logit"), h("table", {}, ...Object.entries(c).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).map(([k, v]) =>
      h("tr", {}, h("td", {}, k), h("td", { class: "num" }, v.toFixed(2)), h("td", {}, h("span", { class: "bar " + (v >= 0 ? "pos" : "neg"), style: `width:${Math.abs(v) / mx * 140}px` }))))));
  }
  const F = $("#feat-out"); F.innerHTML = "";
  F.append(h("table", {}, ...Object.entries(ev.features).map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", { class: "num" }, v.toFixed(3))))));
  if (heat && ev.heat) {
    const red = S.scen.robots[i].alliance === "red";
    const cells = ev.heat.cells.map(([x, y, p]) => [x, y, red ? p : 1 - p]);
    const ref = red ? ev.model.p_red_win : 1 - ev.model.p_red_win;
    drawField($("#field-heat"), { heat: { cells, step: ev.heat.step, ref }, heatRobot: i });
    const best = cells.reduce((a, b) => (b[2] > a[2] ? b : a));
    O.append(h("p", {}, `Best spot for ${S.scen.robots[i].name} by the model: (${best[0].toFixed(2)}, ${best[1].toFixed(2)}), P(win) ${fmtP(best[2])} vs ${fmtP(ref)} where it is.`));
  } else if (heat && !ev.model) toast("Train a value model first");
}
$("#heat-run").addEventListener("click", () => runEval(true));

// ---------- boot ----------
(async function boot() {
  S.meta = await api("/api/meta");
  await refreshScenList();
  const first = $("#scen-list").value;
  if (first) await loadScenario(first); else $("#scen-new").click();
  pollJobs();
  const tab = location.hash.slice(1);
  if (tab && $("#tab-" + tab)) showTab(tab);
})();
