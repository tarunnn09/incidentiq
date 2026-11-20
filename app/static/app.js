"use strict";
/* IncidentIQ dashboard - plain JavaScript, no libraries.
   Safety note: everything that comes from the server (log messages, service names...) is added with
   textContent via the el() helper, never as HTML, so an uploaded file cannot inject scripts. */

const SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];
const SEV_COLOR = { CRITICAL: "#f43f5e", HIGH: "#fb923c", MEDIUM: "#fbbf24", LOW: "#38bdf8" };
const CATEGORIES = ["database", "network", "application", "deployment", "resource", "unclassified"];
const METRICS = [
  { key: "latency_ms", title: "Response time (latency)", unit: " ms", color: "#60a5fa" },
  { key: "error_rate", title: "Error rate", unit: "%", color: "#f87171" },
  { key: "cpu_pct", title: "CPU usage", unit: "%", color: "#fbbf24" },
  { key: "memory_pct", title: "Memory usage", unit: "%", color: "#a78bfa" },
];
const WINDOW = 60; // number of points visible in "live replay" mode

const state = { data: null, list: [], info: null, live: true, pos: 0, filters: { severity: "", category: "" } };

/* ------------------------------------------------------------------ tiny DOM helpers */
function el(tag, attrs, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return node;
}
function svgEl(tag, attrs, ...kids) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  for (const kid of kids) if (kid) node.append(kid);
  return node;
}
const fmtTime = (s) => s.slice(5); // "2026-01-15 09:35" -> "01-15 09:35"
const pct = (x) => Math.round(x * 100) + "%";
const sevBadge = (s) => el("span", { class: "sev sev-" + s }, s);
const tag = (kind) => {
  const map = { ML: ["tag-ml", "ML prediction"], Rule: ["tag-rule", "Rule-based"], Data: ["tag-data", "Data"] };
  return el("span", { class: "tag " + map[kind][0] }, map[kind][1]);
};

async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok) {
    const detail = body && typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`;
    throw new Error(detail);
  }
  return body;
}
function toast(message, kind) {
  const box = document.getElementById("toast");
  box.textContent = message;
  box.className = "toast" + (kind ? " " + kind : "");
  box.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => { box.hidden = true; }, 9000);
}

/* ------------------------------------------------------------------ loading data */
function startPos(n) { return n > WINDOW ? WINDOW : Math.max(5, Math.floor(n / 2)); }

function setData(data) {
  state.data = data;
  state.list = data.incidents;
  state.filters = { severity: "", category: "" };
  state.live = data.source === "demo"; // live replay only makes sense for the demo
  state.pos = startPos(data.timeseries.length);
  document.getElementById("source-badge").textContent =
    data.source === "demo" ? "Demo data (synthetic)" : data.source.replace("upload: ", "Uploaded: ");
  location.hash = "#/";
  render();
}

async function loadDemo() {
  const btn = document.getElementById("demo-btn");
  btn.disabled = true;
  try { setData(await api("/api/demo")); }
  catch (e) { toast("Could not load demo data: " + e.message); }
  finally { btn.disabled = false; }
}

async function uploadCsv(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".csv")) { toast("Please choose a .csv file."); return; }
  const badge = document.getElementById("source-badge");
  badge.textContent = "Analysing " + file.name + "...";
  const form = new FormData();
  form.append("file", file);
  try {
    const data = await api("/api/analyze", { method: "POST", body: form });
    setData(data);
    toast(`Analysed ${data.summary.rows_analyzed} rows and found ${data.summary.total_incidents} incident(s).`, "info");
  } catch (e) {
    toast("Upload failed: " + e.message);
    badge.textContent = state.data ? (state.data.source === "demo" ? "Demo data (synthetic)" : state.data.source) : "No data";
  }
}

async function refreshList() {
  const q = new URLSearchParams();
  if (state.filters.severity) q.set("severity", state.filters.severity);
  if (state.filters.category) q.set("category", state.filters.category);
  try { state.list = (await api("/api/incidents?" + q.toString())).incidents; }
  catch (e) { toast(e.message); }
  renderTable();
}

/* ------------------------------------------------------------------ charts */
function lineChart(points, metric, limit) {
  const W = 560, H = 170, pad = { l: 44, r: 10, t: 10, b: 22 };
  const values = points.map((p) => p[metric.key]);
  const top = Math.max(limit ? limit * 1.15 : 0, ...values, 1) * 1.05;
  const x = (i) => pad.l + (points.length <= 1 ? 0 : (i * (W - pad.l - pad.r)) / (points.length - 1));
  const y = (v) => pad.t + (H - pad.t - pad.b) * (1 - v / top);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": metric.title + " over time" });

  for (let g = 0; g <= 4; g++) { // grid lines + y labels
    const v = (top * g) / 4;
    svg.append(svgEl("line", { x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v), stroke: "#25314d", "stroke-width": 1 }));
    const label = svgEl("text", { x: pad.l - 6, y: y(v) + 4, "text-anchor": "end", fill: "#93a0bd", "font-size": 10 });
    label.textContent = Math.round(v);
    svg.append(label);
  }
  if (limit) { // alert limit
    svg.append(svgEl("line", { x1: pad.l, x2: W - pad.r, y1: y(limit), y2: y(limit), stroke: "#f43f5e", "stroke-dasharray": "5 4", "stroke-width": 1.2 }));
    const l = svgEl("text", { x: W - pad.r, y: y(limit) - 4, "text-anchor": "end", fill: "#f43f5e", "font-size": 10 });
    l.textContent = "alert limit " + limit;
    svg.append(l);
  }
  if (points.length > 1) {
    const d = points.map((p, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(p[metric.key]).toFixed(1)).join(" ");
    svg.append(svgEl("path", { d, fill: "none", stroke: metric.color, "stroke-width": 2, "stroke-linejoin": "round" }));
  }
  points.forEach((p, i) => { // red dots where the ML model found unusual readings
    if (p.anomalies > 0) {
      const tip = svgEl("title");
      tip.textContent = `${p.t}: ${p[metric.key]}${metric.unit} - ML flagged ${p.anomalies} unusual reading(s)`;
      svg.append(svgEl("circle", { cx: x(i), cy: y(p[metric.key]), r: 3.2, fill: "#f43f5e" }, tip));
    }
  });
  [0, Math.floor((points.length - 1) / 2), points.length - 1].forEach((i, n) => { // x labels
    if (!points[i]) return;
    const t = svgEl("text", { x: x(i), y: H - 6, "text-anchor": n === 0 ? "start" : n === 1 ? "middle" : "end", fill: "#93a0bd", "font-size": 10 });
    t.textContent = points[i].t;
    svg.append(t);
  });
  return svg;
}

function drawCharts() {
  const box = document.getElementById("charts");
  if (!box || !state.data) return;
  const all = state.data.timeseries;
  const shown = state.live ? all.slice(Math.max(0, state.pos - WINDOW), state.pos) : all;
  box.replaceChildren(...METRICS.map((m) => {
    const limit = state.data.thresholds[m.key] ? state.data.thresholds[m.key].limit : null;
    const last = shown.length ? shown[shown.length - 1][m.key] : "-";
    return el("div", { class: "chart-card" },
      el("div", { class: "chart-top" }, el("span", { class: "muted" }, m.title), el("b", {}, last + m.unit)),
      lineChart(shown, m, limit));
  }));
  const live = document.getElementById("live-label");
  if (live) { live.textContent = state.live ? "LIVE REPLAY" : "FULL PERIOD"; live.className = "live" + (state.live ? " on" : ""); }
}
setInterval(() => { // makes the demo charts move like a live feed
  if (!state.live || !state.data || !document.getElementById("charts")) return;
  const n = state.data.timeseries.length;
  state.pos = state.pos >= n ? startPos(n) : state.pos + 1;
  drawCharts();
}, 800);

/* ------------------------------------------------------------------ dashboard view */
function introCard() {
  const item = (title, text) => el("div", {}, el("b", {}, title + ": "), text);
  return el("details", { class: "card intro", open: "" },
    el("summary", {}, "New to AIOps? Start here (30-second guide)"),
    el("div", { class: "glossary" },
      item("AIOps", "using AI to watch over apps, servers and databases and spot problems early."),
      item("Logs", "text messages a program writes about what it is doing, like a diary."),
      item("Metrics", "numbers measured every minute: latency (how slow), error rate, CPU and memory use."),
      item("Anomaly", "a moment that looks very different from normal. An ML model (Isolation Forest) finds these."),
      item("Incident", "a group of related unusual events from one service, shown as one problem."),
      item("Severity", "how serious it is: LOW, MEDIUM, HIGH or CRITICAL, from a simple points system."),
      item("ML vs Rules", "tags show what a trained model predicted and what readable rules concluded. Causes are 'probable', never confirmed.")));
}

function kpiRow(s) {
  const max = Math.max(1, ...SEVERITIES.map((k) => s.severity_counts[k]));
  const sevRows = SEVERITIES.map((k) => el("button", {
    class: "sevrow", type: "button", title: "Show only " + k + " incidents",
    onclick: () => { state.filters.severity = state.filters.severity === k ? "" : k; syncFilters(); refreshList(); },
  }, el("span", { class: "name" }, sevBadge(k)),
    el("span", { class: "track" }, el("span", { class: "fill", style: `display:block;width:${(s.severity_counts[k] / max) * 100}%;background:${SEV_COLOR[k]}` })),
    el("span", { class: "num" }, s.severity_counts[k])));
  return el("section", { class: "kpis" },
    el("div", { class: "card kpi health " + s.health_status.toLowerCase() },
      el("div", { class: "ring", style: "--p:" + s.health_score }, el("span", {}, s.health_score)),
      el("div", {}, el("h3", {}, "System health"), el("p", { class: "big" }, s.health_status),
        el("p", { class: "muted small" }, `${s.active_incidents} incident(s) still active. Starts at 100 and drops for each ongoing incident.`))),
    el("div", { class: "card kpi" },
      el("div", {}, el("h3", {}, "Total incidents"), el("p", { class: "big" }, s.total_incidents),
        el("p", { class: "muted small" }, `from ${s.rows_analyzed} log lines (${fmtTime(s.time_start)} to ${fmtTime(s.time_end)}). ${s.anomalous_rows} readings were flagged unusual by ML.`))),
    el("div", { class: "card kpi col" }, el("h3", {}, "Severity breakdown"), sevRows,
      el("p", { class: "muted small", style: "margin-top:8px" }, "Click a severity to filter the table.")));
}

function chartsCard() {
  return el("section", { class: "card" },
    el("div", { class: "card-head" },
      el("div", {}, el("h3", {}, "Live monitoring"),
        el("p", { class: "muted small" }, "Each chart shows the worst value among all services at each moment. Red dots = readings the ML model found unusual. Dashed line = alert limit.")),
      el("div", { class: "kv" }, el("span", { id: "live-label", class: "live" }),
        el("button", { class: "btn", type: "button", onclick: () => { state.live = !state.live; if (state.live) state.pos = startPos(state.data.timeseries.length); drawCharts(); } }, "Toggle live replay"))),
    el("div", { id: "charts", class: "charts" }));
}

function servicesCard(s) {
  return el("section", { class: "card" },
    el("div", { class: "card-head" }, el("div", {}, el("h3", {}, "Services"),
      el("p", { class: "muted small" }, "Status is based on incidents that are still active at the end of the data."))),
    el("div", { class: "services" }, s.services.map((v) => el("div", { class: "svc " + v.status },
      el("b", {}, v.service), el("span", { class: "state" }, v.status), el("span", { class: "muted" }, ` - ${v.incidents} incident(s)`),
      el("div", { class: "muted" }, `avg ${v.avg_latency_ms} ms, ${v.avg_error_rate}% errors`),
      el("div", { class: "muted" }, `CPU ${v.avg_cpu_pct}%, memory ${v.avg_memory_pct}%`)))));
}

function select(id, label, options, current, onchange) {
  const node = el("select", { id, "aria-label": label, onchange },
    el("option", { value: "" }, label), options.map((o) => el("option", { value: o }, o)));
  node.value = current;
  return node;
}
function syncFilters() {
  const a = document.getElementById("f-sev"), b = document.getElementById("f-cat");
  if (a) a.value = state.filters.severity;
  if (b) b.value = state.filters.category;
}

function incidentsCard() {
  return el("section", { class: "card" },
    el("div", { class: "card-head" },
      el("div", {}, el("h3", {}, "Incidents"),
        el("p", { class: "muted small" }, "Click a row to see why the system thinks it happened and what to try first.")),
      el("div", { class: "filters" },
        select("f-sev", "All severities", SEVERITIES, state.filters.severity, (e) => { state.filters.severity = e.target.value; refreshList(); }),
        select("f-cat", "All categories", CATEGORIES, state.filters.category, (e) => { state.filters.category = e.target.value; refreshList(); }))),
    el("div", { id: "table-wrap", class: "table-wrap" }));
}

function renderTable() {
  const wrap = document.getElementById("table-wrap");
  if (!wrap) return;
  if (!state.list.length) { wrap.replaceChildren(el("p", { class: "muted" }, "No incidents match. Everything looks normal here.")); return; }
  const open = (id) => { location.hash = "#/incident/" + id; };
  wrap.replaceChildren(el("table", {},
    el("thead", {}, el("tr", {}, ["ID", "Time", "Service", "Type (ML)", "Severity", "Events", "Anomaly score"].map((h) => el("th", {}, h)))),
    el("tbody", {}, state.list.map((i) => el("tr", { class: "click", tabindex: "0", onclick: () => open(i.id),
        onkeydown: (e) => { if (e.key === "Enter") open(i.id); } },
      el("td", {}, el("a", { href: "#/incident/" + i.id }, i.id)),
      el("td", {}, `${fmtTime(i.start)} - ${i.end.slice(11)}`, el("div", { class: "muted small" }, i.duration_minutes + " min")),
      el("td", {}, i.service),
      el("td", {}, i.category, el("div", { class: "muted small" }, pct(i.category_confidence) + " confidence")),
      el("td", {}, sevBadge(i.severity)),
      el("td", {}, i.event_count),
      el("td", {}, i.max_anomaly_score.toFixed(2)))))));
}

function aboutCard() {
  const info = state.info;
  if (!info) return null;
  const a = info.anomaly_model, c = info.classifier, r = info.rules;
  return el("section", { class: "card" },
    el("div", { class: "card-head" }, el("div", {}, el("h3", {}, "How the AI works"),
      el("p", { class: "muted small" }, "Two small machine-learning models plus readable rules."))),
    el("div", { class: "grid-2" },
      el("div", {}, tag("ML"), el("p", { class: "cause", style: "font-size:15px" }, "1. Isolation Forest (finds unusual numbers)"),
        el("p", { class: "muted small" }, `Trained on ${a.training_rows} normal readings only. On new synthetic data it caught ${pct(a.holdout.recall)} of incident rows with ${pct(a.holdout.precision)} precision.`)),
      el("div", {}, tag("ML"), el("p", { class: "cause", style: "font-size:15px" }, "2. TF-IDF + Logistic Regression (reads logs)"),
        el("p", { class: "muted small" }, `Trained on ${c.training_messages} labelled messages; ${pct(c.holdout_accuracy)} accuracy on unseen synthetic messages. Real logs are messier, so expect lower accuracy.`)),
      el("div", {}, tag("Rule"), el("p", { class: "cause", style: "font-size:15px" }, "3. Rules (severity, grouping, causes)"),
        el("p", { class: "muted small" }, `Alert limits: latency ${r.thresholds.latency_ms} ms, errors ${r.thresholds.error_rate}%, CPU ${r.thresholds.cpu_pct}%, memory ${r.thresholds.memory_pct}%. Events from one service within ${r.group_gap_minutes} minutes are grouped into one incident.`))));
}

function renderDashboard(root) {
  const d = state.data;
  const notes = d.data_quality_notes.length ? el("section", { class: "card" }, el("h3", {}, "Data quality notes"),
    el("ul", { class: "clean" }, d.data_quality_notes.map((n) => el("li", {}, n)))) : null;
  // replaceChildren would print the word "null", so we drop empty sections first
  const sections = [introCard(), kpiRow(d.summary), chartsCard(), servicesCard(d.summary), incidentsCard(), notes, aboutCard()];
  root.replaceChildren(...sections.filter(Boolean));
  drawCharts();
  renderTable();
}

/* ------------------------------------------------------------------ incident detail view */
function metricTiles(inc) {
  const baseline = state.info ? state.info.anomaly_model.baseline : null;
  return el("div", { class: "tiles" }, METRICS.map((m) => {
    const limit = state.data.thresholds[m.key].limit;
    const hot = inc.peaks[m.key] >= limit;
    return el("div", { class: "tile" + (hot ? " hot" : "") }, el("span", { class: "muted small" }, m.title),
      el("b", {}, inc.peaks[m.key] + m.unit),
      el("span", { class: "muted small" }, (baseline ? `normal ~${Math.round(baseline[m.key].mean)}${m.unit}, ` : "") + `limit ${limit}${m.unit}`));
  }));
}

function card(title, who, ...body) {
  return el("section", { class: "card" }, el("div", { class: "card-head" }, el("h3", {}, title), who ? tag(who) : null), body);
}

function renderIncident(root, id) {
  const inc = state.data.incidents.find((i) => i.id === id);
  if (!inc) {
    root.replaceChildren(el("a", { class: "back", href: "#/" }, "< Back to dashboard"),
      el("section", { class: "card" }, el("p", {}, "That incident was not found. It may belong to a different dataset.")));
    return;
  }
  const flagged = inc.max_anomaly_score >= inc.anomaly_threshold;
  const totalPoints = inc.severity_points.reduce((a, p) => a + p.points, 0);
  root.replaceChildren(
    el("div", {}, el("a", { class: "back", href: "#/" }, "< Back to dashboard"),
      el("div", { class: "title-row" }, el("h2", {}, `${inc.id} - ${inc.service}`), sevBadge(inc.severity),
        el("span", { class: "pill" }, inc.category), el("span", { class: "muted" }, `${fmtTime(inc.start)} to ${inc.end.slice(11)} (${inc.duration_minutes} min)`))),
    el("div", { class: "callout" }, el("b", {}, "In simple words: "), inc.summary),
    el("div", { class: "grid-2" },
      el("div", { class: "stack" },
        card("Original log message", null,
          el("p", { class: "muted small" }, `Most serious of ${inc.event_count} grouped log line(s), written at ${fmtTime(inc.log_time)}.`),
          el("div", { class: "logbox" }, el("b", {}, inc.log_level + "  "), inc.log_message)),
        card("Anomaly score", "ML",
          el("p", {}, `Score ${inc.max_anomaly_score.toFixed(2)} - the metrics are `, el("b", {}, flagged ? "ANOMALOUS (unusual)" : "within the normal range"), "."),
          el("div", { class: "gauge" }, el("div", { class: "gauge-mark", style: `left:${inc.anomaly_threshold * 100}%` }),
            el("div", { class: "gauge-dot", style: `left:${Math.min(inc.max_anomaly_score, 1) * 100}%` })),
          el("div", { class: "gauge-labels" }, el("span", {}, "0 normal"), el("span", {}, `dashed line = flag threshold (${inc.anomaly_threshold.toFixed(2)})`), el("span", {}, "1 very unusual")),
          el("p", { class: "muted small", style: "margin-top:8px" }, "Isolation Forest scores how easy it is to isolate this reading from normal data. Higher means more unusual.")),
        card("Peak metrics during the incident", "Data", metricTiles(inc)),
        card("Predicted incident type", "ML",
          el("p", { class: "cause" }, inc.category === "unclassified" ? "Unclear (model not confident)" : inc.category,
            el("span", { class: "muted", style: "font-weight:400;font-size:14px" }, `  ${pct(inc.category_confidence)} confidence`)),
          inc.runner_up ? el("p", { class: "muted small" }, `Second guess: ${inc.runner_up.category} (${pct(inc.runner_up.confidence)}).`) : null,
          el("p", { class: "muted small", style: "margin-top:6px" }, "The model learned which words in logs point to which problem type. This is a prediction, not proof."))),
      el("div", { class: "stack" },
        card("Severity: " + inc.severity, "Rule",
          el("p", { class: "muted small" }, `Points system: ${totalPoints} point(s) in total (8+ CRITICAL, 6+ HIGH, 4+ MEDIUM, otherwise LOW).`),
          el("ul", { class: "clean" }, inc.severity_points.map((p) => el("li", {}, `+${p.points}  ${p.reason}`)))),
        card("Probable root cause", "Rule",
          el("p", { class: "cause" }, inc.probable_cause), el("p", { class: "disclaimer" }, inc.cause_disclaimer)),
        card("Evidence", null,
          el("ul", { class: "plain" }, inc.evidence.map((e) => el("li", {}, tag(e.source), el("span", {}, e.text))))),
        card("Recommended troubleshooting steps", "Rule",
          el("ol", { class: "clean" }, inc.recommendations.map((s) => el("li", {}, s)))))),
    card("All grouped log lines", null,
      el("div", { class: "table-wrap" }, el("table", {},
        el("thead", {}, el("tr", {}, ["Time", "Level", "Message", "Latency", "Errors", "CPU", "Memory", "Score"].map((h) => el("th", {}, h)))),
        el("tbody", {}, inc.events.map((e) => el("tr", {},
          el("td", {}, e.timestamp.slice(11)), el("td", {}, e.log_level), el("td", { class: "msg" }, e.message),
          el("td", {}, e.latency_ms + " ms"), el("td", {}, e.error_rate + "%"), el("td", {}, e.cpu_pct + "%"),
          el("td", {}, e.memory_pct + "%"), el("td", {}, e.anomaly_score.toFixed(2)))))))));
  window.scrollTo(0, 0);
}

/* ------------------------------------------------------------------ routing + start-up */
function render() {
  const root = document.getElementById("app");
  if (!state.data) return;
  const match = location.hash.match(/^#\/incident\/(.+)$/);
  if (match) renderIncident(root, decodeURIComponent(match[1]));
  else renderDashboard(root);
}

window.addEventListener("hashchange", render);
document.getElementById("demo-btn").addEventListener("click", loadDemo);
document.getElementById("csv-input").addEventListener("change", (e) => { uploadCsv(e.target.files[0]); e.target.value = ""; });

(async function start() {
  const keepHash = location.hash; // so a reloaded incident link still works
  try { state.info = await api("/api/model-info"); } catch (e) { /* the page still works without it */ }
  try {
    const data = await api("/api/demo");
    setData(data);
    if (keepHash.startsWith("#/incident/")) { location.hash = keepHash; }
  } catch (e) {
    document.getElementById("app").replaceChildren(el("p", { class: "center" }, "Could not load the dashboard: " + e.message));
  }
})();
