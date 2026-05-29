const FAST_REFRESH_MS = 1000;
const CHART_HISTORY_REFRESH_MS = 60000;
// Roadmap tab polls less often than fast-tick — it changes on minute
// boundaries (when the agent finishes a task), not second boundaries.
const ROADMAP_REFRESH_MS = 30000;

function fmtMoney(v) {
  if (v == null || !isFinite(v)) return "—";
  const sign = v < 0 ? "-" : "";
  const abs = Math.abs(v);
  return sign + "$" + abs.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function fmtPct(v) {
  if (v == null || !isFinite(v)) return "—";
  return (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
}
function fmtNum(v, d = 2) {
  if (v == null || !isFinite(v)) return "—";
  return Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
}
function colorClass(v) {
  if (v == null || !isFinite(v) || v === 0) return "";
  return v > 0 ? "pos-up" : "pos-down";
}
function hbClass(age) {
  if (age == null) return "hb-dead";
  if (age < 120) return "hb-fresh";
  if (age < 600) return "hb-stale";
  return "hb-dead";
}
function fmt12(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-US", {
    month: "numeric", day: "numeric",
    hour: "numeric", minute: "2-digit", second: "2-digit",
    hour12: true,
  });
}
function fmt12Time(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString("en-US", {
    hour: "numeric", minute: "2-digit", second: "2-digit", hour12: true,
  });
}
function fmt12HHMMSS(s) {
  // Convert an "HH:MM:SS" (or "HH:MM") local-naive string into "h:mm:ss AM/PM".
  if (!s) return "—";
  const parts = s.split(":");
  if (parts.length < 2) return s;
  let h = parseInt(parts[0], 10);
  if (isNaN(h)) return s;
  const mm = parts[1];
  const ss = parts[2];
  const ap = h >= 12 ? "PM" : "AM";
  h = h % 12; if (h === 0) h = 12;
  return ss != null ? `${h}:${mm}:${ss} ${ap}` : `${h}:${mm} ${ap}`;
}
async function fetchJSON(path) {
  try {
    const r = await fetch(path, { cache: "no-store" });
    if (r.status === 204) return null;
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

async function refreshSummary() {
  const s = await fetchJSON("/api/summary");
  if (!s) return;
  document.getElementById("env-badge").textContent = s.environment || "paper";
  document.getElementById("stat-portfolio").textContent = fmtMoney(s.portfolio_value);
  const daily = s.realized_pnl_daily;
  const dailyEl = document.getElementById("stat-daily-pnl");
  dailyEl.textContent = fmtMoney(daily);
  dailyEl.className = colorClass(daily);
  // All-time P&L = current portfolio_value − initial deposit. Replaces the
  // old "Unrealized" tile, which only showed open-position drift and made
  // a losing portfolio look profitable when winners on open positions
  // happened to exceed locked-in losses on closed ones.
  const atp = s.all_time_pnl_usd;
  const atpPct = s.all_time_pnl_pct;
  const unEl = document.getElementById("stat-unrealized");
  if (atp == null) {
    unEl.textContent = "—";
    unEl.className = "";
  } else {
    unEl.textContent = fmtMoney(atp) + (atpPct != null ? " / " + fmtPct(atpPct * 100) : "");
    unEl.className = colorClass(atp);
  }
  const dd = s.drawdown_usd;
  const ddEl = document.getElementById("stat-drawdown");
  ddEl.textContent = fmtMoney(dd) + (s.drawdown_pct != null ? " / " + fmtPct(s.drawdown_pct * 100) : "");
  ddEl.className = (dd != null && dd < 0) ? "pos-down" : "";
  document.getElementById("stat-positions").textContent =
    s.open_position_count != null ? s.open_position_count : "—";
  document.getElementById("stat-regime").textContent = s.regime || "—";
  const killEl = document.getElementById("stat-kill");
  if (s.kill_switch_active === true) {
    killEl.innerHTML = '<span class="pill on">ACTIVE</span>';
  } else if (s.kill_switch_active === false) {
    killEl.innerHTML = '<span class="pill off">off</span>';
  } else {
    killEl.textContent = "—";
  }
  document.getElementById("stat-model").textContent = s.model_version || "—";
  const hb = s.heartbeat_age_s;
  const hbEl = document.getElementById("stat-heartbeat");
  if (hb == null) {
    hbEl.textContent = "none";
    hbEl.className = "hb-dead";
  } else {
    const mins = Math.floor(hb / 60);
    const secs = Math.floor(hb % 60);
    hbEl.textContent = mins > 0 ? `${mins}m ${secs}s ago` : `${secs}s ago`;
    hbEl.className = hbClass(hb);
  }
}

async function refreshPositions() {
  const rows = await fetchJSON("/api/positions") || [];
  document.getElementById("positions-count").textContent = `(${rows.length})`;
  const tbody = document.querySelector("#positions-table tbody");
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="muted">No open positions.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.symbol}</td>
      <td>${r.qty}</td>
      <td>${fmtNum(r.avg_entry_price)}</td>
      <td>${fmtNum(r.current_price)}</td>
      <td>${fmtMoney(r.market_value)}</td>
      <td class="${colorClass(r.unrealized_pnl)}">${fmtMoney(r.unrealized_pnl)}</td>
      <td class="${colorClass(r.unrealized_pnl_pct)}">${fmtPct((r.unrealized_pnl_pct ?? 0) * 100)}</td>
      <td>${fmtNum(r.hard_stop)}</td>
    </tr>`).join("");
}

async function refreshOrders() {
  const rows = await fetchJSON("/api/orders/pending") || [];
  document.getElementById("orders-count").textContent = `(${rows.length})`;
  const tbody = document.querySelector("#orders-table tbody");
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="muted">No pending orders.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.symbol}</td>
      <td><span class="pill ${r.side}">${r.side}</span></td>
      <td>${r.order_type}</td>
      <td>${r.qty}</td>
      <td>${r.filled_qty}</td>
      <td>${r.limit_price != null ? fmtNum(r.limit_price) : "—"}</td>
      <td>${r.status}</td>
      <td>${fmt12(r.submitted_at)}</td>
    </tr>`).join("");
}

function fmtFillPnL(pnl) {
  if (pnl === null || pnl === undefined || pnl === "") {
    return '<span class="muted">—</span>';
  }
  const v = Number(pnl);
  if (!Number.isFinite(v)) return '<span class="muted">—</span>';
  const cls = v > 0 ? "pos-up" : v < 0 ? "pos-down" : "muted";
  const sign = v > 0 ? "+" : "";
  return `<span class="${cls}">${sign}$${Math.abs(v).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>`;
}

async function refreshFills() {
  const rows = await fetchJSON("/api/fills/recent?n=50") || [];
  document.getElementById("fills-count").textContent = `(${rows.length})`;
  const tbody = document.querySelector("#fills-table tbody");
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="muted">No fills yet.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.date_ct} ${fmt12HHMMSS(r.time_ct)} CT</td>
      <td>${r.symbol}</td>
      <td><span class="pill ${r.side}">${r.side}</span></td>
      <td>${r.shares}</td>
      <td>${fmtNum(parseFloat(r.fill_price))}</td>
      <td>${r.order_type}</td>
      <td>${fmtFillPnL(r.pnl)}</td>
    </tr>`).join("");
}

async function refreshPlan() {
  const plan = await fetchJSON("/api/plan/today");
  const body = document.getElementById("plan-body");
  const dateEl = document.getElementById("plan-date");
  if (!plan || !Object.keys(plan).length) {
    dateEl.textContent = "";
    body.innerHTML = '<div class="muted">No plan for today yet.</div>';
    return;
  }
  dateEl.textContent = `${plan.plan_date || ""} • ${plan.regime || "?"} • ${plan.model_version || ""}`;
  const entries = plan.entries || [];
  const exits = plan.exits || [];
  const holds = plan.holds || [];
  let html = `<h3>Entries (${entries.length})</h3>`;
  if (entries.length) {
    html += `<table><thead><tr><th>Strategy</th><th>Symbol</th><th>Shares</th><th>Target</th><th>Stop</th><th>Score</th></tr></thead><tbody>`;
    for (const e of entries) {
      html += `<tr><td>${e.strategy_id ?? "?"}</td><td>${e.symbol}</td><td>${e.shares}</td><td>${fmtNum(e.target_price)}</td><td>${fmtNum(e.stop)}</td><td>${fmtNum(e.signal_score ?? 0, 3)}</td></tr>`;
    }
    html += `</tbody></table>`;
  } else {
    html += '<div class="muted">None.</div>';
  }
  html += `<h3>Exits (${exits.length})</h3>`;
  html += exits.length
    ? `<div>${exits.join(", ")}</div>`
    : '<div class="muted">None.</div>';
  html += `<h3>Holds (${holds.length})</h3>`;
  html += holds.length
    ? `<div>${holds.map(h => typeof h === "string" ? h : h.symbol).join(", ")}</div>`
    : '<div class="muted">None.</div>';
  body.innerHTML = html;
}

async function refreshRisk() {
  const rows = await fetchJSON("/api/risk/events?n=20") || [];
  const list = document.getElementById("risk-list");
  if (!rows.length) {
    list.innerHTML = '<li class="muted">No risk events.</li>';
    return;
  }
  list.innerHTML = rows.map(r => `
    <li class="sev-${r.severity || "INFO"}">
      <span class="sev-tag ${r.severity || "INFO"}">${r.severity || "INFO"}</span>
      <div>
        <div><strong>${r.event_type || "event"}</strong> <span class="muted">${fmt12(r.ts)}</span></div>
        <div>${r.description || ""}</div>
      </div>
    </li>`).join("");
}

async function refreshHealth() {
  const h = await fetchJSON("/api/health/detail");
  const list = document.getElementById("health-list");
  if (!h || !h.checks || !h.checks.length) {
    list.innerHTML = '<li class="muted">No health checks registered.</li>';
    return;
  }
  list.innerHTML = h.checks.map(c => `
    <li>
      <span class="${c.healthy ? "health-ok" : "health-bad"}">${c.healthy ? "●" : "●"}</span>
      <div>
        <div><strong>${c.name}</strong> <span class="muted">${c.latency_ms ? c.latency_ms.toFixed(0) + "ms" : ""}</span></div>
        <div class="muted">${c.message || ""}</div>
      </div>
    </li>`).join("");
}

let chart;
let chartHistoryRows = [];
let lastSummaryValue = null;

function renderChart() {
  const base = chartHistoryRows.map(r => {
    let lbl = "";
    if (r.ts) {
      const d = new Date(r.ts);
      if (!isNaN(d.getTime())) {
        lbl = d.toLocaleString("en-US", {
          month: "numeric", day: "numeric",
          hour: "numeric", minute: "2-digit", hour12: true,
        });
      }
    }
    return { label: lbl, value: r.portfolio_value };
  });
  if (lastSummaryValue != null) {
    const now = new Date();
    const lbl = now.toLocaleTimeString("en-US", {
      hour: "numeric", minute: "2-digit", second: "2-digit", hour12: true,
    });
    base.push({ label: `now ${lbl}`, value: lastSummaryValue });
  }
  const labels = base.map(b => b.label);
  const values = base.map(b => b.value);
  const ctx = document.getElementById("portfolio-chart").getContext("2d");
  if (!chart) {
    chart = new Chart(ctx, {
      type: "line",
      data: { labels, datasets: [{
        label: "Portfolio $",
        data: values,
        borderColor: "#58a6ff",
        backgroundColor: "rgba(88,166,255,0.08)",
        fill: true, tension: 0.2, pointRadius: 0,
      }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#8a98a7", maxTicksLimit: 8 }, grid: { color: "#1d252f" } },
          y: { ticks: { color: "#8a98a7", callback: v => "$" + v.toLocaleString() }, grid: { color: "#1d252f" } },
        },
      },
    });
  } else {
    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.update("none");
  }
}

async function refreshChartHistory() {
  const rows = await fetchJSON("/api/portfolio/history?days=30");
  if (rows) chartHistoryRows = rows;
  renderChart();
}

function fmtDuration(s) {
  if (s == null || !isFinite(s)) return "—";
  const sign = s < 0 ? "-" : "";
  s = Math.abs(Math.floor(s));
  const d = Math.floor(s / 86400); s %= 86400;
  const h = Math.floor(s / 3600); s %= 3600;
  const m = Math.floor(s / 60); s %= 60;
  if (d) return `${sign}${d}d ${h}h`;
  if (h) return `${sign}${h}h ${m}m`;
  if (m) return `${sign}${m}m ${s}s`;
  return `${sign}${s}s`;
}

function fmtRelETA(secs) {
  if (secs == null) return "—";
  if (secs < 0) return `${fmtDuration(-secs)} ago`;
  return `in ${fmtDuration(secs)}`;
}

function statusChipClass(status, success) {
  if (status === "success" || success === true) return "status-ok";
  if (status === "error" || success === false) return "status-err";
  if (status === "missed") return "status-warn";
  if (status === "submitted") return "status-run";
  return "status-idle";
}

const DAILY_STATE_META = {
  error:     { label: "ERROR",    cls: "status-err"  },
  running:   { label: "RUNNING",  cls: "status-run"  },
  pending:   { label: "PENDING",  cls: "status-warn" },
  recurring: { label: "RECURRING",cls: "status-ok"   },
  done:      { label: "DONE",     cls: "status-ok"   },
  not_today: { label: "NOT TODAY",cls: "status-idle" },
};

async function refreshDailyChecklist() {
  const rows = await fetchJSON("/api/scheduler/daily_checklist") || [];
  const tbody = document.querySelector("#daily-checklist-table tbody");
  const badge = document.getElementById("daily-checklist-badge");
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="muted">No jobs registered.</td></tr>';
    badge.textContent = "";
    return;
  }
  const tally = { error: 0, running: 0, pending: 0, recurring: 0, done: 0, not_today: 0 };
  for (const r of rows) tally[r.status] = (tally[r.status] || 0) + 1;
  const bits = [];
  if (tally.error)     bits.push(`<span class="status-err">${tally.error} error</span>`);
  if (tally.running)   bits.push(`<span class="status-run">${tally.running} running</span>`);
  if (tally.pending)   bits.push(`<span class="status-warn">${tally.pending} pending</span>`);
  if (tally.recurring) bits.push(`${tally.recurring} recurring`);
  if (tally.done)      bits.push(`${tally.done} done`);
  if (tally.not_today) bits.push(`<span class="muted">${tally.not_today} not today</span>`);
  badge.innerHTML = `(${bits.join(" • ")})`;

  tbody.innerHTML = rows.map(r => {
    const meta = DAILY_STATE_META[r.status] || { label: r.status, cls: "status-idle" };
    const nextLbl = r.next_run_label
      ? `${r.next_run_label}${r.seconds_until_next != null ? ` <span class="muted">(${fmtRelETA(r.seconds_until_next)})</span>` : ""}`
      : '<span class="muted">—</span>';
    const lastLbl = r.last_run_today
      ? `${r.last_run_label}${r.last_duration_ms != null ? ` <span class="muted">• ${r.last_duration_ms}ms</span>` : ""}`
      : '<span class="muted">—</span>';
    const runsCell = r.runs_today
      ? `${r.runs_ok_today} ok` +
        (r.runs_err_today ? ` / <span class="status-err">${r.runs_err_today} err</span>` : "") +
        (r.runs_missed_today ? ` / <span class="status-warn">${r.runs_missed_today} miss</span>` : "")
      : '<span class="muted">0</span>';
    let notes = "";
    if (r.status === "error" && r.last_error) {
      notes = `<span class="status-err" title="${(r.last_error || "").replace(/"/g, "&quot;")}">${r.last_error.slice(0, 120)}</span>`;
    } else if (r.status === "pending") {
      notes = "scheduled, has not fired yet";
    } else if (r.status === "recurring") {
      notes = "fires multiple times today";
    } else if (r.status === "not_today") {
      notes = `<span class="muted">next ${r.next_run_label || "—"}</span>`;
    }
    return `
    <tr class="job-row">
      <td><span class="status-chip ${meta.cls}">${meta.label}</span></td>
      <td><strong>${r.func_name || r.job_id}</strong><div class="muted">${r.job_id}</div></td>
      <td>${nextLbl}</td>
      <td>${lastLbl}</td>
      <td>${runsCell}</td>
      <td>${notes}</td>
    </tr>`;
  }).join("");
}

async function refreshExecutions() {
  const rows = await fetchJSON("/api/scheduler/executions?n=25") || [];
  const list = document.getElementById("executions-list");
  if (!rows.length) {
    list.innerHTML = '<li class="muted">No executions recorded yet.</li>';
    return;
  }
  list.innerHTML = rows.map(r => {
    const cls = statusChipClass(r.status);
    const when = fmt12(r.started_ts);
    const dur = r.duration_ms != null ? `${r.duration_ms}ms` : "";
    const err = r.error ? `<div class="muted">${r.error.slice(0, 180)}</div>` : "";
    return `
    <li>
      <span class="status-chip ${cls}">${r.status}</span>
      <div>
        <div><strong>${r.func_name}</strong> <span class="muted">${when} • ${dur}</span></div>
        ${err}
      </div>
    </li>`;
  }).join("");
}

async function refreshAgentStatus() {
  const rows = await fetchJSON("/api/agents/status") || [];
  const grid = document.getElementById("agent-tiles");
  const muted = document.getElementById("agent-status-muted");
  if (!rows.length) {
    grid.innerHTML = '<div class="muted">No agents registered.</div>';
    muted.textContent = "";
    return;
  }
  let totalCost = 0, totalCalls = 0;
  for (const a of rows) {
    totalCost += a.cost_usd_today || 0;
    totalCalls += a.calls_today || 0;
  }
  muted.textContent = `(${rows.length} agents • $${totalCost.toFixed(2)} / ${totalCalls} calls today)`;
  grid.innerHTML = rows.map(a => {
    const age = a.last_entry_age_s;
    const ageCls = age == null ? "status-idle"
      : age < 3600 ? "status-ok"
      : age < 86400 ? "status-warn"
      : "status-err";
    const ageStr = age == null ? "never" : `${fmtDuration(age)} ago`;
    const lastTitle = a.last_entry_title
      ? `<div class="muted tile-title" title="${a.last_entry_title.replace(/"/g, "&quot;")}">${a.last_entry_title.slice(0, 80)}</div>`
      : '<div class="muted">—</div>';
    return `
    <div class="agent-tile">
      <div class="tile-head">
        <strong>${a.agent_id}</strong>
        <span class="muted">${a.role || ""} • ${a.model_tier || ""}</span>
      </div>
      <div class="tile-row">
        <span class="status-chip ${ageCls}">${a.last_entry_kind || "—"}</span>
        <span class="muted">${ageStr}</span>
      </div>
      ${lastTitle}
      <div class="tile-stats">
        <span>Today: <strong>${a.entries_today}</strong> entries</span>
        <span>${a.calls_today} calls</span>
        <span>$${(a.cost_usd_today || 0).toFixed(2)}</span>
      </div>
    </div>`;
  }).join("");
}

async function refreshProgramInfo() {
  const p = await fetchJSON("/api/program/info");
  const list = document.getElementById("program-info-list");
  if (!p) { list.innerHTML = '<li class="muted">Unavailable.</li>'; return; }
  const rows = [
    ["Environment", p.environment || "—"],
    ["Host", p.hostname || "—"],
    ["Platform", p.platform || "—"],
    ["Python", p.python_version || "—"],
    ["Uptime", p.uptime_s != null ? fmtDuration(p.uptime_s) : "—"],
    ["Git SHA", p.git_sha || "—"],
    ["Server time", fmt12(p.now)],
  ];
  list.innerHTML = rows.map(([k, v]) => `<li><div><strong>${k}</strong></div><div>${v}</div></li>`).join("");
}

let _fastTickInFlight = false;
async function fastTick() {
  // 1-second cadence — skip if the previous tick is still running so slow
  // network responses don't stack up parallel fetches and pile load on the
  // monitor service.
  if (_fastTickInFlight) return;
  _fastTickInFlight = true;
  try {
    await Promise.all([
      refreshSummary(), refreshPositions(), refreshOrders(),
      refreshFills(), refreshPlan(), refreshRisk(), refreshHealth(),
      refreshDailyChecklist(), refreshExecutions(), refreshAgentStatus(),
      refreshProgramInfo(),
    ]);
    // Pull the live portfolio value and append to chart as a moving "now" point.
    const s = await fetchJSON("/api/summary");
    if (s && s.portfolio_value != null) {
      lastSummaryValue = s.portfolio_value;
      renderChart();
    }
    document.getElementById("last-refresh").textContent =
      "Refreshed " + new Date().toLocaleTimeString("en-US", {
        hour: "numeric", minute: "2-digit", second: "2-digit", hour12: true,
      });
  } finally {
    _fastTickInFlight = false;
  }
}

// ─── Roadmap tab ─────────────────────────────────────────────────────────

// Map of task_id → task. Refreshed every poll. Read by the inline-edit
// handlers to check dependencies before allowing a status transition.
let _roadmapTasksById = {};
// Set of expanded task ids — preserved across re-renders so opening the
// details on a row doesn't snap shut on the next 30s poll.
const _roadmapExpanded = new Set();
// Set of collapsed week numbers (persisted to localStorage so the
// user's choice survives a hard reload).
const _roadmapCollapsedWeeks = new Set(
  JSON.parse(localStorage.getItem("rm-collapsed-weeks") || "[]")
);

function _persistCollapsedWeeks() {
  localStorage.setItem(
    "rm-collapsed-weeks", JSON.stringify([..._roadmapCollapsedWeeks])
  );
}

function _statusLabel(s) {
  return { not_started: "not started", in_progress: "in progress",
           complete: "complete", blocked: "blocked" }[s] || s;
}

// Next status in the click-cycle. Blocked is reached via the modal, not
// the cycle, so a normal click never silently puts a task into blocked.
function _nextStatus(current) {
  if (current === "not_started") return "in_progress";
  if (current === "in_progress") return "complete";
  if (current === "complete")    return "not_started";
  return "in_progress";  // blocked → in_progress when user clicks
}

function _depsComplete(task) {
  const deps = task.dependencies || [];
  for (const id of deps) {
    const dep = _roadmapTasksById[id];
    if (!dep || dep.status !== "complete") return false;
  }
  return true;
}

function _missingDeps(task) {
  const deps = task.dependencies || [];
  return deps.filter(id => {
    const d = _roadmapTasksById[id];
    return !d || d.status !== "complete";
  });
}

async function _patchTask(taskId, body) {
  try {
    const resp = await fetch(`/api/roadmap/${taskId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      alert(`Failed to update ${taskId}: ${err.error || resp.statusText}`);
      return null;
    }
    return await resp.json();
  } catch (e) {
    alert(`Network error updating ${taskId}: ${e}`);
    return null;
  }
}

async function refreshRoadmap() {
  const payload = await fetchJSON("/api/roadmap");
  if (!payload) return;
  const { tasks, summary } = payload;
  _roadmapTasksById = Object.fromEntries(tasks.map(t => [t.id, t]));

  // KPI tiles.
  document.getElementById("rm-kpi-total").textContent       = summary.total;
  document.getElementById("rm-kpi-complete").textContent    =
    `${summary.complete} (${summary.total ? Math.round(100 * summary.complete / summary.total) : 0}%)`;
  document.getElementById("rm-kpi-inprogress").textContent  = summary.in_progress;
  document.getElementById("rm-kpi-blocked").textContent     = summary.blocked;
  const blockedTile = document.getElementById("rm-kpi-blocked-tile");
  if (summary.blocked > 0) blockedTile.classList.add("has-blocked");
  else blockedTile.classList.remove("has-blocked");

  document.getElementById("roadmap-summary-badge").textContent =
    `(${summary.complete}/${summary.total} complete)`;

  // Group tasks by week.
  const byWeek = new Map();
  for (const t of tasks) {
    if (!byWeek.has(t.week)) byWeek.set(t.week, []);
    byWeek.get(t.week).push(t);
  }
  const weeks = [...byWeek.keys()].sort((a, b) => a - b);
  const body = document.getElementById("roadmap-body");
  body.innerHTML = weeks.map(w => _renderWeek(w, byWeek.get(w))).join("");

  // Wire up event handlers (delegated would be cleaner, but the DOM
  // size here is small enough that this is fine).
  body.querySelectorAll("[data-rm-week-toggle]").forEach(el => {
    el.addEventListener("click", () => {
      const wk = Number(el.dataset.rmWeekToggle);
      if (_roadmapCollapsedWeeks.has(wk)) _roadmapCollapsedWeeks.delete(wk);
      else _roadmapCollapsedWeeks.add(wk);
      _persistCollapsedWeeks();
      refreshRoadmap();
    });
  });
  body.querySelectorAll("[data-rm-task-title]").forEach(el => {
    el.addEventListener("click", () => {
      const id = el.dataset.rmTaskTitle;
      if (_roadmapExpanded.has(id)) _roadmapExpanded.delete(id);
      else _roadmapExpanded.add(id);
      refreshRoadmap();
    });
  });
  body.querySelectorAll("[data-rm-status]").forEach(el => {
    el.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      if (el.classList.contains("disabled")) return;
      const id = el.dataset.rmStatus;
      const task = _roadmapTasksById[id];
      const next = _nextStatus(task.status);
      const updated = await _patchTask(id, { status: next });
      if (updated) refreshRoadmap();
    });
  });
  body.querySelectorAll("[data-rm-block]").forEach(el => {
    el.addEventListener("click", (ev) => {
      ev.stopPropagation();
      _openBlockModal(el.dataset.rmBlock);
    });
  });
  body.querySelectorAll("[data-rm-progress-input]").forEach(el => {
    el.addEventListener("change", async () => {
      const id = el.dataset.rmProgressInput;
      const v = Math.max(0, Math.min(100, Number(el.value) || 0));
      const updated = await _patchTask(id, { progress_pct: v });
      if (updated) refreshRoadmap();
    });
  });
  body.querySelectorAll("[data-rm-notes-save]").forEach(el => {
    el.addEventListener("click", async () => {
      const id = el.dataset.rmNotesSave;
      const ta = document.querySelector(`[data-rm-notes-input="${id}"]`);
      const updated = await _patchTask(id, { notes: ta.value });
      if (updated) refreshRoadmap();
    });
  });
}

function _renderWeek(week, tasks) {
  const collapsed = _roadmapCollapsedWeeks.has(week);
  const total = tasks.length;
  const complete = tasks.filter(t => t.status === "complete").length;
  const tasksHtml = tasks.map(t => _renderTask(t)).join("");
  return `
    <div class="rm-week ${collapsed ? "collapsed" : ""}">
      <div class="rm-week-hdr" data-rm-week-toggle="${week}">
        <div>Week ${week} <span class="rm-week-counts">— ${complete}/${total} complete</span></div>
        <span class="caret">▾</span>
      </div>
      <div class="rm-week-body">${tasksHtml}</div>
    </div>
  `;
}

function _renderTask(t) {
  const expanded = _roadmapExpanded.has(t.id);
  const depsOk = _depsComplete(t);
  // Status pill is only disabled when transitioning OUT of not_started
  // AND deps are not all complete. Once a task is in_progress (e.g.
  // user wants to revert), the cycle remains live.
  const cycleDisabled = !depsOk && t.status === "not_started";
  const missing = _missingDeps(t);
  const tooltip = cycleDisabled
    ? `Dependencies still open: ${missing.join(", ")}`
    : `Click to advance to '${_statusLabel(_nextStatus(t.status))}'`;
  const pct = Math.max(0, Math.min(100, t.progress_pct || 0));

  return `
    <div class="rm-task ${expanded ? "expanded" : ""}" data-rm-task="${t.id}">
      <div class="rm-task-row">
        <div class="rm-task-title" data-rm-task-title="${t.id}">
          <span class="rm-task-id">${t.id}</span>${escapeHtml(t.title)}
        </div>
        <span class="rm-cat-badge cat-${t.category}">${t.category}</span>
        <div class="rm-progress" title="${pct}% complete">
          <div class="rm-progress-bar" style="width: ${pct}%"></div>
        </div>
        <span class="rm-status-pill rm-status-${t.status} ${cycleDisabled ? "disabled" : ""}"
              data-rm-status="${t.id}" title="${escapeHtml(tooltip)}">
          ${_statusLabel(t.status)}
        </span>
      </div>
      <div class="rm-task-details">
        <h4>Description</h4>
        <div>${escapeHtml(t.description)}</div>
        <h4>Acceptance criteria</h4>
        <div>${escapeHtml(t.acceptance_criteria)}</div>
        ${_renderDeps(t)}
        ${_renderDeliverables(t)}
        ${t.blocked_reason
          ? `<div class="rm-blocked-reason">Blocker: ${escapeHtml(t.blocked_reason)}</div>`
          : ""}
        <h4>Notes</h4>
        <textarea data-rm-notes-input="${t.id}" rows="3">${escapeHtml(t.notes || "")}</textarea>
        <div class="rm-notes-actions">
          <button class="rm-btn" data-rm-notes-save="${t.id}">Save notes</button>
          ${t.status !== "blocked"
            ? `<button class="rm-btn" data-rm-block="${t.id}">Mark blocked</button>`
            : ""}
          <span class="muted" style="font-size:11px; align-self:center;">
            progress&nbsp;
            <input type="number" min="0" max="100" value="${pct}"
                   data-rm-progress-input="${t.id}"
                   style="width:50px; background:var(--card); color:var(--text); border:1px solid var(--border); border-radius:3px; padding:1px 4px;">
            %
          </span>
        </div>
      </div>
    </div>
  `;
}

function _renderDeps(t) {
  if (!t.dependencies || t.dependencies.length === 0) return "";
  const items = t.dependencies.map(id => {
    const dep = _roadmapTasksById[id];
    const cls = dep && dep.status === "complete" ? "complete" : "pending";
    const status = dep ? _statusLabel(dep.status) : "missing";
    return `<li class="${cls}"><span class="rm-task-id">${escapeHtml(id)}</span> — ${status}</li>`;
  }).join("");
  return `<h4>Dependencies</h4><ul class="rm-deps">${items}</ul>`;
}

function _renderDeliverables(t) {
  if (!t.deliverables || t.deliverables.length === 0) return "";
  const items = t.deliverables.map(d => `<li>${escapeHtml(d)}</li>`).join("");
  return `<h4>Deliverables</h4><ul>${items}</ul>`;
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

let _modalTaskId = null;
function _openBlockModal(taskId) {
  _modalTaskId = taskId;
  document.getElementById("rm-modal-task-id").textContent = taskId;
  document.getElementById("rm-modal-reason").value = "";
  document.getElementById("roadmap-modal").style.display = "flex";
  setTimeout(() => document.getElementById("rm-modal-reason").focus(), 50);
}

function _closeBlockModal() {
  document.getElementById("roadmap-modal").style.display = "none";
  _modalTaskId = null;
}

document.getElementById("rm-modal-cancel").addEventListener("click", _closeBlockModal);
document.getElementById("rm-modal-save").addEventListener("click", async () => {
  if (!_modalTaskId) return;
  const reason = document.getElementById("rm-modal-reason").value.trim();
  if (!reason) {
    alert("Blocked reason is required.");
    return;
  }
  const updated = await _patchTask(_modalTaskId, {
    status: "blocked", blocked_reason: reason,
  });
  _closeBlockModal();
  if (updated) refreshRoadmap();
});


refreshChartHistory();
fastTick();
refreshRoadmap();
setInterval(fastTick, FAST_REFRESH_MS);
setInterval(refreshChartHistory, CHART_HISTORY_REFRESH_MS);
setInterval(refreshRoadmap, ROADMAP_REFRESH_MS);
