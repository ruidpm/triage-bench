/*
 * Triage Bench dashboard.
 *
 * Renders the server's TickEvent frames. The server computes every metric in
 * `totals`; this file does no metric maths except the routing panel, which
 * mirrors domain/routing.py confidence_gated_report so the threshold slider can
 * be explored without a round trip. All stream text reaches the DOM through
 * textContent, never innerHTML.
 *
 * Replay pacing happens here, not on the server: the stream is opened at the server's
 * maximum speed (max_speed_tps from /api/meta), frames wait in a queue, and a timer renders
 * them at the speed slider's current rate, so the slider changes the pace mid-run.
 * Pause stops that timer and keeps the queue (the stream keeps buffering), so Resume
 * continues from the next unrendered frame; Restart resets and replays from ticket 1.
 * The pacing state machine lives in pacer.js and the chart points in latency-points.js,
 * both pure and unit-tested (tests/web/js/).
 */

import { appendLatencyPoints, isErrorDecision } from "./latency-points.js";
import { createPacer } from "./pacer.js";

const API_META = "/api/meta";
const API_STREAM = "/api/stream";
const DONE_EVENT = "done";
const MODE_LIVE = "live";
const MS_PER_SECOND = 1000;

const PLACEHOLDER = "–";
const PERCENT = 100;
const LOG_BASE = 10;
// Floating-point slack when matching a tick's mantissa (e.g. 0.3 / 0.1 = 2.9999999999999996).
const LOG_TICK_TOLERANCE = 1e-9;
// Screen-reader summary cadence: announce progress every N ticks, plus start, pause, resume
// and finish.
const ANNOUNCE_EVERY_TICKS = 25;
const EXACT_DECIMALS = 100;

const DIGITS = {
  accuracy: 1,
  confidence: 0,
  cost: 4,
  ece: 3,
  latency: 0,
  share: 0,
  speed: 1,
  threshold: 2,
};

const ROUTING_PRIMARY = "von";
const ROUTING_FALLBACK = "haiku";

const CONTESTANTS = {
  von:   { title: "Von",              model: "wfzyx/von-1.0 · local",   free: true },
  haiku: { title: "Claude Haiku 4.5", model: "claude-haiku-4-5 · API" },
  luna:  { title: "GPT-5.6 Luna",     model: "gpt-5.6-luna · API" },
};

const OUTCOMES = {
  ok:   { className: "ok",   glyph: "✓", word: "correct" },
  miss: { className: "miss", glyph: "✗", word: "wrong" },
  err:  { className: "err",  glyph: "!", word: "error" },
};
const OUTCOME_CLASSES = Object.values(OUTCOMES).map((o) => o.className);

const BUTTON_TEXT = {
  start: "Start",
  pause: "Pause",
  resume: "Resume",
  running: "Running",
  finished: "Finished",
};
const LIVE_RESTART_TITLE = "a live run cannot be restarted from the browser";
const FREE_COST_TEXT = "local";
const MESSAGES = {
  disconnected: "stream disconnected",
  liveRefused:
    "the server refused the stream: this live run already started (reload or second tab). " +
    "One live run per process; restart the command for another run.",
  badFrame: "stream sent a frame the dashboard could not read",
  metaFailed: "could not load run details from the server",
  chartMissing: "chart library failed to load; the latency chart is unavailable",
  routingIncomplete: "a ticket is missing a Von or Haiku decision; routing panel stopped",
};

// The chart keeps every ticket (200 x 3 points), so it draws lines without point markers
// (markers appear on hover), skips crowded x labels, and does not animate per-tick updates.
// Each point carries its own ticket number ({x: tick, y: ms}) on a linear axis, so there is
// no separate labels array that could drift out of step with the data.
const CHART = {
  lineWidth: 1.5,
  pointRadius: 0,
  pointHoverRadius: 4,
  xLabelGapPx: 12,
  pointRingWidth: 2,
  logTickMantissas: [1, 2, 5],
  fontSize: 12,
  legendKeyWidth: 18,
  legendKeyHeight: 2,
};
const LIGHT_SCHEME_QUERY = "(prefers-color-scheme: light)";

// The inline script in index.html's <head> reads the same key and values before first paint.
const THEME = {
  storageKey: "triage-bench-theme",
  system: "system",
  explicit: ["light", "dark"],
};

/* ---------- DOM ---------- */

const $ = (id) => document.getElementById(id);
const els = {
  mode: $("mode"),
  runName: $("run-name"),
  tick: $("tick"),
  play: $("play"),
  restart: $("restart"),
  speed: $("speed"),
  speedWrap: $("speed-wrap"),
  speedOut: $("speed-out"),
  threshold: $("threshold"),
  thresholdOut: $("threshold-out"),
  banner: $("banner"),
  bannerText: $("banner-text"),
  cards: $("cards"),
  cardTemplate: $("card-template"),
  ticketText: $("ticket-text"),
  ticketLabel: $("ticket-label"),
  latency: $("latency"),
  routing: $("routing"),
  routingText: $("routing-text"),
  liveStatus: $("live-status"),
  chartCaption: $("chart-caption"),
  rLocal: $("r-local"),
  rAcc: $("r-acc"),
  rCost: $("r-cost"),
  rFallback: $("r-fallback"),
  rSaving: $("r-saving"),
  feedHeadRow: $("feed-head-row"),
  themeInputs: document.querySelectorAll('input[name="theme"]'),
  feedBody: document.querySelector("#feed tbody"),
};

/* ---------- state ---------- */

const state = {
  mode: "",
  contestants: [],
  cards: new Map(),
  history: [],
  lastTotal: 0,
  lastTotals: null,
  lastTick: 0,
  source: null,
  streamOpened: false,
  maxSpeedTps: 0,
  // A run has started and has not finished or stopped (a paused replay is still running).
  running: false,
  chart: null,
  routingBroken: false,
};

/* ---------- formatting (mirrors Python's format spec) ---------- */

/**
 * Format like Python's f"{value:.{digits}f}": round the exact binary value of
 * the double, ties to even. Number.prototype.toFixed rounds ties up, which
 * would make e.g. a 12.5% share read "13%" here and "12%" in the CLI.
 */
function formatFixed(value, digits) {
  const exact = Math.abs(value).toFixed(EXACT_DECIMALS);
  const [intPart, fracPart] = exact.split(".");
  const kept = BigInt(intPart + fracPart.slice(0, digits));
  const rest = fracPart.slice(digits);
  const firstDropped = rest[0];
  const anyNonZeroAfter = /[1-9]/.test(rest.slice(1));
  let roundUp;
  if (firstDropped > "5") {
    roundUp = true;
  } else if (firstDropped < "5") {
    roundUp = false;
  } else {
    roundUp = anyNonZeroAfter || kept % 2n === 1n;
  }
  const scaled = (roundUp ? kept + 1n : kept).toString().padStart(digits + 1, "0");
  const sign = value < 0 ? "-" : "";
  if (digits === 0) {
    return sign + scaled;
  }
  return `${sign}${scaled.slice(0, -digits)}.${scaled.slice(-digits)}`;
}

/** Python's f"{value:.{digits}%}": multiply by 100 as a double, then format. */
function formatPercent(share, digits) {
  return `${formatFixed(share * PERCENT, digits)}%`;
}

function formatCost(usd) {
  return `$${formatFixed(usd, DIGITS.cost)}`;
}

function formatMs(ms) {
  return `${formatFixed(ms, DIGITS.latency)} ms`;
}

/* ---------- small helpers ---------- */

function describe(key) {
  return CONTESTANTS[key] ?? { title: key, model: "" };
}

function outcomeOf(decision) {
  if (isErrorDecision(decision)) {
    return OUTCOMES.err;
  }
  return decision.correct ? OUTCOMES.ok : OUTCOMES.miss;
}

function setOutcomeClass(element, outcome) {
  element.classList.remove(...OUTCOME_CLASSES);
  if (outcome) {
    element.classList.add(outcome.className);
  }
}

function cssToken(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function showBanner(message) {
  els.bannerText.textContent = message;
  els.banner.hidden = false;
}

function hideBanner() {
  els.banner.hidden = true;
  els.bannerText.textContent = "";
}

/* ---------- cards ---------- */

function buildCards() {
  els.cards.replaceChildren();
  state.cards.clear();
  for (const key of state.contestants) {
    const info = describe(key);
    const fragment = els.cardTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".card");
    card.dataset.contestant = key;
    card.setAttribute("aria-label", info.title);
    card.querySelector(".card-name").textContent = info.title;
    card.querySelector(".card-model").textContent = info.model;
    state.cards.set(key, {
      root: card,
      verdict: card.querySelector(".verdict"),
      glyph: card.querySelector(".verdict .glyph"),
      label: card.querySelector(".verdict-label"),
      conf: card.querySelector(".verdict-conf"),
      ring: card.querySelector(".ring"),
      ringValue: card.querySelector(".ring-value"),
      p50: card.querySelector(".p50"),
      cost: card.querySelector(".cost"),
      ece: card.querySelector(".ece"),
      errors: card.querySelector(".errors"),
    });
    els.cards.append(fragment);
  }
}

function resetCard(card) {
  setOutcomeClass(card.verdict, null);
  card.glyph.textContent = "";
  card.label.textContent = PLACEHOLDER;
  card.conf.textContent = "";
  card.ring.style.setProperty("--pct", "0");
  card.ring.setAttribute("aria-label", "accuracy not yet known");
  card.ringValue.textContent = PLACEHOLDER;
  card.p50.textContent = PLACEHOLDER;
  card.cost.textContent = PLACEHOLDER;
  card.ece.textContent = PLACEHOLDER;
  card.errors.textContent = "0";
}

function renderVerdict(card, decision) {
  const outcome = outcomeOf(decision);
  setOutcomeClass(card.verdict, outcome);
  card.glyph.textContent = outcome.glyph;
  if (outcome === OUTCOMES.err) {
    card.label.textContent = outcome.word;
    card.conf.textContent = decision.error;
    return;
  }
  card.label.textContent = decision.label;
  card.conf.textContent = formatPercent(decision.confidence, DIGITS.confidence);
}

function renderTotals(card, key, totals) {
  const accuracyText = formatPercent(totals.accuracy, DIGITS.accuracy);
  card.ring.style.setProperty("--pct", String(totals.accuracy * PERCENT));
  card.ring.setAttribute("aria-label", `accuracy ${accuracyText}`);
  card.ringValue.textContent = accuracyText;
  const hasScorable = totals.count > totals.errors;
  card.p50.textContent = hasScorable ? formatMs(totals.p50_ms) : PLACEHOLDER;
  card.ece.textContent = hasScorable ? formatFixed(totals.ece, DIGITS.ece) : PLACEHOLDER;
  card.cost.textContent = describe(key).free ? FREE_COST_TEXT : formatCost(totals.cost_usd);
  card.errors.textContent = String(totals.errors);
}

/* ---------- chart ---------- */

function chartColours() {
  return {
    ink: cssToken("--ink-2"),
    muted: cssToken("--ink-muted"),
    grid: cssToken("--hairline"),
    axis: cssToken("--axis"),
    surface: cssToken("--surface"),
    series: (key) => cssToken(`--series-${key}`) || cssToken("--ink-muted"),
  };
}

function isRoundLogTick(value) {
  const mantissa = value / LOG_BASE ** Math.floor(Math.log10(value));
  // Compare with a tolerance, not Math.round: rounding would let 1.5 (1500 ms) pass as 2.
  return CHART.logTickMantissas.some((m) => Math.abs(mantissa - m) < LOG_TICK_TOLERANCE);
}

function keepRoundLogTicks(axis) {
  axis.ticks = axis.ticks.filter((tick) => isRoundLogTick(tick.value));
}

function tooltipLabel(context) {
  const value = context.parsed.y;
  const text = value === null ? OUTCOMES.err.word : formatMs(value);
  return `${context.dataset.label}: ${text}`;
}

function buildChart() {
  if (typeof Chart === "undefined") {
    console.error("Chart.js did not load from the CDN; the latency chart is disabled.");
    showBanner(MESSAGES.chartMissing);
    return;
  }
  const colours = chartColours();
  Chart.defaults.font.family = cssToken("--font");
  Chart.defaults.font.size = CHART.fontSize;
  state.chart = new Chart(els.latency, {
    type: "line",
    data: {
      datasets: state.contestants.map((key) => ({
        contestant: key,
        label: describe(key).title,
        data: [],
        borderColor: colours.series(key),
        backgroundColor: colours.series(key),
        pointBorderColor: colours.surface,
        borderWidth: CHART.lineWidth,
        pointRadius: CHART.pointRadius,
        pointHoverRadius: CHART.pointHoverRadius,
        pointBorderWidth: CHART.pointRingWidth,
        borderCapStyle: "round",
        borderJoinStyle: "round",
        spanGaps: false,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          type: "linear",
          bounds: "data",
          title: { display: true, text: "ticket" },
          grid: { display: false },
          ticks: {
            precision: 0,
            autoSkip: true,
            autoSkipPadding: CHART.xLabelGapPx,
            maxRotation: 0,
          },
        },
        y: {
          type: "logarithmic",
          title: { display: true, text: "ms" },
          afterBuildTicks: keepRoundLogTicks,
          ticks: { callback: (value) => String(value) },
        },
      },
      plugins: {
        legend: {
          position: "top",
          align: "start",
          labels: { boxWidth: CHART.legendKeyWidth, boxHeight: CHART.legendKeyHeight },
        },
        tooltip: { callbacks: { label: tooltipLabel } },
      },
    },
  });
  applyChartTheme();
}

function applyChartTheme() {
  const chart = state.chart;
  if (!chart) {
    return;
  }
  const colours = chartColours();
  for (const dataset of chart.data.datasets) {
    dataset.borderColor = colours.series(dataset.contestant);
    dataset.backgroundColor = colours.series(dataset.contestant);
    dataset.pointBorderColor = colours.surface;
  }
  const { x, y } = chart.options.scales;
  for (const axis of [x, y]) {
    axis.ticks.color = colours.muted;
    axis.title.color = colours.muted;
    axis.border = { color: colours.axis };
  }
  y.grid = { color: colours.grid };
  chart.options.plugins.legend.labels.color = colours.ink;
  chart.update("none");
}

function pushLatency(tick, decisions) {
  const chart = state.chart;
  if (!chart) {
    return;
  }
  appendLatencyPoints(chart.data.datasets, tick, decisions);
  chart.update("none");
}

function resetChart() {
  const chart = state.chart;
  if (!chart) {
    return;
  }
  for (const dataset of chart.data.datasets) {
    dataset.data = [];
  }
  chart.update("none");
}

/* ---------- feed ---------- */

function buildFeedHeads() {
  els.feedHeadRow.querySelectorAll("th[data-contestant]").forEach((th) => th.remove());
  for (const key of state.contestants) {
    const th = document.createElement("th");
    th.scope = "col";
    th.dataset.contestant = key;
    th.textContent = describe(key).title;
    els.feedHeadRow.append(th);
  }
}

function textCell(text, className) {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) {
    td.className = className;
  }
  return td;
}

function outcomeCell(key, decision) {
  const td = document.createElement("td");
  const title = describe(key).title;
  td.className = "outcome";
  td.dataset.name = title;
  if (!decision) {
    td.textContent = PLACEHOLDER;
    return td;
  }
  const outcome = outcomeOf(decision);
  td.classList.add(outcome.className);
  const inner = document.createElement("span");
  inner.className = "outcome-inner";
  const glyph = document.createElement("span");
  glyph.className = "glyph";
  glyph.setAttribute("aria-hidden", "true");
  glyph.textContent = outcome.glyph;
  const label = document.createElement("span");
  label.className = "outcome-label";
  // Glyph and label stay together; only the confidence may wrap onto its own line.
  const answer = document.createElement("span");
  answer.className = "outcome-answer";
  answer.append(glyph, label);
  inner.append(answer);
  td.append(inner);
  if (isErrorDecision(decision)) {
    // Errors show their message, so nothing is reachable only by hovering.
    label.textContent = decision.error;
    td.setAttribute("aria-label", `${title}, ${outcome.word}: ${decision.error}`);
    return td;
  }
  const confidence = formatPercent(decision.confidence, DIGITS.confidence);
  label.textContent = decision.label;
  const conf = document.createElement("span");
  conf.className = "outcome-conf";
  conf.textContent = confidence;
  inner.append(conf);
  td.setAttribute("aria-label", `${title}, ${outcome.word}: ${decision.label}, ${confidence}`);
  return td;
}

/** Adds one row on top; earlier rows are never rebuilt, so 200 tickets stay cheap. */
function prependFeedRow(event) {
  const row = document.createElement("tr");
  row.append(
    textCell(String(event.tick), "num"),
    textCell(event.ticket.text, "message"),
  );
  const truth = document.createElement("td");
  truth.className = "truth";
  const code = document.createElement("code");
  code.textContent = event.ticket.label;
  truth.append(code);
  row.append(truth);
  for (const key of state.contestants) {
    row.append(outcomeCell(key, event.decisions[key]));
  }
  els.feedBody.prepend(row);
}

/* ---------- routing (mirrors domain/routing.py) ---------- */

function routingAvailable() {
  return state.contestants.includes(ROUTING_PRIMARY) && state.contestants.includes(ROUTING_FALLBACK);
}

function accuracyOf(decisions) {
  if (decisions.length === 0) {
    return 0;
  }
  return decisions.filter((d) => d.correct).length / decisions.length;
}

function sumCost(decisions) {
  return decisions.reduce((total, d) => total + d.cost_usd, 0);
}

/** Same rule as confidence_gated_report: keep primary when !error && confidence >= threshold. */
function confidenceGatedReport(history, threshold) {
  const primary = history.map((h) => h.decisions[ROUTING_PRIMARY]);
  const fallback = history.map((h) => h.decisions[ROUTING_FALLBACK]);
  const routed = [];
  let local = 0;
  primary.forEach((p, index) => {
    if (!isErrorDecision(p) && p.confidence >= threshold) {
      routed.push(p);
      local += 1;
    } else {
      routed.push(fallback[index]);
    }
  });
  const n = routed.length;
  const routedCost = sumCost(routed);
  const fallbackCost = sumCost(fallback);
  return {
    threshold,
    handledLocallyShare: n ? local / n : 0,
    routedAccuracy: accuracyOf(routed),
    routedCostUsd: routedCost,
    fallbackAccuracy: accuracyOf(fallback),
    fallbackCostUsd: fallbackCost,
    costSavingShare: fallbackCost ? (fallbackCost - routedCost) / fallbackCost : 0,
  };
}

function readThreshold() {
  return Number(els.threshold.value);
}

function clearRoutingValues() {
  for (const dd of [els.rLocal, els.rAcc, els.rCost, els.rFallback, els.rSaving]) {
    dd.textContent = PLACEHOLDER;
  }
}

function renderRouting() {
  const threshold = readThreshold();
  const thresholdText = formatFixed(threshold, DIGITS.threshold);
  els.thresholdOut.textContent = thresholdText;
  if (!routingAvailable()) {
    els.routing.hidden = true;
    return;
  }
  const primary = describe(ROUTING_PRIMARY).title;
  const fallback = describe(ROUTING_FALLBACK).title;
  els.routingText.textContent =
    `${primary} answers when its confidence is at least ${thresholdText}, otherwise ${fallback} answers.`;
  if (state.history.length === 0 || state.routingBroken) {
    clearRoutingValues();
    return;
  }
  const incomplete = state.history.find(
    (h) => !h.decisions[ROUTING_PRIMARY] || !h.decisions[ROUTING_FALLBACK],
  );
  if (incomplete) {
    console.error("Routing needs both decisions for every ticket; stopping the routing panel.", {
      ticketId: incomplete.ticket.id,
      present: Object.keys(incomplete.decisions),
      needed: [ROUTING_PRIMARY, ROUTING_FALLBACK],
    });
    state.routingBroken = true;
    showBanner(MESSAGES.routingIncomplete);
    clearRoutingValues();
    return;
  }
  const r = confidenceGatedReport(state.history, threshold);
  els.rLocal.textContent = formatPercent(r.handledLocallyShare, DIGITS.share);
  els.rAcc.textContent = formatPercent(r.routedAccuracy, DIGITS.accuracy);
  els.rCost.textContent = formatCost(r.routedCostUsd);
  els.rFallback.textContent =
    `${formatPercent(r.fallbackAccuracy, DIGITS.accuracy)} · ${formatCost(r.fallbackCostUsd)}`;
  els.rSaving.textContent = formatPercent(r.costSavingShare, DIGITS.share);
}

/* ---------- tick rendering ---------- */

function renderTickCounter(tick, total) {
  els.tick.textContent = `${tick} / ${total}`;
}

function renderTicket(ticket) {
  els.ticketText.textContent = ticket.text;
  els.ticketLabel.textContent = ticket.label;
}

function announce(message) {
  els.liveStatus.textContent = message;
}

function accuracySummary(totals) {
  return state.contestants
    .filter((key) => totals[key])
    .map((key) => `${describe(key).title} ${formatPercent(totals[key].accuracy, DIGITS.accuracy)}`)
    .join(", ");
}

function onTick(event) {
  state.lastTotal = event.total;
  state.history.push({ ticket: event.ticket, decisions: event.decisions });
  renderTickCounter(event.tick, event.total);
  renderTicket(event.ticket);
  for (const [key, card] of state.cards) {
    const decision = event.decisions[key];
    const totals = event.totals[key];
    if (decision && totals) {
      renderVerdict(card, decision);
      renderTotals(card, key, totals);
    }
  }
  pushLatency(event.tick, event.decisions);
  prependFeedRow(event);
  renderRouting();
  state.lastTotals = event.totals;
  state.lastTick = event.tick;
  if (event.tick % ANNOUNCE_EVERY_TICKS === 0) {
    announce(`Ticket ${event.tick} of ${event.total}. Accuracy: ${accuracySummary(event.totals)}.`);
  }
}

function resetDisplay() {
  state.history = [];
  state.routingBroken = false;
  state.lastTotals = null;
  state.lastTick = 0;
  hideBanner();
  renderTickCounter(0, state.lastTotal);
  els.ticketText.textContent = "Waiting for the first ticket…";
  els.ticketLabel.textContent = PLACEHOLDER;
  for (const card of state.cards.values()) {
    resetCard(card);
  }
  resetChart();
  els.feedBody.replaceChildren();
  renderRouting();
}

/* ---------- stream control ---------- */

function setButton(text, { disabled = false, title = "" } = {}) {
  // The label itself names the action (Pause / Resume), so the button is not a toggle and
  // carries no aria-pressed.
  els.play.textContent = text;
  els.play.disabled = disabled;
  if (title) {
    els.play.title = title;
  } else {
    els.play.removeAttribute("title");
  }
}

function closeSource() {
  if (state.source) {
    state.source.close();
    state.source = null;
  }
}

function closeStream() {
  closeSource();
  pacer.reset();
  state.running = false;
}

/* ---------- replay pacing ---------- */

function tickIntervalMs() {
  return MS_PER_SECOND / Number(els.speed.value);
}

const pacer = createPacer({
  now: () => performance.now(),
  schedule: (callback, delayMs) => setTimeout(callback, delayMs),
  cancel: (timer) => clearTimeout(timer),
  intervalMs: tickIntervalMs,
  render: renderFrame,
  onDrained: finishRun,
});

/** Render one frame; a frame that parses but cannot be rendered stops the run visibly. */
function renderFrame(event) {
  try {
    onTick(event);
  } catch (error) {
    console.error("Could not render a stream frame; stopping the stream.", error, event);
    stopWithBanner(MESSAGES.badFrame);
  }
}

function enqueueTick(event) {
  if (state.mode === MODE_LIVE) {
    renderFrame(event);
    return;
  }
  pacer.enqueue(event);
}

/** The slider moved: re-time the pending render from the last one at the new rate. */
function repacePendingTick() {
  pacer.retime();
}

/* ---------- stream events ---------- */

function handleMessage(message) {
  let event;
  try {
    event = JSON.parse(message.data);
  } catch (error) {
    console.error("Unreadable stream frame; stopping the stream.", error, message.data);
    stopWithBanner(MESSAGES.badFrame);
    return;
  }
  enqueueTick(event);
}

function finishRun() {
  closeStream();
  setButton(BUTTON_TEXT.finished, { disabled: true });
  const summary = state.lastTotals ? ` Accuracy: ${accuracySummary(state.lastTotals)}.` : "";
  announce(`Run finished after ${state.lastTick} tickets.${summary}`);
}

function handleDone() {
  // Close now so EventSource does not reconnect when the server ends the response; a replay
  // finishes once the pacer has rendered every queued frame.
  closeSource();
  if (state.mode === MODE_LIVE) {
    finishRun();
  } else {
    pacer.markDone();
  }
}

function handleStreamOpen() {
  state.streamOpened = true;
}

function handleStreamError(error) {
  // EventSource retries on its own; close it so a dropped stream is visible, not silent.
  // It cannot read the HTTP status, but a live stream that fails before opening was refused
  // by the server's one-run-per-process guard (HTTP 409) or the server is gone.
  const refused = state.mode === MODE_LIVE && !state.streamOpened;
  console.error(refused ? "Live stream refused." : "Event stream disconnected.", error);
  stopWithBanner(refused ? MESSAGES.liveRefused : MESSAGES.disconnected);
}

function stopWithBanner(message) {
  closeStream();
  showBanner(message);
  if (state.mode === MODE_LIVE) {
    setButton(BUTTON_TEXT.start, { disabled: true, title: LIVE_RESTART_TITLE });
  } else {
    // A replay keeps what it rendered; Start (or Restart) replays from ticket 1.
    setButton(BUTTON_TEXT.start);
  }
}

/* ---------- start / pause / resume / restart ---------- */

function streamUrl() {
  // Live runs are never paced; replays arrive as fast as the server allows and the pacer
  // renders them at the slider's rate.
  if (state.mode === MODE_LIVE) {
    return API_STREAM;
  }
  return `${API_STREAM}?speed=${encodeURIComponent(state.maxSpeedTps)}`;
}

function startStream() {
  resetDisplay();
  pacer.reset();
  const source = new EventSource(streamUrl());
  state.streamOpened = false;
  source.addEventListener("open", handleStreamOpen);
  source.addEventListener("message", handleMessage);
  source.addEventListener(DONE_EVENT, handleDone);
  source.addEventListener("error", handleStreamError);
  state.source = source;
  state.running = true;
  if (state.mode === MODE_LIVE) {
    setButton(BUTTON_TEXT.running, { disabled: true, title: LIVE_RESTART_TITLE });
  } else {
    setButton(BUTTON_TEXT.pause);
    els.restart.disabled = false;
  }
  announce("Run started.");
}

/** Freeze rendering; the queue and everything on screen stay as they are. */
function pauseReplay() {
  pacer.pause();
  setButton(BUTTON_TEXT.resume);
  announce(`Paused at ticket ${state.lastTick} of ${state.lastTotal}.`);
}

/** Continue from the next unrendered frame. */
function resumeReplay() {
  setButton(BUTTON_TEXT.pause);
  announce(`Resumed at ticket ${state.lastTick} of ${state.lastTotal}.`);
  pacer.resume();
}

function restartReplay() {
  closeStream();
  startStream();
}

function onPlayClick() {
  if (state.mode === MODE_LIVE) {
    // The button is disabled from Start on, so a click here always starts the live run.
    startStream();
  } else if (!state.running) {
    startStream();
  } else if (pacer.isPaused()) {
    resumeReplay();
  } else {
    pauseReplay();
  }
}

/* ---------- theme ---------- */

function isExplicitTheme(choice) {
  return THEME.explicit.includes(choice);
}

/** The theme the <head> script applied from storage, or "system". */
function appliedTheme() {
  const theme = document.documentElement.dataset.theme;
  return isExplicitTheme(theme) ? theme : THEME.system;
}

function saveTheme(choice) {
  try {
    if (isExplicitTheme(choice)) {
      localStorage.setItem(THEME.storageKey, choice);
    } else {
      localStorage.removeItem(THEME.storageKey);
    }
  } catch (error) {
    // Storage can be blocked (private mode, site settings); the theme still applies this visit.
    console.warn("Could not save the theme choice; it applies until reload.", error);
  }
}

function applyTheme(choice) {
  if (isExplicitTheme(choice)) {
    document.documentElement.dataset.theme = choice;
  } else {
    delete document.documentElement.dataset.theme;
  }
  applyChartTheme();
}

function onThemeChange(event) {
  const choice = event.target.value;
  applyTheme(choice);
  saveTheme(choice);
}

function renderThemeChoice() {
  const current = appliedTheme();
  for (const input of els.themeInputs) {
    input.checked = input.value === current;
  }
}

/* ---------- controls ---------- */

function renderSpeed() {
  const speed = Number(els.speed.value);
  const text = Number.isInteger(speed) ? String(speed) : formatFixed(speed, DIGITS.speed);
  els.speedOut.textContent = `${text}×`;
}

function wireControls() {
  els.play.addEventListener("click", onPlayClick);
  els.restart.addEventListener("click", restartReplay);
  els.speed.addEventListener("input", renderSpeed);
  els.speed.addEventListener("input", repacePendingTick);
  els.threshold.addEventListener("input", renderRouting);
  const lightScheme = window.matchMedia(LIGHT_SCHEME_QUERY);
  lightScheme.addEventListener("change", applyChartTheme);
  for (const input of els.themeInputs) {
    input.addEventListener("change", onThemeChange);
  }
}

/* ---------- boot ---------- */

async function loadMeta() {
  const response = await fetch(API_META);
  if (!response.ok) {
    throw new Error(`${API_META} answered HTTP ${response.status}`);
  }
  return response.json();
}

function applyMeta(meta) {
  if (meta.mode !== MODE_LIVE && !(Number.isFinite(meta.max_speed_tps) && meta.max_speed_tps > 0)) {
    throw new Error(`${API_META} sent an unusable max_speed_tps: ${meta.max_speed_tps}`);
  }
  state.mode = meta.mode;
  state.maxSpeedTps = meta.max_speed_tps;
  state.contestants = meta.contestants;
  els.mode.textContent = meta.mode;
  els.runName.textContent = meta.run_name;
  els.speedWrap.hidden = meta.mode === MODE_LIVE;
  els.restart.hidden = meta.mode === MODE_LIVE;
  els.chartCaption.textContent = "ms, log scale";
  buildCards();
  buildFeedHeads();
  buildChart();
  renderRouting();
}

async function boot() {
  renderThemeChoice();
  renderSpeed();
  wireControls();
  els.play.disabled = true;
  try {
    applyMeta(await loadMeta());
    els.play.disabled = false;
  } catch (error) {
    console.error("Failed to load run metadata.", error);
    showBanner(MESSAGES.metaFailed);
  }
}

boot();
