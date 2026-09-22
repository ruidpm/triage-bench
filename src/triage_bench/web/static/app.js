"use strict";

/*
 * Triage Bench dashboard.
 *
 * Renders the server's TickEvent frames. The server computes every metric in
 * `totals`; this file does no metric maths except the routing panel, which
 * mirrors domain/routing.py confidence_gated_report so the threshold slider can
 * be explored without a round trip. All stream text reaches the DOM through
 * textContent, never innerHTML.
 */

const API_META = "/api/meta";
const API_STREAM = "/api/stream";
const DONE_EVENT = "done";
const MODE_LIVE = "live";

const MAX_POINTS = 50;
const MAX_FEED_ROWS = 12;
const MAX_FEED_CHARS = 90;
const ELLIPSIS = "…";
const PLACEHOLDER = "–";
const PERCENT = 100;
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

const BUTTON_TEXT = { start: "Start", pause: "Pause", running: "Running", finished: "Finished" };
const LIVE_RESTART_TITLE = "a live run cannot be restarted from the browser";
const FREE_COST_TEXT = "local";
const MESSAGES = {
  disconnected: "stream disconnected",
  badFrame: "stream sent a frame the dashboard could not read",
  metaFailed: "could not load run details from the server",
  chartMissing: "chart library failed to load; the latency chart is unavailable",
};

const CHART = {
  animationMs: 250,
  lineWidth: 2,
  pointRadius: 3,
  pointHoverRadius: 5,
  pointRingWidth: 2,
  logTickMantissas: [1, 2, 5],
  fontSize: 12,
  legendKeyWidth: 18,
  legendKeyHeight: 2,
};
const LIGHT_SCHEME_QUERY = "(prefers-color-scheme: light)";

/* ---------- DOM ---------- */

const $ = (id) => document.getElementById(id);
const els = {
  mode: $("mode"),
  runName: $("run-name"),
  tick: $("tick"),
  play: $("play"),
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
  rLocal: $("r-local"),
  rAcc: $("r-acc"),
  rCost: $("r-cost"),
  rFallback: $("r-fallback"),
  rSaving: $("r-saving"),
  feedHeadRow: $("feed-head-row"),
  feedBody: document.querySelector("#feed tbody"),
};

/* ---------- state ---------- */

const state = {
  mode: "",
  contestants: [],
  cards: new Map(),
  history: [],
  lastTotal: 0,
  source: null,
  playing: false,
  finished: false,
  chart: null,
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

function truncate(text, maxChars) {
  const chars = Array.from(text);
  if (chars.length <= maxChars) {
    return text;
  }
  return chars.slice(0, maxChars - 1).join("").trimEnd() + ELLIPSIS;
}

/* ---------- small helpers ---------- */

function describe(key) {
  return CONTESTANTS[key] ?? { title: key, model: "" };
}

function isErrorDecision(decision) {
  return decision.error !== null && decision.error !== undefined;
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
  card.conf.removeAttribute("title");
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
    card.conf.title = decision.error;
    return;
  }
  card.label.textContent = decision.label;
  card.conf.textContent = formatPercent(decision.confidence, DIGITS.confidence);
  card.conf.removeAttribute("title");
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
  const mantissa = value / 10 ** Math.floor(Math.log10(value));
  return CHART.logTickMantissas.includes(Math.round(mantissa));
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
      labels: [],
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
      animation: { duration: CHART.animationMs },
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          title: { display: true, text: "ticket" },
          grid: { display: false },
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
  chart.data.labels.push(String(tick));
  for (const dataset of chart.data.datasets) {
    const decision = decisions[dataset.contestant];
    // Errors, and non-positive latencies a log axis cannot place, break the line.
    const plottable = decision && !isErrorDecision(decision) && decision.latency_ms > 0;
    dataset.data.push(plottable ? decision.latency_ms : null);
  }
  if (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    for (const dataset of chart.data.datasets) {
      dataset.data.shift();
    }
  }
  chart.update();
}

function resetChart() {
  const chart = state.chart;
  if (!chart) {
    return;
  }
  chart.data.labels = [];
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
  label.textContent = isErrorDecision(decision) ? outcome.word : decision.label;
  inner.append(glyph, label);
  td.append(inner);
  td.title = isErrorDecision(decision) ? decision.error : `${outcome.word}: ${decision.label}`;
  td.setAttribute("aria-label", `${title}, ${td.title}`);
  return td;
}

function prependFeedRow(event) {
  const row = document.createElement("tr");
  row.append(
    textCell(String(event.tick), "num"),
    textCell(truncate(event.ticket.text, MAX_FEED_CHARS), "message"),
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
  while (els.feedBody.rows.length > MAX_FEED_ROWS) {
    els.feedBody.lastElementChild.remove();
  }
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
  if (state.history.length === 0) {
    for (const dd of [els.rLocal, els.rAcc, els.rCost, els.rFallback, els.rSaving]) {
      dd.textContent = PLACEHOLDER;
    }
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
}

function resetDisplay() {
  state.history = [];
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

function setButton(text, { pressed = false, disabled = false, title = "" } = {}) {
  els.play.textContent = text;
  els.play.setAttribute("aria-pressed", String(pressed));
  els.play.disabled = disabled;
  if (title) {
    els.play.title = title;
  } else {
    els.play.removeAttribute("title");
  }
}

function closeStream() {
  if (state.source) {
    state.source.close();
    state.source = null;
  }
  state.playing = false;
  els.speed.disabled = false;
}

function handleMessage(message) {
  let event;
  try {
    event = JSON.parse(message.data);
  } catch (error) {
    console.error("Unreadable stream frame; stopping the stream.", error, message.data);
    stopWithBanner(MESSAGES.badFrame);
    return;
  }
  onTick(event);
}

function handleDone() {
  closeStream();
  state.finished = true;
  setButton(BUTTON_TEXT.finished, { disabled: true });
}

function handleStreamError(error) {
  // EventSource retries on its own; close it so a dropped stream is visible, not silent.
  console.error("Event stream disconnected.", error);
  stopWithBanner(MESSAGES.disconnected);
}

function stopWithBanner(message) {
  closeStream();
  showBanner(message);
  if (state.mode === MODE_LIVE) {
    setButton(BUTTON_TEXT.start, { disabled: true, title: LIVE_RESTART_TITLE });
  } else {
    setButton(BUTTON_TEXT.start);
  }
}

function startStream() {
  resetDisplay();
  const speed = Number(els.speed.value);
  const source = new EventSource(`${API_STREAM}?speed=${encodeURIComponent(speed)}`);
  source.addEventListener("message", handleMessage);
  source.addEventListener(DONE_EVENT, handleDone);
  source.addEventListener("error", handleStreamError);
  state.source = source;
  state.playing = true;
  state.finished = false;
  els.speed.disabled = true;
  if (state.mode === MODE_LIVE) {
    setButton(BUTTON_TEXT.running, { pressed: true, disabled: true, title: LIVE_RESTART_TITLE });
  } else {
    setButton(BUTTON_TEXT.pause, { pressed: true });
  }
}

function pauseStream() {
  closeStream();
  setButton(BUTTON_TEXT.start);
}

function onPlayClick() {
  if (state.playing) {
    pauseStream();
  } else {
    startStream();
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
  els.speed.addEventListener("input", renderSpeed);
  els.threshold.addEventListener("input", renderRouting);
  const lightScheme = window.matchMedia(LIGHT_SCHEME_QUERY);
  lightScheme.addEventListener("change", applyChartTheme);
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
  state.mode = meta.mode;
  state.contestants = meta.contestants;
  els.mode.textContent = meta.mode;
  els.runName.textContent = meta.run_name;
  els.speedWrap.hidden = meta.mode === MODE_LIVE;
  buildCards();
  buildFeedHeads();
  buildChart();
  renderRouting();
}

async function boot() {
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
