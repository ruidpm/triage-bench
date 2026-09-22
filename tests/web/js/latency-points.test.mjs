// Latency chart point accumulation. Run: node --test 'tests/web/js/*.test.mjs'
import { test } from "node:test";
import assert from "node:assert/strict";

import { appendLatencyPoints } from "../../../src/triage_bench/web/static/latency-points.js";

const ok = (latencyMs) => ({ label: "card_arrival", confidence: 0.9, latency_ms: latencyMs, error: null });
const failed = { label: null, confidence: null, latency_ms: 12, error: "rate limited" };

function emptySeries() {
  return [
    { contestant: "von", data: [] },
    { contestant: "haiku", data: [] },
    { contestant: "luna", data: [] },
  ];
}

test("appends one point per contestant, keyed by the ticket number", () => {
  const series = emptySeries();
  appendLatencyPoints(series, 7, { von: ok(150), haiku: ok(900), luna: ok(1200) });
  assert.deepEqual(series.map((s) => s.data), [
    [{ x: 7, y: 150 }],
    [{ x: 7, y: 900 }],
    [{ x: 7, y: 1200 }],
  ]);
});

test("errors, missing decisions and non-positive latencies break the line", () => {
  const series = emptySeries();
  appendLatencyPoints(series, 3, { von: ok(0), haiku: failed });
  assert.deepEqual(series.map((s) => s.data), [
    [{ x: 3, y: null }],
    [{ x: 3, y: null }],
    [{ x: 3, y: null }],
  ]);
});

test("200 ticks keep every series the same length with contiguous x", () => {
  const series = emptySeries();
  const run = 200;
  for (let tick = 1; tick <= run; tick += 1) {
    appendLatencyPoints(series, tick, { von: ok(tick), haiku: ok(tick), luna: failed });
  }
  for (const s of series) {
    assert.equal(s.data.length, run);
    assert.ok(s.data.every((point, index) => point.x === index + 1));
  }
});
