/*
 * Latency chart points. Pure: no Chart.js, no DOM (tests/web/js/latency-points.test.mjs).
 *
 * Each point carries its own ticket number ({x: tick, y: ms}), so every series stays
 * aligned by construction; there is no separate labels array to fall out of step.
 */

export function isErrorDecision(decision) {
  return decision.error !== null && decision.error !== undefined;
}

/** Latency in ms, or null when a log axis cannot place it (error, missing, non-positive). */
function plottableLatency(decision) {
  if (!decision || isErrorDecision(decision) || !(decision.latency_ms > 0)) {
    return null;
  }
  return decision.latency_ms;
}

/** Appends this ticket's point to every series ({contestant, data}); null y breaks the line. */
export function appendLatencyPoints(series, tick, decisions) {
  for (const s of series) {
    s.data.push({ x: tick, y: plottableLatency(decisions[s.contestant]) });
  }
}
