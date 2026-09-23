/*
 * Dashboard copy that names the benchmark item: "ticket" for triage, "review" for sentiment.
 *
 * Pure: given the noun from /api/meta it returns every string that mentions the item, so the
 * headings, ARIA labels and screen-reader announcements cannot drift apart
 * (tests/web/js/item-copy.test.mjs).
 */

export function itemCopy(noun) {
  const plural = `${noun}s`;
  const capital = noun.charAt(0).toUpperCase() + noun.slice(1);
  return {
    counterLabel: `${plural} processed`,
    restartLabel: `Restart from ${noun} 1`,
    currentHeading: `Current ${noun}`,
    waiting: `Waiting for the first ${noun}…`,
    chartHeading: `Latency per ${noun}`,
    chartAria: `Latency per ${noun}, one line per contestant. The contestant cards list p50 latency.`,
    chartAxis: noun,
    feedHeading: plural.charAt(0).toUpperCase() + plural.slice(1),
    routingIncomplete: `a ${noun} is missing a Von or Haiku decision; routing panel stopped`,
    progress: (tick, total) => `${capital} ${tick} of ${total}.`,
    finished: (count) => `Run finished after ${count} ${plural}.`,
    paused: (tick, total) => `Paused at ${noun} ${tick} of ${total}.`,
    resumed: (tick, total) => `Resumed at ${noun} ${tick} of ${total}.`,
  };
}
