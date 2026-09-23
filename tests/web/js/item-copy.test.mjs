// Dashboard strings that name the benchmark item. Run: node --test 'tests/web/js/*.test.mjs'
import { test } from "node:test";
import assert from "node:assert/strict";

import { itemCopy } from "../../../src/triage_bench/web/static/item-copy.js";

test("ticket copy reproduces the triage dashboard's original strings", () => {
  const copy = itemCopy("ticket");
  assert.equal(copy.counterLabel, "tickets processed");
  assert.equal(copy.restartLabel, "Restart from ticket 1");
  assert.equal(copy.currentHeading, "Current ticket");
  assert.equal(copy.waiting, "Waiting for the first ticket…");
  assert.equal(copy.chartHeading, "Latency per ticket");
  assert.equal(copy.chartAria,
    "Latency per ticket, one line per contestant. The contestant cards list p50 latency.");
  assert.equal(copy.chartAxis, "ticket");
  assert.equal(copy.feedHeading, "Tickets");
  assert.equal(copy.routingIncomplete,
    "a ticket is missing a Von or Haiku decision; routing panel stopped");
  assert.equal(copy.progress(120, 200), "Ticket 120 of 200.");
  assert.equal(copy.finished(200), "Run finished after 200 tickets.");
  assert.equal(copy.paused(7, 200), "Paused at ticket 7 of 200.");
  assert.equal(copy.resumed(7, 200), "Resumed at ticket 7 of 200.");
});

test("review copy capitalises headings and pluralises", () => {
  const copy = itemCopy("review");
  assert.equal(copy.currentHeading, "Current review");
  assert.equal(copy.feedHeading, "Reviews");
  assert.equal(copy.progress(17, 200), "Review 17 of 200.");
  assert.equal(copy.finished(200), "Run finished after 200 reviews.");
});

test("every string mentions the noun", () => {
  const copy = itemCopy("review");
  for (const value of Object.values(copy)) {
    if (typeof value === "string") {
      assert.match(value.toLowerCase(), /review/);
    }
  }
});
