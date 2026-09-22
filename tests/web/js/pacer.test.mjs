// Replay pacer transitions, driven by a fake clock so every timing is deterministic.
// Run: node --test 'tests/web/js/*.test.mjs'
import { test } from "node:test";
import assert from "node:assert/strict";

import { createPacer } from "../../../src/triage_bench/web/static/pacer.js";

const TWO_TPS_MS = 500;
const RUN_LENGTH = 200;

/** A manual clock: timers fire only when advance() passes their due time. */
function fakeClock() {
  let now = 0;
  let nextId = 1;
  const timers = new Map();
  return {
    now: () => now,
    schedule(callback, delayMs) {
      const id = nextId++;
      timers.set(id, { dueAt: now + delayMs, callback });
      return id;
    },
    cancel(id) {
      timers.delete(id);
    },
    pendingTimers: () => timers.size,
    advance(ms) {
      const end = now + ms;
      for (;;) {
        let dueId = null;
        for (const [id, timer] of timers) {
          if (timer.dueAt <= end && (dueId === null || timer.dueAt < timers.get(dueId).dueAt)) {
            dueId = id;
          }
        }
        if (dueId === null) {
          break;
        }
        const { dueAt, callback } = timers.get(dueId);
        timers.delete(dueId);
        now = dueAt;
        callback();
      }
      now = end;
    },
  };
}

function frames(first, last) {
  return Array.from({ length: last - first + 1 }, (_, index) => ({ tick: first + index }));
}

/** A pacer wired to a fake clock, recording what it rendered and when it drained. */
function harness({ intervalMs = TWO_TPS_MS } = {}) {
  const clock = fakeClock();
  const rendered = [];
  const events = [];
  let interval = intervalMs;
  const pacer = createPacer({
    now: clock.now,
    schedule: (callback, delayMs) => clock.schedule(callback, delayMs),
    cancel: (id) => clock.cancel(id),
    intervalMs: () => interval,
    render: (frame) => {
      rendered.push(frame.tick);
      events.push(`render ${frame.tick}`);
    },
    onDrained: () => events.push("drained"),
  });
  return {
    clock,
    pacer,
    rendered,
    events,
    setInterval: (ms) => {
      interval = ms;
    },
  };
}

const drainedCount = (events) => events.filter((e) => e === "drained").length;

test("renders every frame once, in order, then drains once", () => {
  const { clock, pacer, rendered, events } = harness();
  frames(1, RUN_LENGTH).forEach((frame) => pacer.enqueue(frame));
  pacer.markDone();
  clock.advance(RUN_LENGTH * TWO_TPS_MS);
  assert.deepEqual(rendered, frames(1, RUN_LENGTH).map((f) => f.tick));
  assert.equal(drainedCount(events), 1);
  assert.equal(events.at(-1), "drained");
});

test("renders the first frame at once, then one per interval", () => {
  const { clock, pacer, rendered } = harness();
  frames(1, 3).forEach((frame) => pacer.enqueue(frame));
  clock.advance(0);
  assert.deepEqual(rendered, [1]);
  clock.advance(TWO_TPS_MS - 1);
  assert.deepEqual(rendered, [1]);
  clock.advance(1);
  assert.deepEqual(rendered, [1, 2]);
});

test("pause and resume neither duplicate nor skip a tick", () => {
  const { clock, pacer, rendered } = harness();
  frames(1, 50).forEach((frame) => pacer.enqueue(frame));
  clock.advance(10 * TWO_TPS_MS);
  pacer.pause();
  const pausedAt = rendered.length;
  assert.equal(clock.pendingTimers(), 0, "pause cancels the pending step");
  frames(51, 100).forEach((frame) => pacer.enqueue(frame)); // the stream keeps buffering
  clock.advance(100 * TWO_TPS_MS);
  assert.equal(rendered.length, pausedAt, "nothing renders while paused");
  assert.equal(clock.pendingTimers(), 0, "buffering while paused schedules nothing");

  pacer.resume();
  clock.advance(0);
  assert.equal(rendered.at(-1), pausedAt + 1, "resume continues with the next unrendered frame");
  clock.advance(100 * TWO_TPS_MS);
  assert.deepEqual(rendered, frames(1, 100).map((f) => f.tick));
});

test("done while playing waits for the queue to drain", () => {
  const { clock, pacer, events } = harness();
  frames(1, 5).forEach((frame) => pacer.enqueue(frame));
  pacer.markDone();
  clock.advance(2 * TWO_TPS_MS);
  assert.equal(drainedCount(events), 0);
  clock.advance(10 * TWO_TPS_MS);
  assert.deepEqual(events.slice(-2), ["render 5", "drained"]);
  assert.equal(drainedCount(events), 1);
});

test("done arriving while paused waits for resume and the rest of the queue", () => {
  const { clock, pacer, rendered, events } = harness();
  frames(1, 10).forEach((frame) => pacer.enqueue(frame));
  clock.advance(2 * TWO_TPS_MS);
  pacer.pause();
  pacer.markDone();
  clock.advance(100 * TWO_TPS_MS);
  assert.equal(drainedCount(events), 0, "no finish while frames are still queued");

  pacer.resume();
  clock.advance(100 * TWO_TPS_MS);
  assert.deepEqual(rendered, frames(1, 10).map((f) => f.tick));
  assert.equal(events.at(-1), "drained");
  assert.equal(drainedCount(events), 1);
});

test("done with nothing queued drains straight away, even when paused", () => {
  const { clock, pacer, events } = harness();
  pacer.enqueue({ tick: 1 });
  clock.advance(0);
  pacer.pause();
  pacer.markDone();
  assert.equal(drainedCount(events), 1);
});

test("reset clears the queue, the pending step and the done flag", () => {
  const { clock, pacer, rendered, events } = harness();
  frames(1, 10).forEach((frame) => pacer.enqueue(frame));
  clock.advance(0);
  pacer.markDone();
  pacer.reset();
  assert.equal(clock.pendingTimers(), 0);
  assert.equal(pacer.queuedCount(), 0);
  clock.advance(100 * TWO_TPS_MS);
  assert.deepEqual(rendered, [1]);

  // A restarted run renders from its own first frame and finishes only on its own done.
  pacer.enqueue({ tick: 1 });
  clock.advance(100 * TWO_TPS_MS);
  assert.deepEqual(rendered, [1, 1]);
  assert.equal(drainedCount(events), 0);
});

test("reset also clears a pause", () => {
  const { clock, pacer, rendered } = harness();
  pacer.enqueue({ tick: 1 });
  pacer.pause();
  pacer.reset();
  assert.equal(pacer.isPaused(), false);
  pacer.enqueue({ tick: 1 });
  clock.advance(0);
  assert.deepEqual(rendered, [1]);
});

test("a speed change re-times the pending step from the last render", () => {
  const slow = 1000;
  const fast = 200;
  const { clock, pacer, rendered, setInterval } = harness({ intervalMs: slow });
  frames(1, 3).forEach((frame) => pacer.enqueue(frame));
  clock.advance(0);
  clock.advance(100);
  setInterval(fast);
  pacer.retime();
  clock.advance(fast - 100 - 1);
  assert.deepEqual(rendered, [1]);
  clock.advance(1);
  assert.deepEqual(rendered, [1, 2], "the next frame follows the new interval, not the old one");
});

test("a speed change while paused schedules nothing", () => {
  const { clock, pacer } = harness();
  frames(1, 3).forEach((frame) => pacer.enqueue(frame));
  clock.advance(0);
  pacer.pause();
  pacer.retime();
  assert.equal(clock.pendingTimers(), 0);
});

test("a reset from inside render stops the run", () => {
  const clock = fakeClock();
  const rendered = [];
  const pacer = createPacer({
    now: clock.now,
    schedule: (callback, delayMs) => clock.schedule(callback, delayMs),
    cancel: (id) => clock.cancel(id),
    intervalMs: () => TWO_TPS_MS,
    render: (frame) => {
      rendered.push(frame.tick);
      if (frame.tick === 3) {
        pacer.reset(); // what the dashboard does when a frame cannot be rendered
      }
    },
    onDrained: () => assert.fail("a stopped run must not finish"),
  });
  frames(1, 10).forEach((frame) => pacer.enqueue(frame));
  pacer.markDone();
  clock.advance(100 * TWO_TPS_MS);
  assert.deepEqual(rendered, [1, 2, 3]);
  assert.equal(clock.pendingTimers(), 0);
});
