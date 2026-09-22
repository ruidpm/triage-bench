/*
 * Replay pacer: renders queued stream frames one per interval.
 *
 * Pure state machine with no DOM and no built-in timers: the caller injects the clock
 * (now), the timer functions (schedule / cancel) and the current interval, so the
 * transitions can be tested deterministically (tests/web/js/pacer.test.mjs).
 *
 * - enqueue: a frame arrived; it renders when its turn comes.
 * - markDone: the stream ended; onDrained fires once every queued frame has rendered.
 * - pause / resume: freeze and continue; the queue is kept, so nothing is skipped or repeated.
 * - retime: the interval changed; the pending step is re-timed from the last render.
 * - reset: drop everything (restart, stop on error). Safe to call from inside render.
 */

export function createPacer({ now, schedule, cancel, intervalMs, render, onDrained }) {
  let queue = [];
  let timer = null;
  let lastRenderAt = -Infinity;
  let streamDone = false;
  let paused = false;

  function cancelStep() {
    if (timer !== null) {
      cancel(timer);
      timer = null;
    }
  }

  function scheduleStep() {
    cancelStep();
    const wait = Math.max(0, lastRenderAt + intervalMs() - now());
    timer = schedule(step, wait);
  }

  function continueOrDrain() {
    if (queue.length > 0) {
      scheduleStep();
    } else if (streamDone) {
      onDrained();
    }
  }

  function step() {
    timer = null;
    const frame = queue.shift();
    if (frame !== undefined) {
      lastRenderAt = now();
      render(frame);
    }
    // Read the state after render: render may have reset the pacer (stop on error).
    continueOrDrain();
  }

  return {
    enqueue(frame) {
      queue.push(frame);
      if (timer === null && !paused) {
        scheduleStep();
      }
    },

    markDone() {
      streamDone = true;
      if (queue.length === 0 && timer === null) {
        onDrained();
      }
    },

    pause() {
      cancelStep();
      paused = true;
    },

    resume() {
      paused = false;
      continueOrDrain();
    },

    retime() {
      if (timer !== null) {
        scheduleStep();
      }
    },

    reset() {
      cancelStep();
      queue = [];
      lastRenderAt = -Infinity;
      streamDone = false;
      paused = false;
    },

    isPaused: () => paused,
    queuedCount: () => queue.length,
  };
}
