// Capture the README screenshots from a running replay dashboard.
//
// Dependency-free: Node 22+ (global fetch and WebSocket) and a local Google Chrome.
// It launches headless Chrome with a throwaway profile, drives the dashboard over the
// Chrome DevTools Protocol, and writes PNGs to docs/screenshots/.
//
// Usage, from the repo root:
//   uv run triage-bench replay --port 8130 &          # no API keys needed
//   node scripts/capture_screenshots.mjs --url http://127.0.0.1:8130
//   kill %1
//
// Options:
//   --url <url>      dashboard to capture (default http://127.0.0.1:8000)
//   --out <dir>      output directory (default docs/screenshots)
//   --chrome <path>  Chrome binary (default: $CHROME, else the macOS app path)
//
// Each shot reloads the page, so every run starts from ticket 1.

import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const DEFAULT_URL = "http://127.0.0.1:8000";
const DEFAULT_OUT = "docs/screenshots";
const MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const DEVTOOLS_PORT_FILE = "DevToolsActivePort";

const POLL_MS = 50;
const CHROME_START_TIMEOUT_MS = 15_000;
const PAGE_READY_TIMEOUT_MS = 15_000;
const RUN_TIMEOUT_MS = 180_000;
// Mid-run shots are taken this long after a fresh tick: time for the chart and layout to
// settle after the viewport resize, and still before the next tick at 2x (500 ms).
const SETTLE_MS = 350;
// Finished shots have no next tick to race, so they can wait longer for layout to settle.
const FINISHED_SETTLE_MS = 1000;
// Gap kept above the element a clipped shot stops at.
const CLIP_GAP_PX = 4;

const DESKTOP_WIDTH = 1280;
const DESKTOP_HEIGHT = 800;
const DESKTOP_SCALE = 1;
const PHONE_WIDTH = 390;
const PHONE_HEIGHT = 844;
const PHONE_SCALE = 2;

const MID_RUN_TICK = 120;
const REPLAY_SPEED_DEFAULT = 2;
const REPLAY_SPEED_FAST = 10;
const ALT_THRESHOLD = 0.95;
const FINISHED_TEXT = "Finished";

// stopAbove: CSS selector the shot is cut just above; omitted means the full page.
const FEED_SELECTOR = ".feed";

const SHOTS = [
  {
    file: "dashboard-mid-run.png",
    width: DESKTOP_WIDTH, height: DESKTOP_HEIGHT, scale: DESKTOP_SCALE, mobile: false,
    speed: REPLAY_SPEED_DEFAULT, until: { tick: MID_RUN_TICK }, stopAbove: FEED_SELECTOR,
  },
  {
    file: "dashboard-finished.png",
    width: DESKTOP_WIDTH, height: DESKTOP_HEIGHT, scale: DESKTOP_SCALE, mobile: false,
    speed: REPLAY_SPEED_FAST, until: { finished: true },
  },
  {
    file: "dashboard-threshold-095.png",
    width: DESKTOP_WIDTH, height: DESKTOP_HEIGHT, scale: DESKTOP_SCALE, mobile: false,
    speed: REPLAY_SPEED_FAST, until: { finished: true }, threshold: ALT_THRESHOLD,
    stopAbove: FEED_SELECTOR,
  },
  {
    file: "dashboard-phone.png",
    width: PHONE_WIDTH, height: PHONE_HEIGHT, scale: PHONE_SCALE, mobile: true,
    speed: REPLAY_SPEED_DEFAULT, until: { tick: MID_RUN_TICK }, stopAbove: FEED_SELECTOR,
  },
];

function parseArgs(argv) {
  const args = { url: DEFAULT_URL, out: DEFAULT_OUT, chrome: process.env.CHROME || MAC_CHROME };
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, "");
    if (!(key in args) || argv[i + 1] === undefined) {
      throw new Error(`unknown or incomplete option ${argv[i]}`);
    }
    args[key] = argv[i + 1];
  }
  return args;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitFor(check, timeoutMs, what) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await check();
    if (value) {
      return value;
    }
    await sleep(POLL_MS);
  }
  throw new Error(`timed out after ${timeoutMs} ms waiting for ${what}`);
}

async function launchChrome(chromePath) {
  const profile = await mkdtemp(join(tmpdir(), "triage-bench-chrome-"));
  const chrome = spawn(chromePath, [
    "--headless=new",
    "--remote-debugging-port=0",
    `--user-data-dir=${profile}`,
    "--no-first-run",
    "--no-default-browser-check",
    "--hide-scrollbars",
    "about:blank",
  ], { stdio: "ignore" });
  const portFile = join(profile, DEVTOOLS_PORT_FILE);
  const port = await waitFor(async () => {
    try {
      return (await readFile(portFile, "utf8")).split("\n")[0].trim() || null;
    } catch (error) {
      if (error.code === "ENOENT") {
        return null; // Chrome has not written the file yet; keep polling.
      }
      throw error;
    }
  }, CHROME_START_TIMEOUT_MS, "Chrome to open its DevTools port");
  return { chrome, profile, port };
}

class Cdp {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    socket.addEventListener("message", (message) => {
      const data = JSON.parse(message.data);
      const waiter = this.pending.get(data.id);
      if (!waiter) {
        return; // protocol events we did not subscribe to
      }
      this.pending.delete(data.id);
      if (data.error) {
        waiter.reject(new Error(`${waiter.method}: ${data.error.message}`));
      } else {
        waiter.resolve(data.result);
      }
    });
  }

  static async connect(url) {
    const socket = new WebSocket(url);
    await new Promise((resolve, reject) => {
      socket.addEventListener("open", resolve, { once: true });
      socket.addEventListener("error", () => reject(new Error(`cannot connect to ${url}`)),
        { once: true });
    });
    return new Cdp(socket);
  }

  send(method, params = {}) {
    const id = this.nextId++;
    this.socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject, method }));
  }

  async evaluate(expression) {
    const { result, exceptionDetails } = await this.send("Runtime.evaluate", {
      expression, returnByValue: true, awaitPromise: true,
    });
    if (exceptionDetails) {
      throw new Error(`page script failed: ${exceptionDetails.text} in ${expression}`);
    }
    return result.value;
  }

  close() {
    this.socket.close();
  }
}

async function openPage(port) {
  const response = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" });
  if (!response.ok) {
    throw new Error(`Chrome refused a new tab: HTTP ${response.status}`);
  }
  const target = await response.json();
  return Cdp.connect(target.webSocketDebuggerUrl);
}

function setSlider(id, value) {
  return `(() => { const el = document.getElementById(${JSON.stringify(id)});
    el.value = ${JSON.stringify(String(value))};
    el.dispatchEvent(new Event("input", { bubbles: true })); return el.value; })()`;
}

const READ_TICK = `Number(document.getElementById("tick").textContent.split("/")[0])`;
const READ_BUTTON = `document.getElementById("play").textContent`;

async function capture(page, shot, url, outDir) {
  await page.send("Emulation.setDeviceMetricsOverride", {
    width: shot.width, height: shot.height, deviceScaleFactor: shot.scale, mobile: shot.mobile,
  });
  await page.send("Page.navigate", { url });
  await waitFor(
    () => page.evaluate(`document.readyState === "complete" && !document.getElementById("play").disabled`),
    PAGE_READY_TIMEOUT_MS, "the dashboard to load its run metadata",
  );
  await page.evaluate(setSlider("speed", shot.speed));
  await page.evaluate(`document.getElementById("play").click()`);

  const midRun = Boolean(shot.until.tick);
  if (midRun) {
    // Stop one tick short: the viewport is resized next, then we wait for a fresh tick.
    await waitFor(async () => (await page.evaluate(READ_TICK)) >= shot.until.tick - 1,
      RUN_TIMEOUT_MS, `tick ${shot.until.tick - 1}`);
  } else {
    await waitFor(async () => (await page.evaluate(READ_BUTTON)) === FINISHED_TEXT,
      RUN_TIMEOUT_MS, "the run to finish");
  }
  if (shot.threshold !== undefined) {
    await page.evaluate(setSlider("threshold", shot.threshold));
  }

  // Grow the viewport to the whole page so the capture never triggers a late resize.
  const pageHeight = Math.ceil(await page.evaluate(`document.documentElement.scrollHeight`));
  await page.send("Emulation.setDeviceMetricsOverride", {
    width: shot.width, height: pageHeight, deviceScaleFactor: shot.scale, mobile: shot.mobile,
  });
  if (midRun) {
    const before = await page.evaluate(READ_TICK);
    await waitFor(async () => (await page.evaluate(READ_TICK)) > before,
      RUN_TIMEOUT_MS, "the next tick");
    await sleep(SETTLE_MS);
  } else {
    await sleep(FINISHED_SETTLE_MS);
  }

  const height = shot.stopAbove
    ? Math.floor(await page.evaluate(
      `document.querySelector(${JSON.stringify(shot.stopAbove)}).getBoundingClientRect().top`,
    )) - CLIP_GAP_PX
    : Math.ceil(await page.evaluate(`document.documentElement.scrollHeight`));
  const { data } = await page.send("Page.captureScreenshot", {
    format: "png",
    clip: { x: 0, y: 0, width: shot.width, height, scale: 1 },
  });
  const path = join(outDir, shot.file);
  await writeFile(path, Buffer.from(data, "base64"));
  const tick = await page.evaluate(READ_TICK);
  console.log(`wrote ${path} (${shot.width}x${height} css px @${shot.scale}x, tick ${tick})`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  await mkdir(args.out, { recursive: true });
  const { chrome, profile, port } = await launchChrome(args.chrome);
  let page;
  try {
    page = await openPage(port);
    await page.send("Page.enable");
    await page.send("Emulation.setEmulatedMedia", {
      features: [{ name: "prefers-color-scheme", value: "dark" }],
    });
    for (const shot of SHOTS) {
      await capture(page, shot, args.url, args.out);
    }
  } finally {
    page?.close();
    if (chrome.exitCode === null && chrome.signalCode === null) {
      const exited = new Promise((resolve) => chrome.once("exit", resolve));
      chrome.kill();
      await exited;
    }
    await rm(profile, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(`capture failed: ${error.message}`);
  process.exitCode = 1;
});
