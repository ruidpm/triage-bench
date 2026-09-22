"""FastAPI app: static dashboard plus a Server-Sent Events stream of tick events."""

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from triage_bench.application.runner import TickEvent

EventSource = Callable[[], Iterator[TickEvent]]

STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"
DEFAULT_SPEED_TPS = 2.0
MIN_SPEED_TPS = 0.1
MAX_SPEED_TPS = 20.0
MODE_REPLAY = "replay"
MODE_LIVE = "live"
SSE_MEDIA_TYPE = "text/event-stream"
LIVE_ALREADY_STARTED = (
    "this live run has already started (one live run per process); "
    "restart the command for another run"
)
_END = object()


@dataclass(frozen=True)
class RunMeta:
    mode: str
    run_name: str
    contestants: list[str]


def _sse(data: dict[str, object], event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data)}\n\n"


class _SingleUse:
    """Thread-safe latch: claim() succeeds exactly once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._claimed = False

    def claim(self) -> bool:
        with self._lock:
            if self._claimed:
                return False
            self._claimed = True
            return True


def create_app(source: EventSource, meta: RunMeta) -> FastAPI:
    app = FastAPI(title="Triage Bench")
    # Each live stream calls the paid APIs and appends to the same run file, so a reload or
    # second tab must not start another run.
    live_run = _SingleUse()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(INDEX_FILE)

    @app.get("/api/meta")
    async def get_meta() -> dict[str, object]:
        # The dashboard streams replays at the maximum speed and paces them client-side.
        return {**asdict(meta), "max_speed_tps": MAX_SPEED_TPS}

    @app.get("/api/stream")
    async def stream(
        speed: float = Query(DEFAULT_SPEED_TPS, ge=MIN_SPEED_TPS, le=MAX_SPEED_TPS),
    ) -> StreamingResponse:
        if meta.mode == MODE_LIVE and not live_run.claim():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=LIVE_ALREADY_STARTED)
        pause_s = 1.0 / speed if meta.mode == MODE_REPLAY else 0.0

        async def events() -> AsyncIterator[str]:
            iterator = source()
            while True:
                item = await asyncio.to_thread(next, iterator, _END)
                if item is _END:
                    break
                assert isinstance(item, TickEvent)
                yield _sse(item.to_dict())
                if pause_s:
                    await asyncio.sleep(pause_s)
            yield _sse({}, event="done")

        return StreamingResponse(events(), media_type=SSE_MEDIA_TYPE)

    return app
