"""
Pipeline progress tracking with SSE streaming.

Each pipeline run gets a unique request_id. Stages report progress
which is streamed to the frontend via Server-Sent Events.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

STAGES = [
    ("upload", "Uploading image", 0.02),
    ("sam", "SAM segmentation", 0.50),
    ("shadows", "Shadow detection", 0.10),
    ("sun", "Sun geometry", 0.03),
    ("dav2", "DAv2 depth", 0.15),
    ("heights", "Height estimation", 0.15),
    ("finalize", "Finalizing results", 0.05),
]

STAGE_WEIGHTS = {s[0]: s[2] for s in STAGES}
STAGE_NAMES = {s[0]: s[1] for s in STAGES}
STAGE_ORDER = [s[0] for s in STAGES]


@dataclass
class PipelineProgress:
    request_id: str
    start_time: float = field(default_factory=time.perf_counter)
    current_stage: str = "upload"
    stage_start: float = field(default_factory=time.perf_counter)
    completed_stages: dict = field(default_factory=dict)
    done: bool = False
    error: Optional[str] = None
    _waiters: list = field(default_factory=list, repr=False)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    @property
    def overall_pct(self) -> float:
        pct = 0.0
        for stage_key in STAGE_ORDER:
            if stage_key in self.completed_stages:
                pct += STAGE_WEIGHTS.get(stage_key, 0)
        return min(pct * 100, 100.0)

    @property
    def eta_s(self) -> Optional[float]:
        pct = self.overall_pct
        if pct < 1:
            return None
        elapsed = self.elapsed
        return max(0, elapsed / (pct / 100.0) - elapsed)

    def to_event(self) -> dict:
        stage_name = STAGE_NAMES.get(self.current_stage, self.current_stage)
        return {
            "request_id": self.request_id,
            "stage": self.current_stage,
            "stage_name": stage_name,
            "pct": round(self.overall_pct, 1),
            "elapsed_s": round(self.elapsed, 1),
            "eta_s": round(self.eta_s, 1) if self.eta_s is not None else None,
            "done": self.done,
            "error": self.error,
            "completed": {k: round(v, 2) for k, v in self.completed_stages.items()},
        }

    def start_stage(self, stage: str):
        self.current_stage = stage
        self.stage_start = time.perf_counter()
        self._notify()

    def finish_stage(self, stage: str):
        duration = time.perf_counter() - self.stage_start
        self.completed_stages[stage] = duration
        self._notify()

    def finish(self):
        self.done = True
        self._notify()

    def fail(self, error: str):
        self.error = error
        self.done = True
        self._notify()

    def _notify(self):
        for loop, fut in self._waiters:
            if not fut.done():
                loop.call_soon_threadsafe(fut.set_result, True)
        self._waiters = []

    async def wait_for_update(self, timeout: float = 30.0):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._waiters.append((loop, fut))
        try:
            await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            pass


_progress_store: dict[str, PipelineProgress] = {}


def create_progress(request_id: str) -> PipelineProgress:
    prog = PipelineProgress(request_id=request_id)
    _progress_store[request_id] = prog
    if len(_progress_store) > 20:
        oldest = sorted(_progress_store.keys(), key=lambda k: _progress_store[k].start_time)
        for k in oldest[:10]:
            del _progress_store[k]
    return prog


def get_progress(request_id: str) -> Optional[PipelineProgress]:
    return _progress_store.get(request_id)
