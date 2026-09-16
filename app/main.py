import json
import logging
import struct
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .estimators.base import HeightEstimator
from .estimators.synthetic import SyntheticEstimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"

app = FastAPI(title="DepthWizard")

# ── Estimator registry ────────────────────────────────────────────────────────
_synthetic = SyntheticEstimator()
_depth_anything: HeightEstimator | None = None
_depth_anything_failed = False
_fallback_reason: str = ""

def _load_depth_anything() -> None:
    global _depth_anything, _depth_anything_failed, _fallback_reason
    try:
        from .estimators.depth_anything import DepthAnythingEstimator
        _depth_anything = DepthAnythingEstimator()
    except Exception as exc:
        _depth_anything_failed = True
        _fallback_reason = str(exc)
        logger.error(
            "=" * 60 + "\n"
            "DEPTH-ANYTHING FAILED TO LOAD — falling back to synthetic.\n"
            "Reason: %s\n" + "=" * 60,
            exc,
        )


@app.on_event("startup")
async def startup():
    _load_depth_anything()


def _pick_estimator(name: str) -> tuple[HeightEstimator, bool, str]:
    """Return (estimator, is_fallback, warning_message)."""
    if name == "synthetic":
        return _synthetic, False, ""
    # name == "depth" (default)
    if _depth_anything is not None:
        return _depth_anything, False, ""
    return _synthetic, True, f"depth-anything unavailable ({_fallback_reason}); using synthetic"


# ── Upload endpoint ───────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    estimator: Literal["depth", "synthetic"] = Query(default="depth"),
):
    image_bytes = await file.read()
    est, is_fallback, warning = _pick_estimator(estimator)
    height_array, meta = est.estimate(image_bytes)

    meta_dict = meta.to_dict()
    if is_fallback:
        meta_dict["warning"] = warning
        logger.warning("Serving fallback: %s", warning)

    meta_bytes = json.dumps(meta_dict).encode("utf-8")
    prefix = struct.pack("<I", len(meta_bytes))
    raw = height_array.astype("float32").tobytes()

    return Response(
        content=prefix + meta_bytes + raw,
        media_type="application/octet-stream",
        headers={"X-Meta-Length": str(len(meta_bytes))},
    )


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
