import json
import logging
import os
import struct
import threading
from enum import Enum

# Force HuggingFace to use the standard HTTP download path instead of the Xet
# CDN (cas-server.xethub.hf.co), which 503s on this corporate network.
# Must be set before any huggingface_hub / transformers import.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .estimators.base import HeightEstimator
from .estimators.synthetic import SyntheticEstimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"

app = FastAPI(title="DepthWizard")


# ── Model state ───────────────────────────────────────────────────────────────
class ModelStatus(str, Enum):
    LOADING = "loading"
    READY   = "ready"
    FAILED  = "failed"


_synthetic = SyntheticEstimator()
_midas: HeightEstimator | None = None
_model_status: ModelStatus = ModelStatus.LOADING
_model_error: str = ""


def _load_midas() -> None:
    global _midas, _model_status, _model_error
    try:
        from .estimators.midas import MidasEstimator
        _midas = MidasEstimator()
        _model_status = ModelStatus.READY
        logger.info("=" * 60)
        logger.info("DepthWizard ready — MiDaS_small loaded, accepting requests.")
        logger.info("=" * 60)
    except Exception as exc:
        _model_status = ModelStatus.FAILED
        _model_error = str(exc)
        logger.error("=" * 60)
        logger.error("MiDaS failed to load — /upload?estimator=midas will fall back to synthetic.")
        logger.error("Reason: %s", exc)
        logger.error("=" * 60)


@app.on_event("startup")
async def startup() -> None:
    t = threading.Thread(target=_load_midas, name="model-loader", daemon=True)
    t.start()
    logger.info("Server ready. MiDaS loading in background — watch for the 'DepthWizard ready' line.")


# ── Estimator selection ───────────────────────────────────────────────────────
def _pick_estimator(name: str) -> tuple[HeightEstimator, str]:
    """Return (estimator, warning).  Raises HTTPException 503 if model not ready yet."""
    if name == "synthetic":
        return _synthetic, ""

    if _model_status is ModelStatus.LOADING:
        raise HTTPException(
            status_code=503,
            detail="Model still loading — please try again in a moment.",
        )
    if _model_status is ModelStatus.FAILED:
        return _synthetic, f"MiDaS failed to load ({_model_error}); using synthetic instead."
    return _midas, ""  # type: ignore[return-value]


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> JSONResponse:
    body = {"model_status": _model_status.value}
    if _model_error:
        body["model_error"] = _model_error
    status_code = 200 if _model_status is ModelStatus.READY else 503
    return JSONResponse(content=body, status_code=status_code)


# ── /upload ───────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    estimator: Literal["midas", "synthetic"] = Query(default="midas"),
) -> Response:
    est, warning = _pick_estimator(estimator)
    image_bytes = await file.read()
    height_array, meta = est.estimate(image_bytes)

    meta_dict = meta.to_dict()
    if warning:
        meta_dict["warning"] = warning
        logger.warning("Fallback active: %s", warning)

    meta_bytes = json.dumps(meta_dict).encode("utf-8")
    prefix = struct.pack("<I", len(meta_bytes))
    raw = height_array.astype("float32").tobytes()

    return Response(
        content=prefix + meta_bytes + raw,
        media_type="application/octet-stream",
        headers={"X-Meta-Length": str(len(meta_bytes))},
    )


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
