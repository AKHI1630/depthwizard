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

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .calibration import (
    CalibrationResult,
    calibrate_with_reference,
    detect_geotiff,
    read_geotiff_elevation,
)
from .estimators.base import HeightEstimator
from .estimators.synthetic import SyntheticEstimator
from .validation import validate as run_validation

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


def _pack_response(height_array: np.ndarray, meta_dict: dict) -> Response:
    meta_bytes = json.dumps(meta_dict).encode("utf-8")
    prefix = struct.pack("<I", len(meta_bytes))
    raw = height_array.astype("float32").tobytes()
    return Response(
        content=prefix + meta_bytes + raw,
        media_type="application/octet-stream",
        headers={"X-Meta-Length": str(len(meta_bytes))},
    )


_last_prediction: dict = {}


MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


# ── /upload ───────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    estimator: Literal["midas", "synthetic"] = Query(default="midas"),
) -> Response:
    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file uploaded.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File too large ({len(image_bytes) / 1024 / 1024:.1f} MB). Max {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
        )

    geo = detect_geotiff(image_bytes)
    if geo is not None:
        logger.info("GeoTIFF detected: %s CRS=%s", geo["dtype"], geo["crs"])
        result = read_geotiff_elevation(image_bytes)
        if result is not None:
            return _pack_response(result.heights, result.to_meta_dict())
        logger.warning("GeoTIFF elevation read failed — falling back to estimator")

    est, warning = _pick_estimator(estimator)

    try:
        height_array, meta = est.estimate(image_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Estimator failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Depth estimation failed: {e}")

    meta_dict = meta.to_dict()
    meta_dict["provenance"] = "relative"
    if warning:
        meta_dict["warning"] = warning
        logger.warning("Fallback active: %s", warning)

    _last_prediction["heights"] = height_array
    _last_prediction["meta"] = meta_dict

    return _pack_response(height_array, meta_dict)


# ── /calibrate ───────────────────────────────────────────────────────────────
@app.post("/calibrate")
async def calibrate(reference: UploadFile = File(...)) -> Response:
    if "heights" not in _last_prediction:
        raise HTTPException(400, "No depth prediction to calibrate — upload an image first.")

    ref_bytes = await reference.read()
    result = calibrate_with_reference(_last_prediction["heights"], ref_bytes)
    if result is None:
        raise HTTPException(400, "Reference is not a valid elevation GeoTIFF, or calibration failed.")

    return _pack_response(result.heights, result.to_meta_dict())


# ── /validate ────────────────────────────────────────────────────────────────
@app.post("/validate")
async def validate_endpoint(reference: UploadFile = File(...)) -> JSONResponse:
    if "heights" not in _last_prediction:
        raise HTTPException(400, "No depth prediction to validate — upload an image first.")

    ref_bytes = await reference.read()
    ref_result = read_geotiff_elevation(ref_bytes, target_size=_last_prediction["heights"].shape[0])
    if ref_result is None:
        raise HTTPException(400, "Reference is not a valid elevation GeoTIFF.")

    vr = run_validation(_last_prediction["heights"], ref_result.heights)

    error_map_list = vr.error_map.flatten().tolist()

    result = vr.to_dict()
    result["error_map"] = error_map_list

    return JSONResponse(content=result)


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
