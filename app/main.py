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
from .structure_dsm import structure_aware_dsm
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
_depth_anything: HeightEstimator | None = None
_midas: HeightEstimator | None = None
_model_status: ModelStatus = ModelStatus.LOADING
_model_error: str = ""
_da_error: str = ""


def _load_models() -> None:
    global _depth_anything, _midas, _model_status, _model_error, _da_error

    # Primary: Depth Anything V2 (local weights)
    try:
        from .estimators.depth_anything import DepthAnythingEstimator
        _depth_anything = DepthAnythingEstimator()
        logger.info("Depth-Anything-V2-Small loaded successfully.")
    except Exception as exc:
        _da_error = str(exc)
        logger.warning("Depth Anything V2 failed to load: %s", exc)

    # Secondary: MiDaS (torch.hub)
    try:
        from .estimators.midas import MidasEstimator
        _midas = MidasEstimator()
        logger.info("MiDaS_small loaded successfully.")
    except Exception as exc:
        logger.warning("MiDaS failed to load: %s", exc)

    if _depth_anything or _midas:
        _model_status = ModelStatus.READY
        primary = "Depth-Anything-V2" if _depth_anything else "MiDaS_small"
        logger.info("=" * 60)
        logger.info("DepthWizard ready — %s loaded, accepting requests.", primary)
        logger.info("=" * 60)
    else:
        _model_status = ModelStatus.FAILED
        _model_error = f"All models failed. DA: {_da_error}"
        logger.error("=" * 60)
        logger.error("All depth models failed — only synthetic available.")
        logger.error("=" * 60)


@app.on_event("startup")
async def startup() -> None:
    t = threading.Thread(target=_load_models, name="model-loader", daemon=True)
    t.start()
    logger.info("Server ready. Models loading in background — watch for the 'DepthWizard ready' line.")


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

    if name == "depth_anything":
        if _depth_anything:
            return _depth_anything, ""
        if _midas:
            return _midas, f"Depth Anything V2 unavailable ({_da_error}); using MiDaS instead."
        return _synthetic, f"All depth models failed ({_da_error}); using synthetic."

    if name == "midas":
        if _midas:
            return _midas, ""
        if _depth_anything:
            return _depth_anything, "MiDaS unavailable; using Depth Anything instead."
        return _synthetic, f"All depth models failed ({_model_error}); using synthetic."

    if _depth_anything:
        return _depth_anything, ""
    if _midas:
        return _midas, f"Depth Anything V2 unavailable ({_da_error}); using MiDaS."
    return _synthetic, f"All depth models failed; using synthetic."


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> JSONResponse:
    body = {
        "model_status": _model_status.value,
        "depth_anything": "ready" if _depth_anything else ("failed" if _da_error else "not_loaded"),
        "midas": "ready" if _midas else "not_loaded",
    }
    if _model_error:
        body["model_error"] = _model_error
    if _da_error:
        body["da_error"] = _da_error
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
_last_image_bytes: bytes = b""


MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


# ── /upload ───────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    estimator: Literal["depth_anything", "midas", "synthetic"] = Query(default="depth_anything"),
    detrend: bool = Query(default=True),
    structure: bool = Query(default=False),
) -> Response:
    global _last_image_bytes
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
        kwargs = {}
        if hasattr(est, 'estimate') and 'detrend' in est.estimate.__code__.co_varnames:
            kwargs['detrend'] = detrend
        height_array, meta = est.estimate(image_bytes, **kwargs)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Estimator failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Depth estimation failed: {e}")

    _last_image_bytes = image_bytes

    if structure:
        from PIL import Image as PILImage
        import io as _io
        img_pil = PILImage.open(_io.BytesIO(image_bytes)).convert("RGB")
        structured, class_map, stats = structure_aware_dsm(height_array, img_pil)
        height_array = structured
        _last_prediction["class_map"] = class_map
        _last_prediction["structure_stats"] = stats

    meta_dict = meta.to_dict()
    meta_dict["provenance"] = "relative"
    if structure:
        meta_dict["structure_stats"] = _last_prediction.get("structure_stats", {})
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


# ── /structure-map ──────────────────────────────────────────────────────────
@app.get("/structure-map")
async def structure_map() -> Response:
    """Return the last segmentation class map as a color-coded PNG."""
    import io as _io
    from PIL import Image as PILImage

    if "class_map" not in _last_prediction:
        raise HTTPException(400, "No structure map — upload with ?structure=true first.")

    cmap = _last_prediction["class_map"]
    h, w = cmap.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # BUILDING=0 red, ROAD=1 gray, VEGETATION=2 green, GROUND=3 brown
    colors = {
        0: (220, 50, 50, 180),
        1: (140, 140, 140, 180),
        2: (50, 180, 50, 180),
        3: (160, 120, 80, 180),
    }
    for cls, color in colors.items():
        mask = cmap == cls
        rgba[mask] = color

    img = PILImage.fromarray(rgba, "RGBA")
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
