import struct
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .estimators.synthetic import SyntheticEstimator

STATIC_DIR = Path(__file__).parent.parent / "static"

app = FastAPI(title="DepthWizard")

_estimator = SyntheticEstimator()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    image_bytes = await file.read()
    height_array, meta = _estimator.estimate(image_bytes)

    # Return multipart-style: 4-byte little-endian int (metadata JSON length),
    # then the JSON bytes, then the raw float32 binary blob.
    meta_json = meta.to_dict()
    import json
    meta_bytes = json.dumps(meta_json).encode("utf-8")
    prefix = struct.pack("<I", len(meta_bytes))
    raw = height_array.astype("float32").tobytes()

    return Response(
        content=prefix + meta_bytes + raw,
        media_type="application/octet-stream",
        headers={"X-Meta-Length": str(len(meta_bytes))},
    )


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
