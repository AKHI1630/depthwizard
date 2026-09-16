# DepthWizard — SIH 26175

## Project
Single-view height estimation and 3D flythrough.

## Working Directory
Fixed to `C:\Users\rtadi001\ClaudeAI\depthwizard`. Do not create or reference files outside this folder.

## Frontend
- Single `index.html` — no bundler, no build step.
- Three.js and OrbitControls load from cdnjs.cloudflare.com CDN.

## Architecture
- Height generation is behind a `HeightEstimator` interface (`estimate(image) -> (array, metadata)`).
- Swap implementations in `estimators/` without touching the frontend.

## Commands
```bash
# Start backend (from repo root, inside venv)
.venv\Scripts\uvicorn.exe app.main:app --reload --host 0.0.0.0 --port 8000
```
Server starts immediately; model loads in background. Watch the log for:
`DepthWizard ready — Depth-Anything-V2-Small loaded, accepting requests.`
Poll `GET /health` to check model status programmatically.

## ⚠ Environment constraints — READ BEFORE TOUCHING TEXTURES
- **This machine has no `OES_texture_float` WebGL support.** Float textures silently clamp to 0 → completely flat terrain. This cost a full debug session.
- All height data **must** travel as RG-packed 16-bit in `THREE.RGBAFormat + THREE.UnsignedByteType` with `NearestFilter` on both min and mag filters. The unpack formula in the vertex shader is: `h = (t.r * 255.0 * 256.0 + t.g * 255.0) / 65535.0`.
- **Never reintroduce `FloatType`, `RedFormat`, or `HalfFloatType` textures.**
- Before any demo on a different machine, verify: `gl.getExtension('OES_texture_float') !== null`.

## What's working (as of 2026-09-17)
- RG-packed 16-bit heightmap rendering in Three.js — hills and buildings visible, hillshade lighting, height colour ramp, vertical-exaggeration slider.
- Depth-Anything-V2-Small-hf estimator (`app/estimators/depth_anything.py`) loading on CPU with background thread startup — server accepts requests immediately.
- `GET /health` endpoint returning `{model_status: loading|ready|failed}`.
- Synthetic fallback (`app/estimators/synthetic.py`) — always available, used when model is loading or failed.
- `POST /upload?estimator=depth|synthetic` — dropdown in UI to switch between them.
- Fallback warning surfaced in amber in the UI when depth model is unavailable.
- Raw depth min/max/mean logged before normalisation so model output range is visible.

## What's untested / known risks
- **Depth inversion may be wrong for some scenes** — buildings could render as craters if the model's near/far convention doesn't match what we assumed. Needs a real photo of a known scene to verify. The inversion is in `depth_anything.py`: `depth_inv = raw_max - depth_raw`.
- Depth output is uncalibrated and normalised to 0–150 m (`units="relative"`). This is a placeholder — not metric.
- Only tested on the dev machine browser (no `OES_texture_float`). Behaviour on a machine with float texture support is untested.

## What's next
- **Calibration module** — map relative depth to metric scale using known reference heights or SRTM ground truth.
- **SRTM terrain decomposition** — separate model-predicted relative heights from an absolute terrain base layer.
- **GAMUS accuracy numbers** — run against benchmark dataset, report SI, AbsRel, δ1 metrics.
- **Camera modes and flythrough** — animated path, first-person fly mode, cinematic orbit.
- **Validate panel** — side-by-side 2D image vs 3D render with measurement overlays.
- **PPT slide deck** — project summary for SIH 26175 presentation.
- **Demo video** — screen capture of flythrough with narration.

## Depth-Anything-V2-Small model notes
- Model: `depth-anything/Depth-Anything-V2-Small-hf` (~99 MB, cached in `~/.cache/huggingface/`).
- Internally resizes input to **518×518** before inference. The 1024×1024 output is bicubic interpolation — **do not overclaim output resolution**. Real spatial detail is limited to ~518 px.
- Normalised to 0–150 m with `units="relative"` — uncalibrated, not metric.

## Conventions
- Python venv at `.venv/`
- Static frontend at `static/`
- Do not auto-commit without asking.
