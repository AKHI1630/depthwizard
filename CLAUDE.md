# DepthWizard — SIH 26175

## Project
Single-view height estimation and 3D flythrough.

## Working Directory
Fixed to `C:\Users\rtadi001\ClaudeAI\depthwizard`. Do not create or reference files outside this folder.

## Frontend
- Single `index.html` — no bundler, no build step.
- Three.js r134 and OrbitControls load from cdnjs.cloudflare.com / cdn.jsdelivr.net CDN.

## Architecture
- Height generation is behind a `HeightEstimator` interface (`estimate(image) -> (array, metadata)`).
- Swap implementations in `estimators/` without touching the frontend.
- Calibration: `app/calibration.py` — GeoTIFF detection, SRTM RANSAC calibration, DSM export.
- Validation: `app/validation.py` — MAE, RMSE, R², error map, scatter plot.
- Tiling: `app/tiling.py` — overlapping tile merge for large images.

## Commands
```bash
# Start backend (from repo root, inside venv) — NO --reload flag
.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000
```
Server starts immediately; model loads in background. Watch the log for:
`DepthWizard ready — MiDaS_small loaded, accepting requests.`
Poll `GET /health` to check model status programmatically.

## API Endpoints
- `GET /health` — model status
- `POST /upload?estimator=midas|synthetic` — upload image, returns binary heightmap
- `POST /calibrate` — upload reference GeoTIFF to RANSAC-calibrate last prediction
- `POST /validate` — upload reference GeoTIFF to compute error stats vs last prediction

## ⚠ Environment constraints — READ BEFORE TOUCHING TEXTURES
- **This machine has no `OES_texture_float` WebGL support.** Float textures silently clamp to 0 → completely flat terrain. This cost a full debug session.
- All height data **must** travel as RG-packed 16-bit in `THREE.RGBAFormat + THREE.UnsignedByteType` with `NearestFilter` on both min and mag filters. The unpack formula in the vertex shader is: `h = (t.r * 255.0 * 256.0 + t.g * 255.0) / 65535.0`.
- **Never reintroduce `FloatType`, `RedFormat`, or `HalfFloatType` textures.**

## ⚠ Network constraints
- **HuggingFace Xet CDN (us.aws.cdn.hf.co) returns 503** on this corporate network. `HF_HUB_DISABLE_XET=1` is set in main.py but may not help.
- **MiDaS via torch.hub works** — GitHub releases CDN is reachable. Trust bypass: `torch.hub._check_repo_is_trusted = lambda *a, **k: None`.
- Do NOT use `--reload` with uvicorn — it watches `.venv/` and restarts during pip installs.

## What's working (as of 2026-09-17)
- **MiDaS_small** depth estimator via torch.hub (14.7s load, 0.37s inference).
- RG-packed 16-bit heightmap rendering — hills and buildings visible.
- **Viewer features**: viridis/cividis/terrain palettes, slope overlay, hillshade-only mode.
- **Camera modes**: orbit, first-person (WASD+mouse), aerial (top-down), waypoint flythrough.
- **Click-to-measure**: real-time height readout at cursor position.
- **Cross-section profile**: click two points, see 2D elevation profile.
- **Calibration pipeline**: GeoTIFF auto-detection, SRTM RANSAC calibration, provenance badges.
- **Validation panel**: MAE/RMSE/R²/bias stats, signed error map, pred-vs-ref scatter plot.
- **Robustness**: 50MB upload limit, image size validation (16px–8192px), proper error messages, tiling module.
- Synthetic fallback always available. Estimator dropdown in UI.

## Estimator notes
- MiDaS_small: 81.8 MB checkpoint from GitHub releases. Resizes internally to 256×256. Output interpolated to 1024×1024. Uncalibrated 0–150m range.
- Depth-Anything-V2-Small: NOT working (HF CDN blocked). Code exists but not wired in.
- Synthetic: deterministic Gaussian hills + box buildings, 1024×1024.

## Conventions
- Python venv at `.venv/`
- Static frontend at `static/`
- Do not auto-commit without asking.
- Never embed GitHub tokens in URLs or command lines.
