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
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Environment constraints
- This machine has no `OES_texture_float` support — WebGL clamps float textures to 0, producing a flat terrain.
- All height data **must** travel as RG-packed 16-bit in an `RGBA / UnsignedByteType` texture with `NearestFilter` on both min and mag. Never reintroduce `FloatType` or `RedFormat` textures.
- Before presenting on any demo machine, verify float texture support with `gl.getExtension('OES_texture_float')` and confirm it is non-null.

## Conventions
- Python venv at `.venv/`
- Static frontend at `static/`
- Do not auto-commit without asking.
