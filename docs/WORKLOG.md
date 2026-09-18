# DepthWizard Work Log

## 2026-09-18 Session: Shadow-Height Rearchitecture

### Section 0 — Diagnosis

**What:** Ran DAv2 on a synthetic satellite tile (512×512) with realistic texture properties — flat-roof buildings, tree canopy patches, roads, ground. Measured mean disparity per region class.

**Finding:** Vegetation disparity (58.88) > building disparity (50.39). Within-class std (~30) is 3.5× the between-class difference (~8.5). DAv2 output is effectively noise w.r.t. building heights on nadir imagery. Confirms the structural failure hypothesis in the handoff brief.

**Decision:** No real satellite tile was available in the project. Used a synthetic tile with representative texture properties. Findings are directionally correct but should be re-validated on real imagery.

**Artifacts:** `docs/DIAGNOSIS.md`, `docs/diag_*.png`

---

### P0-1: SAM integration + instance segmentation

**What:** Integrated SAM ViT-B for automatic mask generation on uploaded tiles. Created `app/sam_segmentation.py`.

**Key decisions:**
- CPU-only (no CUDA on this machine). Initial config (32 points, 1 crop layer) took ~288s. Reduced to 16 points, 0 crop layers → ~48s.
- `pred_iou_thresh=0.88` to filter low-quality masks.
- SAM weights downloaded to `models/sam_vit_b_01ec64.pth` (374.8 MB, gitignored).

**Artifacts:** `app/sam_segmentation.py`, `models/sam_vit_b_01ec64.pth`

---

### P0-2: Shape + colour classification

**What:** Added feature computation and scoring-based classifier in `sam_segmentation.py`.

**Features:** rectangularity, solidity, aspect ratio, excess green index (ExG), texture variance.
**Classes:** building (rectangular, solid, not green), tree (green, textured, non-rectangular), road (elongated, not green).

**Decision:** No brightness or depth thresholds — purely geometric + spectral. This was explicit in the handoff brief.

---

### P0-3: Polygon regularisation

**What:** Douglas-Peucker simplification + edge snapping to 2 dominant orientations (from length-weighted angle histogram). 15° snap threshold.

**Artifacts:** `regularise_polygon()` in `app/sam_segmentation.py`

---

### P0-4: Footprint validation — SKIPPED

**Blocked:** No real satellite tile with hand-labelled ground truth. See `docs/BLOCKERS.md`.

---

### P0-5: Shadow detection

**What:** Created `app/shadow_detection.py`. LAB lightness < 0.35 + HSV saturation < 0.25. Excludes building interiors. Morphological cleanup (open 3×3 + close 5×5). Adjacency requirement: shadow must overlap dilated building mask.

---

### P0-6: Shadow ↔ building matching

**What:** Dilated mask overlap heuristic in `match_shadows_to_buildings()`. Each shadow blob assigned to the nearest building whose dilated mask it overlaps.

---

### P0-7: Sun geometry (3-source hierarchy)

**What:** Created `app/sun_geometry.py`.
1. EXIF GPS+DateTime → pysolar computation
2. Lat/lon/datetime → pysolar (with azimuth convention fix: pysolar 0°=S → geographic 0°=N)
3. Estimate from shadow directions — circular mean of building→shadow vectors

**Fallback chain:** metadata → computed → estimated, with confidence decreasing.

---

### P0-8: Shadow-length measurement

**What:** Projects building centroid and shadow centroid onto solar azimuth direction. Bins by perpendicular coordinate. Median of per-scanline shadow lengths for robustness.

---

### P0-9: Per-building height with confidence

**What:** `h = L_shadow * tan(θ_sun)`. Full uncertainty propagation:
- Shadow length variance (std across scanlines)
- Sun angle uncertainty: ±5° for estimated, ±2° for high-confidence
- Sanity clamp: [1m, 300m]

Missing heights interpolated from neighbourhood median (marked as `interpolated` confidence).

**Artifacts:** `app/height_estimation.py`

---

### P0-10: Server endpoints (SAM pipeline)

**What:** Added to `app/main.py`:
- `POST /segment` — SAM + classification
- `GET /segment-overlay` — RGBA PNG of classification
- `GET /instances` — JSON instance data
- `POST /estimate-heights?gsd=&lat=&lon=&sun_elevation=&sun_azimuth=` — full pipeline
- `GET /shadow-overlay` — shadow detection PNG

SAM loads in background thread to not block server startup.

---

### P0-11: LOD1 Three.js rendering

**What:** Added to `static/index.html`:
- `buildLOD1Scene()`: ExtrudeGeometry buildings from polygons (or bbox fallback), CylinderGeometry + SphereGeometry trees, road polygons, ground plane with source image texture.
- "Segment & Build 3D" button triggers full pipeline.
- Sun elevation + GSD sliders.
- Mousemove raycasting for metric height tooltip in LOD1 mode.

---

### P1-12: Flood inundation model

**What:** Created `app/flood_model.py`. Per-building: `submerged = clamp(W - ground_elev, 0, building_height)`. Status: dry / partially_inundated / fully_submerged. Aggregate: counts, areas, freeboard stats, approximate water volume.

---

### P1-13: Water surface + interactive slider

**What:** Added to `static/index.html`:
- Water level slider (0–30m) in FLOOD ANALYSIS section
- Semi-transparent blue PlaneGeometry at water level × vertical exaggeration
- Debounced fetch to `/flood` endpoint
- Buildings colored by status: green (dry), yellow (partial), red (submerged)
- Stat display: affected count, mean freeboard, inundated area, volume

---

### P1-14: CSV export

**What:** Export button in flood section. Downloads per-building CSV with: building height, water depth, submerged depth, freeboard, status, area.

---

### P1-15: Landing page with orbital intro

**What:** Created `static/landing.html`:
- Dark starfield (2000 particles) with pointer-move parallax
- Earth sphere with NASA Blue Marble texture (2K, 2.5MB) + fresnel rim glow shader
- Low-poly satellite on tilted orbital path
- "Launch Workspace" button → camera zoom + fade-to-black → redirect to workspace
- `?fast=1` query param skips landing entirely
- Root URL `/` redirects to `/landing.html`
- Building-rise animation on workspace load (staggered cubic ease per building)
- SIH 26175 badge, method statement

**Decision:** Kept landing page as separate HTML file rather than embedding in index.html — cleaner separation, no loading overhead on workspace.
