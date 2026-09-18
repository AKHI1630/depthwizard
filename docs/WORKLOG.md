# DepthWizard Work Log

## 2026-09-18 Session: Shadow-Height Rearchitecture

### Section 0 — Diagnosis

**What:** Ran DAv2 on a synthetic satellite tile (512×512) with realistic texture properties — flat-roof buildings, tree canopy patches, roads, ground. Measured mean disparity per region class.

**Finding:** Vegetation disparity (58.88) > building disparity (50.39). Within-class std (~30) is 3.5× the between-class difference (~8.5). DAv2 output is effectively noise w.r.t. building heights on nadir imagery. Confirms the structural failure hypothesis in the handoff brief.

**Decision:** No real satellite tile was available in the project. Used a synthetic tile with representative texture properties. Findings are directionally correct but should be re-validated on real imagery.

**Artifacts:** `docs/DIAGNOSIS.md`, `docs/diag_*.png`
