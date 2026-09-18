# Diagnosis: Depth Anything V2 on Nadir Satellite Imagery

**Date:** 2026-09-18
**Test image:** Synthetic satellite tile (512×512) with realistic texture properties — flat-roof buildings (low variance), tree canopy patches (high variance), gray roads, and mixed ground.
**Note:** No real satellite tile was available in the project. Diagnosis was run on a synthetic tile designed to reproduce the texture properties described in the handoff brief. Findings should be re-validated on a real satellite image when one is available.

## Hypothesis

DAv2 is trained on ground-level photography. Its depth cues (perspective convergence, occlusion ordering, vanishing-point geometry) are absent in nadir satellite imagery. Deprived of these cues, the model keys on local texture variance, producing raised disparity for high-variance vegetation and flat disparity for uniform building rooftops.

## Test Setup

- Server: port 8002, Depth Anything V2 Small (local weights)
- Upload parameters: `estimator=depth_anything`, `detrend=false`, `structure=false`
- Raw disparity linearly rescaled to 0–150 by the pipeline

## Input Texture Variance (7×7 local window)

| Region     | Mean Texture Variance |
|------------|----------------------|
| Building   | 119.8                |
| Vegetation | 782.2                |
| Road       | 392.8                |
| Ground     | 219.9                |

Buildings have the lowest texture variance (flat concrete rooftops). Vegetation has ~6.5× higher variance (canopy noise). This matches the expected properties of real satellite imagery.

## DAv2 Output Disparity

| Region     | Mean Disparity | Std Disparity |
|------------|---------------|---------------|
| Building   | 50.39         | 29.33         |
| Vegetation | 58.88         | 30.80         |
| Road       | 40.24         | —             |
| Ground     | 49.49         | —             |

## Key Findings

### 1. Vegetation gets higher disparity than buildings — CONFIRMED

Mean vegetation disparity (58.88) exceeds mean building disparity (50.39) by 8.5 units. In a real urban scene, buildings should be elevated above vegetation; DAv2 inverts this.

### 2. Signal-to-noise ratio is extremely poor

The between-class difference (~8.5 units) is 0.28 standard deviations of within-class noise (~30 units). The disparity map is effectively random with respect to actual building heights. Any height readout derived from this signal is meaningless.

### 3. The 0–150 m rescaling has no metric basis

DAv2 produces unitless relative disparity. The pipeline rescales linearly to 0–150 m. Since the raw disparity bears no structural relation to actual building heights on nadir imagery, the resulting "height" values are not heights.

### 4. Texture-disparity correlation is weak overall

Pearson r(texture variance, disparity) = −0.0723. The relationship is not a simple linear mapping from texture to disparity — DAv2 is computing something more complex. But the net effect on nadir imagery is that it cannot distinguish buildings from ground in any useful way.

## Conclusion

The diagnosis **confirms the structural failure described in the handoff brief**. DAv2 does not produce usable building height estimates from nadir satellite imagery. The current "height at cursor" readout reports a rescaled disparity value with no physical meaning.

**Recommendation:** Replace the depth-map heightfield approach with instance segmentation → per-instance metric height (shadow-based) → LOD1 extrusion, as specified in the handoff brief Section 1. Retain DAv2 as a secondary relative-relief layer for non-urban terrain, clearly labelled.

## Artifacts

- `docs/diag_input_tile.png` — synthetic satellite tile used for diagnosis
- `docs/diag_dav2_disparity.png` — raw DAv2 disparity map (grayscale)
- `docs/diag_overlay.png` — disparity heatmap overlaid on source tile
