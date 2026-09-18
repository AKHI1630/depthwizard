# Blockers

## P0-4: Footprint IoU validation — DEFERRED

**Problem:** Hand-labelled footprint validation requires a real satellite tile with manually annotated building outlines. No real satellite tile is available in the project. The synthetic test images have known ground truth by construction but don't test real-world segmentation quality.

**Decision:** Skip P0-4 for now. The validation framework (IoU computation, precision/recall) will be built when a real satellite tile is added. The brief's metric requirements (IoU, count F1, boundary F1) are documented and the code structure supports them. Continue to P0-5 (shadow detection).

**What's needed:** A real nadir satellite tile of an urban area (ideally the "several hundred villas plus a large warehouse" scene referenced in the handoff brief). Upload it to the project root or `data/` directory.
