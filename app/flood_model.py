"""
Flood inundation model.

Input: water level W (metres above local ground datum).
Per building:
  - submerged = clamp(W - ground_elev, 0, building_height)
  - freeboard = building_height - submerged
  - status = dry | partially_inundated | fully_submerged

Aggregate stats: count and area per status, inundated area,
approximate water volume.
"""
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FloodResult:
    water_level_m: float
    buildings: list[dict]
    aggregate: dict


def compute_flood(
    instances: list,
    water_level_m: float,
    ground_elev_m: float = 0.0,
    gsd_m: float = 0.5,
) -> FloodResult:
    """
    Compute flood impact for each building at a given water level.

    Args:
        instances: list of instance dicts (from instances_to_json or InstanceMask objects)
        water_level_m: water level in metres above ground datum
        ground_elev_m: ground elevation datum
        gsd_m: ground sampling distance for area computation

    Returns:
        FloodResult with per-building status and aggregate stats
    """
    results = []
    total_buildings = 0
    dry_count = 0
    partial_count = 0
    submerged_count = 0
    total_footprint_area_m2 = 0
    inundated_area_m2 = 0
    freeboard_values = []

    for inst in instances:
        label = inst.label if hasattr(inst, 'label') else inst.get('label', '')
        if label != 'building':
            continue

        total_buildings += 1
        height_m = inst.height_m if hasattr(inst, 'height_m') else inst.get('height_m')
        area_px = inst.area if hasattr(inst, 'area') else inst.get('area_px', 0)
        area_m2 = area_px * (gsd_m ** 2)
        total_footprint_area_m2 += area_m2

        if height_m is None:
            height_m = 10.0  # default for buildings without measured height

        water_depth = max(0.0, water_level_m - ground_elev_m)
        submerged_depth = min(water_depth, height_m)
        freeboard = max(0.0, height_m - submerged_depth)

        if water_depth <= 0:
            status = "dry"
            dry_count += 1
        elif submerged_depth >= height_m:
            status = "fully_submerged"
            submerged_count += 1
            inundated_area_m2 += area_m2
        else:
            status = "partially_inundated"
            partial_count += 1
            inundated_area_m2 += area_m2

        if status != "dry":
            freeboard_values.append(freeboard)

        bbox = inst.bbox if hasattr(inst, 'bbox') else inst.get('bbox', [0, 0, 0, 0])

        results.append({
            "building_height_m": round(height_m, 2),
            "water_depth_m": round(water_depth, 2),
            "submerged_depth_m": round(submerged_depth, 2),
            "freeboard_m": round(freeboard, 2),
            "status": status,
            "area_m2": round(area_m2, 1),
            "bbox": list(bbox) if hasattr(bbox, '__iter__') else bbox,
        })

    # Approximate water volume (integrate water_depth over inundated footprint)
    water_depth = max(0.0, water_level_m - ground_elev_m)
    approx_volume_m3 = water_depth * inundated_area_m2

    aggregate = {
        "total_buildings": total_buildings,
        "dry": dry_count,
        "partially_inundated": partial_count,
        "fully_submerged": submerged_count,
        "affected_pct": round(100 * (partial_count + submerged_count) / max(total_buildings, 1), 1),
        "total_footprint_area_m2": round(total_footprint_area_m2, 1),
        "inundated_area_m2": round(inundated_area_m2, 1),
        "mean_freeboard_m": round(float(np.mean(freeboard_values)), 2) if freeboard_values else None,
        "min_freeboard_m": round(float(np.min(freeboard_values)), 2) if freeboard_values else None,
        "approx_water_volume_m3": round(approx_volume_m3, 1),
    }

    logger.info(
        "Flood at %.1f m: %d/%d affected (%.1f%%), mean freeboard %.1f m",
        water_level_m, partial_count + submerged_count, total_buildings,
        aggregate["affected_pct"],
        aggregate["mean_freeboard_m"] or 0,
    )

    return FloodResult(
        water_level_m=water_level_m,
        buildings=results,
        aggregate=aggregate,
    )
