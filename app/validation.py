"""
Validation: compare predicted heights against a reference DSM.
Produces error statistics, error map, and data for a scatter plot.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    mae: float
    rmse: float
    median_ae: float
    r_squared: float
    bias: float
    error_map: np.ndarray       # signed error (pred - ref), same shape as inputs
    pred_sample: list           # subsampled for scatter plot
    ref_sample: list
    width: int
    height: int

    def to_dict(self) -> dict:
        return {
            "mae": round(self.mae, 3),
            "rmse": round(self.rmse, 3),
            "median_ae": round(self.median_ae, 3),
            "r_squared": round(self.r_squared, 4),
            "bias": round(self.bias, 3),
            "width": self.width,
            "height": self.height,
            "scatter_pred": self.pred_sample,
            "scatter_ref": self.ref_sample,
        }


def validate(
    predicted: np.ndarray,
    reference: np.ndarray,
    max_scatter_points: int = 2000,
) -> ValidationResult:
    if predicted.shape != reference.shape:
        from PIL import Image
        ref_img = Image.fromarray(reference)
        reference = np.array(
            ref_img.resize((predicted.shape[1], predicted.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        )

    mask = np.isfinite(predicted) & np.isfinite(reference)
    p = predicted[mask].flatten()
    r = reference[mask].flatten()

    if p.size == 0:
        raise ValueError("No overlapping valid pixels between predicted and reference")

    error = p - r
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    median_ae = float(np.median(np.abs(error)))
    bias = float(np.mean(error))

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((r - r.mean()) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    error_map = (predicted - reference).astype(np.float32)
    error_map[~mask] = 0.0

    idx = np.random.choice(p.size, min(max_scatter_points, p.size), replace=False)
    pred_sample = [round(float(v), 2) for v in p[idx]]
    ref_sample = [round(float(v), 2) for v in r[idx]]

    logger.info(
        "Validation: MAE=%.2f RMSE=%.2f R²=%.4f bias=%.2f n=%d",
        mae, rmse, r_squared, bias, p.size,
    )

    return ValidationResult(
        mae=mae, rmse=rmse, median_ae=median_ae,
        r_squared=r_squared, bias=bias,
        error_map=error_map,
        pred_sample=pred_sample, ref_sample=ref_sample,
        width=predicted.shape[1], height=predicted.shape[0],
    )
