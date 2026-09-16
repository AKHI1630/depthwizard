from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class HeightMetadata:
    def __init__(self, width: int, height: int, units: str, min_height: float, max_height: float):
        self.width = width
        self.height = height
        self.units = units
        self.min_height = min_height
        self.max_height = max_height

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "units": self.units,
            "min_height": self.min_height,
            "max_height": self.max_height,
        }


class HeightEstimator(ABC):
    @abstractmethod
    def estimate(self, image: bytes) -> Tuple[np.ndarray, HeightMetadata]:
        """
        Estimate height from raw image bytes.
        Returns (float32 array of shape [H, W], HeightMetadata).
        """
