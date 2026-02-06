"""Delta-T hotspot detection for thermal boiler inspections.

Example:
    import numpy as np
    from layer2_hotspot_detection import DeltaTHotspotDetector

    detector = DeltaTHotspotDetector()
    thermal_frame = np.random.uniform(20.0, 120.0, size=(512, 640)).astype(np.float32)
    hotspots = detector.process_frame(thermal_frame)
    for roi in hotspots:
        print(roi)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class HotspotConfig:
    """Configuration for Delta-T hotspot detection.

    The defaults reflect conservative industry standards for boiler insulation
    and refractory inspection at typical inspection distances.
    """

    baseline_kernel_size: int = 81
    delta_t_threshold_c: float = 8.0
    min_area_px: int = 400
    morph_kernel_size: int = 3
    morph_iterations: int = 1


@dataclass(frozen=True)
class HotspotROI:
    """Detected hotspot region of interest with thermometric metrics."""

    bbox: Tuple[int, int, int, int]
    tmax: float
    tmean: float
    delta_t: float
    area_px: int
    centroid: Tuple[float, float]


class DeltaTHotspotDetector:
    """Detect ΔT-based hotspots in radiometrically corrected thermal frames."""

    def __init__(self, config: HotspotConfig | None = None) -> None:
        self.config = config or HotspotConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.baseline_kernel_size <= 1:
            raise ValueError("baseline_kernel_size must be > 1")
        if self.config.delta_t_threshold_c <= 0.0:
            raise ValueError("delta_t_threshold_c must be > 0")
        if self.config.min_area_px <= 0:
            raise ValueError("min_area_px must be > 0")
        if self.config.morph_kernel_size <= 0:
            raise ValueError("morph_kernel_size must be > 0")
        if self.config.morph_iterations <= 0:
            raise ValueError("morph_iterations must be > 0")

    def process_frame(self, thermal_frame: np.ndarray) -> List[HotspotROI]:
        """Process a thermal frame and return detected hotspot ROIs."""

        if not isinstance(thermal_frame, np.ndarray):
            raise TypeError("thermal_frame must be a numpy ndarray")
        if thermal_frame.ndim != 2:
            raise ValueError("thermal_frame must be a 2D array")
        if thermal_frame.size == 0:
            return []
        if not np.issubdtype(thermal_frame.dtype, np.floating):
            raise TypeError("thermal_frame must have a float dtype")
        if not np.all(np.isfinite(thermal_frame)):
            raise ValueError("thermal_frame contains non-finite values")

        kernel_size = self._ensure_odd(self.config.baseline_kernel_size)
        local_baseline = cv2.GaussianBlur(
            thermal_frame,
            (kernel_size, kernel_size),
            sigmaX=0,
            borderType=cv2.BORDER_REFLECT,
        )
        delta_t = thermal_frame - local_baseline

        mask = (delta_t >= self.config.delta_t_threshold_c).astype(np.uint8) * 255

        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._ensure_odd(self.config.morph_kernel_size),) * 2,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            morph_kernel,
            iterations=self.config.morph_iterations,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            morph_kernel,
            iterations=self.config.morph_iterations,
        )

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        hotspots: List[HotspotROI] = []
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < self.config.min_area_px:
                continue
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            component_mask = labels == label_idx
            component_temps = thermal_frame[component_mask]
            component_delta_t = delta_t[component_mask]
            tmax = float(component_temps.max())
            tmean = float(component_temps.mean())
            delta_t_max = float(component_delta_t.max())
            cx, cy = centroids[label_idx]
            hotspot = HotspotROI(
                bbox=(x, y, w, h),
                tmax=tmax,
                tmean=tmean,
                delta_t=delta_t_max,
                area_px=area,
                centroid=(float(cx), float(cy)),
            )
            hotspots.append(hotspot)

        return hotspots

    @staticmethod
    def _ensure_odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1
