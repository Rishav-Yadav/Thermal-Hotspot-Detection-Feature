"""Layer 1 service: thermal data conditioning.

This service performs only signal conditioning:
1) Spatial median denoising
2) Temporal EMA smoothing

It does not perform detection, thresholding, segmentation, or visualization.
"""

from __future__ import annotations

from .spatial_filter import SpatialMedianFilter
from .temporal_filter import TemporalEMAFilter


class Layer1Service:
    """Condition thermometric frames for stable downstream analytics."""

    def __init__(self, spatial_kernel_size: int = 3, temporal_alpha: float = 0.7):
        self.spatial_filter = SpatialMedianFilter(kernel_size=spatial_kernel_size)
        self.temporal_filter = TemporalEMAFilter(alpha=temporal_alpha)

    def process(self, thermal_frame):
        """Condition and return the same thermal frame object.

        The method updates only ``thermal_frame.temperature_matrix`` and leaves
        all metadata untouched to preserve interface compatibility.
        """
        spatially_conditioned = self.spatial_filter.apply(thermal_frame.temperature_matrix)
        temporally_conditioned = self.temporal_filter.apply(spatially_conditioned)
        thermal_frame.temperature_matrix = temporally_conditioned
        return thermal_frame
