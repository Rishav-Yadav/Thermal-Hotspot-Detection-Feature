"""Spatial conditioning for radiometric thermal frames.

Layer 1 performs signal conditioning only. This module provides a median filter
that reduces impulsive pixel noise while preserving edges and thermal gradients.
"""

from __future__ import annotations

import cv2
import numpy as np


class SpatialMedianFilter:
    """Apply spatial median filtering on thermometric matrices.

    The filter operates directly on temperature values in degrees Celsius and
    never normalizes or rescales the data to display ranges.
    """

    def __init__(self, kernel_size: int = 3):
        """Create a spatial median filter.

        Args:
            kernel_size: Odd kernel size for median filtering. Typical industry
                values are 3 or 5 for thermal denoising with hotspot retention.

        Raises:
            ValueError: If kernel_size is not an odd integer >= 3.
        """
        if kernel_size < 3 or kernel_size % 2 == 0:
            msg = "kernel_size must be an odd integer >= 3"
            raise ValueError(msg)

        self.kernel_size = kernel_size

    def apply(self, temperature_matrix: np.ndarray) -> np.ndarray:
        """Return a spatially conditioned temperature matrix.

        Args:
            temperature_matrix: 2D thermal matrix in degrees Celsius.

        Returns:
            2D thermal matrix after median denoising.

        Raises:
            ValueError: If the input is not a 2D array.
        """
        if temperature_matrix.ndim != 2:
            msg = "temperature_matrix must be a 2D ndarray"
            raise ValueError(msg)

        original_dtype = temperature_matrix.dtype

        # OpenCV medianBlur supports CV_32F for single-channel data and is
        # efficient on Jetson-class hardware.
        matrix_32f = np.asarray(temperature_matrix, dtype=np.float32)
        filtered = cv2.medianBlur(matrix_32f, self.kernel_size)

        if np.issubdtype(original_dtype, np.floating):
            return filtered.astype(original_dtype, copy=False)

        return filtered
