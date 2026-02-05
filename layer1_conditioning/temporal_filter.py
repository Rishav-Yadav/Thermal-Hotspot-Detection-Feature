"""Temporal conditioning for radiometric thermal frames.

This module implements exponential moving average (EMA) smoothing to reduce
inter-frame flicker with minimal state and real-time suitability.
"""

from __future__ import annotations

import numpy as np


class TemporalEMAFilter:
    """EMA-based temporal smoother for thermometric matrices."""

    def __init__(self, alpha: float = 0.7):
        """Create a temporal EMA filter.

        Args:
            alpha: Weight of the current frame in EMA, in (0, 1].

        Raises:
            ValueError: If alpha is outside (0, 1].
        """
        if not (0.0 < alpha <= 1.0):
            msg = "alpha must be in the range (0, 1]"
            raise ValueError(msg)

        self.alpha = alpha
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        """Reset internal state for a new stream/session."""
        self._previous = None

    def apply(self, temperature_matrix: np.ndarray) -> np.ndarray:
        """Return a temporally conditioned temperature matrix.

        Safe initialization behavior:
        - First frame is passed through unchanged.
        - Shape changes reinitialize the EMA state for robustness.
        """
        current = np.asarray(temperature_matrix, dtype=np.float32)

        if self._previous is None or self._previous.shape != current.shape:
            smoothed = current
        else:
            smoothed = (self.alpha * current) + ((1.0 - self.alpha) * self._previous)

        # Keep internal state in float32 for deterministic compute and low memory.
        self._previous = smoothed.copy()

        if np.issubdtype(temperature_matrix.dtype, np.floating):
            return smoothed.astype(temperature_matrix.dtype, copy=False)

        return smoothed
