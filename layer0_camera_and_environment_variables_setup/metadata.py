from dataclasses import dataclass

import numpy as np


@dataclass
class ThermalFrame:
    frame_id: int
    timestamp: float
    temperature_matrix: np.ndarray
    emissivity: float
    distance: float
    ambient_temp: float
    humidity: float
    correction_enabled: bool
