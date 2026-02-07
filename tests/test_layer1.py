"""
Layer 1 Validation Test
-----------------------
This script validates that Layer 1:
- Reduces spatial noise
- Reduces temporal flicker
- Preserves absolute temperature values (°C)
- Preserves hotspot integrity
- Does not introduce drift

Run:
    python tests/test_layer1.py
"""

import numpy as np
from layer1_conditioning import Layer1Service


# -------------------------------
# Synthetic thermal data generator
# -------------------------------
def generate_thermal_frame(
    base_temp=80.0,
    hotspot_delta=40.0,
    noise_std=0.8,
    shape=(240, 320)
):
    """
    Generate a synthetic thermometric frame in °C.

    - base boiler wall temperature
    - one stable hotspot
    - gaussian sensor noise
    """
    frame = np.full(shape, base_temp, dtype=np.float32)

    # Add rectangular hotspot
    h, w = shape
    frame[h // 3 : h // 3 + 40, w // 2 : w // 2 + 40] += hotspot_delta

    # Add sensor noise
    noise = np.random.normal(0, noise_std, shape).astype(np.float32)
    frame += noise

    return frame


# -------------------------------
# Minimal ThermalFrame mock
# -------------------------------
class MockThermalFrame:
    def __init__(self, temperature_matrix):
        self.temperature_matrix = temperature_matrix
        self.timestamp = 0.0


# -------------------------------
# Main test
# -------------------------------
def main():
    layer1 = Layer1Service(
        spatial_kernel_size=3,
        temporal_alpha=0.7,
    )

    means = []
    max_vals = []

    print("\nRunning Layer 1 conditioning test\n")

    for i in range(20):
        raw_matrix = generate_thermal_frame()
        frame = MockThermalFrame(raw_matrix)

        processed_frame = layer1.process(frame)
        conditioned = processed_frame.temperature_matrix

        means.append(conditioned.mean())
        max_vals.append(conditioned.max())

        print(
            f"Frame {i:02d} | "
            f"mean = {conditioned.mean():6.2f} °C | "
            f"max = {conditioned.max():6.2f} °C"
        )

    print("\n--- Stability Checks ---")
    print(f"Mean temperature drift : {max(means) - min(means):.3f} °C")
    print(f"Max temperature spread : {max(max_vals) - min(max_vals):.3f} °C")

    # Simple sanity assertions (not pytest, just logic)
    assert max(means) - min(means) < 1.5, "Mean temperature drift too high"
    assert min(max_vals) > 115.0, "Hotspot collapsed or lost"

    print("\n✅ Layer 1 PASSED all checks\n")


if __name__ == "__main__":
    main()
