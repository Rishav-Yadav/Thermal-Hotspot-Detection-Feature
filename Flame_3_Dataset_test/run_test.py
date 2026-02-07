"""Test harness for Layer 1-3 thermal hotspot detection.

This script loads two thermal TIFF frames and two RGB frames, runs the
existing Layer 1 conditioning, Layer 2 ΔT detection, and Layer 3 tracking
+ visualization in sequence, then saves annotated RGB outputs.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import tifffile

from layer0_camera_and_environment_variables_setup.metadata import ThermalFrame
from layer1_conditioning import Layer1Service
from layer2_hotspot_detection.delta_t_detector import DeltaTHotspotDetector
from layer3_tracking import HotspotTracker


def _resolve_input_paths(base_dir: Path) -> tuple[list[Path], list[Path]]:
    """Resolve two thermal TIFFs and two RGB images under data/.

    Expected layout (customize if needed):
      data/thermal/*.tif(f)
      data/rgb/*.png|*.jpg|*.jpeg
    """
    thermal_dir = base_dir / "data" / "thermal"
    rgb_dir = base_dir / "data" / "rgb"

    thermal_paths = sorted(
        list(thermal_dir.glob("*.tif")) + list(thermal_dir.glob("*.tiff"))
    )
    rgb_paths = sorted(
        list(rgb_dir.glob("*.png"))
        + list(rgb_dir.glob("*.jpg"))
        + list(rgb_dir.glob("*.jpeg"))
    )

    if len(thermal_paths) != 2:
        raise FileNotFoundError(
            f"Expected exactly 2 thermal TIFFs in {thermal_dir}, found {len(thermal_paths)}"
        )
    if len(rgb_paths) != 2:
        raise FileNotFoundError(
            f"Expected exactly 2 RGB images in {rgb_dir}, found {len(rgb_paths)}"
        )

    return thermal_paths, rgb_paths


def _load_thermal_frame(path: Path) -> np.ndarray:
    """Load a thermal TIFF, using only the first page if multi-page."""
    # Use TiffFile for explicit page handling (first page only).
    with tifffile.TiffFile(path) as tif:
        thermal = tif.pages[0].asarray()
    if thermal.ndim != 2:
        raise ValueError(f"Thermal frame at {path} must be 2D, got shape {thermal.shape}")
    # Ensure float32 without rescaling or normalization.
    if thermal.dtype != np.float32:
        thermal = thermal.astype(np.float32, copy=False)
    return thermal


def _load_rgb_frame(path: Path) -> np.ndarray:
    """Load an RGB image (kept in OpenCV BGR order)."""
    # OpenCV returns BGR by default; keep as-is for downstream drawing.
    rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Failed to read RGB image at {path}")
    return rgb


def main() -> None:
    # Resolve paths relative to this script for portability.
    base_dir = Path(__file__).resolve().parent
    thermal_paths, rgb_paths = _resolve_input_paths(base_dir)

    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    layer1 = Layer1Service()
    detector = DeltaTHotspotDetector()
    tracker = HotspotTracker()

    for frame_idx, (thermal_path, rgb_path) in enumerate(
        zip(thermal_paths, rgb_paths)
    ):
        # Load thermal and RGB frames in matching order.
        thermal_frame = _load_thermal_frame(thermal_path)

        # Wrap thermal data in the pipeline's expected metadata container.
        thermal_meta = ThermalFrame(
            frame_id=frame_idx,
            timestamp=time.time(),
            temperature_matrix=thermal_frame,
            emissivity=1.0,
            distance=1.0,
            ambient_temp=20.0,
            humidity=0.0,
            correction_enabled=False,
        )

        # Layer 1: conditioning.
        conditioned = layer1.process(thermal_meta)
        conditioned_matrix = conditioned.temperature_matrix

        # Layer 2: ΔT hotspot detection (frozen).
        rois = detector.process_frame(conditioned_matrix)
        roi_dicts = [
            {
                "bbox": roi.bbox,
                "centroid": roi.centroid,
                "Tmax": roi.tmax,
                "Tmean": roi.tmean,
                "ΔT": roi.delta_t,
                "area_px": roi.area_px,
            }
            for roi in rois
        ]

        # Layer 3: tracking + annotation.
        rgb_frame = _load_rgb_frame(rgb_path)
        tracker.update(roi_dicts, frame_idx)
        annotated = tracker.draw(rgb_frame.copy())

        print(
            "Frame",
            frame_idx,
            "| thermal dtype:",
            conditioned_matrix.dtype,
            "| min:",
            float(conditioned_matrix.min()),
            "| max:",
            float(conditioned_matrix.max()),
            "| ROIs:",
            len(rois),
        )

        # Persist annotated output for inspection.
        output_path = output_dir / f"annotated_frame_{frame_idx}.png"
        if not cv2.imwrite(str(output_path), annotated):
            raise IOError(f"Failed to write annotated output to {output_path}")


if __name__ == "__main__":
    main()
