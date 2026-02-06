"""Layer 3 hotspot tracking, anti-flicker, and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]
Centroid = Tuple[float, float]


@dataclass
class TrackedHotspot:
    """Tracked hotspot state across frames."""

    id: int
    bbox: BBox
    centroid: Centroid
    tmax: float
    tmean: float
    delta_t: float
    area_px: int
    age_frames: int
    missed_frames: int
    first_seen_frame: int
    last_seen_frame: int
    visible: bool = False

    def to_output_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "Tmax": self.tmax,
            "Tmean": self.tmean,
            "ΔT": self.delta_t,
            "area_px": self.area_px,
            "age_frames": self.age_frames,
        }


def compute_iou(box_a: BBox, box_b: BBox) -> float:
    """Compute Intersection-over-Union between two bounding boxes."""

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def _centroid_distance(ca: Centroid, cb: Centroid) -> float:
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def match_rois_to_tracks(
    rois: List[Dict[str, object]],
    tracks: List[TrackedHotspot],
    iou_threshold: float = 0.3,
    distance_threshold: float = 40.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy match ROIs to existing tracks."""

    candidates: List[Tuple[float, float, int, int]] = []
    for track_idx, track in enumerate(tracks):
        for roi_idx, roi in enumerate(rois):
            roi_bbox = roi["bbox"]
            roi_centroid = roi["centroid"]
            iou = compute_iou(track.bbox, roi_bbox)
            distance = _centroid_distance(track.centroid, roi_centroid)
            if iou >= iou_threshold or distance <= distance_threshold:
                candidates.append((iou, distance, track_idx, roi_idx))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

    matched_tracks = set()
    matched_rois = set()
    matches: List[Tuple[int, int]] = []

    for iou, distance, track_idx, roi_idx in candidates:
        if track_idx in matched_tracks or roi_idx in matched_rois:
            continue
        matches.append((track_idx, roi_idx))
        matched_tracks.add(track_idx)
        matched_rois.add(roi_idx)

    unmatched_tracks = [
        idx for idx in range(len(tracks)) if idx not in matched_tracks
    ]
    unmatched_rois = [idx for idx in range(len(rois)) if idx not in matched_rois]

    return matches, unmatched_tracks, unmatched_rois


def apply_antiflicker(tracks: Iterable[TrackedHotspot]) -> None:
    """Apply anti-flicker visibility rules to tracks in-place."""

    for track in tracks:
        if track.age_frames >= 3 and track.missed_frames == 0:
            track.visible = True
        if track.missed_frames >= 6:
            track.visible = False


def update_tracks(
    tracks: List[TrackedHotspot],
    rois: List[Dict[str, object]],
    frame_index: int,
    next_track_id: int,
) -> Tuple[List[TrackedHotspot], int]:
    """Update tracked hotspots based on new ROIs."""

    matches, unmatched_tracks, unmatched_rois = match_rois_to_tracks(rois, tracks)

    for track in tracks:
        track.age_frames += 1

    for track_idx, roi_idx in matches:
        track = tracks[track_idx]
        roi = rois[roi_idx]
        track.bbox = roi["bbox"]
        track.centroid = roi["centroid"]
        track.tmax = float(roi["Tmax"])
        track.tmean = float(roi["Tmean"])
        track.delta_t = float(roi["ΔT"])
        track.area_px = int(roi["area_px"])
        track.missed_frames = 0
        track.last_seen_frame = frame_index

    for track_idx in unmatched_tracks:
        track = tracks[track_idx]
        track.missed_frames += 1

    for roi_idx in unmatched_rois:
        roi = rois[roi_idx]
        new_track = TrackedHotspot(
            id=next_track_id,
            bbox=roi["bbox"],
            centroid=roi["centroid"],
            tmax=float(roi["Tmax"]),
            tmean=float(roi["Tmean"]),
            delta_t=float(roi["ΔT"]),
            area_px=int(roi["area_px"]),
            age_frames=1,
            missed_frames=0,
            first_seen_frame=frame_index,
            last_seen_frame=frame_index,
            visible=False,
        )
        tracks.append(new_track)
        next_track_id += 1

    apply_antiflicker(tracks)

    tracks = [track for track in tracks if track.missed_frames < 12]

    return tracks, next_track_id


def draw_visualization(frame: np.ndarray, tracks: Iterable[TrackedHotspot]) -> np.ndarray:
    """Draw bounding boxes and labels for visible hotspots."""

    for track in tracks:
        if not track.visible:
            continue
        x, y, w, h = track.bbox
        if track.delta_t < 15.0:
            color = (0, 255, 255)
        elif track.delta_t < 30.0:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"HS-{track.id} | Tmax:{track.tmax:.1f}C | ΔT:{track.delta_t:.1f}"
        cv2.putText(
            frame,
            label,
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


class HotspotTracker:
    """Layer 3 hotspot tracker with anti-flicker and visualization helpers."""

    def __init__(self) -> None:
        self._tracks: List[TrackedHotspot] = []
        self._next_track_id = 1

    @property
    def tracks(self) -> List[TrackedHotspot]:
        return list(self._tracks)

    def update(self, rois: List[Dict[str, object]], frame_index: int) -> List[Dict[str, object]]:
        self._tracks, self._next_track_id = update_tracks(
            self._tracks, rois, frame_index, self._next_track_id
        )
        return [track.to_output_dict() for track in self._tracks if track.visible]

    def draw(self, frame: np.ndarray) -> np.ndarray:
        return draw_visualization(frame, self._tracks)


if __name__ == "__main__":
    tracker = HotspotTracker()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    sample_rois = [
        {
            "bbox": (120, 80, 60, 50),
            "centroid": (150.0, 105.0),
            "Tmax": 92.4,
            "Tmean": 80.1,
            "ΔT": 22.5,
            "area_px": 2800,
        },
        {
            "bbox": (320, 200, 50, 40),
            "centroid": (345.0, 220.0),
            "Tmax": 120.7,
            "Tmean": 98.3,
            "ΔT": 35.2,
            "area_px": 2400,
        },
    ]

    for frame_idx in range(1, 6):
        visible = tracker.update(sample_rois, frame_idx)
        frame = blank_frame.copy()
        tracker.draw(frame)
        print(f"Frame {frame_idx} visible hotspots: {visible}")
