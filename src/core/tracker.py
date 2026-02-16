"""
tracker.py - Person Tracking
=============================
Handles multi-person tracking across frames.
"""

from math import hypot
from typing import List, Dict, Tuple, Optional
from collections import deque
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exercises.base import ExerciseType


class PersonTracker:
    """
    Tracks multiple people across video frames using centroid matching.
    """
    
    # Tracking configuration
    MAX_TRACK_DISTANCE = 0.25
    TRACK_LOST_FRAMES = 30
    ANGLE_HISTORY_SIZE = 8
    
    def __init__(self):
        """Initialize the tracker."""
        self.tracks = {}
        self.next_track_id = 1
        self.frame_idx = 0
    
    @staticmethod
    def centroid_from_box(box: np.ndarray) -> Tuple[float, float]:
        """Calculate centroid from bounding box."""
        y_min, x_min, y_max, x_max, _ = box
        return ((x_min + x_max) / 2, (y_min + y_max) / 2)
    
    @staticmethod
    def normalized_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return hypot(a[0] - b[0], a[1] - b[1])
    
    def match_detections_to_tracks(self, detections: List[Dict]) -> List[Optional[int]]:
        """
        Match new detections to existing tracks.
        
        Args:
            detections: List of detection dictionaries with 'centroid' key
            
        Returns:
            List of track IDs for each detection (None if no match)
        """
        det_count = len(detections)
        track_ids = list(self.tracks.keys())
        result = [None] * det_count

        if not track_ids:
            return result

        # Calculate distance matrix
        distances = []
        for det in detections:
            row = []
            for tid in track_ids:
                dist = self.normalized_distance(det['centroid'], self.tracks[tid]['centroid'])
                row.append(dist)
            distances.append(row)

        # Greedy matching
        used_tracks = set()
        for _ in range(min(det_count, len(track_ids))):
            best_dist = float('inf')
            best_det_idx = -1
            best_track_idx = -1

            for i in range(det_count):
                if result[i] is not None:
                    continue
                for j, tid in enumerate(track_ids):
                    if tid in used_tracks:
                        continue
                    if distances[i][j] < best_dist:
                        best_dist = distances[i][j]
                        best_det_idx = i
                        best_track_idx = j

            if best_det_idx == -1:
                break

            if best_dist <= self.MAX_TRACK_DISTANCE:
                track_id = track_ids[best_track_idx]
                result[best_det_idx] = track_id
                used_tracks.add(track_id)
            else:
                break

        return result
    
    def create_track(self, detection: Dict) -> int:
        """
        Create a new track for a detection.
        
        Args:
            detection: Detection dictionary
            
        Returns:
            New track ID
        """
        track_id = self.next_track_id
        self.next_track_id += 1

        self.tracks[track_id] = {
            "centroid": detection["centroid"],
            "last_seen": self.frame_idx,
            "lost": 0,
            "reps": 0,
            "current_exercise": ExerciseType.UNKNOWN,
            "confirmed_exercise": ExerciseType.UNKNOWN,
            "exercise_confidence": 0.0,
            "exercise_frames": 0,
            "bicep_angle_history": deque(maxlen=self.ANGLE_HISTORY_SIZE),
            "lateral_angle_history": deque(maxlen=self.ANGLE_HISTORY_SIZE),
            "current_state": "down",
            "debounce_up": 0,
            "debounce_down": 0,
            "last_bicep_angle": detection.get("bicep_angle", 0),
            "last_lateral_angle": detection.get("lateral_angle", 0),
            "exercise_pattern_history": deque(maxlen=20),
            "detection_history": deque(maxlen=20),
            "debug_log": deque(maxlen=50),
            
            # Shadow tracking for two-rep locking
            "bicep_reps": 0,
            "bicep_state": "down",
            "bicep_debounce_up": 0,
            "bicep_debounce_down": 0,
            "lateral_reps": 0,
            "lateral_state": "down",
            "lateral_debounce_up": 0,
            "lateral_debounce_down": 0
        }

        # Initialize history
        for _ in range(2):
            self.tracks[track_id]["bicep_angle_history"].append(detection.get("bicep_angle", 0))
            self.tracks[track_id]["lateral_angle_history"].append(detection.get("lateral_angle", 0))
            self.tracks[track_id]["exercise_pattern_history"].append(detection.get("exercise_type", ExerciseType.UNKNOWN))

        return track_id
    
    def update_track(self, track_id: int, detection: Dict):
        """
        Update an existing track with new detection.
        
        Args:
            track_id: ID of the track to update
            detection: New detection data
        """
        track = self.tracks[track_id]
        track["centroid"] = detection["centroid"]
        track["last_seen"] = self.frame_idx
        track["lost"] = 0
    
    def mark_tracks_lost(self, active_tracks: set):
        """Mark tracks that weren't matched as lost."""
        for track_id in self.tracks:
            if track_id not in active_tracks:
                self.tracks[track_id]["lost"] += 1
    
    def clean_lost_tracks(self):
        """Remove tracks that have been lost for too long."""
        lost_tracks = [
            tid for tid, track in self.tracks.items()
            if track["lost"] > self.TRACK_LOST_FRAMES
        ]
        for tid in lost_tracks:
            del self.tracks[tid]
    
    def increment_frame(self):
        """Increment frame counter."""
        self.frame_idx += 1
    
    def get_track(self, track_id: int) -> Optional[Dict]:
        """Get a track by ID."""
        return self.tracks.get(track_id)
    
    def reset(self):
        """Reset all tracking state."""
        self.tracks = {}
        self.next_track_id = 1
        self.frame_idx = 0
        print("🔄 [PersonTracker] Tracking state reset")
