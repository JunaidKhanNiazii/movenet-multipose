"""
bicep_curl.py - Bicep Curl Detection Logic
==========================================
Handles detection and rep counting for bicep curl exercises.
"""

import numpy as np
from typing import Dict, List, Tuple
from .base import AngleCalculator, ExerciseType


class BicepCurlDetector:
    """Detects and counts bicep curl repetitions."""
    
    # ===== CONFIGURATION =====
    ANGLE_DOWN_THRESHOLD = 160  # Arm extended
    ANGLE_UP_THRESHOLD = 50     # Arm curled
    ANGLE_RANGE = (40, 170)     # Valid range for bicep curls
    
    MIN_KEYPOINT_SCORE = 0.3
    DEBOUNCE_FRAMES = 2
    
    def __init__(self):
        self.angle_calculator = AngleCalculator()
    
    def detect(self, keypoints: np.ndarray, kp_scores: List[float]) -> Tuple[float, bool, Dict]:
        """
        Detect bicep curl exercise from keypoints.
        
        Args:
            keypoints: Array of keypoints (17, 3) with [y, x, score]
            kp_scores: List of keypoint confidence scores
            
        Returns:
            Tuple of (average_angle, is_valid, debug_info)
        """
        angles = []
        debug_info = {
            "left_angle": 0, 
            "right_angle": 0, 
            "left_valid": False, 
            "right_valid": False
        }

        # Check left arm (shoulder=5, elbow=7, wrist=9)
        if (kp_scores[5] > self.MIN_KEYPOINT_SCORE and
            kp_scores[7] > self.MIN_KEYPOINT_SCORE and
            kp_scores[9] > self.MIN_KEYPOINT_SCORE):

            shoulder = (keypoints[5, 1], keypoints[5, 0])
            elbow = (keypoints[7, 1], keypoints[7, 0])
            wrist = (keypoints[9, 1], keypoints[9, 0])

            left_angle = self.angle_calculator.calculate_angle(shoulder, elbow, wrist)
            angles.append(left_angle)
            debug_info["left_angle"] = left_angle
            debug_info["left_valid"] = self.ANGLE_RANGE[0] <= left_angle <= self.ANGLE_RANGE[1]

        # Check right arm (shoulder=6, elbow=8, wrist=10)
        if (kp_scores[6] > self.MIN_KEYPOINT_SCORE and
            kp_scores[8] > self.MIN_KEYPOINT_SCORE and
            kp_scores[10] > self.MIN_KEYPOINT_SCORE):

            shoulder = (keypoints[6, 1], keypoints[6, 0])
            elbow = (keypoints[8, 1], keypoints[8, 0])
            wrist = (keypoints[10, 1], keypoints[10, 0])

            right_angle = self.angle_calculator.calculate_angle(shoulder, elbow, wrist)
            angles.append(right_angle)
            debug_info["right_angle"] = right_angle
            debug_info["right_valid"] = self.ANGLE_RANGE[0] <= right_angle <= self.ANGLE_RANGE[1]

        if not angles:
            return 0, False, debug_info

        avg_angle = sum(angles) / len(angles)
        is_valid = debug_info["left_valid"] or debug_info["right_valid"]

        return avg_angle, is_valid, debug_info
    
    def update_track(self, track: Dict, angle: float, track_id: int) -> bool:
        """
        Update track state for bicep curl and count reps.
        
        Args:
            track: Track dictionary with state
            angle: Current bicep angle
            track_id: ID for logging
            
        Returns:
            True if a rep was completed
        """
        track["bicep_angle_history"].append(angle)
        smoothed_angle = float(np.mean(track["bicep_angle_history"]))
        track["last_bicep_angle"] = smoothed_angle

        rep_completed = False

        if smoothed_angle > self.ANGLE_DOWN_THRESHOLD:
            track["debounce_down"] += 1
            track["debounce_up"] = 0
        elif smoothed_angle < self.ANGLE_UP_THRESHOLD:
            track["debounce_up"] += 1
            track["debounce_down"] = 0
        else:
            track["debounce_up"] = max(0, track["debounce_up"] - 1)
            track["debounce_down"] = max(0, track["debounce_down"] - 1)

        if track["current_state"] == "down" and track["debounce_up"] >= self.DEBOUNCE_FRAMES:
            track["current_state"] = "up"

        elif track["current_state"] == "up" and track["debounce_down"] >= self.DEBOUNCE_FRAMES:
            track["reps"] += 1
            print(f"💪 Track {track_id}: Bicep Curl Rep #{track['reps']}")
            track["current_state"] = "down"
            track["debounce_up"] = 0
            track["debounce_down"] = 0
            rep_completed = True
            
        return rep_completed
