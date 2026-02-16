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
    ANGLE_RANGE = (5.0, 180.0)  # Broadened range for tight curls and full extension
    
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
    
    def update_track(self, track: Dict, angle: float, track_id: int, 
                     prefix: str = "", update_history: bool = True) -> bool:
        """
        Update track state for bicep curl and count reps.
        
        Args:
            track: Track dictionary with state
            angle: Current bicep angle
            track_id: ID for logging
            prefix: Prefix for state keys (e.g., "bicep_") for shadow counting
            update_history: Whether to update angle history
            
        Returns:
            True if a rep was completed
        """
        # Select keys
        state_key = f"{prefix}current_state" if not prefix else f"{prefix}state"
        reps_key = f"{prefix}reps"
        up_key = f"{prefix}debounce_up"
        down_key = f"{prefix}debounce_down"
        
        if update_history:
            track["bicep_angle_history"].append(angle)
            smoothed_angle = float(np.mean(track["bicep_angle_history"]))
            track["last_bicep_angle"] = smoothed_angle
        else:
            smoothed_angle = angle

        rep_completed = False

        if smoothed_angle > self.ANGLE_DOWN_THRESHOLD:
            track[down_key] += 1
            track[up_key] = 0
        elif smoothed_angle < self.ANGLE_UP_THRESHOLD:
            track[up_key] += 1
            track[down_key] = 0
        else:
            track[up_key] = max(0, track[up_key] - 1)
            track[down_key] = max(0, track[down_key] - 1)

        if track[state_key] == "down" and track[up_key] >= self.DEBOUNCE_FRAMES:
            track[state_key] = "up"

        elif track[state_key] == "up" and track[down_key] >= self.DEBOUNCE_FRAMES:
            track[reps_key] += 1
            if not prefix: # Only print for confirmed exercise
                print(f"💪 Track {track_id}: Bicep Curl Rep #{track[reps_key]}")
            track[state_key] = "down"
            track[up_key] = 0
            track[down_key] = 0
            rep_completed = True
            
        return rep_completed
