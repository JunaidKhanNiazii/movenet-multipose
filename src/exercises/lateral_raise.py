"""
lateral_raise.py - Lateral Raise Detection Logic
=================================================
Handles detection and rep counting for lateral raise exercises.
"""

import numpy as np
from typing import Dict, List, Tuple
from .base import AngleCalculator, ExerciseType


class LateralRaiseDetector:
    """Detects and counts lateral raise repetitions."""
    
    # ===== CONFIGURATION =====
    ANGLE_DOWN_THRESHOLD = 25    # Arms down (more lenient)
    ANGLE_UP_THRESHOLD = 60      # Arms raised
    ELBOW_ANGLE_MIN = 130        # Elbow should be relatively straight
    HEIGHT_THRESHOLD = -0.10     # Wrist should be higher than shoulder
    
    MIN_KEYPOINT_SCORE = 0.3
    DEBOUNCE_FRAMES = 2
    
    def __init__(self):
        self.angle_calculator = AngleCalculator()
    
    def detect(self, keypoints: np.ndarray, kp_scores: List[float]) -> Tuple[float, bool, Dict]:
        """
        Detect lateral raise exercise from keypoints.
        
        Args:
            keypoints: Array of keypoints (17, 3) with [y, x, score]
            kp_scores: List of keypoint confidence scores
            
        Returns:
            Tuple of (average_elevation_angle, is_valid, debug_info)
        """
        elevation_angles = []
        debug_info = {
            "left_elevation": 0, "right_elevation": 0,
            "left_height_diff": 0, "right_height_diff": 0,
            "left_elbow_angle": 0, "right_elbow_angle": 0,
            "left_valid": False, "right_valid": False
        }
        valid_count = 0

        # Check left arm (shoulder=5, elbow=7, wrist=9)
        if (kp_scores[5] > self.MIN_KEYPOINT_SCORE and
            kp_scores[7] > self.MIN_KEYPOINT_SCORE and
            kp_scores[9] > self.MIN_KEYPOINT_SCORE):

            shoulder = (keypoints[5, 1], keypoints[5, 0])
            elbow = (keypoints[7, 1], keypoints[7, 0])
            wrist = (keypoints[9, 1], keypoints[9, 0])

            left_elbow_angle = self.angle_calculator.calculate_angle(shoulder, elbow, wrist)
            left_elevation = self.angle_calculator.calculate_arm_elevation_angle(shoulder, wrist)
            left_height_diff = self.angle_calculator.calculate_height_difference(shoulder, wrist)

            debug_info["left_elbow_angle"] = left_elbow_angle
            debug_info["left_elevation"] = left_elevation
            debug_info["left_height_diff"] = left_height_diff

            # More lenient validation
            arm_straight_enough = left_elbow_angle > self.ELBOW_ANGLE_MIN
            arm_raised = left_elevation > self.ANGLE_DOWN_THRESHOLD

            # Accept if arm is raised, even if elbow isn't perfectly straight
            left_valid = arm_raised and (arm_straight_enough or left_elevation > 40)

            debug_info["left_valid"] = left_valid
            if left_valid:
                elevation_angles.append(left_elevation)
                valid_count += 1

        # Check right arm (shoulder=6, elbow=8, wrist=10)
        if (kp_scores[6] > self.MIN_KEYPOINT_SCORE and
            kp_scores[8] > self.MIN_KEYPOINT_SCORE and
            kp_scores[10] > self.MIN_KEYPOINT_SCORE):

            shoulder = (keypoints[6, 1], keypoints[6, 0])
            elbow = (keypoints[8, 1], keypoints[8, 0])
            wrist = (keypoints[10, 1], keypoints[10, 0])

            right_elbow_angle = self.angle_calculator.calculate_angle(shoulder, elbow, wrist)
            right_elevation = self.angle_calculator.calculate_arm_elevation_angle(shoulder, wrist)
            right_height_diff = self.angle_calculator.calculate_height_difference(shoulder, wrist)

            debug_info["right_elbow_angle"] = right_elbow_angle
            debug_info["right_elevation"] = right_elevation
            debug_info["right_height_diff"] = right_height_diff

            arm_straight_enough = right_elbow_angle > self.ELBOW_ANGLE_MIN
            arm_raised = right_elevation > self.ANGLE_DOWN_THRESHOLD

            right_valid = arm_raised and (arm_straight_enough or right_elevation > 40)

            debug_info["right_valid"] = right_valid
            if right_valid:
                elevation_angles.append(right_elevation)
                valid_count += 1

        if valid_count > 0 and elevation_angles:
            avg_elevation = sum(elevation_angles) / len(elevation_angles)
            return avg_elevation, True, debug_info

        return 0, False, debug_info
    
    def update_track(self, track: Dict, angle: float, track_id: int) -> bool:
        """
        Update track state for lateral raise and count reps.
        
        Args:
            track: Track dictionary with state
            angle: Current lateral angle
            track_id: ID for logging
            
        Returns:
            True if a rep was completed
        """
        track["lateral_angle_history"].append(angle)
        smoothed_angle = float(np.mean(track["lateral_angle_history"]))
        track["last_lateral_angle"] = smoothed_angle

        rep_completed = False

        if smoothed_angle < self.ANGLE_DOWN_THRESHOLD:
            track["debounce_down"] += 1
            track["debounce_up"] = 0
        elif smoothed_angle > self.ANGLE_UP_THRESHOLD:
            track["debounce_up"] += 1
            track["debounce_down"] = 0
        else:
            track["debounce_up"] = max(0, track["debounce_up"] - 1)
            track["debounce_down"] = max(0, track["debounce_down"] - 1)

        if track["current_state"] == "down" and track["debounce_up"] >= self.DEBOUNCE_FRAMES:
            track["current_state"] = "up"

        elif track["current_state"] == "up" and track["debounce_down"] >= self.DEBOUNCE_FRAMES:
            track["reps"] += 1
            print(f"🏋️ Track {track_id}: Lateral Raise Rep #{track['reps']}")
            track["current_state"] = "down"
            track["debounce_up"] = 0
            track["debounce_down"] = 0
            rep_completed = True
            
        return rep_completed
