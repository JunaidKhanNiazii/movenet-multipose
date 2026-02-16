"""
base.py - Base classes and utilities for exercise detection
============================================================
Contains common functionality shared across all exercise types.
"""

import numpy as np
import enum
from math import atan2, degrees
from typing import Tuple, List, Optional
from collections import deque


class ExerciseType(enum.Enum):
    """Enum for different exercise types"""
    BICEP_CURL = "Bicep Curl"
    LATERAL_RAISE = "Lateral Raise"
    UNKNOWN = "Unknown"


class AngleCalculator:
    """Utility class for calculating angles from keypoints."""
    
    @staticmethod
    def calculate_angle(a: Tuple[float, float],
                       b: Tuple[float, float],
                       c: Tuple[float, float]) -> float:
        """
        Calculate angle ABC (at point B) in degrees.
        
        Args:
            a, b, c: Points as (y, x) normalized coordinates
            
        Returns:
            Angle in degrees (0-180)
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180:
            angle = 360 - angle
        return angle

    @staticmethod
    def calculate_arm_elevation_angle(shoulder: Tuple[float, float],
                                     wrist: Tuple[float, float]) -> float:
        """
        Calculate how high the arm is raised (0-90°).
        0° = arm straight down, 90° = arm straight out to side.
        
        Args:
            shoulder: (y, x) normalized coordinates
            wrist: (y, x) normalized coordinates
            
        Returns:
            Elevation angle in degrees
        """
        # MoveNet MultiPose returns keypoints as [y, x, score].
        # In this project, they are often swapped or handled as (x, y).
        # Standardizing unpacking here based on system usage:
        shoulder_x, shoulder_y = shoulder
        wrist_x, wrist_y = wrist

        # Calculate horizontal and vertical displacement
        # (y increases downwards in images)
        dy = wrist_y - shoulder_y  # Positive if wrist is below shoulder
        dx = wrist_x - shoulder_x 

        # Calculate angle from vertical (0 degrees = straight down)
        # Using atan2(horizontal distance, vertical distance)
        # We use abs(dx) as we only care about elevation from the vertical axis.
        angle = degrees(atan2(abs(dx), max(dy, 0.001)))

        return angle

    @staticmethod
    def calculate_height_difference(shoulder: Tuple[float, float],
                                   wrist: Tuple[float, float]) -> float:
        """
        Calculate height difference (shoulder_y - wrist_y).
        
        Returns:
            Positive if wrist is higher than shoulder
        """
        shoulder_y, _ = shoulder
        wrist_y, _ = wrist
        return shoulder_y - wrist_y


class ExerciseTrackState:
    """Represents the tracking state for a person's exercise."""
    
    def __init__(self, angle_history_size: int = 8):
        self.reps = 0
        self.current_state = "down"  # "up" or "down"
        self.debounce_up = 0
        self.debounce_down = 0
        self.angle_history = deque(maxlen=angle_history_size)
        self.last_angle = 0.0
        
    def reset(self):
        """Reset the track state."""
        self.reps = 0
        self.current_state = "down"
        self.debounce_up = 0
        self.debounce_down = 0
        self.angle_history.clear()
        self.last_angle = 0.0
