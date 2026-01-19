"""
Core Processing Module
======================
Contains pose estimation, tracking, and exercise detection logic.
"""

from .pose_estimator import PoseEstimator
from .tracker import PersonTracker
from .exercise_detector import ExerciseDetector
from .streamer import VideoStreamer

# Alias for convenience
Tracker = PersonTracker

__all__ = [
    'PoseEstimator',
    'PersonTracker',
    'Tracker',
    'ExerciseDetector',
    'VideoStreamer'
]
