"""
Movenet Multipose - Fitness Zone System
========================================
A pose-based exercise detection and tracking system using MoveNet MultiPose.
"""

from .core import ExerciseDetector, PoseEstimator, Tracker
from .face import FaceRecognizer
from .exercises import BicepCurl, LateralRaise, BaseExercise

__version__ = "1.0.0"
__all__ = [
    "ExerciseDetector",
    "PoseEstimator", 
    "Tracker",
    "FaceRecognizer",
    "BicepCurl",
    "LateralRaise",
    "BaseExercise",
]
