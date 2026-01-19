"""
Exercise Logic Module
=====================
Contains exercise detection logic for different exercise types.
"""

from .base import ExerciseType, AngleCalculator
from .bicep_curl import BicepCurlDetector
from .lateral_raise import LateralRaiseDetector

__all__ = [
    'ExerciseType',
    'AngleCalculator',
    'BicepCurlDetector',
    'LateralRaiseDetector'
]
