"""
exercise_detector.py - Main Exercise Detection
===============================================
Combines pose estimation, tracking, and exercise detection.
"""

import numpy as np
import time
from typing import List, Dict, Tuple
from collections import deque

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exercises.base import ExerciseType
from exercises.bicep_curl import BicepCurlDetector
from exercises.lateral_raise import LateralRaiseDetector
from .pose_estimator import PoseEstimator
from .tracker import PersonTracker


class ExerciseDetector:
    """
    Main exercise detector that combines all modules.
    Handles multi-person pose estimation, tracking, and exercise detection.
    """
    
    # Detection configuration
    MIN_DETECTION_SCORE = 0.3
    MIN_KEYPOINT_SCORE = 0.3
    
    # Exercise confirmation
    EXERCISE_CONFIRMATION_FRAMES = 8
    MIN_EXERCISE_CONFIDENCE = 0.55
    
    def __init__(self, model_path: str):
        """
        Initialize the exercise detector.
        
        Args:
            model_path: Path to the MoveNet SavedModel directory
        """
        print("🔹 [ExerciseDetector] Initializing...")
        
        # Initialize sub-modules
        self.pose_estimator = PoseEstimator(model_path)
        self.tracker = PersonTracker()
        self.bicep_detector = BicepCurlDetector()
        self.lateral_detector = LateralRaiseDetector()
        
        # Debug tracking
        self.debug_frame_count = 0
        self.last_debug_output = 0
        self.debug_interval = 30
        
        print("✅ [ExerciseDetector] Ready!")
    
    @property
    def tracks(self):
        """Access tracker's tracks for compatibility."""
        return self.tracker.tracks
    
    def _print_detection_debug(self, track_id: int, detection: Dict, bicep_debug: Dict, lateral_debug: Dict):
        """Print detailed debug information."""
        current_time = time.time()
        if current_time - self.last_debug_output < 3.0:
            return

        self.last_debug_output = current_time

        print(f"\n🔍 FRAME {self.tracker.frame_idx} - TRACK {track_id}:")
        print(f"   BICEP: {detection['bicep_angle']:.1f}° (valid={detection['bicep_valid']})")
        print(f"   LATERAL: {detection['lateral_angle']:.1f}° (valid={detection['lateral_valid']})")

        if detection["bicep_valid"] and not detection["lateral_valid"]:
            likely = "BICEP CURL ✓"
        elif detection["lateral_valid"] and not detection["bicep_valid"]:
            likely = "LATERAL RAISE ✓"
        elif detection["bicep_valid"] and detection["lateral_valid"]:
            likely = "BOTH (resolving)"
        else:
            likely = "NONE"

        print(f"   LIKELY: {likely}")
    
    def _determine_exercise_for_detection(self, detection: Dict, bicep_debug: Dict, lateral_debug: Dict) -> ExerciseType:
        """Determine the most likely exercise type from a single detection."""
        bicep_valid = detection["bicep_valid"]
        lateral_valid = detection["lateral_valid"]

        # LATERAL RAISE EXCLUSION: If elbow is bent (< 150), it's NOT a lateral raise
        is_elbow_bent = (
            (bicep_debug["left_valid"] and bicep_debug["left_angle"] < 150) or
            (bicep_debug["right_valid"] and bicep_debug["right_angle"] < 150)
        )

        if is_elbow_bent and lateral_valid:
            lateral_valid = False
            detection["lateral_valid"] = False

        if bicep_valid and not lateral_valid:
            return ExerciseType.BICEP_CURL

        if lateral_valid and not bicep_valid:
            return ExerciseType.LATERAL_RAISE

        if bicep_valid and lateral_valid:
            # If both seem valid, prioritize Bicep Curl if elbow is bent at all
            if is_elbow_bent:
                return ExerciseType.BICEP_CURL
            
            # Use elevation as tie-breaker for straight-arm movements
            if detection["lateral_angle"] > 40:
                return ExerciseType.LATERAL_RAISE
            
            return ExerciseType.BICEP_CURL

        return ExerciseType.UNKNOWN
    
    def _determine_best_exercise_for_track(self, track_id: int, detection: Dict,
                                          bicep_debug: Dict, lateral_debug: Dict) -> ExerciseType:
        """Determine the best exercise type using history (pre-confirmation)."""
        track = self.tracker.tracks[track_id]

        if track["confirmed_exercise"] != ExerciseType.UNKNOWN:
            return track["confirmed_exercise"]

        current_exercise = self._determine_exercise_for_detection(detection, bicep_debug, lateral_debug)
        track["exercise_pattern_history"].append(current_exercise)

        if len(track["exercise_pattern_history"]) < 5:
            return ExerciseType.UNKNOWN

        exercise_counts = {
            ExerciseType.BICEP_CURL: 0,
            ExerciseType.LATERAL_RAISE: 0,
            ExerciseType.UNKNOWN: 0
        }

        for ex in track["exercise_pattern_history"]:
            exercise_counts[ex] += 1

        total_frames = len(track["exercise_pattern_history"])
        bicep_ratio = exercise_counts[ExerciseType.BICEP_CURL] / total_frames
        lateral_ratio = exercise_counts[ExerciseType.LATERAL_RAISE] / total_frames

        if bicep_ratio > 0.7: 
            return ExerciseType.BICEP_CURL
        elif lateral_ratio > 0.8: 
            return ExerciseType.LATERAL_RAISE
        else:
            return ExerciseType.UNKNOWN
    
    def _update_exercise_confirmation(self, track_id: int, new_exercise: ExerciseType):
        """Update exercise confirmation state."""
        track = self.tracker.tracks[track_id]

        if new_exercise == track["current_exercise"]:
            track["exercise_frames"] += 1
            track["exercise_confidence"] = min(1.0, track["exercise_confidence"] + 0.08)
        else:
            track["current_exercise"] = new_exercise
            track["exercise_frames"] = 1
            track["exercise_confidence"] = 0.4

        if (track["exercise_frames"] >= self.EXERCISE_CONFIRMATION_FRAMES and
            track["exercise_confidence"] >= self.MIN_EXERCISE_CONFIDENCE and
            track["current_exercise"] != ExerciseType.UNKNOWN):

            if track["confirmed_exercise"] != track["current_exercise"]:
                track["confirmed_exercise"] = track["current_exercise"]
                print(f"\n✅ TRACK {track_id} CONFIRMED: {track['confirmed_exercise'].value}")
    
    def _update_track_state(self, track_id: int, detection: Dict, bicep_debug: Dict, lateral_debug: Dict):
        """Update track exercise state and count reps."""
        track = self.tracker.tracks[track_id]

        # 1. PRINT DEBUG (Optional, throttled)
        self._print_detection_debug(track_id, detection, bicep_debug, lateral_debug)

        # 2. IF NOT CONFIRMED, RUN SHADOW TRACKING
        if track["confirmed_exercise"] == ExerciseType.UNKNOWN:
            # Update background counts for both
            self.bicep_detector.update_track(track, angle=detection["bicep_angle"], 
                                             track_id=track_id, prefix="bicep_", update_history=True)
            self.lateral_detector.update_track(track, angle=detection["lateral_angle"], 
                                               track_id=track_id, prefix="lateral_", update_history=True)

            # Check if either reached 2 reps
            if track["bicep_reps"] >= 2:
                track["confirmed_exercise"] = ExerciseType.BICEP_CURL
                track["reps"] = track["bicep_reps"]
                track["current_state"] = track["bicep_state"]
                track["debounce_up"] = track["bicep_debounce_up"]
                track["debounce_down"] = track["bicep_debounce_down"]
                print(f"\n✅ TRACK {track_id} LOCKED: Bicep Curl (after 2 reps)")
                
            elif track["lateral_reps"] >= 2:
                track["confirmed_exercise"] = ExerciseType.LATERAL_RAISE
                track["reps"] = track["lateral_reps"]
                track["current_state"] = track["lateral_state"]
                track["debounce_up"] = track["lateral_debounce_up"]
                track["debounce_down"] = track["lateral_debounce_down"]
                print(f"\n✅ TRACK {track_id} LOCKED: Lateral Raise (after 2 reps)")
            
            # While not confirmed, we can still use heuristic for "likely" display
            best_exercise = self._determine_best_exercise_for_track(track_id, detection, bicep_debug, lateral_debug)
            track["current_exercise"] = best_exercise
            
        else:
            # 3. IF CONFIRMED, UPDATE MAIN DETECTOR
            track["current_exercise"] = track["confirmed_exercise"]
            if track["confirmed_exercise"] == ExerciseType.BICEP_CURL:
                self.bicep_detector.update_track(track, detection["bicep_angle"], track_id)
            elif track["confirmed_exercise"] == ExerciseType.LATERAL_RAISE:
                self.lateral_detector.update_track(track, detection["lateral_angle"], track_id)
    
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a single frame and return detection results.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of detection dictionaries with tracking info
        """
        self.tracker.increment_frame()
        h, w = frame.shape[:2]

        # Run pose estimation
        persons = self.pose_estimator.infer_poses(frame)

        # Process detections
        detections = []
        for person_data in persons:
            keypoints_raw = person_data[:51].reshape((17, 3))
            box = person_data[51:56]

            if box[4] < self.MIN_DETECTION_SCORE:
                continue

            keypoints = [(keypoints_raw[i, 0], keypoints_raw[i, 1]) for i in range(17)]
            kp_scores = [keypoints_raw[i, 2] for i in range(17)]
            centroid = self.tracker.centroid_from_box(box)

            # Detect exercises
            bicep_angle, bicep_valid, bicep_debug = self.bicep_detector.detect(keypoints_raw, kp_scores)
            lateral_angle, lateral_valid, lateral_debug = self.lateral_detector.detect(keypoints_raw, kp_scores)

            detection_data = {
                "box": box,
                "keypoints": keypoints,
                "kp_scores": kp_scores,
                "centroid": centroid,
                "bicep_angle": bicep_angle,
                "lateral_angle": lateral_angle,
                "bicep_valid": bicep_valid,
                "lateral_valid": lateral_valid,
                "bicep_debug": bicep_debug,
                "lateral_debug": lateral_debug,
                "exercise_type": ExerciseType.UNKNOWN
            }

            detections.append(detection_data)

        # Match detections to tracks
        matches = self.tracker.match_detections_to_tracks(detections)
        active_tracks = set()

        for det_idx, detection in enumerate(detections):
            track_id = matches[det_idx]

            if track_id is None:
                track_id = self.tracker.create_track(detection)

            active_tracks.add(track_id)
            self.tracker.update_track(track_id, detection)
            self._update_track_state(track_id, detection, detection["bicep_debug"], detection["lateral_debug"])

        # Clean up lost tracks
        self.tracker.mark_tracks_lost(active_tracks)
        self.tracker.clean_lost_tracks()

        # Build results
        results = []
        for det_idx, detection in enumerate(detections):
            track_id = matches[det_idx]
            if track_id is None:
                for tid, track in self.tracker.tracks.items():
                    if tid in active_tracks and track["last_seen"] == self.tracker.frame_idx:
                        if track["centroid"] == detection["centroid"]:
                            track_id = tid
                            break

            if track_id is None:
                continue

            track = self.tracker.tracks[track_id]
            box = detection["box"]

            y_min, x_min, y_max, x_max, _ = box
            x1, y1 = int(x_min * w), int(y_min * h)
            x2, y2 = int(x_max * w), int(y_max * h)

            keypoints_pixel = []
            for i, (ky, kx) in enumerate(detection["keypoints"]):
                if detection["kp_scores"][i] > self.MIN_KEYPOINT_SCORE:
                    keypoints_pixel.append((int(kx * w), int(ky * h)))
                else:
                    keypoints_pixel.append(None)

            # Only return official exercise and reps if CONFIRMED (2 reps completed)
            is_confirmed = (track["confirmed_exercise"] != ExerciseType.UNKNOWN)
            
            results.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "reps": track["reps"] if is_confirmed else 0,
                "exercise_type": track["confirmed_exercise"].value if is_confirmed else "Detecting...",
                "bicep_angle": track["last_bicep_angle"],
                "lateral_angle": track["last_lateral_angle"],
                "keypoints": keypoints_pixel,
                "keypoint_scores": detection["kp_scores"],
                "confidence": track["exercise_confidence"],
                "confirmed": is_confirmed
            })

        return results
    
    def reset(self):
        """Reset all tracking state."""
        self.tracker.reset()
        print("🔄 [ExerciseDetector] Tracking state reset")
    
    def print_final_debug_summary(self):
        """Print final debug summary."""
        print("\n" + "="*70)
        print("🎯 FINAL EXERCISE DETECTION SUMMARY")
        print("="*70)

        for track_id, track in self.tracker.tracks.items():
            print(f"\nTrack {track_id}:")
            print(f"  Exercise: {track['confirmed_exercise'].value}")
            print(f"  Total reps: {track['reps']}")
            print(f"  Confidence: {track['exercise_confidence']:.2f}")
