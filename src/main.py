"""
main.py - UPDATED WITH MULTIPLE EXERCISES
Main controller with Bicep Curls and Lateral Raises detection
"""

import cv2
import numpy as np
import time
import threading
import argparse
from typing import Dict, List, Tuple, Optional
from collections import deque
import os
import sys
import io

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import from new modular structure
from core import ExerciseDetector, VideoStreamer
from face import FaceRecognizer

# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================

# Path to your MoveNet MultiPose model file (use .onnx for best performance on Windows)
MOVENET_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'movenet_multipose.onnx'))

# Path to your face database pickle file
FACE_DATABASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'face_db.pkl'))

# Video source - Default to videos folder (one level up)
VIDEO_SOURCE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'videos', 'J Biceps S shoulder.mp4'))

# Optional: Set to a path to save output video (saved to output folder)
OUTPUT_VIDEO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'output_result.mp4'))

# Display window (set to False for headless mode)
SHOW_DISPLAY = True


# ============================================================================


class FitZoneSystem:
    """
    Main system with multiple exercise detection.
    """

    # ===== VISUALIZATION CONFIG =====
    COLOR_BBOX_IDENTIFIED = (0, 255, 0)  # Green for identified persons
    COLOR_BBOX_UNIDENTIFIED = (0, 255, 255)  # Yellow for unknown persons
    COLOR_FACE_BBOX = (255, 0, 255)  # Magenta for face boxes
    COLOR_SKELETON = (255, 0, 0)  # Blue for pose skeleton
    COLOR_KEYPOINTS = (0, 255, 0)  # Green for keypoints
    COLOR_TEXT = (255, 255, 255)  # White text

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LARGE = 0.8
    FONT_SCALE_MEDIUM = 0.6
    FONT_SCALE_SMALL = 0.5
    FONT_THICKNESS = 2

    # Skeleton connections for pose visualization
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (11, 12), (5, 11), (6, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    def __init__(self,
                 movenet_model_path: str,
                 face_database_path: str,
                 video_source: str = 0):
        """
        Initialize system.
        """
        print("\n" + "=" * 70)
        print("🏋️  FITZONE: Multiple Exercise Detection System")
        print("⚡ Models: MoveNet MultiPose + InsightFace (ArcFace)")
        print("=" * 70 + "\n")

        # Check if files exist
        print("🔍 Checking file paths...")
        if not os.path.exists(movenet_model_path):
            print(f"❌ MoveNet model not found at: {movenet_model_path}")
            raise FileNotFoundError(f"MoveNet model not found at: {movenet_model_path}")

        if not os.path.exists(face_database_path):
            print(f"❌ Face database not found at: {face_database_path}")
            raise FileNotFoundError(f"Face database not found at: {face_database_path}")

        # Check video source
        if isinstance(video_source, str) and not os.path.exists(video_source):
            # Try to find video in parent directory
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            possible_path = os.path.join(parent_dir, video_source)
            if os.path.exists(possible_path):
                video_source = possible_path
                print(f"✅ Found video at: {video_source}")
            else:
                print(f"❌ Video not found at: {video_source}")
                print(f"   Also checked: {possible_path}")
                raise FileNotFoundError(f"Video not found: {video_source}")

        print("✅ All file paths verified")

        # Initialize exercise detector (UPDATED)
        print("🔹 Initializing Exercise Detector...")
        self.exercise_detector = ExerciseDetector(movenet_model_path)

        # Initialize face recognizer
        print("🔹 Initializing Face Recognizer...")
        self.face_recognizer = FaceRecognizer(face_database_path)

        # Video capture
        print(f"🎥 Opening video source: {video_source}")
        self.streamer = VideoStreamer(video_source)

        print(f"✅ Video opened: {self.streamer.width}x{self.streamer.height} @ {self.streamer.fps:.1f} FPS")
        print(f"📊 Known identities: {', '.join(self.face_recognizer.get_database_names())}")

        # ===== OPTIMIZATION ADDITIONS =====
        # Track which persons have been identified
        self.identified_tracks = {}  # track_id -> {"name": str, "similarity": float, "face_bbox": tuple}

        # Face recognition scheduling
        self.frame_count = 0
        self.last_face_recognition_frame = 0
        self.face_recognition_interval = 30  # Run face recognition every 30 frames max
        self.face_recognition_in_progress = False
        self.recognition_lock = threading.Lock()

        # Performance tracking
        self.processing_times = deque(maxlen=100)

        # Current face results (cached)
        self.current_face_results = []

        print("\n" + "=" * 70)
        print("⚡ DETECTING EXERCISES:")
        print("  • Bicep Curls")
        print("  • Lateral Raises")
        print("  • Face recognition: InsightFace (Background Thread)")
        print("  • Pose estimation: MoveNet MultiPose (C++ Execution)")
        print("=" * 70 + "\n")

    @staticmethod
    def _is_point_in_bbox(point: Tuple[float, float],
                          bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check if a point is inside a bounding box.
        """
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def _match_faces_to_bodies(self,
                               face_results: List[Dict],
                               body_results: List[Dict]) -> Dict[int, Dict]:
        """
        Match detected faces to tracked bodies.
        Uses centroid distance for tie-breaking when multiple bodies qualify.
        """
        matches = {}

        for face in face_results:
            face_bbox = face["bbox"]
            fx1, fy1, fx2, fy2 = face_bbox
            face_center = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)

            best_track_id = None
            min_dist = float('inf')
            best_face_info = None

            # Check which body bbox contains this face center
            for body in body_results:
                body_bbox = body["bbox"]

                if self._is_point_in_bbox(face_center, body_bbox):
                    # Calculate distance to body center for better tie-breaking
                    bx1, by1, bx2, by2 = body_bbox
                    body_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)
                    dist = ((face_center[0] - body_center[0])**2 + (face_center[1] - body_center[1])**2)**0.5

                    if dist < min_dist:
                        min_dist = dist
                        best_track_id = body["track_id"]
                        best_face_info = {
                            "name": face["name"],
                            "similarity": face["similarity"],
                            "face_bbox": face_bbox,
                            "confidence": face.get("confidence", 0)
                        }

            if best_track_id is not None:
                # If multiple faces match same body, keep the one with higher similarity
                if best_track_id in matches:
                    if best_face_info["similarity"] > matches[best_track_id]["similarity"]:
                        matches[best_track_id] = best_face_info
                else:
                    matches[best_track_id] = best_face_info

        return matches

    def _check_if_needs_face_recognition(self, body_results: List[Dict]) -> bool:
        """
        Check if we need to run face recognition.
        """
        with self.recognition_lock:
            # Don't run if already in progress
            if self.face_recognition_in_progress:
                return False

            # Check if enough frames have passed
            frames_since_last = self.frame_count - self.last_face_recognition_frame
            if frames_since_last < self.face_recognition_interval:
                return False

            # Check if there are unidentified tracks
            for body in body_results:
                track_id = body["track_id"]
                if track_id not in self.identified_tracks:
                    return True

            return False

    def _run_face_recognition_async(self, frame: np.ndarray):
        """
        Run face recognition in background thread.
        """
        def recognize_task():
            try:
                # Run face recognition
                face_results = self.face_recognizer.process_frame(frame)

                with self.recognition_lock:
                    self.current_face_results = face_results
                    self.last_face_recognition_frame = self.frame_count
                    self.face_recognition_in_progress = False

                    # Print recognition results
                    if face_results:
                        print(f"🔍 Found {len(face_results)} face(s)")
                        for face in face_results:
                            if face["name"]:
                                print(f"   ✅ Recognized: {face['name']} (similarity: {face['similarity']:.2f})")
                            else:
                                print(f"   ❓ Unknown face (similarity: {face['similarity']:.2f})")

            except Exception as e:
                print(f"⚠️ Face recognition error: {str(e)}")
                with self.recognition_lock:
                    self.face_recognition_in_progress = False

        # Start recognition in background
        thread = threading.Thread(target=recognize_task, daemon=True)
        thread.start()
        self.face_recognition_in_progress = True

    def _process_frame_optimized(self, frame: np.ndarray) -> Tuple[List[Dict], Dict[int, Dict]]:
        """
        Optimized frame processing.
        """
        start_time = time.time()

        # STEP 1: ALWAYS run pose estimation (EVERY FRAME)
        body_results = self.exercise_detector.process_frame(frame)

        # STEP 2: Check if we need face recognition
        needs_recognition = self._check_if_needs_face_recognition(body_results)

        if needs_recognition:
            self._run_face_recognition_async(frame)

        # STEP 3: Update identified tracks with new recognition results
        with self.recognition_lock:
            if self.current_face_results:
                # Match faces to bodies
                face_body_matches = self._match_faces_to_bodies(self.current_face_results, body_results)

                # Update identified tracks
                for track_id, face_info in face_body_matches.items():
                    if face_info["name"]:  # Only update if we got a name
                        self.identified_tracks[track_id] = {
                            "name": face_info["name"],
                            "similarity": face_info["similarity"],
                            "face_bbox": face_info["face_bbox"],
                            "confidence": face_info.get("confidence", 0)
                        }

                # Clear results after processing
                self.current_face_results = []
            else:
                # Use cached matches
                face_body_matches = {}
                for track_id, identity in self.identified_tracks.items():
                    face_body_matches[track_id] = {
                        "name": identity["name"],
                        "similarity": identity["similarity"],
                        "face_bbox": identity["face_bbox"],
                        "confidence": identity.get("confidence", 0)
                    }

        # Track processing time
        self.processing_times.append(time.time() - start_time)
        self.frame_count += 1

        return body_results, face_body_matches

    def _draw_skeleton(self,
                       frame: np.ndarray,
                       keypoints: List[Optional[Tuple[int, int]]],
                       keypoint_scores: List[float]):
        """
        Draw pose skeleton.
        """
        # Draw skeleton lines
        for idx_a, idx_b in self.SKELETON_CONNECTIONS:
            if keypoints[idx_a] is not None and keypoints[idx_b] is not None:
                cv2.line(frame, keypoints[idx_a], keypoints[idx_b],
                         self.COLOR_SKELETON, 2)

        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if kp is not None and keypoint_scores[i] > 0.3:
                cv2.circle(frame, kp, 4, self.COLOR_KEYPOINTS, -1)

    def _get_exercise_color(self, exercise_type: str) -> Tuple[int, int, int]:
        """
        Get color for exercise type.
        """
        if exercise_type == "Bicep Curl":
            return (0, 165, 255)  # Orange
        elif exercise_type == "Lateral Raise":
            return (255, 0, 255)  # Magenta
        else:
            return (200, 200, 200)  # Gray

    def _render_frame(self,
                      frame: np.ndarray,
                      body_results: List[Dict],
                      face_body_matches: Dict[int, Dict],
                      fps: float) -> np.ndarray:
        """
        Render frame with exercise information.
        """
        output = frame.copy()

        # Draw each tracked person
        for body in body_results:
            track_id = body["track_id"]
            bbox = body["bbox"]
            reps = body["reps"]
            exercise_type = body["exercise_type"]
            bicep_angle = body.get("bicep_angle", 0)
            lateral_angle = body.get("lateral_angle", 0)

            x1, y1, x2, y2 = bbox

            # Check if this person is identified
            is_identified = track_id in face_body_matches
            face_info = face_body_matches.get(track_id, {})
            person_name = face_info.get("name", None)

            # Choose color based on identification status
            bbox_color = self.COLOR_BBOX_IDENTIFIED if is_identified else self.COLOR_BBOX_UNIDENTIFIED

            # Get exercise color
            exercise_color = self._get_exercise_color(exercise_type)

            # Draw body bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), bbox_color, 2)

            # Draw face bounding box if available
            if "face_bbox" in face_info:
                fx1, fy1, fx2, fy2 = face_info["face_bbox"]
                cv2.rectangle(output, (fx1, fy1), (fx2, fy2), self.COLOR_FACE_BBOX, 2)

            # Draw pose skeleton
            self._draw_skeleton(output, body["keypoints"], body["keypoint_scores"])

            # Prepare text labels
            if person_name:
                # Identified person
                label_name = f"{person_name}"
                label_exercise = f"Exercise: {exercise_type}"
                label_reps = f"Reps: {reps}"

                # Add angle information based on exercise
                if exercise_type == "Bicep Curl":
                    label_angle = f"Elbow Angle: {int(bicep_angle)}°"
                elif exercise_type == "Lateral Raise":
                    label_angle = f"Arm Angle: {int(lateral_angle)}°"
                else:
                    label_angle = ""

                # Draw name
                cv2.putText(output, label_name, (x1, y1 - 80),
                            self.FONT, self.FONT_SCALE_LARGE, (0, 255, 0),
                            self.FONT_THICKNESS + 1)

                # Draw exercise info with color
                cv2.putText(output, label_exercise, (x1, y1 - 55),
                            self.FONT, self.FONT_SCALE_MEDIUM, exercise_color,
                            self.FONT_THICKNESS)

                # Draw reps
                cv2.putText(output, label_reps, (x1, y1 - 30),
                            self.FONT, self.FONT_SCALE_MEDIUM, self.COLOR_TEXT,
                            self.FONT_THICKNESS)

                # Draw angle if available
                if label_angle:
                    cv2.putText(output, label_angle, (x1, y1 - 5),
                                self.FONT, self.FONT_SCALE_SMALL, (200, 200, 200),
                                self.FONT_THICKNESS - 1)

            else:
                # Unidentified person
                label_id = f"Person #{track_id}"
                label_exercise = f"Exercise: {exercise_type}"
                label_reps = f"Reps: {reps}"
                label_unknown = "Unknown"

                # Add angle information based on exercise
                if exercise_type == "Bicep Curl":
                    label_angle = f"Elbow Angle: {int(bicep_angle)}°"
                elif exercise_type == "Lateral Raise":
                    label_angle = f"Arm Angle: {int(lateral_angle)}°"
                else:
                    label_angle = ""

                cv2.putText(output, label_id, (x1, y1 - 80),
                            self.FONT, self.FONT_SCALE_LARGE, self.COLOR_BBOX_UNIDENTIFIED,
                            self.FONT_THICKNESS)

                cv2.putText(output, label_exercise, (x1, y1 - 55),
                            self.FONT, self.FONT_SCALE_MEDIUM, exercise_color,
                            self.FONT_THICKNESS)

                cv2.putText(output, label_reps, (x1, y1 - 30),
                            self.FONT, self.FONT_SCALE_MEDIUM, self.COLOR_TEXT,
                            self.FONT_THICKNESS)

                # Draw angle if available
                if label_angle:
                    cv2.putText(output, label_angle, (x1, y1 - 5),
                                self.FONT, self.FONT_SCALE_SMALL, (200, 200, 200),
                                self.FONT_THICKNESS - 1)

        # Draw FPS counter
        cv2.rectangle(output, (10, 10), (250, 100), (0, 0, 0), -1)
        cv2.putText(output, f"FPS: {fps:.1f}", (20, 40),
                    self.FONT, 0.8, (0, 255, 0), 2)

        # Draw system info
        info_text = f"Tracking: {len(body_results)} person(s)"
        cv2.putText(output, info_text, (20, 70),
                    self.FONT, 0.6, (255, 255, 255), 1)

        # Draw optimization info
        next_scan = max(0, self.face_recognition_interval -
                       (self.frame_count - self.last_face_recognition_frame))
        ident_count = len(self.identified_tracks)
        scan_text = f"Next scan: {next_scan}f | Identified: {ident_count}"
        cv2.putText(output, scan_text, (20, 90),
                    self.FONT, 0.5, (200, 200, 200), 1)

        return output

    def run(self, display: bool = True, save_output: Optional[str] = None):
        """
        Main processing loop.
        """
        # Setup video writer if saving
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, self.streamer.fps,
                                     (self.streamer.width, self.streamer.height))
            print(f"💾 Saving output to: {save_output}")

        # Setup display window
        if display:
            cv2.namedWindow("FitZone - Multiple Exercises", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("FitZone - Multiple Exercises",
                             self.streamer.width, self.streamer.height)

        print("▶️  MULTIPLE EXERCISE DETECTION STARTED...")
        print("   • ESC = Exit")
        print("   • R = Reset tracking")
        print("   • F = Force face recognition now\n")

        frame_count = 0
        start_time = time.time()
        last_fps_time = time.time()
        fps_frames = 0
        current_fps = 0

        # Start streamer thread
        self.streamer.start()

        try:
            while self.streamer.more():
                # Capture frame
                frame = self.streamer.read()
                if frame is None:
                    continue

                frame_count += 1
                loop_start = time.time()

                # ===== PROCESSING =====
                # Process frame with optimized logic
                body_results, face_body_matches = self._process_frame_optimized(frame)

                # Calculate FPS
                loop_time = time.time() - loop_start
                fps = 1.0 / loop_time if loop_time > 0 else 0

                # Update FPS counter
                fps_frames += 1
                if time.time() - last_fps_time >= 1.0:
                    current_fps = fps_frames / (time.time() - last_fps_time)
                    fps_frames = 0
                    last_fps_time = time.time()

                # ===== VISUALIZATION =====
                output_frame = self._render_frame(frame, body_results,
                                                  face_body_matches, current_fps)

                # Display
                if display:
                    cv2.imshow("FitZone - Multiple Exercises", output_frame)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        print("\n⏹️  Stopped by user.")
                        break
                    elif key == ord('r') or key == ord('R'):
                        self.exercise_detector.reset()
                        self.identified_tracks = {}
                        with self.recognition_lock:
                            self.current_face_results = []
                        print("\n🔄 Tracking reset.")
                    elif key == ord('f') or key == ord('F'):
                        # Force face recognition
                        with self.recognition_lock:
                            self.last_face_recognition_frame = 0
                        print("\n🔍 Forcing face recognition...")

                # Save to file
                if writer:
                    writer.write(output_frame)

                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed

                    # Calculate processing time stats
                    avg_process = np.mean(self.processing_times) * 1000 if self.processing_times else 0

                    # Count exercises
                    exercise_counts = {}
                    for body in body_results:
                        ex_type = body["exercise_type"]
                        if ex_type != "Unknown":
                            exercise_counts[ex_type] = exercise_counts.get(ex_type, 0) + 1

                    exercises_str = " | ".join([f"{k}: {v}" for k, v in exercise_counts.items()])

                    print(f"Frame {frame_count:5d} | "
                          f"FPS: {current_fps:5.1f} | "
                          f"Avg FPS: {avg_fps:5.1f} | "
                          f"Process: {avg_process:5.1f}ms | "
                          f"Bodies: {len(body_results)} | "
                          f"Exercises: {exercises_str if exercises_str else 'None'}")

        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user.")

        finally:
            # Cleanup
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0

            print("\n" + "=" * 70)
            print("📊 PROCESSING SUMMARY")
            print("=" * 70)
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Final FPS: {current_fps:.2f}")

            if self.processing_times:
                avg_process = np.mean(self.processing_times) * 1000
                print(f"Average frame processing: {avg_process:.1f}ms")

            # Calculate unique names identified
            unique_names = set(info["name"] for info in self.identified_tracks.values() if info.get("name"))
            print(f"Unique persons identified: {len(unique_names)}")
            if unique_names:
                print(f"   Named identities: {', '.join(unique_names)}")
            print(f"Total track IDs created: {len(self.identified_tracks)}")
            print("=" * 70 + "\n")

            self.streamer.stop()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

    def get_statistics(self) -> Dict:
        """Get current system statistics."""
        return {
            "active_tracks": len(self.exercise_detector.tracks),
            "identified_persons": len(self.identified_tracks),
            "known_persons": len(self.face_recognizer.database),
            "frame_width": self.frame_width,
            "frame_height": self.frame_height
        }


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="FitZone: Multiple Exercise Detection System"
    )

    parser.add_argument('--video', type=str, default=VIDEO_SOURCE,
                        help='Video file path, RTSP URL, or camera index')
    parser.add_argument('--model', type=str, default=MOVENET_MODEL_PATH,
                        help='Path to MoveNet MultiPose model directory')
    parser.add_argument('--db', type=str, default=FACE_DATABASE_PATH,
                        help='Path to face_db.pkl file')
    parser.add_argument('--output', type=str, default=OUTPUT_VIDEO_PATH,
                        help='Optional: path to save output video')
    parser.add_argument('--no-display', action='store_true',
                        help='Run without displaying video window')
    parser.add_argument('--face-interval', type=int, default=30,
                        help='Face recognition interval in frames')

    args = parser.parse_args()

    # Initialize and run system
    try:
        system = FitZoneSystem(
            movenet_model_path=args.model,
            face_database_path=args.db,
            video_source=args.video
        )

        # Set face recognition interval
        system.face_recognition_interval = args.face_interval

        system.run(
            display=not args.no_display,
            save_output=args.output if args.output != "None" else None
        )

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

# python main.py --video "YourVideo.mp4" --model "D:\Movenet multipose\Movenet multipose" --db "face_db.pkl"