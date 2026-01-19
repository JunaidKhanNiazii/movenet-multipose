"""
pose_estimator.py - MoveNet Pose Estimation
============================================
Handles MoveNet model loading and pose inference.
"""

import tensorflow as tf
import numpy as np
import cv2


class PoseEstimator:
    """
    Handles MoveNet MultiPose model loading and inference.
    Provides pose estimation for multiple people in a frame.
    """
    
    INPUT_SIZE = 256
    
    def __init__(self, model_path: str):
        """
        Initialize the pose estimator with MoveNet model.
        
        Args:
            model_path: Path to the SavedModel directory, .tflite, or .onnx file
        """
        self.model_path = model_path
        
        if model_path.endswith('.onnx'):
            import onnxruntime as ort
            print(f"🔹 [PoseEstimator] Loading MoveNet MultiPose ONNX model: {model_path}")
            # Use CPU by default for stability, or DirectML/CUDA if available
            providers = ['CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(model_path, providers=providers)
            self.model_type = 'onnx'
        elif model_path.endswith('.tflite'):
            print(f"🔹 [PoseEstimator] Loading MoveNet MultiPose TFLite model: {model_path}")
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model_type = 'tflite'
        else:
            print(f"🔹 [PoseEstimator] Loading MoveNet MultiPose SavedModel: {model_path}")
            self.model = tf.saved_model.load(model_path)
            self.movenet = self.model.signatures['serving_default']
            self.model_type = 'tf'
            
        print("✅ [PoseEstimator] Model loaded successfully!")
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for MoveNet inference.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Preprocessed array ready for inference
        """
        img = cv2.resize(frame, (self.INPUT_SIZE, self.INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.model_type == 'onnx':
            # ONNX usually expects float32 or int32 depending on export
            # Our movenet_multipose.onnx usually expects int32 [1, 256, 256, 3]
            return np.expand_dims(img.astype(np.int32), axis=0)
        elif self.model_type == 'tflite':
            return np.expand_dims(img.astype(self.input_details[0]['dtype']), axis=0)
        else:
            return np.expand_dims(img.astype(np.int32), axis=0)
    
    def infer_poses(self, frame: np.ndarray) -> np.ndarray:
        """
        Run MoveNet inference on frame.
        """
        input_data = self._preprocess_frame(frame)
        
        if self.model_type == 'onnx':
            input_name = self.ort_session.get_inputs()[0].name
            output = self.ort_session.run(None, {input_name: input_data})[0][0]
        elif self.model_type == 'tflite':
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            input_tensor = tf.convert_to_tensor(input_data)
            output = self.movenet(input_tensor)['output_0'].numpy()[0]
            
        return output
