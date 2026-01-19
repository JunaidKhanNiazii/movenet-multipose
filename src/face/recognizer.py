"""
face_recog.py
=============
Face detection and recognition module using InsightFace.

Responsibilities:
- Face detection in frames
- Face embedding extraction
- Face recognition via cosine similarity
- Loading pre-computed face database

Does NOT handle:
- Pose estimation
- Exercise tracking
- Video I/O or display
"""

import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from insightface.app import FaceAnalysis


class FaceRecognizer:
    """
    Detects and recognizes faces using InsightFace (ArcFace embeddings).

    Matches detected faces against a pre-computed database using cosine similarity.
    """

    # ===== CONFIGURATION =====
    RECOGNITION_THRESHOLD = 0.4  # Cosine similarity threshold for recognition
    DETECTION_SIZE = (640, 640)  # Input size for face detection

    def __init__(self, database_path: str, model_name: str = 'buffalo_l'):
        """
        Initialize face recognizer.

        Args:
            database_path: Path to face_db.pkl containing name->embedding mapping
            model_name: InsightFace model to use (default: buffalo_l)
        """
        print(f"🔹 [FaceRecognizer] Initializing InsightFace ({model_name})...")

        # Initialize InsightFace
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=self.DETECTION_SIZE)

        print(f"✅ [FaceRecognizer] InsightFace initialized")

        # Load face database
        print(f"🔹 [FaceRecognizer] Loading face database from {database_path}...")
        self.database = self._load_database(database_path)
        print(f"✅ [FaceRecognizer] Loaded {len(self.database)} identities")

    def _load_database(self, database_path: str) -> Dict[str, np.ndarray]:
        """
        Load face embedding database from pickle file.

        Args:
            database_path: Path to pickle file

        Returns:
            Dictionary mapping person names to normalized embeddings

        Raises:
            FileNotFoundError: If database file doesn't exist
            ValueError: If database format is invalid
        """
        try:
            with open(database_path, 'rb') as f:
                database = pickle.load(f)

            # Validate database format
            if not isinstance(database, dict):
                raise ValueError("Database must be a dictionary")

            # Validate embeddings
            for name, embedding in database.items():
                if not isinstance(embedding, np.ndarray):
                    raise ValueError(f"Embedding for {name} must be numpy array")
                if embedding.shape != (512,):
                    raise ValueError(f"Embedding for {name} must be shape (512,), got {embedding.shape}")

                # Ensure embedding is normalized
                norm = np.linalg.norm(embedding)
                if not np.isclose(norm, 1.0, atol=1e-5):
                    print(f"⚠️  [FaceRecognizer] Normalizing embedding for {name} (norm was {norm:.4f})")
                    database[name] = embedding / norm

            return database

        except FileNotFoundError:
            raise FileNotFoundError(f"Face database not found at {database_path}")
        except Exception as e:
            raise ValueError(f"Failed to load face database: {str(e)}")

    @staticmethod
    def _cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1, embedding2: Normalized embedding vectors

        Returns:
            Cosine similarity in range [-1, 1], where 1 is identical
        """
        return np.dot(embedding1, embedding2)

    def _recognize_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face by matching against database.

        Args:
            embedding: Face embedding from InsightFace

        Returns:
            Tuple of (name, similarity) or (None, 0) if no match above threshold
        """
        if len(self.database) == 0:
            return None, 0.0

        # Normalize query embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Find best match
        best_similarity = -1.0
        best_name = None

        for name, db_embedding in self.database.items():
            similarity = self._cosine_similarity(embedding, db_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name

        # Check if similarity meets threshold
        if best_similarity >= self.RECOGNITION_THRESHOLD:
            return best_name, best_similarity
        else:
            return None, best_similarity

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and recognize faces in a frame.

        This is the main public API method.

        Args:
            frame: BGR image from OpenCV (any resolution)

        Returns:
            List of dictionaries, each containing:
            {
                "name": str or None,       # Recognized name or None if unknown
                "bbox": (x1, y1, x2, y2),  # Face bounding box in pixels
                "confidence": float,       # Detection confidence (0-1)
                "similarity": float,       # Recognition similarity (0-1, or 0 if unknown)
                "embedding": np.ndarray    # 512-d face embedding
            }
        """
        # Detect faces and extract embeddings
        faces = self.app.get(frame)

        results = []
        for face in faces:
            # Extract bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Get embedding
            embedding = face.embedding

            # Recognize face
            name, similarity = self._recognize_face(embedding)

            results.append({
                "name": name,
                "bbox": (x1, y1, x2, y2),
                "confidence": face.det_score,
                "similarity": similarity,
                "embedding": embedding
            })

        return results

    def add_to_database(self, name: str, embedding: np.ndarray):
        """
        Add or update a person in the database.

        Args:
            name: Person's name
            embedding: Face embedding (will be normalized)
        """
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        self.database[name] = embedding
        print(f"➕ [FaceRecognizer] Added/updated {name} in database")

    def remove_from_database(self, name: str) -> bool:
        """
        Remove a person from the database.

        Args:
            name: Person's name to remove

        Returns:
            True if removed, False if not found
        """
        if name in self.database:
            del self.database[name]
            print(f"➖ [FaceRecognizer] Removed {name} from database")
            return True
        return False

    def save_database(self, output_path: str):
        """
        Save current database to file.

        Args:
            output_path: Path to save pickle file
        """
        with open(output_path, 'wb') as f:
            pickle.dump(self.database, f)
        print(f"💾 [FaceRecognizer] Database saved to {output_path}")

    def get_database_names(self) -> List[str]:
        """
        Get list of all names in database.

        Returns:
            List of person names
        """
        return list(self.database.keys())