import os
import cv2
import pickle
import numpy as np
import sys
import io
from insightface.app import FaceAnalysis

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'faces'))
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'face_db.pkl'))

database = {}

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f"No face found in {img_path}")
            continue

        # Take largest face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        embeddings.append(face.embedding)

    if len(embeddings) > 0:
        database[person_name] = np.mean(embeddings, axis=0)
        print(f"Saved embeddings for {person_name}")

with open(DB_PATH, "wb") as f:
    pickle.dump(database, f)

print("✅ Face database created successfully")
