"""Quick test to check if embedding extraction works."""

import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_processing import (
    load_image_from_bytes,
    normalize_image_orientation,
)
from models.face_detection import FaceDetector
from models.face_embedding import create_embedding_backend
from api.config import Settings

settings = Settings()

detector = FaceDetector()
embedder = create_embedding_backend(
    backend_type=settings.embedding_backend,
    model_path=settings.arcface_model_path,
    use_gpu=settings.use_gpu,
    model_name=settings.insightface_model_name,
)

img_path = "datasets/FGNET_organized/person_037/age_04.jpg"
print(f"Testing: {img_path}")

img_bytes = Path(img_path).read_bytes()
normalized_bytes = normalize_image_orientation(img_bytes)
image_bgr = load_image_from_bytes(normalized_bytes)

faces = detector.detect_faces(
    image_bgr, confidence_threshold=settings.face_confidence_threshold
)
print(f"Faces detected: {len(faces)}")

if faces:
    primary_face = max(faces, key=lambda f: f.get("area", 0.0))
    aligned = detector.align_face(image_bgr, primary_face["keypoints"])
    embedding = embedder.extract_embedding(aligned)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {float(np.linalg.norm(embedding)):.4f}")
    print("[OK] Embedding extraction works")
else:
    print("[ERROR] No faces detected")

