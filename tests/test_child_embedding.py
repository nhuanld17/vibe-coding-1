"""Quick test to debug embedding extraction for child images."""

import sys
from pathlib import Path

# Add repo root to path (BE directory)
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Change to BE directory
import os
os.chdir(repo_root)

# Fix encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from models.face_detection import FaceDetector
from models.face_embedding import create_embedding_backend
from api.config import Settings
from utils.image_processing import normalize_image_orientation
import cv2
import numpy as np

def test_embedding_extraction(image_path: str):
    """Test embedding extraction for a single image."""
    print(f"Testing: {image_path}")
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"[ERROR] File not found: {image_path}")
        return None
    
    try:
        # Initialize
        settings = Settings()
        detector = FaceDetector()
        embedder = create_embedding_backend()
        
        # Read image
        image_bytes = Path(image_path).read_bytes()
        normalized_bytes = normalize_image_orientation(image_bytes)
        
        # Decode
        nparr = np.frombuffer(normalized_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"[ERROR] Failed to decode image")
            return None
        
        print(f"[OK] Image decoded, shape: {image.shape}")
        
        # Detect faces
        faces = detector.detect_faces(image, confidence_threshold=settings.face_confidence_threshold)
        print(f"[INFO] Faces detected: {len(faces)}")
        
        if not faces:
            print(f"[ERROR] No faces detected")
            return None
        
        # Use largest face
        main_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
        print(f"[OK] Using face with bbox: {main_face['bbox']}")
        
        # Align
        aligned = detector.align_face(image, main_face['keypoints'])
        if aligned is None:
            print(f"[ERROR] Face alignment failed")
            return None
        
        print(f"[OK] Face aligned, shape: {aligned.shape}")
        
        # Extract embedding
        embedding = embedder.extract_embedding(aligned)
        if embedding is None:
            print(f"[ERROR] Embedding extraction failed")
            return None
        
        print(f"[OK] Embedding extracted, shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}")
        return embedding
        
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_image = "datasets/FGNET_organized/person_037/age_04.jpg"
    result = test_embedding_extraction(test_image)
    if result is not None:
        print("\n[SUCCESS] Embedding extraction works!")
    else:
        print("\n[FAILED] Embedding extraction failed")

