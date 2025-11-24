"""Quick test to check if embedding extraction works"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import load_image_from_bytes, normalize_image_orientation
from services.face_detector import FaceDetector
from services.embedding_extractor import create_embedding_backend
from api.config import Settings
import cv2

settings = Settings()
detector = FaceDetector()
embedder = create_embedding_backend(settings)

img_path = 'datasets/FGNET_organized/person_037/age_04.jpg'
print(f'Testing: {img_path}')

img_bytes = Path(img_path).read_bytes()
img = normalize_image_orientation(img_bytes)
img_bgr = cv2.imdecode(img, cv2.IMREAD_COLOR)

faces = detector.detect_faces(img_bgr, settings.face_confidence_threshold)
print(f'Faces detected: {len(faces)}')

if faces:
    aligned = detector.align_face(img_bgr, faces[0].landmarks)
    emb = embedder.extract_embedding(aligned)
    print(f'Embedding shape: {emb.shape}')
    print('[OK] Embedding extraction works')
else:
    print('[ERROR] No faces detected')

