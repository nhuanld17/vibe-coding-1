"""Test similarity với các cặp ảnh có độ tuổi gần nhau vs xa nhau."""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix encoding for Windows console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
from models.face_detection import FaceDetector
from models.face_embedding import create_embedding_backend
from utils.image_processing import load_image_from_bytes, normalize_image_orientation
from api.config import Settings
from loguru import logger

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def extract_embedding(image_path, detector, embedder, settings):
    """Extract embedding from image."""
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_bytes = normalize_image_orientation(image_bytes)
        image = load_image_from_bytes(image_bytes)
        if image is None:
            return None
        faces = detector.detect_faces(image, confidence_threshold=settings.face_confidence_threshold)
        if not faces:
            return None
        landmarks = faces[0]['keypoints']
        aligned = detector.align_face(image, landmarks, output_size=(112, 112))
        embedding = embedder.extract_embedding(aligned)
        return embedding
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None

def main():
    print("=" * 80)
    print("TEST SIMILARITY THEO KHOANG CACH TUOI")
    print("=" * 80)
    
    settings = Settings()
    detector = FaceDetector(min_face_size=40, device="CPU:0")
    embedder = create_embedding_backend(
        backend_type=settings.embedding_backend,
        use_gpu=settings.use_gpu,
        model_name=settings.insightface_model_name
    )
    
    # Test với person_001 - các cặp ảnh với khoảng cách tuổi khác nhau
    person_id = "person_001"
    base_path = f"datasets/FGNET_organized/{person_id}"
    
    test_pairs = [
        # (image1, image2, age_gap, description)
        ("age_02.jpg", "age_05.jpg", 3, "Tuổi gần nhau (2 vs 5 tuổi, gap 3 năm)"),
        ("age_05.jpg", "age_08.jpg", 3, "Tuổi gần nhau (5 vs 8 tuổi, gap 3 năm)"),
        ("age_08.jpg", "age_10.jpg", 2, "Tuổi rất gần nhau (8 vs 10 tuổi, gap 2 năm)"),
        ("age_14.jpg", "age_16.jpg", 2, "Tuổi rất gần nhau (14 vs 16 tuổi, gap 2 năm)"),
        ("age_16.jpg", "age_18.jpg", 2, "Tuổi rất gần nhau (16 vs 18 tuổi, gap 2 năm)"),
        ("age_18.jpg", "age_19.jpg", 1, "Tuổi cực gần nhau (18 vs 19 tuổi, gap 1 năm)"),
        ("age_22.jpg", "age_28.jpg", 6, "Tuổi xa nhau (22 vs 28 tuổi, gap 6 năm)"),
        ("age_28.jpg", "age_33.jpg", 5, "Tuổi xa nhau (28 vs 33 tuổi, gap 5 năm)"),
        ("age_02.jpg", "age_14.jpg", 12, "Tuổi rất xa nhau (2 vs 14 tuổi, gap 12 năm)"),
        ("age_02.jpg", "age_33.jpg", 31, "Tuổi cực xa nhau (2 vs 33 tuổi, gap 31 năm)"),
    ]
    
    print(f"\nTesting với {person_id}:\n")
    
    results = {
        'near': [],  # gap <= 3 years
        'medium': [],  # gap 4-10 years
        'far': []  # gap > 10 years
    }
    
    for img1, img2, gap, desc in test_pairs:
        path1 = os.path.join(base_path, img1)
        path2 = os.path.join(base_path, img2)
        
        if not os.path.exists(path1) or not os.path.exists(path2):
            print(f"[SKIP] {desc}: File not found")
            continue
        
        emb1 = extract_embedding(path1, detector, embedder, settings)
        emb2 = extract_embedding(path2, detector, embedder, settings)
        
        if emb1 is None or emb2 is None:
            print(f"[SKIP] {desc}: Failed to extract embeddings")
            continue
        
        similarity = np.dot(emb1, emb2)
        
        # Categorize by age gap
        if gap <= 3:
            results['near'].append(similarity)
            category = "GẦN"
        elif gap <= 10:
            results['medium'].append(similarity)
            category = "XA"
        else:
            results['far'].append(similarity)
            category = "RẤT XA"
        
        print(f"{category:8s} | Gap {gap:2d} năm | Similarity: {similarity:.4f} | {desc}")
    
    print("\n" + "=" * 80)
    print("TỔNG KẾT")
    print("=" * 80)
    
    if results['near']:
        print(f"\nTuổi GẦN NHAU (gap <= 3 năm):")
        print(f"  Count: {len(results['near'])}")
        print(f"  Mean:  {np.mean(results['near']):.4f}")
        print(f"  Min:   {np.min(results['near']):.4f}")
        print(f"  Max:   {np.max(results['near']):.4f}")
        print(f"  Median:{np.median(results['near']):.4f}")
    
    if results['medium']:
        print(f"\nTuổi XA NHAU (gap 4-10 năm):")
        print(f"  Count: {len(results['medium'])}")
        print(f"  Mean:  {np.mean(results['medium']):.4f}")
        print(f"  Min:   {np.min(results['medium']):.4f}")
        print(f"  Max:   {np.max(results['medium']):.4f}")
        print(f"  Median:{np.median(results['medium']):.4f}")
    
    if results['far']:
        print(f"\nTuổi RẤT XA NHAU (gap > 10 năm):")
        print(f"  Count: {len(results['far'])}")
        print(f"  Mean:  {np.mean(results['far']):.4f}")
        print(f"  Min:   {np.min(results['far']):.4f}")
        print(f"  Max:   {np.max(results['far']):.4f}")
        print(f"  Median:{np.median(results['far']):.4f}")
    
    print("\n" + "=" * 80)
    print("KẾT LUẬN")
    print("=" * 80)
    
    if results['near']:
        near_mean = np.mean(results['near'])
        print(f"\n[OK] Với ảnh có tuổi GẦN NHAU: similarity trung bình = {near_mean:.4f}")
        if near_mean >= 0.6:
            print("      -> Rất tốt! Có thể match được dễ dàng")
        elif near_mean >= 0.5:
            print("      -> Tốt! Vẫn match được")
        else:
            print("      -> Cần kiểm tra thêm")
    
    if results['far']:
        far_mean = np.mean(results['far'])
        print(f"\n[INFO] Với ảnh có tuổi XA NHAU: similarity trung bình = {far_mean:.4f}")
        print(f"       -> Thấp hơn nhưng vẫn cao hơn different-person (0.09)")

if __name__ == "__main__":
    main()

