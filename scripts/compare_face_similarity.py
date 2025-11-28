"""
Script tiện dụng để tính độ tương đồng giữa 2 ảnh mặt bằng pipeline ArcFace
production của dự án.

Chạy:
    cd BE
    python scripts/compare_face_similarity.py <anh1> <anh2>

Ảnh có thể đưa vào bằng đường dẫn tuyệt đối hoặc tương đối so với thư mục BE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

# Bảo đảm import được module nội bộ
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from api.config import Settings  # type: ignore
from models.face_detection import FaceDetector  # type: ignore
from models.face_embedding import BaseFaceEmbedder, create_embedding_backend  # type: ignore
from utils.image_processing import load_image_from_bytes, normalize_image_orientation  # type: ignore


def extract_embedding(
    image_path: Path,
    detector: FaceDetector,
    embedder: BaseFaceEmbedder,
    settings: Settings,
) -> Optional[np.ndarray]:
    """Trích embedding ArcFace cho một ảnh duy nhất."""
    try:
        with image_path.open("rb") as f:
            image_bytes = normalize_image_orientation(f.read())

        image = load_image_from_bytes(image_bytes)
        if image is None:
            logger.error(f"Không decode được ảnh: {image_path}")
            return None

        faces = detector.detect_faces(
            image,
            confidence_threshold=settings.face_confidence_threshold,
        )
        if not faces:
            logger.error(f"Không tìm thấy khuôn mặt trong ảnh: {image_path}")
            return None

        landmarks = faces[0]["keypoints"]
        aligned = detector.align_face(image, landmarks, output_size=(112, 112))
        embedding = embedder.extract_embedding(aligned)
        return embedding
    except Exception as exc:
        logger.exception(f"Lỗi khi xử lý ảnh {image_path}: {exc}")
        return None


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Tính cosine similarity giữa hai vector đã L2 normalize."""
    return float(np.dot(emb1, emb2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="So sánh độ tương đồng khuôn mặt giữa 2 ảnh bằng ArcFace.",
    )
    parser.add_argument("image_a", type=Path, help="Ảnh thứ nhất")
    parser.add_argument("image_b", type=Path, help="Ảnh thứ hai")
    parser.add_argument(
        "--device",
        default="CPU:0",
        help="Thiết bị cho FaceDetector (ví dụ CPU:0 hoặc GPU:0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    img_paths = [args.image_a, args.image_b]
    img_paths = [p if p.is_absolute() else (ROOT_DIR / p) for p in img_paths]

    for p in img_paths:
        if not p.exists():
            print(f"Không tìm thấy file: {p}")
            return 1

    print("=" * 80)
    print("SO SANH DO TUONG DONG KHUON MAT (ArcFace)")
    print("=" * 80)

    settings = Settings()
    detector = FaceDetector(min_face_size=40, device=args.device)
    embedder = create_embedding_backend(
        backend_type=settings.embedding_backend,
        use_gpu=settings.use_gpu,
        model_name=(
            settings.insightface_model_name
            if settings.embedding_backend == "insightface"
            else None
        ),
        model_path=(
            settings.arcface_model_path
            if settings.embedding_backend == "onnx"
            else None
        ),
    )

    embeddings = []
    for path in img_paths:
        print(f"\nĐang xử lý: {path}")
        emb = extract_embedding(path, detector, embedder, settings)
        if emb is None:
            print("Không thể trích embedding cho ảnh này, dừng lại.")
            return 1
        embeddings.append(emb)

    cos = cosine_similarity(embeddings[0], embeddings[1])
    scaled_sim = (cos + 1.0) / 2.0

    print("\nKẾT QUẢ:")
    print(f"- Cosine similarity (ArcFace): {cos:.4f}")
    print(f"- Similarity chuẩn hóa [0,1]: {scaled_sim:.4f}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

