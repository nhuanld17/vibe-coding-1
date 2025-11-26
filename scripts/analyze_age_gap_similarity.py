"""
Phân tích ảnh hưởng của khoảng cách tuổi (age gap) tới độ giống nhau khuôn mặt
trên toàn bộ FGNET_organized/test_pairs.txt.

Chạy:
    cd BE
    python scripts/analyze_age_gap_similarity.py

Yêu cầu:
    - Đã chuẩn bị FGNET_organized và file:
        datasets/FGNET_organized/test_pairs.txt
      với định dạng:
        img1,img2,label,age_gap
      trong đó:
        - img1, img2: đường dẫn tương đối bên trong FGNET_organized (dùng \ hoặc /)
        - label: 1 = cùng người, 0 = khác người
        - age_gap: số năm chênh lệch tuổi giữa hai ảnh

Script sẽ:
    1. Đọc toàn bộ các cặp label=1 (cùng người).
    2. Dùng pipeline production: MTCNN + align_face + InsightFace (create_embedding_backend)
       để trích xuất embedding cho mỗi ảnh (có cache theo đường dẫn).
    3. Tính cosine similarity cho từng cặp.
    4. Gom nhóm theo bucket khoảng cách tuổi:
         - 0–5 năm
         - 6–10 năm
         - 11–20 năm
         - 21–30 năm
         - 31–40 năm
         - >40 năm
    5. In thống kê cho từng bucket: số cặp, mean / min / max / median / std.
"""

import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from loguru import logger

# Thêm root của repo vào sys.path để import được các module nội bộ
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.face_detection import FaceDetector  # type: ignore
from models.face_embedding import create_embedding_backend, BaseFaceEmbedder  # type: ignore
from utils.image_processing import load_image_from_bytes, normalize_image_orientation  # type: ignore
from api.config import Settings  # type: ignore


TEST_PAIRS_PATH = ROOT_DIR / "datasets" / "FGNET_organized" / "test_pairs.txt"
FGNET_ROOT = ROOT_DIR / "datasets" / "FGNET_organized"


def load_test_pairs(path: Path) -> List[Dict]:
    """Đọc toàn bộ test_pairs.txt và chỉ lấy các cặp cùng người (label=1)."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file test_pairs: {path}")

    pairs: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                label = int(row["label"])
                if label != 1:
                    continue  # chỉ quan tâm same-person
                img1_rel = row["img1"].strip().replace("\\", "/")
                img2_rel = row["img2"].strip().replace("\\", "/")
                age_gap = int(row.get("age_gap", "0"))
            except Exception as e:
                logger.warning(f"Lỗi parse dòng {row}: {e}")
                continue

            img1_path = FGNET_ROOT / img1_rel
            img2_path = FGNET_ROOT / img2_rel

            if not img1_path.exists() or not img2_path.exists():
                logger.warning(f"Bỏ qua cặp vì thiếu file: {img1_path} hoặc {img2_path}")
                continue

            pairs.append(
                {
                    "img1": img1_path,
                    "img2": img2_path,
                    "age_gap": age_gap,
                }
            )

    logger.info(f"Đã load {len(pairs)} cặp cùng người từ {path}")
    return pairs


def extract_embedding(
    image_path: Path,
    detector: FaceDetector,
    embedder: BaseFaceEmbedder,
    settings: Settings,
) -> Optional[np.ndarray]:
    """Trích xuất embedding từ một ảnh (dùng pipeline production)."""
    try:
        with image_path.open("rb") as f:
            image_bytes = f.read()

        # Chuẩn hóa orientation và decode BGR
        image_bytes = normalize_image_orientation(image_bytes)
        image = load_image_from_bytes(image_bytes)
        if image is None:
            logger.warning(f"Không decode được ảnh: {image_path}")
            return None

        # Detect face
        faces = detector.detect_faces(
            image, confidence_threshold=settings.face_confidence_threshold
        )
        if not faces:
            logger.warning(f"Không tìm thấy mặt trong ảnh: {image_path}")
            return None

        main_face = faces[0]
        landmarks = main_face["keypoints"]

        # Align 112x112
        aligned = detector.align_face(image, landmarks, output_size=(112, 112))

        # Embedding + L2 normalize (embedder đã normalize)
        emb = embedder.extract_embedding(aligned)
        return emb
    except Exception as e:
        logger.error(f"Lỗi khi xử lý {image_path}: {e}")
        return None


def compute_embeddings_cache(
    pairs: List[Dict],
    detector: FaceDetector,
    embedder: BaseFaceEmbedder,
    settings: Settings,
) -> Dict[Path, Optional[np.ndarray]]:
    """Tính embedding cho tất cả ảnh xuất hiện trong danh sách cặp (có cache)."""
    unique_paths = set()  # type: ignore[var-annotated]
    for p in pairs:
        unique_paths.add(p["img1"])
        unique_paths.add(p["img2"])

    cache: Dict[Path, Optional[np.ndarray]] = {}
    total = len(unique_paths)
    logger.info(f"Đang trích xuất embedding cho {total} ảnh (FGNET)...")

    for idx, img_path in enumerate(sorted(unique_paths)):
        emb = extract_embedding(img_path, detector, embedder, settings)
        cache[img_path] = emb
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            logger.info(f"  Đã xử lý {idx + 1}/{total} ảnh...")

    ok = sum(1 for v in cache.values() if v is not None)
    fail = total - ok
    logger.info(f"Embedding: {ok} ảnh thành công, {fail} ảnh lỗi/không thấy mặt")
    return cache


def bucket_for_age_gap(age_gap: int) -> str:
    """Xác định bucket cho khoảng cách tuổi (năm)."""
    if age_gap <= 5:
        return "0-5"
    if age_gap <= 10:
        return "6-10"
    if age_gap <= 20:
        return "11-20"
    if age_gap <= 30:
        return "21-30"
    if age_gap <= 40:
        return "31-40"
    return "41+"


def analyze_similarity_by_age_gap(
    pairs: List[Dict], cache: Dict[Path, Optional[np.ndarray]]
) -> None:
    """Tính cosine similarity cho từng cặp và thống kê theo bucket age gap."""
    buckets: Dict[str, List[float]] = {
        "0-5": [],
        "6-10": [],
        "11-20": [],
        "21-30": [],
        "31-40": [],
        "41+": [],
    }

    skipped = 0
    all_sims: List[Tuple[int, float]] = []

    for p in pairs:
        img1 = p["img1"]
        img2 = p["img2"]
        age_gap = p["age_gap"]

        emb1 = cache.get(img1)
        emb2 = cache.get(img2)
        if emb1 is None or emb2 is None:
            skipped += 1
            continue

        sim = float(np.dot(emb1, emb2))
        all_sims.append((age_gap, sim))

        b = bucket_for_age_gap(age_gap)
        buckets[b].append(sim)

    logger.info(
        f"Tong so cap same-person: {len(pairs)}, dung duoc {len(all_sims)}, bo qua {skipped}"
    )

    print("\n" + "=" * 80)
    print("THONG KE SIMILARITY THEO KHOANG CACH TUOI (FGNET TOAN BO)")
    print("=" * 80)

    def fmt_stats(vals: List[float]) -> str:
        arr = np.array(vals)
        return (
            f"count={len(vals)}, "
            f"mean={arr.mean():.4f}, "
            f"std={arr.std():.4f}, "
            f"min={arr.min():.4f}, "
            f"p25={np.percentile(arr,25):.4f}, "
            f"median={np.median(arr):.4f}, "
            f"p75={np.percentile(arr,75):.4f}, "
            f"max={arr.max():.4f}"
        )

    # In tung bucket
    ordered_keys = ["0-5", "6-10", "11-20", "21-30", "31-40", "41+"]
    for key in ordered_keys:
        vals = buckets[key]
        print(f"\nKhoang cach tuoi {key} nam:")
        if not vals:
            print("  (khong co cap nao)")
        else:
            print("  " + fmt_stats(vals))

    # Thong ke chung theo age_gap tho (optional)
    if all_sims:
        gaps = np.array([g for g, _ in all_sims])
        sims = np.array([s for _, s in all_sims])
        print("\n" + "=" * 80)
        print("TONG KET TOAN BO CAP SAME-PERSON:")
        print("=" * 80)
        print(f"Số cặp: {len(all_sims)}")
        print(f"Age gap: min={gaps.min()} năm, max={gaps.max()} năm, mean={gaps.mean():.2f}, median={np.median(gaps):.2f}")
        print(f"Similarity: mean={sims.mean():.4f}, std={sims.std():.4f}, min={sims.min():.4f}, max={sims.max():.4f}")


def main() -> int:
    print("=" * 80)
    # Dung tieng Viet khong dau de tranh loi encoding tren Windows console mac dinh
    print("PHAN TICH SIMILARITY THEO KHOANG CACH TUOI (FGNET_organized/test_pairs.txt)")
    print("=" * 80)

    # 1. Load cap same-person
    pairs = load_test_pairs(TEST_PAIRS_PATH)
    if not pairs:
        print("Không có cặp same-person nào sau khi lọc. Thoát.")
        return 1

    # 2. Khởi tạo model
    print("\nKhoi tao mo hinh (FaceDetector + InsightFace)...")
    settings = Settings()
    detector = FaceDetector(min_face_size=40, device="CPU:0")
    embedder = create_embedding_backend(
        backend_type=settings.embedding_backend,
        use_gpu=settings.use_gpu,
        model_name=settings.insightface_model_name if settings.embedding_backend == "insightface" else None,
        model_path=settings.arcface_model_path if settings.embedding_backend == "onnx" else None,
    )
    print("Khoi tao xong.\n")

    # 3. Tính cache embedding
    cache = compute_embeddings_cache(pairs, detector, embedder, settings)

    # 4. Phân tích theo age gap
    analyze_similarity_by_age_gap(pairs, cache)

    print("\nHoàn tất phân tích.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


