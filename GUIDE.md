# 📖 HƯỚNG DẪN SỬ DỤNG MISSING PERSON AI

## 🚀 BẮT ĐẦU NHANH

### 1. Download Model AI
```bash
python download_model.py
```

### 2. Khởi động Docker
```bash
docker-compose up -d
```

### 3. Kiểm tra
```bash
curl http://localhost:8000/health
```

### 4. Truy cập
- **Swagger UI**: http://localhost:8000/docs
- **API Root**: http://localhost:8000/

---

## 📋 UPLOAD ẢNH - 2 CÁCH

### Cách 1: Dùng Python Script (KHUYẾN NGHỊ)

**Upload người mất tích:**
```bash
python test_upload.py
```

**Upload người tìm thấy (tự động matching):**
```bash
python test_upload_found.py
```

**Hoặc tự viết script:**
```python
import requests
import json

metadata = {
    "case_id": "MISS_2023_001",
    "name": "Nguyen Van A",
    "age_at_disappearance": 25,
    "year_disappeared": 2023,
    "gender": "male",
    "location_last_seen": "Ha Noi",
    "contact": "test@example.com"
}

with open("image.jpg", 'rb') as f:
    response = requests.post(
        "http://localhost:8000/api/v1/upload/missing",
        files={'image': f},
        data={'metadata': json.dumps(metadata)}
    )
    
print(response.json())
```

### Cách 2: Swagger UI

⚠️ **LƯU Ý QUAN TRỌNG**: Metadata phải là **STRING JSON** (có dấu `"` bao ngoài)

**ĐÚNG:**
```
"{\"case_id\":\"TEST_001\",\"name\":\"Nguyen Van A\",\"age_at_disappearance\":25,\"year_disappeared\":2023,\"gender\":\"male\",\"location_last_seen\":\"Ha Noi\",\"contact\":\"test@example.com\"}"
```

**SAI:**
```json
{
  "case_id": "TEST_001",
  ...
}
```

---

## 📊 API ENDPOINTS

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/api/v1/upload/missing` | POST | Upload người mất tích |
| `/api/v1/upload/found` | POST | Upload người tìm thấy + auto matching |
| `/api/v1/search/missing/{case_id}` | GET | Tìm kiếm theo case_id |
| `/api/v1/search/found/{found_id}` | GET | Tìm kiếm theo found_id |
| `/health` | GET | Health check |

---

## 📝 METADATA FIELDS

### Missing Person (Bắt buộc)
```json
{
  "case_id": "MISS_2023_001",           // 3-50 ký tự
  "name": "Nguyen Van A",                // Ít nhất 2 ký tự
  "age_at_disappearance": 25,            // 0-120
  "year_disappeared": 2023,              // 1900-2025
  "gender": "male",                      // male/female/other/unknown
  "location_last_seen": "Ha Noi",        // Ít nhất 3 ký tự
  "contact": "test@example.com"          // Email hoặc số điện thoại
}
```

### Found Person (Bắt buộc)
```json
{
  "found_id": "FOUND_001",
  "current_age_estimate": 30,
  "gender": "male",
  "current_location": "TP HCM",
  "finder_contact": "finder@example.com"
}
```

---

## 🎯 WORKFLOW SỬ DỤNG

### Scenario: Tìm người mất tích

**Bước 1:** Upload ảnh người mất tích
```bash
python test_upload.py
# Sửa metadata trong file trước khi chạy
```

**Bước 2:** Khi tìm thấy người nghi ngờ
```bash
python test_upload_found.py
```

**Bước 3:** Xem kết quả
```json
{
  "potential_matches": [
    {
      "face_similarity": 0.88,           // 88% giống nhau
      "confidence_level": "HIGH",         // Mức độ tin cậy cao
      "confidence_score": 0.85,
      "contact": "0912345678",            // Liên hệ ngay!
      "explanation": {
        "summary": "Rất có thể đúng người",
        "recommendations": ["Liên hệ gia đình ngay"]
      }
    }
  ]
}
```

**Quyết định:**
- **Confidence > 0.9**: Rất chắc chắn → Liên hệ ngay
- **Confidence 0.7-0.9**: Khả năng cao → Xác minh thêm
- **Confidence < 0.7**: Cần kiểm tra kỹ

---

## 🧪 TEST VỚI FG-NET DATASET

FG-NET là dataset với ảnh của cùng người qua nhiều năm (age progression).

**Organize dataset:**
```bash
python organize_fgnet.py
```

**Kết quả test:**
- ✅ Upload ảnh tuổi 2
- ✅ Upload ảnh tuổi 22 (cùng người, 20 năm sau)
- ✅ **Face Similarity: 88%** - Xuất sắc!
- ✅ Matching thành công dù age gap 20 năm

---

## 🔧 DOCKER COMMANDS

```bash
# Xem trạng thái
docker-compose ps

# Xem logs
docker-compose logs api -f

# Restart
docker-compose restart api

# Dừng tất cả
docker-compose down

# Xóa data và rebuild
docker-compose down -v
docker-compose build
docker-compose up -d
```

---

## 💡 TROUBLESHOOTING

### API không chạy
```bash
docker-compose logs api --tail 50
docker-compose restart api
```

### Model không load được
```bash
# Re-download model
python download_model.py

# Kiểm tra file
ls -l models/weights/arcface_r100_v1.onnx
# Phải là ~249MB
```

### Upload lỗi "Invalid JSON"
- Nếu dùng Swagger UI: Metadata phải là string JSON (có `"` bao ngoài)
- **Khuyến nghị**: Dùng Python script thay vì Swagger UI

### Không phát hiện được khuôn mặt
- Kiểm tra ảnh có khuôn mặt rõ ràng
- Ảnh không bị mờ, tối
- Khuôn mặt không bị che khuất

---

## 📈 PERFORMANCE

- Face Detection: ~200ms
- Embedding Extraction: ~150ms
- Vector Search: <100ms
- **Total**: ~500-800ms/image

---

## 🎓 TÀI LIỆU THAM KHẢO

- [ArcFace Paper](https://arxiv.org/abs/1801.07698)
- [MTCNN Paper](https://arxiv.org/abs/1604.02878)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**🎉 Chúc bạn sử dụng thành công!**

Để được hỗ trợ, vui lòng mở issue trên GitHub hoặc xem tài liệu API tại `/docs`.

