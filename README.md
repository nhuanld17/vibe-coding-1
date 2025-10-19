# 🔍 Missing Person AI - Face Recognition System

Hệ thống AI nhận diện khuôn mặt để tìm kiếm người mất tích, sử dụng ArcFace và MTCNN.

## ✨ Tính năng chính

- ✅ **Face Detection** - Phát hiện khuôn mặt tự động (MTCNN)
- ✅ **Face Embedding** - Tạo vector đặc trưng khuôn mặt (ArcFace R100)
- ✅ **Vector Search** - Tìm kiếm similarity siêu nhanh (Qdrant)
- ✅ **Bilateral Matching** - So khớp 2 chiều tự động
- ✅ **Confidence Scoring** - Đánh giá độ tin cậy chi tiết
- ✅ **Age Progression** - Nhận diện qua nhiều năm (test với FG-NET dataset)

## 🚀 Khởi động nhanh

### 1. Yêu cầu
- Docker Desktop
- Python 3.11+

### 2. Download model AI
```bash
python download_model.py
```

### 3. Khởi động với Docker
```bash
docker-compose up -d
```

### 4. Truy cập API
- **Swagger UI**: http://localhost:8000/docs
- **API Root**: http://localhost:8000/
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📖 Hướng dẫn sử dụng

### Upload người mất tích

**Dùng Python:**
```bash
python test_upload.py
```

**Hoặc tự viết:**
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

### Upload người tìm thấy (Tự động matching)

```bash
python test_upload_found.py
```

Hệ thống sẽ **TỰ ĐỘNG** so sánh với tất cả người mất tích và trả về:
- Face similarity score
- Confidence level
- Contact information
- Recommended actions

## 🏗️ Cấu trúc project

```
AI/
├── api/                    # FastAPI application
│   ├── routes/            # API endpoints
│   ├── schemas/           # Pydantic models
│   └── config.py          # Configuration
├── models/                # AI models
│   ├── face_detection.py  # MTCNN detector
│   ├── face_embedding.py  # ArcFace extractor
│   └── weights/           # Model weights (249MB)
├── services/              # Business logic
│   ├── vector_db.py       # Qdrant integration
│   ├── bilateral_search.py # Matching logic
│   └── confidence_scoring.py # Scoring system
├── utils/                 # Utilities
├── tests/                 # Unit tests
├── datasets/              # FG-NET organized data
├── docker-compose.yml     # Docker config
├── Dockerfile            # Container definition
└── requirements.txt      # Python dependencies
```

## 🧪 Testing

### Test với FG-NET Dataset

FG-NET là dataset với 82 người, mỗi người có ảnh từ nhỏ đến lớn (age progression).

**Organize dataset:**
```bash
python organize_fgnet.py
```

**Kết quả:**
- 82 persons
- 988 images
- 5,042 test pairs
- Age span: 11-54 years

**Test matching:**
- Upload ảnh age_02.jpg → Lưu làm missing person
- Upload ảnh age_22.jpg (cùng người, 20 năm sau) → Tự động match
- **Kết quả: 88% similarity!** ✨

## 📊 API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/api/v1/upload/missing` | POST | Upload người mất tích |
| `/api/v1/upload/found` | POST | Upload người tìm thấy + auto matching |
| `/api/v1/search/missing/{case_id}` | GET | Tìm kiếm theo case_id |
| `/api/v1/search/found/{found_id}` | GET | Tìm kiếm theo found_id |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

## 🔧 Quản lý Docker

```bash
# Xem trạng thái
docker-compose ps

# Xem logs
docker-compose logs api -f

# Restart
docker-compose restart api

# Dừng
docker-compose down

# Rebuild
docker-compose build
docker-compose up -d
```

## 📚 Tài liệu

- **[CACH_SU_DUNG_API.md](CACH_SU_DUNG_API.md)** - Hướng dẫn chi tiết API
- **[HUONG_DAN_UPLOAD_SWAGGER.md](HUONG_DAN_UPLOAD_SWAGGER.md)** - Hướng dẫn upload qua Swagger UI
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Tổng quan kỹ thuật

## 🎯 Performance

- **Face Detection**: ~200ms
- **Embedding Extraction**: ~150ms  
- **Vector Search**: <100ms
- **Total Upload + Match**: ~500-800ms

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **AI Models**: MTCNN, ArcFace (ONNX), TensorFlow
- **Vector DB**: Qdrant
- **Image Processing**: OpenCV, Pillow
- **Container**: Docker, Docker Compose

## 📝 License

MIT License

## 🤝 Contributing

Pull requests are welcome!

---

**Developed with ❤️ for finding missing persons**
