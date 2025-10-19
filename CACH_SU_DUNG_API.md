# 🎉 API MISSING PERSON AI ĐÃ CHẠY THÀNH CÔNG!

## ✅ Trạng thái hiện tại

**API đang chạy tại:** http://localhost:8000

**Tất cả endpoints đã hoạt động:**
- ✅ `/api/v1/upload/missing` - Upload ảnh người mất tích  
- ✅ `/api/v1/upload/found` - Upload ảnh người tìm thấy
- ✅ `/api/v1/search/missing/{case_id}` - Tìm kiếm người mất tích
- ✅ `/api/v1/search/found/{found_id}` - Tìm kiếm người tìm thấy
- ✅ `/health` - Kiểm tra health
- ✅ `/docs` - Swagger UI (giao diện web)

---

## 🚀 CÁCH SỬ DỤNG

### 1. Mở Swagger UI (KHUYẾN NGHỊ - DỄ NHẤT)

Mở trình duyệt và vào:
```
http://localhost:8000/docs
```

Tại đây bạn có thể:
- ✅ Xem tất cả API endpoints
- ✅ Test upload ảnh trực tiếp từ trình duyệt  
- ✅ Xem request/response examples
- ✅ Không cần viết code gì cả!

---

### 2. Upload ảnh người mất tích

**Tại Swagger UI:**
1. Click vào `/api/v1/upload/missing`
2. Click "Try it out"
3. Upload file ảnh
4. Điền metadata (JSON):

```json
{
  "case_id": "MISS_2025_001",
  "name": "Nguyễn Văn A",
  "age_at_disappearance": 25,
  "year_disappeared": 2023,
  "gender": "male",
  "location_last_seen": "Hà Nội, Việt Nam",
  "contact": "0912345678",
  "description": "Cao 1m70, nặng 65kg",
  "distinguishing_marks": "Có vết sẹo ở má trái"
}
```

5. Click "Execute"

**Response mẫu:**
```json
{
  "success": true,
  "message": "Missing person 'Nguyễn Văn A' uploaded successfully",
  "point_id": "uuid-here",
  "potential_matches": [],
  "face_quality": {
    "sharpness": 0.85,
    "brightness": 0.75,
    "contrast": 0.80,
    "is_sharp": true,
    "is_bright_enough": true,
    "is_contrasted": true
  },
  "processing_time_ms": 1250.5
}
```

---

### 3. Upload ảnh người tìm thấy (và tự động tìm matching)

**Tại Swagger UI:**
1. Click vào `/api/v1/upload/found`
2. Click "Try it out"
3. Upload file ảnh
4. Điền metadata (JSON):

```json
{
  "found_id": "FOUND_2025_001",
  "location_found": "Hồ Chí Minh, Việt Nam",
  "date_found": "2025-10-19",
  "reporter_contact": "0987654321",
  "description": "Người này được tìm thấy tại...",
  "current_condition": "Sức khỏe tốt"
}
```

5. Click "Execute"

**Response sẽ TỰ ĐỘNG TÌM KIẾM matching:**
```json
{
  "success": true,
  "message": "Found person 'FOUND_2025_001' uploaded successfully",
  "point_id": "uuid-here",
  "potential_matches": [
    {
      "id": "uuid-of-missing-person",
      "face_similarity": 0.95,
      "metadata_similarity": 0.87,
      "combined_score": 0.92,
      "confidence_level": "VERY_HIGH",
      "confidence_score": 0.93,
      "explanation": {
        "confidence_level": "VERY_HIGH",
        "summary": "Rất có thể đây là cùng một người",
        "factors": {
          "face_similarity": {
            "score": 0.95,
            "weight": 0.70,
            "contribution": 0.665,
            "description": "Độ giống khuôn mặt rất cao"
          }
        },
        "recommendations": [
          "Liên hệ ngay với gia đình",
          "Xác minh thông tin thêm"
        ]
      },
      "contact": "0912345678",
      "metadata": {
        "case_id": "MISS_2025_001",
        "name": "Nguyễn Văn A",
        "age_at_disappearance": 25,
        "location_last_seen": "Hà Nội, Việt Nam"
      }
    }
  ],
  "face_quality": {...},
  "processing_time_ms": 2340.8
}
```

---

### 4. Tìm kiếm người mất tích theo case_id

**URL:** `GET /api/v1/search/missing/{case_id}`

**Ví dụ tại Swagger UI:**
1. Click vào `/api/v1/search/missing/{case_id}`
2. Click "Try it out"
3. Nhập `case_id`: `MISS_2025_001`
4. Click "Execute"

---

### 5. Kiểm tra health của hệ thống

```powershell
curl http://localhost:8000/health
```

Hoặc vào: http://localhost:8000/health

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1697654321.123,
  "services": {
    "face_detector": true,
    "embedding_extractor": true,
    "vector_db": true,
    "bilateral_search": true,
    "confidence_scoring": true,
    "vector_db_connection": true,
    "overall": true
  },
  "database_stats": {
    "missing_persons": {"count": 5},
    "found_persons": {"count": 3}
  },
  "version": "1.0.0"
}
```

---

## 📊 HIỂU CONFIDENCE SCORE

**Khi upload found person, hệ thống sẽ tự động:**
1. ✅ Phát hiện khuôn mặt trong ảnh
2. ✅ Tạo face embedding (vector 512 chiều)
3. ✅ So sánh với TẤT CẢ người mất tích trong database
4. ✅ Tính confidence score dựa trên nhiều yếu tố

**Confidence Levels:**
- `VERY_HIGH` (>0.9): Rất chắc chắn đúng người → **Liên hệ ngay!**
- `HIGH` (0.8-0.9): Khả năng cao → Xác minh thêm
- `MEDIUM` (0.7-0.8): Có thể → Kiểm tra kỹ
- `LOW` (<0.7): Khả năng thấp → Cần thêm bằng chứng

**Các yếu tố được tính:**
- Face similarity (70%): Độ giống khuôn mặt
- Metadata similarity (30%): Giới tính, độ tuổi, địa điểm...

---

## 🎯 WORKFLOW SỬ DỤNG THỰC TẾ

### Scenario: Tìm người mất tích

**Bước 1: Thêm người mất tích vào hệ thống**
- Vào http://localhost:8000/docs
- Chọn `/api/v1/upload/missing`
- Upload ảnh + thông tin đầy đủ
- Hệ thống lưu vào database

**Bước 2: Khi có người nghi ngờ**
- Chọn `/api/v1/upload/found`  
- Upload ảnh người tìm thấy
- Hệ thống **TỰ ĐỘNG** so sánh với TẤT CẢ người mất tích

**Bước 3: Xem kết quả**
- Nếu có match với confidence HIGH/VERY_HIGH → Liên hệ ngay
- Xem chi tiết trong response để quyết định

---

## 📋 QUẢN LÝ DOCKER

### Kiểm tra trạng thái
```powershell
docker-compose -f docker-compose-simple.yml ps
```

### Xem logs
```powershell
# Logs API
docker-compose -f docker-compose-simple.yml logs api -f

# Logs Qdrant
docker-compose -f docker-compose-simple.yml logs qdrant -f
```

### Restart API
```powershell
docker-compose -f docker-compose-simple.yml restart api
```

### Dừng tất cả
```powershell
docker-compose -f docker-compose-simple.yml down
```

### Khởi động lại
```powershell
docker-compose -f docker-compose-simple.yml up -d
```

---

## 🔍 DASHBOARD VÀ MONITORING

**Swagger UI:** http://localhost:8000/docs
- Giao diện test API đầy đủ

**ReDoc:** http://localhost:8000/redoc  
- Tài liệu API đẹp hơn

**Qdrant Dashboard:** http://localhost:6333/dashboard
- Xem dữ liệu vectors
- Quản lý collections

---

## 💡 LƯU Ý QUAN TRỌNG

### 1. Model đã được load thành công
- ✅ ArcFace R100: 248.9 MB
- ✅ MTCNN Face Detector
- ✅ TensorFlow backend

### 2. Dữ liệu được lưu persistent
- Ảnh và embeddings lưu trong Qdrant
- Dữ liệu không mất khi restart container
- Để xóa dữ liệu: `docker-compose -f docker-compose-simple.yml down -v`

### 3. Performance
- **Lần đầu upload:** 2-3 giây (load models)
- **Các lần sau:** 0.5-1 giây
- **Search:** < 0.1 giây (vector search rất nhanh)

### 4. Định dạng ảnh hỗ trợ
- JPG, JPEG, PNG, BMP
- Kích thước tối đa: 10 MB
- Nên có khuôn mặt rõ ràng, chính diện

---

## 🎓 VÍ DỤ SỬ DỤNG VỚI POWERSHELL

### Test Health
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" | ConvertTo-Json
```

### Upload missing person
```powershell
$metadata = @{
    case_id = "MISS_2025_001"
    name = "Nguyen Van A"
    age_at_disappearance = 25
    year_disappeared = 2023
    gender = "male"
    location_last_seen = "Hanoi"
    contact = "0912345678"
} | ConvertTo-Json

$form = @{
    metadata = $metadata
    image = Get-Item -Path "path\to\image.jpg"
}

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/upload/missing" -Method Post -Form $form
```

---

## 🛠 TROUBLESHOOTING

### API không phản hồi
```powershell
# Xem logs
docker-compose -f docker-compose-simple.yml logs api --tail 50

# Restart
docker-compose -f docker-compose-simple.yml restart api
```

### Không phát hiện được khuôn mặt
- ✅ Kiểm tra ảnh có khuôn mặt rõ ràng không
- ✅ Ảnh không bị mờ, tối quá
- ✅ Khuôn mặt không bị che khuất

### Confidence score thấp
- Có thể không phải cùng một người
- Hoặc ảnh chất lượng kém
- Hoặc thời gian qua lâu, ngoại hình thay đổi nhiều

---

## 🎉 KẾT LUẬN

**PROJECT ĐÃ HOÀN TOÀN HOẠT ĐỘNG!**

Bạn có thể:
1. ✅ Upload ảnh người mất tích qua Swagger UI
2. ✅ Upload ảnh người tìm thấy và nhận kết quả matching ngay lập tức
3. ✅ Tìm kiếm theo case_id hoặc found_id
4. ✅ Xem confidence score và lý do matching

**Bắt đầu sử dụng ngay:**
👉 http://localhost:8000/docs

---

Chúc bạn sử dụng thành công! 🚀

