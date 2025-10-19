# 📚 API DOCUMENTATION - MISSING PERSON AI

## 🎯 **TỔNG QUAN**

Hệ thống Missing Person AI có **4 API endpoints chính** được chia thành 2 nhóm:

- **UPLOAD APIs**: Upload ảnh và thông tin người
- **SEARCH APIs**: Tìm kiếm thông tin theo ID

---

## 📤 **UPLOAD APIs**

### 1. **POST** `/api/v1/upload/missing`

**Mục đích**: Upload ảnh người mất tích

**Khi nào dùng**:
- Gia đình báo cáo người thân mất tích
- Cần lưu ảnh và thông tin người mất tích
- **Tự động tìm kiếm** trong database xem có người tìm thấy nào tương tự không

**Request**:
```http
POST /api/v1/upload/missing
Content-Type: multipart/form-data

image: [file] - Ảnh khuôn mặt người mất tích
metadata: [string] - JSON string chứa thông tin
```

**Metadata JSON**:
```json
{
  "case_id": "MISS_2023_001",
  "name": "Nguyen Van A",
  "age_at_disappearance": 25,
  "year_disappeared": 2023,
  "gender": "male",
  "location_last_seen": "Ha Noi",
  "contact": "family@example.com"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Missing person 'Nguyen Van A' uploaded successfully",
  "point_id": "552ccaea-5c82-4383-9ae0-bc10c4db931e",
  "potential_matches": [
    {
      "id": "0d42e546-ce76-4c70-b1a8-923a1d598882",
      "face_similarity": 0.9566134,
      "confidence_level": "HIGH",
      "confidence_score": 0.7653067,
      "contact": "finder@example.com",
      "metadata": { ... }
    }
  ],
  "face_quality": { ... },
  "processing_time_ms": 342.95
}
```

---

### 2. **POST** `/api/v1/upload/found`

**Mục đích**: Upload ảnh người tìm thấy

**Khi nào dùng**:
- Ai đó tìm thấy người nghi ngờ là người mất tích
- Cần lưu ảnh và thông tin người tìm thấy
- **Tự động tìm kiếm** trong database xem có người mất tích nào tương tự không

**Request**:
```http
POST /api/v1/upload/found
Content-Type: multipart/form-data

image: [file] - Ảnh khuôn mặt người tìm thấy
metadata: [string] - JSON string chứa thông tin
```

**Metadata JSON**:
```json
{
  "found_id": "FOUND_001",
  "current_age_estimate": 30,
  "gender": "male",
  "current_location": "TP HCM",
  "finder_contact": "finder@example.com"
}
```

**Response**: Tương tự như `/upload/missing`

---

## 🔍 **SEARCH APIs**

### 3. **GET** `/api/v1/search/missing/{case_id}`

**Mục đích**: Tìm kiếm thông tin người mất tích theo ID

**Khi nào dùng**:
- Muốn xem thông tin chi tiết của một case mất tích cụ thể
- Kiểm tra trạng thái tìm kiếm
- Xem các potential matches đã tìm được

**Request**:
```http
GET /api/v1/search/missing/MISS_2023_001?limit=5&include_similar=true
```

**Query Parameters**:
- `limit` (optional): Số lượng kết quả tối đa (default: 1, max: 100)
- `include_similar` (optional): Bao gồm các matches tương tự (default: false)

**Response**:
```json
{
  "success": true,
  "case_id": "MISS_2023_001",
  "total_matches": 1,
  "matches": [
    {
      "id": "0d42e546-ce76-4c70-b1a8-923a1d598882",
      "face_similarity": 0.9566134,
      "confidence_level": "HIGH",
      "confidence_score": 0.7653067,
      "contact": "finder@example.com",
      "metadata": { ... }
    }
  ],
  "search_time_ms": 45.2
}
```

---

### 4. **GET** `/api/v1/search/found/{found_id}`

**Mục đích**: Tìm kiếm thông tin người tìm thấy theo ID

**Khi nào dùng**:
- Muốn xem thông tin chi tiết của một người tìm thấy cụ thể
- Kiểm tra các potential matches đã tìm được

**Request**:
```http
GET /api/v1/search/found/FOUND_001?limit=5&include_similar=true
```

**Response**: Tương tự như `/search/missing/{case_id}`

---

## 🚀 **WORKFLOW THỰC TẾ**

### **Scenario 1: Gia đình báo cáo mất tích**

1. **Gia đình upload ảnh mất tích**:
   ```bash
   POST /api/v1/upload/missing
   ```
   - Hệ thống tự động tìm trong database người tìm thấy
   - Trả về potential matches (nếu có)

2. **Theo dõi tiến độ**:
   ```bash
   GET /api/v1/search/missing/MISS_2023_001
   ```
   - Xem các matches mới
   - Kiểm tra trạng thái case

### **Scenario 2: Ai đó tìm thấy người nghi ngờ**

1. **Upload ảnh người tìm thấy**:
   ```bash
   POST /api/v1/upload/found
   ```
   - Hệ thống tự động tìm trong database người mất tích
   - Trả về potential matches (nếu có)

2. **Xem kết quả matching**:
   ```bash
   GET /api/v1/search/found/FOUND_001
   ```
   - Xem các matches đã tìm được
   - Liên hệ với gia đình

### **Scenario 3: Kết nối thành công**

- Hai API upload sẽ **tự động matching với nhau**
- Khi có match → liên hệ giữa gia đình và người tìm thấy
- Confidence score cao → khả năng đúng người cao

---

## 📊 **CONFIDENCE SCORES**

| Confidence Level | Score Range | Ý nghĩa | Hành động |
|------------------|-------------|---------|-----------|
| **VERY_HIGH** | 0.9 - 1.0 | Rất chắc chắn | Liên hệ ngay |
| **HIGH** | 0.75 - 0.9 | Khả năng cao | Xác minh thêm |
| **MEDIUM** | 0.6 - 0.75 | Khả năng trung bình | Kiểm tra kỹ |
| **LOW** | 0.4 - 0.6 | Khả năng thấp | Cần thêm thông tin |
| **VERY_LOW** | 0.0 - 0.4 | Khả năng rất thấp | Không khuyến nghị |

---

## 🔧 **TESTING APIs**

### **Sử dụng Python scripts**:
```bash
# Test upload missing person
python test_upload.py

# Test upload found person  
python test_upload_found.py
```

### **Sử dụng Swagger UI**:
- Truy cập: http://localhost:8000/docs
- **Lưu ý**: Metadata phải là **string JSON** (có dấu `"` bao ngoài)

### **Sử dụng curl**:
```bash
# Upload missing
curl -X POST "http://localhost:8000/api/v1/upload/missing" \
  -F "image=@photo.jpg" \
  -F "metadata={\"case_id\":\"TEST_001\",\"name\":\"Test\"}"

# Search missing
curl "http://localhost:8000/api/v1/search/missing/TEST_001"
```

---

## ⚠️ **LƯU Ý QUAN TRỌNG**

1. **Metadata phải là JSON string** khi dùng Swagger UI
2. **Ảnh phải có khuôn mặt rõ ràng** để AI có thể nhận diện
3. **Face similarity > 0.9** = khả năng đúng người rất cao
4. **Location proximity = 0** không có nghĩa là không đúng người (có thể di chuyển)
5. **Age progression** được tính toán dựa trên khoảng cách thời gian

---

## 🎉 **KẾT LUẬN**

Hệ thống AI tự động matching giữa missing và found persons với độ chính xác cao. Sử dụng đúng API theo workflow sẽ giúp kết nối gia đình với người tìm thấy một cách hiệu quả.

**Để được hỗ trợ**: Xem thêm `GUIDE.md` hoặc truy cập Swagger UI tại `/docs`
