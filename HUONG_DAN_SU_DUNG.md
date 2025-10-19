# 🎯 HƯỚNG DẪN SỬ DỤNG MISSING PERSON AI

## ✅ Trạng thái hiện tại
Project đã chạy thành công với Docker! 🎉

- ✅ Docker containers đang chạy
- ✅ Qdrant vector database: http://localhost:6333
- ✅ API Server: http://localhost:8000
- ✅ Swagger UI: http://localhost:8000/docs

---

## 📋 CÁC LỆNH QUẢN LÝ DOCKER

### 1. Kiểm tra trạng thái containers
```bash
docker-compose -f docker-compose-simple.yml ps
```

### 2. Xem logs
```bash
# Xem logs tất cả services
docker-compose -f docker-compose-simple.yml logs

# Xem logs của API
docker-compose -f docker-compose-simple.yml logs api -f

# Xem logs của Qdrant
docker-compose -f docker-compose-simple.yml logs qdrant -f
```

### 3. Dừng và khởi động lại
```bash
# Dừng tất cả
docker-compose -f docker-compose-simple.yml down

# Khởi động lại
docker-compose -f docker-compose-simple.yml up -d

# Khởi động lại chỉ 1 service
docker-compose -f docker-compose-simple.yml restart api
```

### 4. Xóa tất cả và reset
```bash
# Dừng và xóa containers + volumes
docker-compose -f docker-compose-simple.yml down -v

# Rebuild lại image
docker-compose -f docker-compose-simple.yml build --no-cache
docker-compose -f docker-compose-simple.yml up -d
```

---

## 🔍 SỬ DỤNG API

### 1. Truy cập Swagger UI (Giao diện Web)
Mở trình duyệt và vào: **http://localhost:8000/docs**

Tại đây bạn có thể:
- ✅ Xem tất cả API endpoints
- ✅ Test API trực tiếp từ trình duyệt
- ✅ Xem request/response examples

### 2. Test API với PowerShell

#### Kiểm tra health
```powershell
curl http://localhost:8000/health
```

#### Kiểm tra root endpoint
```powershell
curl http://localhost:8000/
```

### 3. Upload ảnh người mất tích (Missing Person)

**Chuẩn bị:**
- Có 1 file ảnh (ví dụ: `person.jpg`)
- Thông tin metadata về người mất tích

**Upload bằng PowerShell:**
```powershell
# Tạo file metadata.json
$metadata = @{
    case_id = "MISS_2023_001"
    name = "Nguyễn Văn A"
    age_at_disappearance = 25
    year_disappeared = 2023
    gender = "male"
    location_last_seen = "Hanoi, Vietnam"
    contact = "contact@email.com"
}

# Convert to JSON
$metadataJson = $metadata | ConvertTo-Json

# Upload ảnh
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/upload/missing" `
    -Method Post `
    -Form @{
        image = Get-Item -Path "C:\path\to\your\image.jpg"
        metadata = $metadataJson
    }
```

**Upload bằng cURL (nếu có cài Git Bash):**
```bash
curl -X POST "http://localhost:8000/api/v1/upload/missing" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/image.jpg" \
  -F 'metadata={"case_id":"MISS_2023_001","name":"Nguyen Van A","age_at_disappearance":25,"year_disappeared":2023,"gender":"male","location_last_seen":"Hanoi","contact":"test@email.com"}'
```

### 4. Upload ảnh người được tìm thấy (Found Person)

```powershell
$metadata = @{
    location_found = "Ho Chi Minh City, Vietnam"
    date_found = "2023-10-19"
    reporter_contact = "reporter@email.com"
}

$metadataJson = $metadata | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/upload/found" `
    -Method Post `
    -Form @{
        image = Get-Item -Path "C:\path\to\found_image.jpg"
        metadata = $metadataJson
    }
```

### 5. Tìm kiếm người mất tích

**Sau khi upload ảnh found person, API sẽ tự động:**
1. ✅ Phát hiện khuôn mặt trong ảnh
2. ✅ Tạo face embedding (vector đặc trưng)
3. ✅ Tìm kiếm trong database người mất tích
4. ✅ Trả về kết quả matching với confidence score

**Response mẫu:**
```json
{
    "upload_id": "uuid-here",
    "faces_detected": 1,
    "matches": [
        {
            "case_id": "MISS_2023_001",
            "name": "Nguyễn Văn A",
            "confidence_score": 0.95,
            "similarity": 0.98,
            "age_at_disappearance": 25,
            "year_disappeared": 2023,
            "location_last_seen": "Hanoi, Vietnam"
        }
    ]
}
```

---

## 📊 QDRANT DASHBOARD

Truy cập Qdrant Dashboard: **http://localhost:6333/dashboard**

Tại đây bạn có thể:
- ✅ Xem collections (bảng dữ liệu)
- ✅ Xem số lượng vectors đã lưu
- ✅ Quản lý dữ liệu

---

## 🔧 TROUBLESHOOTING

### 1. API không chạy
```bash
# Xem logs để kiểm tra lỗi
docker-compose -f docker-compose-simple.yml logs api

# Restart API
docker-compose -f docker-compose-simple.yml restart api
```

### 2. Qdrant không kết nối được
```bash
# Kiểm tra Qdrant có chạy không
docker-compose -f docker-compose-simple.yml ps

# Kiểm tra logs
docker-compose -f docker-compose-simple.yml logs qdrant
```

### 3. Lỗi khi upload ảnh
- ✅ Kiểm tra file ảnh có tồn tại
- ✅ Kiểm tra định dạng ảnh (JPG, PNG)
- ✅ Kiểm tra metadata có đúng format JSON
- ✅ Xem logs API để biết lỗi chi tiết

### 4. Docker Desktop không khởi động
- ✅ Restart máy tính
- ✅ Chạy Docker Desktop với quyền Administrator
- ✅ Kiểm tra Hyper-V đã enable chưa (Windows)

---

## 📝 LƯU Ý QUAN TRỌNG

1. **Model weights**: Đảm bảo file `models/weights/arcface_r100_v1.onnx` tồn tại
   - Nếu chưa có, chạy: `python download_model.py`

2. **Dữ liệu lưu trữ**: 
   - Ảnh và embeddings được lưu trong Qdrant database
   - Dữ liệu được persist trong Docker volume `qdrant_data`
   - Để xóa tất cả dữ liệu: `docker-compose -f docker-compose-simple.yml down -v`

3. **Performance**:
   - Lần đầu detect face có thể chậm (load model)
   - Các lần sau sẽ nhanh hơn
   - CPU: ~1-2 giây/ảnh
   - GPU: ~0.1-0.3 giây/ảnh (nếu có)

---

## 🚀 WORKFLOW SỬ DỤNG THỰC TẾ

### Scenario: Tìm người mất tích

1. **Upload ảnh người mất tích vào hệ thống**
   - Sử dụng endpoint `/api/v1/upload/missing`
   - Cung cấp đầy đủ thông tin: tên, tuổi, nơi mất tích, v.v.

2. **Khi tìm thấy người nghi ngờ**
   - Upload ảnh qua endpoint `/api/v1/upload/found`
   - Hệ thống tự động so sánh với database

3. **Xem kết quả**
   - Nếu confidence > 0.9: Rất có thể đúng người
   - Nếu confidence 0.7-0.9: Khả năng cao, cần xác minh thêm
   - Nếu confidence < 0.7: Có thể không phải, cần kiểm tra kỹ

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề:
1. Kiểm tra logs: `docker-compose -f docker-compose-simple.yml logs`
2. Restart services: `docker-compose -f docker-compose-simple.yml restart`
3. Xem file `logs/app.log` để debug chi tiết
4. Kiểm tra `/docs` để xem API documentation

---

## 🎓 TÀI LIỆU BỔ SUNG

- `README.md` - Tổng quan project
- `PROJECT_SUMMARY.md` - Chi tiết kỹ thuật
- `QUICK_START.md` - Hướng dẫn khởi động nhanh
- `HOW_TO_RUN.md` - Các cách chạy khác nhau

---

**🎉 Chúc bạn sử dụng thành công!**

