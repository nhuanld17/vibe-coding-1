# 🚀 Cách chạy Missing Person AI System

## ✅ Hệ thống đã sẵn sàng!

Kiểm tra cơ bản đã PASS! Bây giờ bạn có thể chạy hệ thống theo 2 cách:

## Cách 1: Chạy với Docker (Khuyến nghị)

### Bước 1: Khởi động Docker Desktop
- Mở Docker Desktop trên Windows
- Đợi cho đến khi Docker sẵn sàng

### Bước 2: Khởi động services
```bash
docker-compose up -d
```

### Bước 3: Kiểm tra
```bash
# Xem status
docker-compose ps

# Kiểm tra logs
docker-compose logs api

# Test API
curl http://localhost:8000/health
```

### Bước 4: Sử dụng API
- API Documentation: http://localhost:8000/docs
- Qdrant Dashboard: http://localhost:6333/dashboard
- MinIO Console: http://localhost:9001

## Cách 2: Chạy trực tiếp (Development)

### Bước 1: Khởi động Qdrant (cần Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant:v1.7.4
```

### Bước 2: Chạy API
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Bước 3: Test
```bash
curl http://localhost:8000/health
```

## 🧪 Test đơn giản

Nếu gặp vấn đề, chạy test cơ bản:
```bash
python test_basic.py
```

## 📋 API Usage Examples

### Upload Missing Person
```bash
curl -X POST "http://localhost:8000/api/v1/upload/missing" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@your_photo.jpg" \
  -F 'metadata={
    "case_id": "MISS_2023_001",
    "name": "Nguyen Van A",
    "age_at_disappearance": 25,
    "year_disappeared": 2020,
    "gender": "male",
    "location_last_seen": "Ha Noi",
    "contact": "family@email.com"
  }'
```

### Upload Found Person
```bash
curl -X POST "http://localhost:8000/api/v1/upload/found" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@found_photo.jpg" \
  -F 'metadata={
    "found_id": "FOUND_2023_001",
    "current_age_estimate": 30,
    "gender": "male",
    "current_location": "Ho Chi Minh City",
    "finder_contact": "finder@email.com"
  }'
```

## 🔧 Troubleshooting

### Vấn đề thường gặp:

1. **Docker không khởi động được**
   - Kiểm tra Docker Desktop đã chạy chưa
   - Restart Docker Desktop

2. **Port đã được sử dụng**
   - Đổi port trong docker-compose.yml
   - Hoặc stop service đang dùng port

3. **API không phản hồi**
   - Kiểm tra logs: `docker-compose logs api`
   - Restart API: `docker-compose restart api`

4. **Model không tìm thấy**
   - Chạy: `python download_model.py`
   - Hoặc tải manual từ InsightFace GitHub

## 📊 System Status

Kiểm tra trạng thái hệ thống:
```bash
python check_system.py
```

## 🎯 Next Steps

1. **Production Setup**:
   - Tải model ArcFace thật
   - Cấu hình SSL/HTTPS
   - Setup monitoring

2. **Customization**:
   - Điều chỉnh threshold trong .env
   - Thêm custom validation
   - Tích hợp database khác

3. **Scaling**:
   - Load balancer
   - Multiple API instances
   - GPU acceleration

---

**🎉 Chúc mừng! Hệ thống Missing Person AI đã sẵn sàng giúp đoàn tụ các gia đình!**
