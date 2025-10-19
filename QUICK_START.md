# 🚀 Hướng dẫn chạy Missing Person AI

## Bước 1: Kiểm tra hệ thống
```bash
python check_system.py
```

## Bước 2: Khởi động services với Docker
```bash
# Khởi động tất cả services (Qdrant, PostgreSQL, MinIO, API)
docker-compose up -d

# Xem logs
docker-compose logs -f api
```

## Bước 3: Kiểm tra API
```bash
# Kiểm tra health
curl http://localhost:8000/health

# Hoặc mở browser
start http://localhost:8000/docs
```

## Bước 4: Test upload (nếu có ảnh)
```bash
# Upload missing person
curl -X POST "http://localhost:8000/api/v1/upload/missing" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@your_image.jpg" \
  -F 'metadata={"case_id":"MISS_2023_001","name":"Test Person","age_at_disappearance":25,"year_disappeared":2020,"gender":"male","location_last_seen":"Hanoi","contact":"test@email.com"}'
```

## Các lệnh hữu ích:
```bash
# Xem status services
docker-compose ps

# Stop services
docker-compose down

# Xem logs chi tiết
docker-compose logs api

# Restart API service
docker-compose restart api
```
