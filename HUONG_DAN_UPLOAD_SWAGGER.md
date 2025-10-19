# 📖 HƯỚNG DẪN UPLOAD QUA SWAGGER UI

## ⚠️ LƯU Ý QUAN TRỌNG

Khi upload qua Swagger UI, field **metadata** phải là **STRING JSON**, KHÔNG phải JSON object!

---

## ✅ CÁCH ĐÚNG - Upload Missing Person

### Bước 1: Mở Swagger UI
Vào: http://localhost:8000/docs

### Bước 2: Chọn endpoint
Click vào: **POST /api/v1/upload/missing**

### Bước 3: Click "Try it out"

### Bước 4: Upload ảnh
Click "Choose File" và chọn ảnh

### Bước 5: Điền metadata (QUAN TRỌNG!)

**⚠️ PHẢI ĐIỀN DẠNG STRING JSON (có dấu ngoặc kép ngoài cùng):**

```json
"{\"case_id\":\"TEST_001\",\"name\":\"Nguyen Van A\",\"age_at_disappearance\":25,\"year_disappeared\":2023,\"gender\":\"male\",\"location_last_seen\":\"Ha Noi\",\"contact\":\"test@example.com\"}"
```

**HOẶC** (Dễ đọc hơn - viết trên nhiều dòng):
```
"{
  \"case_id\": \"TEST_001\",
  \"name\": \"Nguyen Van A\",
  \"age_at_disappearance\": 25,
  \"year_disappeared\": 2023,
  \"gender\": \"male\",
  \"location_last_seen\": \"Ha Noi\",
  \"contact\": \"test@example.com\"
}"
```

### Bước 6: Click "Execute"

---

## ✅ CÁCH ĐÚNG - Upload Found Person

**Metadata dạng STRING JSON:**

```json
"{\"found_id\":\"FOUND_001\",\"current_age_estimate\":30,\"gender\":\"male\",\"current_location\":\"TP HCM\",\"finder_contact\":\"finder@example.com\"}"
```

**HOẶC:**
```
"{
  \"found_id\": \"FOUND_001\",
  \"current_age_estimate\": 30,
  \"gender\": \"male\",
  \"current_location\": \"TP HCM\",
  \"finder_contact\": \"finder@example.com\"
}"
```

---

## ❌ SAI - ĐỪNG ĐIỀN NHƯ NÀY

**SAI:** (Không có dấu ngoặc kép ngoài cùng)
```json
{
  "case_id": "TEST_001",
  "name": "Nguyen Van A",
  ...
}
```

**ĐÚNG:** (Phải có dấu ngoặc kép " bao bọc toàn bộ)
```json
"{\"case_id\":\"TEST_001\",\"name\":\"Nguyen Van A\",...}"
```

---

## 💡 CÁCH DỄ HƠN - DÙNG PYTHON SCRIPT

Nếu thấy phức tạp, dùng Python script đã tạo sẵn:

### Upload Missing Person:
```bash
python test_upload.py
```

### Upload Found Person:
```bash
python test_upload_found.py
```

### Hoặc tự tạo script:
```python
import requests
import json

# Metadata (object bình thường)
metadata = {
    "case_id": "TEST_001",
    "name": "Nguyen Van A",
    "age_at_disappearance": 25,
    "year_disappeared": 2023,
    "gender": "male",
    "location_last_seen": "Ha Noi",
    "contact": "test@example.com"
}

# Upload
with open("image.jpg", 'rb') as f:
    files = {'image': f}
    data = {'metadata': json.dumps(metadata)}  # Convert to JSON string
    
    response = requests.post(
        "http://localhost:8000/api/v1/upload/missing",
        files=files,
        data=data
    )
    
print(response.json())
```

---

## 📋 CÁC FIELD BẮT BUỘC

### Missing Person:
- ✅ `case_id` - ID (3-50 ký tự, chữ số, gạch ngang, gạch dưới)
- ✅ `name` - Tên (ít nhất 2 ký tự)
- ✅ `age_at_disappearance` - Tuổi khi mất tích (0-120)
- ✅ `year_disappeared` - Năm mất tích (1900-2025)
- ✅ `gender` - Giới tính: "male", "female", "other", "unknown"
- ✅ `location_last_seen` - Địa điểm (ít nhất 3 ký tự)
- ✅ `contact` - Email hoặc số điện thoại

### Found Person:
- ✅ `found_id` - ID (3-50 ký tự)
- ✅ `current_age_estimate` - Tuổi ước tính (0-120)
- ✅ `gender` - Giới tính
- ✅ `current_location` - Nơi tìm thấy
- ✅ `finder_contact` - Email hoặc số điện thoại

---

## 🔧 TOOLS HỖ TRỢ

### Online JSON String Encoder:
1. Viết JSON bình thường
2. Vào: https://www.freeformatter.com/json-escape.html
3. Paste JSON vào
4. Click "Escape JSON"
5. Copy kết quả (đã có dấu " bao ngoài)
6. Paste vào Swagger UI

### Hoặc dùng Python:
```python
import json

metadata = {
    "case_id": "TEST_001",
    "name": "Nguyen Van A",
    # ... các field khác
}

# Tạo JSON string để paste vào Swagger
json_string = json.dumps(metadata)
print(json_string)
```

---

## 🎯 KHUYẾN NGHỊ

**Để đơn giản nhất:**
1. ✅ Dùng **Python script** (`test_upload.py`, `test_upload_found.py`)
2. ✅ Hoặc dùng **curl** command
3. ⚠️ Swagger UI hơi phức tạp với metadata

**Swagger UI tốt nhất để:**
- Xem tài liệu API
- Test các endpoint đơn giản (/health, /search)
- Hiểu cấu trúc request/response

---

Bạn có muốn tôi tạo thêm script Python để upload nhiều ảnh cùng lúc không?

