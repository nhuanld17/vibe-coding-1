# 🔍 Family Finder AI - Missing Person Recognition System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119.0-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.12.0-red.svg)](https://qdrant.tech)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AI-powered facial recognition system for matching missing and found persons using advanced computer vision and vector database technology**

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Testing](#-testing)
- [Architecture](#-architecture)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**Family Finder AI** is an intelligent facial recognition system designed to help reunite missing persons with their families. The system leverages state-of-the-art AI models and vector database technology to automatically match uploaded photos of missing and found individuals.

### How It Works

1. **Upload Missing Person**: Families upload photos and information about missing loved ones
2. **AI Processing**: The system extracts facial features using the ArcFace model
3. **Automatic Matching**: When a found person is uploaded, the system automatically searches for matches
4. **Confidence Scoring**: Advanced algorithms provide confidence levels for each match
5. **Notification**: High-confidence matches are flagged for human review

---

## ✨ Features

### Core Capabilities
- 🤖 **Advanced Face Recognition** - 95%+ accuracy using ArcFace ResNet100
- 🔄 **Bilateral Matching** - Automatic search when uploading either missing or found persons
- 📊 **Confidence Scoring** - Intelligent multi-factor confidence assessment
- 🎯 **Age-Invariant Matching** - Can match faces across different ages
- 🗄️ **Vector Database** - Lightning-fast similarity search using Qdrant
- 🌐 **RESTful API** - Complete API for easy integration
- 📱 **Interactive Documentation** - Built-in Swagger UI
- 🐳 **Docker Ready** - One-command deployment

### Advanced Features
- **Face Quality Assessment** - Checks sharpness, brightness, and contrast
- **Metadata Matching** - Considers age, gender, location, and distinctive marks
- **Multi-factor Confidence** - Weighted scoring based on multiple factors
- **Persistent Storage** - All data persists across restarts
- **Comprehensive Logging** - Full audit trail of all operations

---

## 🚀 Quick Start

Get the system running in **3 simple steps**:

### Step 1: Download AI Model
```bash
python download_model.py
```

### Step 2: Start Docker Services
```bash
docker-compose up -d
```

### Step 3: Verify Installation
```bash
curl http://localhost:8000/health
```

### ✅ You're Ready!
- **API Server**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Docker**: 20.10+ with Docker Compose
- **Python**: 3.11+ (for model download and testing)

### Recommended
- **RAM**: 8GB+
- **CPU**: 4+ cores
- **Storage**: 5GB+ for images and logs
- **GPU**: NVIDIA GPU with CUDA (optional, for acceleration)

---

## 🛠️ Installation

### Prerequisites

#### Windows
```powershell
# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop

# Install Python 3.11+
# Download from: https://www.python.org/downloads/
```

#### macOS
```bash
# Install Docker Desktop
brew install --cask docker

# Install Python
brew install python@3.11
```

#### Linux (Ubuntu)
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Python
sudo apt update
sudo apt install python3.11 python3-pip
```

### Installation Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/thndev05/family-finder-aI4life.git
   cd family-finder-aI4life
   ```

2. **Download AI Model**
   ```bash
   python download_model.py
   ```
   This downloads the ArcFace model (~250MB) to `models/weights/`

3. **Start Services**
   ```bash
   docker-compose up -d
   ```
   This starts:
   - Qdrant vector database (port 6333)
   - FastAPI server (port 8000)

4. **Verify Installation**
   ```bash
   curl http://localhost:8000/health
   ```

---

## 📚 Usage

### Option 1: Python Scripts (Recommended)

**Upload Missing Person**
```bash
python test_upload.py
```

**Upload Found Person**
```bash
python test_upload_found.py
```

**Test All Endpoints**
```bash
python test_all_endpoints.py
```

### Option 2: Python Code

```python
import requests
import json

# Upload missing person
metadata = {
    "case_id": "MISS_2024_001",
    "name": "John Doe",
    "age_at_disappearance": 25,
    "year_disappeared": 2024,
    "gender": "male",
    "location_last_seen": "New York, NY",
    "contact": "family@example.com",
    "description": "Brown hair, blue eyes"
}

with open("photo.jpg", 'rb') as f:
    response = requests.post(
        "http://localhost:8000/api/v1/upload/missing",
        files={'image': f},
        data={'metadata': json.dumps(metadata)}
    )

result = response.json()
print(f"Uploaded: {result['point_id']}")
print(f"Potential matches: {len(result['potential_matches'])}")
```

### Option 3: Swagger UI (Easiest)

1. Open browser: http://localhost:8000/docs
2. Click on any endpoint to expand
3. Click "Try it out"
4. Upload image and enter metadata
5. Click "Execute"

**⚠️ Important**: In Swagger UI, metadata must be a **JSON string** (with quotes around it):
```
"{\"case_id\":\"TEST_001\",\"name\":\"John Doe\",\"age_at_disappearance\":25,\"year_disappeared\":2024,\"gender\":\"male\",\"location_last_seen\":\"New York\",\"contact\":\"test@example.com\"}"
```

### Option 4: cURL

**PowerShell:**
```powershell
# Upload missing person
$metadata = @{
    case_id = "MISS_001"
    name = "John Doe"
    age_at_disappearance = 25
    year_disappeared = 2024
    gender = "male"
    location_last_seen = "New York"
    contact = "family@example.com"
} | ConvertTo-Json -Compress

curl -X POST http://localhost:8000/api/v1/upload/missing `
  -F "image=@photo.jpg" `
  -F "metadata=$metadata"
```

**Bash:**
```bash
# Upload missing person
curl -X POST http://localhost:8000/api/v1/upload/missing \
  -F "image=@photo.jpg" \
  -F 'metadata={"case_id":"MISS_001","name":"John Doe","age_at_disappearance":25,"year_disappeared":2024,"gender":"male","location_last_seen":"New York","contact":"family@example.com"}'
```

---

## 🔌 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check and service status |
| `GET` | `/` | API information |
| `POST` | `/api/v1/upload/missing` | Upload missing person + auto-search found persons |
| `POST` | `/api/v1/upload/found` | Upload found person + auto-search missing persons |
| `GET` | `/api/v1/search/missing/{case_id}` | Search for specific missing person |
| `GET` | `/api/v1/search/found/{found_id}` | Search for specific found person |

### Upload Missing Person

**Endpoint:** `POST /api/v1/upload/missing`

**Request:**
- `image` (file): Photo of the missing person
- `metadata` (string): JSON string with person information

**Metadata Fields:**
```json
{
  "case_id": "MISS_2024_001",          // Required: Unique case ID
  "name": "John Doe",                  // Required: Full name
  "age_at_disappearance": 25,          // Required: Age when disappeared
  "year_disappeared": 2024,            // Required: Year of disappearance
  "gender": "male",                    // Required: male/female/other
  "location_last_seen": "New York",    // Required: Last known location
  "contact": "family@example.com",     // Required: Contact information
  "description": "Brown hair...",      // Optional: Physical description
  "distinctive_marks": "Scar on...",   // Optional: Distinctive features
  "last_seen_clothing": "Blue shirt"   // Optional: Clothing description
}
```

**Response:**
```json
{
  "success": true,
  "message": "Missing person 'John Doe' uploaded successfully",
  "point_id": "uuid-here",
  "potential_matches": [
    {
      "id": "match-uuid",
      "face_similarity": 0.8796,
      "confidence_level": "MEDIUM",
      "confidence_score": 0.6307,
      "contact": "finder@example.com",
      "metadata": { ... },
      "explanation": {
        "summary": "Strong match based on facial similarity",
        "confidence_level": "MEDIUM",
        "reasons": ["High facial similarity", "Gender match"],
        "recommendations": ["Verify with family"]
      }
    }
  ],
  "face_quality": {
    "sharpness": 0.85,
    "brightness": 0.75,
    "contrast": 0.80,
    "is_sharp": true,
    "is_bright_enough": true,
    "is_contrasted": true
  },
  "processing_time_ms": 342.95
}
```

### Upload Found Person

**Endpoint:** `POST /api/v1/upload/found`

**Metadata Fields:**
```json
{
  "found_id": "FOUND_001",              // Required: Unique found ID
  "current_age_estimate": 30,           // Required: Estimated current age
  "gender": "male",                     // Required: male/female/other
  "current_location": "Los Angeles",    // Required: Where found
  "finder_contact": "finder@email.com", // Required: Finder's contact
  "description": "Adult male...",       // Optional: Physical description
  "physical_condition": "Good health",  // Optional: Health status
  "distinctive_marks": "Tattoo on..."   // Optional: Distinctive features
}
```

**Response:** Same format as upload missing person

### Search by ID

**Endpoint:** `GET /api/v1/search/missing/{case_id}`

**Parameters:**
- `case_id` (path): The case ID to search for
- `limit` (query, optional): Maximum results (default: 1)
- `include_similar` (query, optional): Include similar records (default: false)

**Response:**
```json
{
  "success": true,
  "query": "MISS_2024_001",
  "total_results": 1,
  "matches": [
    {
      "id": "uuid",
      "face_similarity": 1.0,
      "confidence_level": "VERY_HIGH",
      "metadata": { ... }
    }
  ],
  "processing_time_ms": 15.23
}
```

### Confidence Levels

| Level | Score Range | Meaning |
|-------|-------------|---------|
| `VERY_HIGH` | 0.80 - 1.00 | Extremely likely match - immediate review recommended |
| `HIGH` | 0.70 - 0.79 | Strong potential match - investigation recommended |
| `MEDIUM` | 0.60 - 0.69 | Moderate match - further verification needed |
| `LOW` | 0.50 - 0.59 | Weak match - use caution |
| `VERY_LOW` | 0.00 - 0.49 | Unlikely match - low priority |

---

## 🧪 Testing

### Automated Tests

Run comprehensive tests:
```bash
python test_all_endpoints.py
```

This tests:
- ✅ Health check
- ✅ API info
- ✅ Upload missing person
- ✅ Upload found person with automatic matching
- ✅ Search by case ID
- ✅ Search by found ID
- ✅ Error handling (404, validation)

### Manual Testing

1. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Upload Test Data**
   ```bash
   python test_upload.py
   python test_upload_found.py
   ```

3. **View in Swagger**
   - Open http://localhost:8000/docs
   - Try each endpoint interactively

4. **Check Qdrant**
   - Open http://localhost:6333/dashboard
   - View stored vectors and collections

---

## 🏗️ Architecture

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI | REST API server |
| **Face Detection** | MTCNN | Detect and align faces |
| **Face Embedding** | ArcFace ResNet100 | Extract 512-d facial features |
| **Vector DB** | Qdrant | Store and search embeddings |
| **Image Processing** | OpenCV, Pillow | Image manipulation |
| **Runtime** | Docker, Docker Compose | Containerization |
| **Language** | Python 3.11+ | Main programming language |

### System Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │─────▶│  FastAPI     │─────▶│   MTCNN     │
│  (Upload)   │      │   Server     │      │   Detector  │
└─────────────┘      └──────────────┘      └─────────────┘
                            │                      │
                            ▼                      ▼
                     ┌──────────────┐      ┌─────────────┐
                     │   ArcFace    │      │   OpenCV    │
                     │   Extractor  │      │  Processing │
                     └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Qdrant     │
                     │  Vector DB   │
                     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Bilateral   │
                     │    Search    │
                     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Confidence  │
                     │   Scoring    │
                     └──────────────┘
```

### Directory Structure

```
family-finder-aI4life/
├── api/                      # FastAPI application
│   ├── main.py              # Main application entry
│   ├── config.py            # Configuration management
│   ├── dependencies.py      # Dependency injection
│   ├── routes/              # API route handlers
│   │   ├── upload.py        # Upload endpoints
│   │   └── search.py        # Search endpoints
│   └── schemas/             # Pydantic models
│       └── models.py        # Request/response schemas
├── models/                   # AI models
│   ├── face_detection.py    # MTCNN face detector
│   ├── face_embedding.py    # ArcFace embeddings
│   └── weights/             # Model weights
│       └── arcface_r100_v1.onnx
├── services/                 # Business logic
│   ├── vector_db.py         # Qdrant operations
│   ├── bilateral_search.py  # Matching logic
│   └── confidence_scoring.py # Confidence calculation
├── utils/                    # Utilities
│   ├── image_processing.py  # Image manipulation
│   ├── validation.py        # Input validation
│   └── logger.py            # Logging setup
├── datasets/                 # Test datasets
│   └── FGNET_organized/     # FGNET test images
├── docker-compose.yml        # Docker orchestration
├── Dockerfile               # Docker image definition
├── requirements.txt         # Python dependencies
├── download_model.py        # Model download script
├── test_all_endpoints.py    # Comprehensive tests
├── test_upload.py           # Upload test script
├── test_upload_found.py     # Found person test
└── README.md                # This file
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Docker Not Running
**Error:** `Cannot connect to Docker daemon`

**Solution:**
```bash
# Windows/Mac: Start Docker Desktop
# Linux:
sudo systemctl start docker
```

#### 2. Port Already in Use
**Error:** `Port 8000 is already allocated`

**Solution:**
```powershell
# Windows: Find and kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or change port in docker-compose.yml:
# ports: "8001:8000"
```

#### 3. Model Not Found
**Error:** `ArcFace model not found`

**Solution:**
```bash
# Download model again
python download_model.py

# Verify file exists
ls models/weights/arcface_r100_v1.onnx
```

#### 4. No Face Detected
**Error:** `No face detected in the image`

**Causes:**
- Face too small or too large
- Face not visible or obscured
- Poor image quality
- Face angle > 45 degrees

**Solutions:**
- Use clear, front-facing photos
- Ensure face is well-lit
- Crop image to focus on face
- Use higher resolution images

#### 5. Poor Match Quality
**Issue:** Low confidence scores or no matches

**Solutions:**
- Use high-quality images
- Ensure faces are clearly visible
- Provide accurate metadata
- Verify age, gender, location info

#### 6. Slow Performance
**Issue:** API responses are slow

**Solutions:**
```bash
# Check Docker resources
docker stats

# Increase Docker memory (Docker Desktop settings)
# Recommended: 4GB+ RAM

# Check logs for bottlenecks
docker-compose logs -f api
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# API only
docker-compose logs -f api

# Qdrant only
docker-compose logs -f qdrant

# Last 100 lines
docker-compose logs --tail 100
```

### Resetting the System

```bash
# Stop and remove everything
docker-compose down -v

# Restart fresh
docker-compose up -d

# Wait for startup
sleep 10
curl http://localhost:8000/health
```

---

## 🔧 Docker Management

### Basic Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose stop

# Restart services
docker-compose restart

# Stop and remove containers
docker-compose down

# Stop and remove everything (including data)
docker-compose down -v

# View running containers
docker-compose ps

# View logs
docker-compose logs -f

# Rebuild after code changes
docker-compose up -d --build

# Execute command in container
docker-compose exec api bash
```

### Monitoring

```bash
# Check resource usage
docker stats

# Check container health
docker-compose ps

# Test API health
curl http://localhost:8000/health
```

---

## 🔐 Security Considerations

### Production Deployment

⚠️ **This is a development setup. For production:**

1. **Change Default Secrets**
   - Update `SECRET_KEY` in config
   - Use strong passwords
   - Enable authentication

2. **Enable HTTPS**
   - Use reverse proxy (nginx, traefik)
   - Obtain SSL certificate
   - Configure CORS properly

3. **Secure Database**
   - Enable Qdrant authentication
   - Use encrypted connections
   - Regular backups

4. **Rate Limiting**
   - Implement request limits
   - Add authentication/authorization
   - Monitor for abuse

5. **Data Privacy**
   - Comply with GDPR/privacy laws
   - Encrypt sensitive data
   - Implement data retention policies
   - Get proper consent for data processing

---

## 📊 Performance

### Benchmarks

| Operation | Average Time | Notes |
|-----------|-------------|-------|
| Face Detection | 100-300ms | Depends on image size |
| Face Embedding | 50-150ms | Using CPU |
| Vector Search | 10-50ms | Up to 10k vectors |
| Full Upload | 500-1000ms | End-to-end |
| Search by ID | 10-30ms | Metadata search |

### Optimization Tips

1. **Use GPU** - Set `USE_GPU=true` for 10x faster processing
2. **Image Size** - Resize large images before upload
3. **Batch Processing** - Process multiple images together
4. **Caching** - Implement Redis for frequently accessed data
5. **Indexing** - Qdrant automatically optimizes indexes

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Getting Started

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to functions
- Include type hints
- Write unit tests
- Update documentation

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/family-finder-aI4life.git
cd family-finder-aI4life

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_all_endpoints.py

# Start development server
docker-compose up -d
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Thndev05** - *Initial work* - [GitHub](https://github.com/thndev05)

---

## 🙏 Acknowledgments

- **ArcFace** - Face recognition model
- **MTCNN** - Face detection model
- **Qdrant** - Vector database
- **FastAPI** - Web framework
- **FGNET** - Test dataset

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/thndev05/family-finder-aI4life/issues)
- **Documentation**: http://localhost:8000/docs (when running)
- **Email**: thndev05@example.com

---

## 🗺️ Roadmap

### Planned Features

- [ ] Web UI for non-technical users
- [ ] Mobile app (iOS/Android)
- [ ] Multi-language support
- [ ] Advanced age progression
- [ ] Video face extraction
- [ ] Real-time notification system
- [ ] Integration with law enforcement databases
- [ ] Blockchain for data integrity

### Version History

- **v1.0.0** (2024-11) - Initial release
  - Face recognition with ArcFace
  - Bilateral matching
  - Confidence scoring
  - Docker deployment

---

<div align="center">

**Made with ❤️ for families searching for their loved ones**

[⭐ Star us on GitHub](https://github.com/thndev05/family-finder-aI4life) | [🐛 Report Bug](https://github.com/thndev05/family-finder-aI4life/issues) | [💡 Request Feature](https://github.com/thndev05/family-finder-aI4life/issues)

</div>
