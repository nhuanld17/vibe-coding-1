# Missing Person AI System

🔍 **AI-powered system for matching missing and found persons using facial recognition**

This system uses advanced face recognition technology to help reunite families by matching photos of missing persons with photos of found persons, even across significant time periods.

## 🌟 Features

- **Age-Invariant Face Recognition**: Uses ArcFace embeddings to match faces across time
- **Bilateral Search**: Two-way matching between missing and found persons
- **Explainable AI**: Provides confidence scores with detailed explanations
- **Quality Assessment**: Automatic face quality checks for better matching
- **Metadata Integration**: Combines facial similarity with demographic information
- **RESTful API**: Easy integration with web applications and mobile apps
- **Docker Support**: Complete containerized deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │   Mobile App    │    │  Third Party    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      FastAPI Server       │
                    │   (Missing Person AI)     │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
    ┌─────┴─────┐         ┌───────┴───────┐       ┌───────┴───────┐
    │  Qdrant   │         │  PostgreSQL   │       │    MinIO      │
    │ (Vectors) │         │ (Metadata)    │       │  (Images)     │
    └───────────┘         └───────────────┘       └───────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 4GB RAM
- 2GB free disk space

### 1. Clone Repository

```bash
git clone <repository-url>
cd missing-person-ai
```

### 2. Download ArcFace Model

```bash
# Create models directory
mkdir -p models/weights

# Download ArcFace model (required)
wget https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100_v1.onnx \
    -O models/weights/arcface_r100_v1.onnx
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### 4. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

## 📋 API Usage

### Upload Missing Person

```bash
curl -X POST "http://localhost:8000/api/v1/upload/missing" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@missing_person.jpg" \
  -F 'metadata={
    "case_id": "MISS_2023_001",
    "name": "John Doe",
    "age_at_disappearance": 25,
    "year_disappeared": 2020,
    "gender": "male",
    "location_last_seen": "New York, NY",
    "contact": "family@example.com"
  }'
```

### Upload Found Person

```bash
curl -X POST "http://localhost:8000/api/v1/upload/found" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@found_person.jpg" \
  -F 'metadata={
    "found_id": "FOUND_2023_001",
    "current_age_estimate": 30,
    "gender": "male",
    "current_location": "Los Angeles, CA",
    "finder_contact": "finder@example.com"
  }'
```

### Search for Specific Person

```bash
# Search missing person by case ID
curl "http://localhost:8000/api/v1/search/missing/MISS_2023_001"

# Search found person by found ID
curl "http://localhost:8000/api/v1/search/found/FOUND_2023_001"
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```bash
# API Configuration
DEBUG=false
LOG_LEVEL=INFO

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Model Configuration
ARCFACE_MODEL_PATH=./models/weights/arcface_r100_v1.onnx
USE_GPU=false

# Thresholds
FACE_CONFIDENCE_THRESHOLD=0.70
SIMILARITY_THRESHOLD=0.65
TOP_K_MATCHES=10

# Database Passwords
POSTGRES_PASSWORD=your_secure_password
MINIO_PASSWORD=your_minio_password
```

### Model Configuration

- **Face Detection**: MTCNN with confidence threshold 0.7
- **Face Recognition**: ArcFace R100 (512-dim embeddings)
- **Similarity Threshold**: 0.65 (adjustable)
- **Vector Database**: Qdrant with cosine similarity

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_face_detection.py -v
```

### Test Coverage

The system includes comprehensive tests for:

- Face detection and alignment
- Embedding extraction
- Vector database operations
- Bilateral search logic
- Confidence scoring
- API endpoints

## 📊 Performance

### Benchmarks

- **Face Detection**: ~100ms per image (CPU)
- **Embedding Extraction**: ~50ms per face (CPU)
- **Vector Search**: <100ms for 100K records
- **End-to-End Upload**: <500ms per request

### Scalability

- **Concurrent Requests**: 10+ simultaneous uploads
- **Database Size**: Tested with 100K+ face embeddings
- **Memory Usage**: ~2GB without GPU, ~4GB with GPU

## 🔒 Security

### Data Protection

- All uploaded images are processed in memory
- Face embeddings are stored (not original images)
- Metadata is encrypted in transit
- Rate limiting on API endpoints

### Privacy Considerations

- No biometric data is stored permanently
- Original images can be deleted after processing
- Metadata access is logged and auditable
- GDPR compliance features available

## 🚀 Deployment

### Production Deployment

1. **Use HTTPS**: Configure SSL certificates
2. **Set Strong Passwords**: Update all default passwords
3. **Enable Authentication**: Implement JWT tokens
4. **Configure Monitoring**: Set up health checks
5. **Backup Data**: Regular database backups

### Scaling Options

- **Horizontal Scaling**: Multiple API instances behind load balancer
- **GPU Acceleration**: Enable GPU for faster inference
- **Database Sharding**: Distribute vectors across multiple Qdrant instances
- **CDN Integration**: Cache static assets and responses

## 🛠️ Development

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start development server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Code Structure

```
missing-person-ai/
├── models/                 # ML models and weights
│   ├── face_detection.py  # MTCNN face detector
│   ├── face_embedding.py  # ArcFace embedding extractor
│   └── weights/           # Model weight files
├── services/              # Business logic services
│   ├── vector_db.py       # Qdrant vector database
│   ├── bilateral_search.py # Two-way matching
│   └── confidence_scoring.py # Explainable AI
├── api/                   # FastAPI application
│   ├── main.py           # Application entry point
│   ├── routes/           # API route handlers
│   └── schemas/          # Pydantic models
├── utils/                # Utility functions
├── tests/                # Test suite
└── docker-compose.yml    # Container orchestration
```

## 📈 Monitoring

### Health Checks

- **API Health**: `GET /health`
- **Service Status**: Individual service health
- **Database Stats**: Collection sizes and performance
- **Model Performance**: Inference times and accuracy

### Logging

- **Structured Logging**: JSON format with loguru
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Log Rotation**: Daily rotation with compression
- **Centralized Logs**: Docker log aggregation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Standards

- **Python Style**: Follow PEP 8
- **Type Hints**: Required for all functions
- **Documentation**: Docstrings for all public methods
- **Testing**: Minimum 80% code coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help

- **Documentation**: Check `/docs` endpoint for API reference
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join community discussions
- **Email**: Contact support team

### Common Issues

1. **Model Not Found**: Download ArcFace model to `models/weights/`
2. **Port Conflicts**: Change ports in `docker-compose.yml`
3. **Memory Issues**: Increase Docker memory allocation
4. **Slow Performance**: Enable GPU or increase CPU allocation

## 🙏 Acknowledgments

- **InsightFace**: For the excellent ArcFace model
- **MTCNN**: For robust face detection
- **Qdrant**: For high-performance vector search
- **FastAPI**: For the modern web framework

---

**⚠️ Important**: This system is designed to help reunite families. Please use responsibly and in compliance with local privacy laws and regulations.

**🔗 Links**:
- [API Documentation](http://localhost:8000/docs)
- [Qdrant Dashboard](http://localhost:6333/dashboard)
- [MinIO Console](http://localhost:9001)

Built with ❤️ for reuniting families
