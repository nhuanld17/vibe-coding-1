"""
Dependency injection for Missing Person AI API.

This module provides FastAPI dependencies for services and configurations.
"""

from functools import lru_cache
from typing import Annotated
from fastapi import Depends, HTTPException, status
from loguru import logger

from .config import Settings, get_settings
from models.face_detection import FaceDetector
from models.face_embedding import (
    FaceEmbeddingExtractor,  # Backward compatibility
    create_embedding_backend,
    BaseFaceEmbedder
)
from services.vector_db import VectorDatabaseService
from services.bilateral_search import BilateralSearchService
from services.confidence_scoring import ConfidenceScoringService
from services.cloudinary_service import configure_cloudinary


# Global service instances (initialized on startup)
_face_detector: FaceDetector = None
_embedding_extractor: BaseFaceEmbedder = None  # Use base class for type hint
_vector_db: VectorDatabaseService = None
_bilateral_search: BilateralSearchService = None
_confidence_scoring: ConfidenceScoringService = None


@lru_cache()
def get_cached_settings() -> Settings:
    """Get cached settings instance."""
    return get_settings()


def initialize_services(settings: Settings) -> None:
    """
    Initialize all services on application startup.
    
    Args:
        settings: Application settings
        
    Raises:
        RuntimeError: If service initialization fails
    """
    global _face_detector, _embedding_extractor, _vector_db, _bilateral_search, _confidence_scoring
    
    try:
        logger.info("Initializing services...")
        
        # Initialize Face Detector
        logger.info("Initializing face detector...")
        _face_detector = FaceDetector(
            min_face_size=40,
            device="GPU:0" if settings.use_gpu else "CPU:0"
        )
        logger.info("Face detector initialized successfully")
        
        # Initialize Face Embedding Extractor
        logger.info("Initializing face embedding extractor...")
        logger.info(f"Using embedding backend: {settings.embedding_backend}")
        
        try:
            if settings.embedding_backend.lower() == "insightface":
                # Use InsightFace backend (recommended)
                _embedding_extractor = create_embedding_backend(
                    backend_type="insightface",
                    use_gpu=settings.use_gpu,
                    model_name=settings.insightface_model_name
                )
                logger.info("InsightFace embedding backend initialized successfully")
                
            elif settings.embedding_backend.lower() == "onnx":
                # Use ONNX backend (deprecated)
                if not settings.validate_model_path():
                    raise RuntimeError(f"ArcFace model not found at {settings.arcface_model_path}")
                
                _embedding_extractor = create_embedding_backend(
                    backend_type="onnx",
                    model_path=settings.arcface_model_path,
                    use_gpu=settings.use_gpu,
                    deprecated=True
                )
                logger.warning("ONNX embedding backend initialized (DEPRECATED - may produce incorrect embeddings)")
                
            else:
                raise ValueError(
                    f"Unknown embedding backend: {settings.embedding_backend}. "
                    f"Supported: 'insightface', 'onnx'"
                )
                
        except ImportError as e:
            if "insightface" in str(e).lower():
                logger.error(
                    "InsightFace library is not installed. "
                    "Please install it with: pip install insightface"
                )
                raise RuntimeError(
                    "InsightFace library is required but not installed. "
                    "Install with: pip install insightface"
                )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize embedding backend: {str(e)}")
            raise RuntimeError(f"Embedding backend initialization failed: {str(e)}")
        
        # Initialize Vector Database
        logger.info("Initializing vector database...")
        _vector_db = VectorDatabaseService(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout
        )
        
        # Initialize collections
        _vector_db.initialize_collections(vector_size=512)
        logger.info("Vector database initialized successfully")
        
        # Initialize Bilateral Search Service
        logger.info("Initializing bilateral search service...")
        logger.info(f"Using face_search_threshold={settings.face_search_threshold} for missing-person search")
        _bilateral_search = BilateralSearchService(
            vector_db=_vector_db,
            face_threshold=settings.face_search_threshold,  # Use face_search_threshold as primary threshold
            metadata_weight=0.3,
            initial_search_threshold=settings.initial_search_threshold,
            combined_score_threshold=settings.combined_score_threshold,
            face_metadata_fallback_threshold=settings.face_metadata_fallback_threshold
        )
        logger.info("Bilateral search service initialized successfully")
        
        # Initialize Confidence Scoring Service
        logger.info("Initializing confidence scoring service...")
        _confidence_scoring = ConfidenceScoringService(
            face_weight=0.5,
            metadata_weight=0.2,
            age_weight=0.15,
            location_weight=0.1,
            features_weight=0.05
        )
        logger.info("Confidence scoring service initialized successfully")
        
        # Initialize Cloudinary (if credentials provided)
        if settings.cloudinary_cloud_name and settings.cloudinary_api_key and settings.cloudinary_api_secret:
            logger.info("Initializing Cloudinary...")
            configure_cloudinary(
                cloud_name=settings.cloudinary_cloud_name,
                api_key=settings.cloudinary_api_key,
                api_secret=settings.cloudinary_api_secret
            )
            logger.info("Cloudinary initialized successfully")
        else:
            logger.warning("Cloudinary credentials not provided. Image uploads to Cloudinary will be skipped.")
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Service initialization failed: {str(e)}")
        raise RuntimeError(f"Service initialization failed: {str(e)}")


def get_face_detector() -> FaceDetector:
    """
    Get face detector dependency.
    
    Returns:
        FaceDetector instance
        
    Raises:
        HTTPException: If service not initialized
    """
    if _face_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face detector service not available"
        )
    return _face_detector


def get_embedding_extractor() -> BaseFaceEmbedder:
    """
    Get face embedding extractor dependency.
    
    Returns:
        BaseFaceEmbedder instance (InsightFaceEmbedder or OnnxArcFaceEmbedder)
        
    Raises:
        HTTPException: If service not initialized
    """
    if _embedding_extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face embedding extractor service not available"
        )
    return _embedding_extractor


def get_vector_db() -> VectorDatabaseService:
    """
    Get vector database dependency.
    
    Returns:
        VectorDatabaseService instance
        
    Raises:
        HTTPException: If service not initialized
    """
    if _vector_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database service not available"
        )
    return _vector_db


def get_bilateral_search() -> BilateralSearchService:
    """
    Get bilateral search service dependency.
    
    Returns:
        BilateralSearchService instance
        
    Raises:
        HTTPException: If service not initialized
    """
    if _bilateral_search is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bilateral search service not available"
        )
    return _bilateral_search


def get_confidence_scoring() -> ConfidenceScoringService:
    """
    Get confidence scoring service dependency.
    
    Returns:
        ConfidenceScoringService instance
        
    Raises:
        HTTPException: If service not initialized
    """
    if _confidence_scoring is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Confidence scoring service not available"
        )
    return _confidence_scoring


def check_services_health() -> dict:
    """
    Check health of all services.
    
    Returns:
        Dictionary with service health status
    """
    health_status = {
        "face_detector": _face_detector is not None,
        "embedding_extractor": _embedding_extractor is not None,
        "vector_db": _vector_db is not None,
        "bilateral_search": _bilateral_search is not None,
        "confidence_scoring": _confidence_scoring is not None
    }
    
    # Check vector database connection
    if _vector_db is not None:
        try:
            db_health = _vector_db.health_check()
            health_status["vector_db_connection"] = db_health.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Vector DB health check failed: {str(e)}")
            health_status["vector_db_connection"] = False
    
    # Overall health
    health_status["overall"] = all(health_status.values())
    
    return health_status


# Type aliases for dependency injection
SettingsDep = Annotated[Settings, Depends(get_cached_settings)]
FaceDetectorDep = Annotated[FaceDetector, Depends(get_face_detector)]
EmbeddingExtractorDep = Annotated[BaseFaceEmbedder, Depends(get_embedding_extractor)]
VectorDBDep = Annotated[VectorDatabaseService, Depends(get_vector_db)]
BilateralSearchDep = Annotated[BilateralSearchService, Depends(get_bilateral_search)]
ConfidenceScoringDep = Annotated[ConfidenceScoringService, Depends(get_confidence_scoring)]
