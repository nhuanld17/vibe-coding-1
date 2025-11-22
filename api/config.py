"""
Configuration settings for Missing Person AI API.

This module provides centralized configuration management using
Pydantic Settings with environment variable support.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    app_name: str = Field(default="Missing Person AI API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    
    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    qdrant_timeout: float = Field(default=60.0, description="Qdrant timeout in seconds")
    
    # PostgreSQL Configuration (for future use)
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_user: str = Field(default="missing_person_user", description="PostgreSQL user")
    postgres_password: str = Field(default="secure_password_123", description="PostgreSQL password")
    postgres_db: str = Field(default="missing_person_db", description="PostgreSQL database")
    
    # Model Configuration
    arcface_model_path: str = Field(
        default="./models/weights/arcface_r100_v1.onnx",
        description="Path to ArcFace ONNX model"
    )
    use_gpu: bool = Field(default=False, description="Use GPU for inference")
    
    # Detection and Matching Thresholds
    face_confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for face detection"
    )
    similarity_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for face matching"
    )
    top_k_matches: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of matches to return"
    )
    
    # MinIO Configuration (for future use)
    minio_endpoint: str = Field(default="localhost:9000", description="MinIO endpoint")
    minio_access_key: str = Field(default="minioadmin", description="MinIO access key")
    minio_secret_key: str = Field(default="minioadmin123", description="MinIO secret key")
    minio_bucket_name: str = Field(default="missing-person-images", description="MinIO bucket name")
    minio_use_ssl: bool = Field(default=False, description="Use SSL for MinIO")
    
    # Upload Configuration
    max_upload_size_mb: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum upload size in MB"
    )
    allowed_image_types: str = Field(
        default="jpg,jpeg,png,gif,webp",
        description="Allowed image file types (comma-separated)"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/app.log", description="Log file path")
    
    # CORS Configuration
    cors_origins: str = Field(
        default="*",
        description="CORS allowed origins (comma-separated)"
    )
    cors_methods: str = Field(
        default="GET,POST,PUT,DELETE,OPTIONS",
        description="CORS allowed methods (comma-separated)"
    )
    cors_headers: str = Field(
        default="*",
        description="CORS allowed headers (comma-separated)"
    )
    
    # Rate Limiting (for future implementation)
    rate_limit_requests: int = Field(
        default=100,
        description="Rate limit: requests per minute"
    )
    rate_limit_window: int = Field(
        default=60,
        description="Rate limit window in seconds"
    )
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT tokens"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration in minutes"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables
        
    def get_cors_origins(self) -> list:
        """Get CORS origins as list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    def get_cors_methods(self) -> list:
        """Get CORS methods as list."""
        return [method.strip() for method in self.cors_methods.split(",")]
    
    def get_cors_headers(self) -> list:
        """Get CORS headers as list."""
        if self.cors_headers == "*":
            return ["*"]
        return [header.strip() for header in self.cors_headers.split(",")]
    
    def get_allowed_image_types(self) -> set:
        """Get allowed image types as set."""
        return {ext.strip().lower() for ext in self.allowed_image_types.split(",")}
    
    def get_max_upload_size_bytes(self) -> int:
        """Get maximum upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024
    
    def validate_model_path(self) -> bool:
        """Validate that the ArcFace model exists."""
        return os.path.exists(self.arcface_model_path)
    
    def get_database_url(self) -> str:
        """Get PostgreSQL database URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    def get_qdrant_config(self) -> dict:
        """Get Qdrant configuration dictionary."""
        return {
            "host": self.qdrant_host,
            "port": self.qdrant_port,
            "api_key": self.qdrant_api_key,
            "timeout": self.qdrant_timeout
        }
    
    def get_model_config(self) -> dict:
        """Get model configuration dictionary."""
        return {
            "arcface_model_path": self.arcface_model_path,
            "use_gpu": self.use_gpu,
            "face_confidence_threshold": self.face_confidence_threshold,
            "similarity_threshold": self.similarity_threshold
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get application settings instance.
    
    Returns:
        Settings instance
    """
    return settings


# Validate critical settings on import
if not settings.validate_model_path():
    import warnings
    warnings.warn(
        f"ArcFace model not found at {settings.arcface_model_path}. "
        f"Please download the model before starting the API.",
        UserWarning
    )
