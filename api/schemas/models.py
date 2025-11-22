"""
Pydantic models for Missing Person AI API.

This module defines request and response schemas for API endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class GenderEnum(str, Enum):
    """Gender enumeration."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class ConfidenceLevelEnum(str, Enum):
    """Confidence level enumeration."""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


# Request Models

class MissingPersonMetadata(BaseModel):
    """Missing person metadata schema."""
    case_id: Optional[str] = Field(None, description="Unique case identifier (auto-generated if not provided)", example="MISS_2023_001")
    name: str = Field(..., min_length=2, max_length=100, description="Full name", example="John Doe")
    age_at_disappearance: int = Field(..., ge=0, le=120, description="Age when disappeared", example=25)
    year_disappeared: int = Field(..., ge=1900, le=2024, description="Year of disappearance", example=2020)
    gender: GenderEnum = Field(..., description="Gender", example="male")
    location_last_seen: str = Field(..., min_length=3, max_length=200, description="Last known location", example="New York, NY")
    contact: str = Field(..., description="Contact information", example="family@example.com")
    height_cm: Optional[int] = Field(None, ge=50, le=250, description="Height in centimeters", example=175)
    birthmarks: Optional[List[str]] = Field(None, max_items=10, description="List of birthmarks/scars", example=["scar on left arm"])
    additional_info: Optional[str] = Field(None, max_length=1000, description="Additional information")

    @validator('birthmarks')
    def validate_birthmarks(cls, v):
        if v is not None:
            return [mark.strip() for mark in v if mark.strip()]
        return v


class FoundPersonMetadata(BaseModel):
    """Found person metadata schema."""
    found_id: Optional[str] = Field(None, description="Unique found person identifier (auto-generated if not provided)", example="FOUND_2023_001")
    name: Optional[str] = Field(None, min_length=2, max_length=100, description="Name of the found person (optional)", example="John Doe")
    current_age_estimate: int = Field(..., ge=0, le=120, description="Estimated current age", example=30)
    gender: GenderEnum = Field(..., description="Gender", example="male")
    current_location: str = Field(..., min_length=3, max_length=200, description="Current location", example="Los Angeles, CA")
    finder_contact: str = Field(..., description="Finder contact information", example="finder@example.com")
    visible_marks: Optional[List[str]] = Field(None, max_items=10, description="List of visible marks/scars", example=["scar on left arm"])
    current_condition: Optional[str] = Field(None, max_length=500, description="Current condition/status")
    additional_info: Optional[str] = Field(None, max_length=1000, description="Additional information")

    @validator('visible_marks')
    def validate_visible_marks(cls, v):
        if v is not None:
            return [mark.strip() for mark in v if mark.strip()]
        return v


class SearchFilters(BaseModel):
    """Search filters schema."""
    gender: Optional[GenderEnum] = Field(None, description="Gender filter")
    age_range: Optional[List[int]] = Field(None, min_items=2, max_items=2, description="Age range [min, max]", example=[20, 40])
    location: Optional[str] = Field(None, description="Location filter")

    @validator('age_range')
    def validate_age_range(cls, v):
        if v is not None:
            if len(v) != 2:
                raise ValueError("Age range must have exactly 2 values")
            if v[0] > v[1]:
                raise ValueError("Minimum age must be less than maximum age")
            if not all(0 <= age <= 120 for age in v):
                raise ValueError("Ages must be between 0 and 120")
        return v


class SearchParameters(BaseModel):
    """Search parameters schema."""
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    threshold: float = Field(default=0.65, ge=0.0, le=1.0, description="Similarity threshold")
    filters: Optional[SearchFilters] = Field(None, description="Search filters")


# Response Models

class FaceQualityMetrics(BaseModel):
    """Face quality metrics schema."""
    sharpness: float = Field(..., description="Sharpness score (Laplacian variance)")
    brightness: float = Field(..., description="Brightness score (mean pixel value)")
    contrast: float = Field(..., description="Contrast score (standard deviation)")
    is_sharp: bool = Field(..., description="Whether face is sharp enough")
    is_bright_enough: bool = Field(..., description="Whether face has good brightness")
    is_contrasted: bool = Field(..., description="Whether face has good contrast")


class ConfidenceFactor(BaseModel):
    """Confidence factor schema."""
    score: float = Field(..., ge=0.0, le=1.0, description="Factor score")
    weight: float = Field(..., ge=0.0, le=1.0, description="Factor weight")
    contribution: float = Field(..., ge=0.0, le=1.0, description="Factor contribution to overall score")
    description: str = Field(..., description="Human-readable description")


class ConfidenceExplanation(BaseModel):
    """Confidence explanation schema."""
    confidence_level: ConfidenceLevelEnum = Field(..., description="Confidence level")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    factors: Dict[str, ConfidenceFactor] = Field(..., description="Individual factor scores")
    reasons: List[str] = Field(..., description="List of reasons supporting the match")
    summary: str = Field(..., description="Overall summary of the match")
    recommendations: List[str] = Field(..., description="Recommended next steps")
    threshold_info: Dict[str, float] = Field(..., description="Confidence level thresholds")


class MatchResult(BaseModel):
    """Match result schema."""
    id: str = Field(..., description="Match ID")
    face_similarity: float = Field(..., ge=0.0, le=1.0, description="Face similarity score")
    metadata_similarity: float = Field(..., ge=0.0, le=1.0, description="Metadata similarity score")
    combined_score: float = Field(..., ge=0.0, le=1.0, description="Combined similarity score")
    confidence_level: ConfidenceLevelEnum = Field(..., description="Confidence level")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    explanation: ConfidenceExplanation = Field(..., description="Detailed confidence explanation")
    contact: str = Field(..., description="Contact information for this match")
    metadata: Dict[str, Any] = Field(..., description="Full metadata of the matched record")


class UploadResponse(BaseModel):
    """Upload response schema."""
    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Response message")
    point_id: Optional[str] = Field(None, description="Database point ID")
    potential_matches: List[MatchResult] = Field(default_factory=list, description="Potential matches found")
    face_quality: Optional[FaceQualityMetrics] = Field(None, description="Face quality assessment")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    case_id: Optional[str] = Field(None, description="Auto-generated case identifier (for missing uploads)")
    found_id: Optional[str] = Field(None, description="Auto-generated found identifier (for found uploads)")


class SearchResponse(BaseModel):
    """Search response schema."""
    success: bool = Field(..., description="Whether search was successful")
    message: str = Field(..., description="Response message")
    matches: List[MatchResult] = Field(default_factory=list, description="Search results")
    total_found: int = Field(..., description="Total number of matches found")
    search_parameters: SearchParameters = Field(..., description="Search parameters used")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Overall health status")
    timestamp: float = Field(..., description="Health check timestamp (Unix time)")
    services: Dict[str, bool] = Field(..., description="Individual service health status")
    database_stats: Optional[Dict[str, Any]] = Field(None, description="Database statistics")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response schema."""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class APIInfo(BaseModel):
    """API information schema."""
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")
    documentation_url: str = Field(..., description="Documentation URL")


# Statistics Models

class CollectionStats(BaseModel):
    """Collection statistics schema."""
    collection_name: str = Field(..., description="Collection name")
    points_count: int = Field(..., description="Number of points in collection")
    vector_size: int = Field(..., description="Vector dimension")
    last_updated: Optional[float] = Field(None, description="Last update timestamp (Unix time)")


class SystemStats(BaseModel):
    """System statistics schema."""
    missing_persons: CollectionStats = Field(..., description="Missing persons collection stats")
    found_persons: CollectionStats = Field(..., description="Found persons collection stats")
    total_records: int = Field(..., description="Total number of records")
    total_searches_today: Optional[int] = Field(None, description="Number of searches today")
    average_processing_time_ms: Optional[float] = Field(None, description="Average processing time")
    uptime_seconds: Optional[float] = Field(None, description="System uptime in seconds")


# Validation Models

class ValidationResult(BaseModel):
    """Validation result schema."""
    is_valid: bool = Field(..., description="Whether input is valid")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
