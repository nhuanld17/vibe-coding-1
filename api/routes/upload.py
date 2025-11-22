"""
Upload routes for Missing Person AI API.

This module handles file uploads for missing and found persons,
including image processing, face detection, and similarity matching.
"""

import time
import numpy as np
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from loguru import logger
import json

from ..dependencies import (
    SettingsDep, FaceDetectorDep, EmbeddingExtractorDep, 
    VectorDBDep, BilateralSearchDep, ConfidenceScoringDep
)
from ..schemas.models import (
    MissingPersonMetadata, FoundPersonMetadata, UploadResponse,
    FaceQualityMetrics, MatchResult, ConfidenceExplanation, ConfidenceFactor
)
from utils.validation import validate_file_upload, validate_missing_person_metadata, validate_found_person_metadata
from utils.identifiers import generate_case_id, generate_found_id
from utils.image_processing import load_image_from_bytes, normalize_image_orientation, enhance_image_quality
from services.cloudinary_service import upload_image_to_cloudinary
from services.email_service import send_missing_person_profile_email, send_found_person_profile_email


router = APIRouter()


def process_image_and_extract_face(
    image_bytes: bytes,
    face_detector,
    settings
) -> tuple:
    """
    Process image and extract the largest face.
    
    Returns:
        Tuple of (face_image, face_quality_metrics, error_message)
    """
    try:
        # Normalize image orientation
        normalized_bytes = normalize_image_orientation(image_bytes)
        
        # Load image
        image = load_image_from_bytes(normalized_bytes)
        
        # Enhance image quality
        enhanced_image = enhance_image_quality(image)
        
        # Extract largest face
        face_image = face_detector.extract_largest_face(
            enhanced_image,
            align=True,
            confidence_threshold=settings.face_confidence_threshold
        )
        
        if face_image is None:
            return None, None, "No face detected in the image"
        
        # Check face quality
        is_good_quality, quality_metrics = face_detector.check_face_quality(face_image)
        
        face_quality = FaceQualityMetrics(
            sharpness=quality_metrics['sharpness'],
            brightness=quality_metrics['brightness'],
            contrast=quality_metrics['contrast'],
            is_sharp=quality_metrics['is_sharp'],
            is_bright_enough=quality_metrics['is_bright_enough'],
            is_contrasted=quality_metrics['is_contrasted']
        )
        
        if not is_good_quality:
            logger.warning(f"Poor face quality detected: {quality_metrics}")
        
        return face_image, face_quality, None
        
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        return None, None, f"Image processing failed: {str(e)}"


def format_match_results(
    matches: List[dict],
    confidence_scoring
) -> List[MatchResult]:
    """Format match results with confidence scoring."""
    formatted_matches = []
    
    for match in matches:
        try:
            # Calculate confidence
            confidence_level, confidence_score, explanation = confidence_scoring.calculate_confidence(match)
            
            # Format confidence factors
            factors = {}
            for factor_name, factor_data in explanation.get('factors', {}).items():
                factors[factor_name] = ConfidenceFactor(
                    score=factor_data['score'],
                    weight=factor_data['weight'],
                    contribution=factor_data['contribution'],
                    description=factor_data['description']
                )
            
            # Create confidence explanation
            confidence_explanation = ConfidenceExplanation(
                confidence_level=confidence_level.value,
                confidence_score=confidence_score,
                factors=factors,
                reasons=explanation.get('reasons', []),
                summary=explanation.get('summary', ''),
                recommendations=explanation.get('recommendations', []),
                threshold_info=explanation.get('threshold_info', {})
            )
            
            # Extract contact information
            payload = match.get('payload', {})
            contact = payload.get('contact') or payload.get('finder_contact', 'No contact available')
            
            # Extract image_url from payload (if available)
            match_image_url = payload.get('image_url')
            
            # Create match result
            match_result = MatchResult(
                id=match['id'],
                face_similarity=match['face_similarity'],
                metadata_similarity=match['metadata_similarity'],
                combined_score=match['combined_score'],
                confidence_level=confidence_level.value,
                confidence_score=confidence_score,
                explanation=confidence_explanation,
                contact=contact,
                metadata=payload,
                image_url=match_image_url
            )
            
            formatted_matches.append(match_result)
            
        except Exception as e:
            logger.error(f"Failed to format match result: {str(e)}")
            continue
    
    return formatted_matches


def _generate_timestamp_id(prefix: str) -> str:
    """
    Generate a simple timestamp-based identifier that is URL-safe and
    matches the repository's relaxed validation pattern (A-Za-z0-9_-).

    Example: MISS_1734739200123
    """
    # Use milliseconds to reduce collision chance in quick successive requests
    millis = int(time.time() * 1000)
    return f"{prefix}_{millis}"


@router.post("/missing", response_model=UploadResponse)
async def upload_missing_person(
    settings: SettingsDep,
    face_detector: FaceDetectorDep,
    embedding_extractor: EmbeddingExtractorDep,
    vector_db: VectorDBDep,
    bilateral_search: BilateralSearchDep,
    confidence_scoring: ConfidenceScoringDep,
    image: UploadFile = File(..., description="Face image of the missing person"),
    name: str = Form(..., description="Full name"),
    age_at_disappearance: int = Form(..., description="Age when disappeared"),
    year_disappeared: int = Form(..., description="Year of disappearance"),
    gender: str = Form(..., description="Gender: male, female, other, or unknown"),
    location_last_seen: str = Form(..., description="Last known location"),
    contact: str = Form(..., description="Contact information"),
    height_cm: Optional[int] = Form(None, description="Height in centimeters (optional)"),
    birthmarks: Optional[str] = Form(None, description="List of birthmarks/scars, separated by commas (optional)"),
    additional_info: Optional[str] = Form(None, description="Additional information (optional)")
):
    """
    Upload a missing person's photo and metadata.
    
    This endpoint:
    1. Validates the uploaded image and metadata
    2. Detects and extracts the face from the image
    3. Generates a face embedding
    4. Stores the data in the vector database
    5. Searches for potential matches in found persons
    6. Returns potential matches with confidence scores
    """
    start_time = time.time()
    
    try:
        # Read image file
        image_bytes = await image.read()
        
        # Validate file upload
        is_valid_file, file_error = validate_file_upload(image_bytes, image.filename)
        if not is_valid_file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file: {file_error}"
            )
        
        # Build metadata dictionary from form fields
        metadata_dict = {
            "case_id": _generate_timestamp_id("MISS"),  # Auto-generate case_id
            "name": name,
            "age_at_disappearance": age_at_disappearance,
            "year_disappeared": year_disappeared,
            "gender": gender.lower(),
            "location_last_seen": location_last_seen,
            "contact": contact
        }
        
        # Add optional fields
        
        if height_cm is not None:
            metadata_dict["height_cm"] = height_cm
        
        if birthmarks:
            # Parse comma-separated birthmarks
            birthmarks_list = [mark.strip() for mark in birthmarks.split(",") if mark.strip()]
            if birthmarks_list:
                metadata_dict["birthmarks"] = birthmarks_list
        
        if additional_info:
            metadata_dict["additional_info"] = additional_info
        
        # Validate metadata
        try:
            missing_metadata = MissingPersonMetadata(**metadata_dict)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metadata: {str(e)}"
            )
        
        # Additional metadata validation
        is_valid_metadata, metadata_errors = validate_missing_person_metadata(metadata_dict)
        if not is_valid_metadata:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Metadata validation failed: {'; '.join(metadata_errors)}"
            )
        
        # Process image and extract face
        face_image, face_quality, process_error = process_image_and_extract_face(
            image_bytes, face_detector, settings
        )
        
        if process_error:
            return UploadResponse(
                success=False,
                message=process_error,
                face_quality=face_quality,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Extract face embedding
        try:
            embedding = embedding_extractor.extract_embedding(face_image)
        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract face embedding"
            )
        
        # Upload image to Cloudinary (if configured) - BEFORE storing in database
        image_url = None
        if settings.cloudinary_cloud_name and settings.cloudinary_api_key and settings.cloudinary_api_secret:
            try:
                # Use original image bytes for Cloudinary upload
                public_id = f"missing_{missing_metadata.case_id}"
                upload_result = upload_image_to_cloudinary(
                    image_bytes=image_bytes,
                    folder=settings.cloudinary_folder_missing,
                    public_id=public_id
                )
                image_url = upload_result.get("secure_url")
                logger.info(f"Image uploaded to Cloudinary: {image_url}")
                # Add image_url to metadata to store in database
                metadata_dict['image_url'] = image_url
            except Exception as e:
                logger.warning(f"Failed to upload image to Cloudinary: {str(e)}")
                # Don't fail the upload if Cloudinary upload fails
        
        # Store in vector database
        try:
            point_id = vector_db.insert_missing_person(embedding, metadata_dict)
            logger.info(f"Stored missing person: {missing_metadata.name} (ID: {point_id})")
        except Exception as e:
            logger.error(f"Database insertion failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store data in database"
            )
        
        # Search for potential matches in found persons
        potential_matches = []
        try:
            matches = bilateral_search.search_for_found(
                embedding, metadata_dict, limit=settings.top_k_matches
            )
            potential_matches = format_match_results(matches, confidence_scoring)
            logger.info(f"Found {len(potential_matches)} potential matches for missing person {missing_metadata.name}")
        except Exception as e:
            logger.error(f"Search for matches failed: {str(e)}")
            # Don't fail the upload if search fails
        
        processing_time = (time.time() - start_time) * 1000
        
        # Send email notification if SMTP is configured
        if settings.smtp_enabled and settings.smtp_host and settings.smtp_user and settings.smtp_password and settings.smtp_from_email:
            try:
                send_missing_person_profile_email(
                    smtp_host=settings.smtp_host,
                    smtp_port=settings.smtp_port,
                    smtp_user=settings.smtp_user,
                    smtp_password=settings.smtp_password,
                    from_email=settings.smtp_from_email,
                    contact=contact,
                    metadata=metadata_dict,
                    case_id=missing_metadata.case_id,
                    image_url=image_url,
                    use_tls=settings.smtp_use_tls
                )
                logger.info(f"Email notification sent for missing person profile: {missing_metadata.case_id}")
            except Exception as e:
                logger.error(f"Failed to send email notification: {str(e)}")
                # Don't fail the upload if email fails
        
        return UploadResponse(
            success=True,
            message=f"Missing person '{missing_metadata.name}' uploaded successfully",
            point_id=point_id,
            potential_matches=potential_matches,
            face_quality=face_quality,
            processing_time_ms=processing_time,
            case_id=missing_metadata.case_id,
            image_url=image_url
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload missing person failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during upload"
        )


@router.post("/found", response_model=UploadResponse)
async def upload_found_person(
    settings: SettingsDep,
    face_detector: FaceDetectorDep,
    embedding_extractor: EmbeddingExtractorDep,
    vector_db: VectorDBDep,
    bilateral_search: BilateralSearchDep,
    confidence_scoring: ConfidenceScoringDep,
    image: UploadFile = File(..., description="Face image of the found person"),
    name: Optional[str] = Form(None, description="Name of the found person (optional)"),
    current_age_estimate: int = Form(..., description="Estimated current age"),
    gender: str = Form(..., description="Gender: male, female, other, or unknown"),
    current_location: str = Form(..., description="Current location"),
    finder_contact: str = Form(..., description="Finder contact information"),
    visible_marks: Optional[str] = Form(None, description="List of visible marks/scars, separated by commas (optional)"),
    current_condition: Optional[str] = Form(None, description="Current condition/status (optional)"),
    additional_info: Optional[str] = Form(None, description="Additional information (optional)")
):
    """
    Upload a found person's photo and metadata.
    
    This endpoint:
    1. Validates the uploaded image and metadata
    2. Detects and extracts the face from the image
    3. Generates a face embedding
    4. Stores the data in the vector database
    5. Searches for potential matches in missing persons
    6. Returns potential matches with confidence scores
    """
    start_time = time.time()
    
    try:
        # Read image file
        image_bytes = await image.read()
        
        # Validate file upload
        is_valid_file, file_error = validate_file_upload(image_bytes, image.filename)
        if not is_valid_file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file: {file_error}"
            )
        
        # Build metadata dictionary from form fields
        metadata_dict = {
            "found_id": _generate_timestamp_id("FOUND"),  # Auto-generate found_id
            "current_age_estimate": current_age_estimate,
            "gender": gender.lower(),
            "current_location": current_location,
            "finder_contact": finder_contact
        }
        
        # Add optional fields
        
        if name:
            metadata_dict["name"] = name
        
        if visible_marks:
            # Parse comma-separated visible marks
            marks_list = [mark.strip() for mark in visible_marks.split(",") if mark.strip()]
            if marks_list:
                metadata_dict["visible_marks"] = marks_list
        
        if current_condition:
            metadata_dict["current_condition"] = current_condition
        
        if additional_info:
            metadata_dict["additional_info"] = additional_info
        
        # Validate metadata
        try:
            found_metadata = FoundPersonMetadata(**metadata_dict)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metadata: {str(e)}"
            )
        
        # Additional metadata validation
        is_valid_metadata, metadata_errors = validate_found_person_metadata(metadata_dict)
        if not is_valid_metadata:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Metadata validation failed: {'; '.join(metadata_errors)}"
            )
        
        # Process image and extract face
        face_image, face_quality, process_error = process_image_and_extract_face(
            image_bytes, face_detector, settings
        )
        
        if process_error:
            return UploadResponse(
                success=False,
                message=process_error,
                face_quality=face_quality,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Extract face embedding
        try:
            embedding = embedding_extractor.extract_embedding(face_image)
        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract face embedding"
            )
        
        # Upload image to Cloudinary (if configured) - BEFORE storing in database
        image_url = None
        if settings.cloudinary_cloud_name and settings.cloudinary_api_key and settings.cloudinary_api_secret:
            try:
                # Use original image bytes for Cloudinary upload
                public_id = f"found_{found_metadata.found_id}"
                upload_result = upload_image_to_cloudinary(
                    image_bytes=image_bytes,
                    folder=settings.cloudinary_folder_found,
                    public_id=public_id
                )
                image_url = upload_result.get("secure_url")
                logger.info(f"Image uploaded to Cloudinary: {image_url}")
                # Add image_url to metadata to store in database
                metadata_dict['image_url'] = image_url
            except Exception as e:
                logger.warning(f"Failed to upload image to Cloudinary: {str(e)}")
                # Don't fail the upload if Cloudinary upload fails
        
        # Store in vector database
        try:
            point_id = vector_db.insert_found_person(embedding, metadata_dict)
            logger.info(f"Stored found person: {found_metadata.found_id} (ID: {point_id})")
        except Exception as e:
            logger.error(f"Database insertion failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store data in database"
            )
        
        # Search for potential matches in missing persons
        potential_matches = []
        try:
            logger.info(f"Searching for missing persons matching found person {found_metadata.found_id}")
            logger.info(f"Search parameters: age={metadata_dict.get('current_age_estimate')}, gender={metadata_dict.get('gender')}")
            matches = bilateral_search.search_for_missing(
                embedding, metadata_dict, limit=settings.top_k_matches
            )
            logger.info(f"Bilateral search returned {len(matches)} raw matches")
            potential_matches = format_match_results(matches, confidence_scoring)
            logger.info(f"Formatted {len(potential_matches)} potential matches for found person {found_metadata.found_id}")
            if len(potential_matches) > 0:
                logger.info(f"Top match: face_similarity={potential_matches[0].face_similarity:.3f}, combined_score={potential_matches[0].combined_score:.3f}")
        except Exception as e:
            logger.error(f"Search for matches failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Don't fail the upload if search fails
        
        processing_time = (time.time() - start_time) * 1000
        
        # Send email notification if SMTP is configured
        if settings.smtp_enabled and settings.smtp_host and settings.smtp_user and settings.smtp_password and settings.smtp_from_email:
            try:
                send_found_person_profile_email(
                    smtp_host=settings.smtp_host,
                    smtp_port=settings.smtp_port,
                    smtp_user=settings.smtp_user,
                    smtp_password=settings.smtp_password,
                    from_email=settings.smtp_from_email,
                    finder_contact=finder_contact,
                    metadata=metadata_dict,
                    found_id=found_metadata.found_id,
                    image_url=image_url,
                    use_tls=settings.smtp_use_tls
                )
                logger.info(f"Email notification sent for found person profile: {found_metadata.found_id}")
            except Exception as e:
                logger.error(f"Failed to send email notification: {str(e)}")
                # Don't fail the upload if email fails
        
        return UploadResponse(
            success=True,
            message=f"Found person '{found_metadata.found_id}' uploaded successfully",
            point_id=point_id,
            potential_matches=potential_matches,
            face_quality=face_quality,
            processing_time_ms=processing_time,
            found_id=found_metadata.found_id,
            image_url=image_url
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload found person failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during upload"
        )
