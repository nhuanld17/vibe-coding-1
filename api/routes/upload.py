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
            is_contrasted=quality_metrics['is_contrasted'],
            quality_score=quality_metrics['quality_score']
        )
        
        if not is_good_quality:
            logger.warning(f"Poor face quality detected: {quality_metrics}")
        
        return face_image, face_quality, None
        
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        return None, None, f"Image processing failed: {str(e)}"


def format_match_results(
    matches: List[dict],
    confidence_scoring,
    min_confidence_threshold: float = 0.50  # Allow LOW matches (0.50-0.60) for extra context
) -> List[MatchResult]:
    """
    Format match results with confidence scoring.
    
    Args:
        matches: List of match dictionaries from bilateral search
        confidence_scoring: Confidence scoring service
        min_confidence_threshold: Minimum confidence score to include (default 0.50 = reject VERY_LOW, keep LOW)
    
    Returns:
        List of formatted match results (only matches with confidence >= min_confidence_threshold)
    """
    formatted_matches = []
    rejected_count = 0
    
    for match in matches:
        try:
            # Calculate confidence
            confidence_level, confidence_score, explanation = confidence_scoring.calculate_confidence(match)
            
            # STRICT FILTER: Reject matches with confidence score below threshold
            # This prevents showing very low confidence matches (like 27.4%) to users
            if confidence_score < min_confidence_threshold:
                rejected_count += 1
                logger.debug(
                    f"Rejecting match {match.get('id', 'unknown')} with low confidence: "
                    f"confidence_score={confidence_score:.3f} < threshold={min_confidence_threshold:.3f} "
                    f"(level={confidence_level.value})"
                )
                continue
            
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
    
    if rejected_count > 0:
        logger.info(f"Filtered out {rejected_count} matches with confidence < {min_confidence_threshold:.2f}")
    
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
    gender: str = Form(..., description="Gender: male or female"),
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
    gender: str = Form(..., description="Gender: male or female"),
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


# ============================================================================
# BATCH/MULTI-IMAGE UPLOAD ENDPOINTS
# ============================================================================

import asyncio
from ..schemas.models import MultiImageUploadResponse, UploadedImageInfo, FailedImageInfo, ReferenceImageInfo
from utils.image_helpers import calculate_age_at_photo, compress_image_if_needed


async def process_single_image(
    idx: int,
    image: UploadFile,
    img_meta: Optional[dict],
    shared_params: dict,
    face_detector,
    embedding_extractor,
    settings
) -> dict:
    """
    Process a single image concurrently.
    
    UPDATED: Now saves ALL images regardless of face detection status.
    Images without faces or low quality are marked as "reference only".
    
    Args:
        idx: Image index in batch
        image: Uploaded file
        img_meta: Per-image metadata (photo_year, etc.)
        shared_params: Shared person metadata
        face_detector: Face detector instance
        embedding_extractor: Embedding extractor instance
        settings: Application settings
        
    Returns:
        Result dictionary with success status, validation flags, and data/error
    """
    from datetime import datetime as dt
    
    try:
        # 1. Read & validate FILE (not face - just file format)
        image_bytes = await image.read()
        is_valid, error = validate_file_upload(image_bytes, image.filename)
        if not is_valid:
            # Only reject truly invalid files (corrupt, wrong format, etc.)
            return {"success": False, "idx": idx, "filename": image.filename, "error": error}
        
        # 2. Compress if needed (before Cloudinary upload)
        try:
            compressed_bytes, was_compressed = compress_image_if_needed(
                image_bytes, 
                max_size_mb=2.0,
                quality=85
            )
            if was_compressed:
                logger.info(f"Image {idx} compressed: {len(image_bytes)/(1024*1024):.2f}MB -> {len(compressed_bytes)/(1024*1024):.2f}MB")
                image_bytes = compressed_bytes
        except Exception as e:
            logger.warning(f"Image compression failed for {idx}, using original: {e}")
        
        # 3. Calculate age at photo (ALWAYS, even if no face detected)
        try:
            age_at_photo = calculate_age_at_photo(
                photo_year=img_meta.get('photo_year') if img_meta else None,
                year_disappeared=shared_params['year_disappeared'],
                age_at_disappearance=shared_params['age_at_disappearance']
            )
        except Exception as e:
            logger.error(f"Age calculation failed for image {idx}: {e}")
            # Return None (no age bonus will be applied, but image will still be used)
            age_at_photo = None
        
        # 4. Upload to Cloudinary FIRST (before face detection)
        #    This ensures we save ALL images, even if face detection fails
        image_url = None
        if settings.cloudinary_cloud_name:
            try:
                public_id = f"{shared_params['type']}_{shared_params['case_id']}_img_{idx}"
                result = upload_image_to_cloudinary(
                    image_bytes,
                    settings.cloudinary_folder_missing if shared_params['type'] == 'missing' else settings.cloudinary_folder_found,
                    public_id
                )
                image_url = result.get("secure_url")
                logger.info(f"Image {idx} uploaded to Cloudinary: {public_id}")
            except Exception as e:
                logger.warning(f"Cloudinary upload failed for image {idx}: {e}")
        
        # 5. Try to detect face
        face_image, quality, error_msg = process_image_and_extract_face(
            image_bytes, face_detector, settings
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL CHANGE: Don't reject if no face detected - save as reference
        # ═══════════════════════════════════════════════════════════════════
        
        if face_image is None:
            # Save as reference-only image
            logger.warning(f"Image {idx}: No face detected, saving as reference only")
            return {
                "success": True,  # ← SUCCESS despite no face!
                "idx": idx,
                "filename": image.filename,
                "embedding": None,
                "is_valid_for_matching": False,
                "validation_status": "no_face_detected",
                "validation_details": {
                    "face_detected": False,
                    "detection_confidence": 0.0,
                    "quality_score": 0.0,
                    "reason": error_msg or "MTCNN could not detect face in image. Image saved for reference purposes.",
                    "processing_timestamp": dt.utcnow().isoformat()
                },
                "age_at_photo": age_at_photo,
                "photo_year": img_meta.get('photo_year') if img_meta else None,
                "quality_score": 0.0,
                "image_url": image_url
            }
        
        # 6. Check quality threshold
        quality_score = quality.quality_score if quality else 0.0
        
        if quality_score < 0.60:
            # Low quality but face detected - save as reference only
            logger.warning(f"Image {idx}: Quality low ({quality_score:.2f}), saving as reference only")
            return {
                "success": True,  # ← SUCCESS despite low quality!
                "idx": idx,
                "filename": image.filename,
                "embedding": None,  # Don't extract embedding for low quality
                "is_valid_for_matching": False,
                "validation_status": "low_quality",
                "validation_details": {
                    "face_detected": True,
                    "detection_confidence": quality_score,
                    "quality_score": quality_score,
                    "reason": f"Face quality score {quality_score:.2f} below threshold 0.60. Image saved for reference.",
                    "processing_timestamp": dt.utcnow().isoformat()
                },
                "age_at_photo": age_at_photo,
                "photo_year": img_meta.get('photo_year') if img_meta else None,
                "quality_score": quality_score,
                "image_url": image_url
            }
        
        # 7. Extract embedding (valid image - face detected and good quality)
        embedding = embedding_extractor.extract_embedding(face_image)
        
        return {
            "success": True,
            "idx": idx,
            "filename": image.filename,
            "embedding": embedding,
            "is_valid_for_matching": True,  # ← Valid for matching!
            "validation_status": "valid",
            "validation_details": {
                "face_detected": True,
                "detection_confidence": quality_score,
                "quality_score": quality_score,
                "reason": "Face detected successfully with good quality.",
                "processing_timestamp": dt.utcnow().isoformat()
            },
            "age_at_photo": age_at_photo,
            "photo_year": img_meta.get('photo_year') if img_meta else None,
            "quality_score": quality_score,
            "image_url": image_url
        }
        
    except Exception as e:
        logger.error(f"Critical error processing image {idx} ({image.filename}): {e}", exc_info=True)
        return {"success": False, "idx": idx, "filename": image.filename, "error": str(e)}


@router.post("/missing/batch", response_model=MultiImageUploadResponse)
async def upload_missing_person_batch(
    settings: SettingsDep,
    face_detector: FaceDetectorDep,
    embedding_extractor: EmbeddingExtractorDep,
    vector_db: VectorDBDep,
    bilateral_search: BilateralSearchDep,
    confidence_scoring: ConfidenceScoringDep,
    # Multiple images (1-10)
    images: List[UploadFile] = File(..., description="1-10 images of the missing person"),
    # Shared person metadata
    name: str = Form(...),
    age_at_disappearance: int = Form(..., ge=0, le=120),
    year_disappeared: int = Form(..., ge=1900, le=2100),
    gender: str = Form(...),
    location_last_seen: str = Form(...),
    contact: str = Form(...),
    # Optional metadata
    height_cm: Optional[int] = Form(None, ge=50, le=250),
    birthmarks: Optional[str] = Form(None),
    additional_info: Optional[str] = Form(None),
    # Per-image metadata (JSON string)
    image_metadata_json: Optional[str] = Form(None, description='JSON array like [{"photo_year": 2010}, ...]')
):
    """
    Upload multiple images for a missing person (batch/multi-image upload).
    
    This endpoint supports 1-10 images per person for improved matching across age gaps.
    Images are processed in parallel for optimal performance.
    
    Args:
        images: List of 1-10 image files
        name: Full name of missing person
        age_at_disappearance: Age when person disappeared
        year_disappeared: Year of disappearance  
        gender: Gender (male/female)
        location_last_seen: Last known location
        contact: Contact information
        height_cm: Optional height in cm
        birthmarks: Optional comma-separated birthmarks/scars
        additional_info: Optional additional information
        image_metadata_json: Optional JSON array with per-image metadata
        
    Returns:
        MultiImageUploadResponse with upload results and potential matches
        
    Example:
        ```python
        files = [
            ("images", ("photo1.jpg", open("photo1.jpg", "rb"), "image/jpeg")),
            ("images", ("photo2.jpg", open("photo2.jpg", "rb"), "image/jpeg"))
        ]
        data = {
            "name": "John Doe",
            "age_at_disappearance": 25,
            "year_disappeared": 2020,
            "gender": "male",
            "location_last_seen": "New York, NY",
            "contact": "family@example.com",
            "image_metadata_json": '[{"photo_year": 2015}, {"photo_year": 2018}]'
        }
        response = requests.post("/api/v1/upload/missing/batch", files=files, data=data)
        ```
    """
    start_time = time.time()
    
    try:
        # Validate image count
        if len(images) == 0:
            raise HTTPException(400, "At least 1 image is required")
        if len(images) > 10:
            raise HTTPException(400, f"Maximum 10 images allowed, got {len(images)}")
        
        logger.info(f"Batch upload: {len(images)} images for '{name}'")
        
        # Parse image_metadata_json
        per_image_metadata = []
        if image_metadata_json:
            try:
                per_image_metadata = json.loads(image_metadata_json)
                if len(per_image_metadata) != len(images):
                    raise HTTPException(
                        400, 
                        f"image_metadata_json length ({len(per_image_metadata)}) must match images count ({len(images)})"
                    )
            except json.JSONDecodeError as e:
                raise HTTPException(400, f"Invalid JSON in image_metadata_json: {e}")
        
        # Generate case_id
        case_id = generate_case_id()
        
        # Build shared params for all images
        shared_params = {
            "case_id": case_id,
            "type": "missing",
            "year_disappeared": year_disappeared,
            "age_at_disappearance": age_at_disappearance
        }
        
        # Process all images CONCURRENTLY
        logger.info(f"Processing {len(images)} images in parallel...")
        tasks = [
            process_single_image(
                idx, img,
                per_image_metadata[idx] if idx < len(per_image_metadata) else None,
                shared_params,
                face_detector,
                embedding_extractor,
                settings
            )
            for idx, img in enumerate(images)
        ]
        results = await asyncio.gather(*tasks)
        
        # ═══════════════════════════════════════════════════════════════════
        # NEW: Separate by validation status (valid/reference/failed)
        # ═══════════════════════════════════════════════════════════════════
        valid_images = []  # is_valid_for_matching = True
        reference_images = []  # is_valid_for_matching = False
        failed_images = []  # success = False (file errors)
        
        for result in results:
            if not result["success"]:
                failed_images.append(
                    FailedImageInfo(
                        filename=result["filename"],
                        index=result["idx"],
                        reason=result["error"]
                    )
                )
            elif result.get("is_valid_for_matching", True):
                valid_images.append(result)
            else:
                reference_images.append(result)
        
        logger.info(
            f"Upload results: {len(valid_images)} valid, "
            f"{len(reference_images)} reference, {len(failed_images)} failed"
        )
        
        # Allow upload with reference-only images
        if not valid_images and not reference_images:
            raise HTTPException(
                400,
                f"No processable images. All {len(images)} images failed."
            )
        
        if not valid_images and reference_images:
            logger.warning(
                f"Case {case_id}: No valid images for matching, "
                f"but {len(reference_images)} reference images saved."
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # Build payloads for ALL images (valid + reference)
        # ═══════════════════════════════════════════════════════════════════
        all_images = valid_images + reference_images
        embeddings = []
        payloads = []
        
        for img_result in all_images:
            payload = {
                # Grouping & identification
                "case_id": case_id,
                "image_id": f"{case_id}_img_{img_result['idx']}",
                "image_index": img_result['idx'],
                "total_images": len(all_images),
                
                # Validation fields
                "is_valid_for_matching": img_result.get("is_valid_for_matching", True),
                "validation_status": img_result.get("validation_status", "valid"),
                "validation_details": img_result.get("validation_details", {}),
                
                # Per-image metadata
                "age_at_photo": img_result['age_at_photo'],
                "photo_year": img_result.get('photo_year'),
                "photo_quality_score": img_result.get('quality_score', 0.0),
                "image_url": img_result.get('image_url'),
                
                # Shared person metadata
                "name": name,
                "age_at_disappearance": age_at_disappearance,
                "year_disappeared": year_disappeared,
                "gender": gender.lower(),
                "location_last_seen": location_last_seen,
                "contact": contact,
                "height_cm": height_cm,
                "birthmarks": [m.strip() for m in birthmarks.split(",")] if birthmarks else [],
                "additional_info": additional_info
            }
            
            embeddings.append(img_result.get('embedding'))  # May be None!
            payloads.append(payload)
        
        # Batch insert to Qdrant
        logger.info(f"Inserting {len(embeddings)} records to Qdrant ({len(valid_images)} valid, {len(reference_images)} reference)...")
        point_ids = vector_db.insert_batch(
            collection_name="missing_persons",
            embeddings=embeddings,
            payloads=payloads
        )
        logger.info(f"✅ Inserted {len(point_ids)} records for case {case_id}")
        
        # ═══════════════════════════════════════════════════════════════════
        # Search ONLY with valid images (ignore reference images)
        # ═══════════════════════════════════════════════════════════════════
        potential_matches = []
        if valid_images:  # Only search if we have valid images
            try:
                # Build query_embeddings for multi-image search
                query_embeddings = [
                    {
                        "embedding": r['embedding'],
                        "age_at_photo": r['age_at_photo'],
                        "quality": r['quality_score']
                    }
                    for r in valid_images  # Changed from 'uploaded' to 'valid_images'
                ]
                
                # Use multi-image search for better matching
                logger.info(f"Searching with {len(query_embeddings)} images...")
                matches = bilateral_search.search_for_found_multi_image(
                    query_embeddings=query_embeddings,
                    query_metadata=payloads[0],  # Use first payload as representative
                    limit=settings.top_k_matches
                )
                
                potential_matches = format_match_results(matches, confidence_scoring)
                logger.info(f"Found {len(potential_matches)} potential matches (multi-image)")
                
            except Exception as e:
                logger.error(f"Multi-image search failed: {e}", exc_info=True)
                # Don't fail upload if search fails
        
        # ═══════════════════════════════════════════════════════════════════
        # Build response with valid + reference + failed images
        # ═══════════════════════════════════════════════════════════════════
        processing_time = (time.time() - start_time) * 1000
        
        response = MultiImageUploadResponse(
            success=True,
            message=f"Uploaded {len(valid_images)} valid image(s)" + 
                   (f", {len(reference_images)} reference image(s)" if reference_images else "") +
                   f" for '{name}'" +
                   (f" ({len(failed_images)} failed)" if failed_images else ""),
            case_id=case_id,
            total_images_uploaded=len(all_images),
            total_images_failed=len(failed_images),
            
            # Valid images (used for matching)
            valid_images=[
                UploadedImageInfo(
                    image_id=f"{case_id}_img_{r['idx']}",
                    image_index=r['idx'],
                    image_url=r.get('image_url'),
                    age_at_photo=r['age_at_photo'],
                    photo_year=r.get('photo_year'),
                    quality_score=r.get('quality_score', 0.0),
                    validation_status="valid"
                )
                for r in valid_images
            ],
            
            # Reference images (not used for matching)
            reference_images=[
                ReferenceImageInfo(
                    image_id=f"{case_id}_img_{r['idx']}",
                    image_index=r['idx'],
                    image_url=r.get('image_url'),
                    age_at_photo=r['age_at_photo'],
                    photo_year=r.get('photo_year'),
                    validation_status=r.get('validation_status', 'no_face_detected'),
                    reason=r.get('validation_details', {}).get('reason', 'Unknown')
                )
                for r in reference_images
            ],
            
            failed_images=failed_images,
            
            matching_images_count=len(valid_images),
            reference_images_count=len(reference_images),
            
            potential_matches=potential_matches,
            processing_time_ms=processing_time
        )
        
        logger.info(
            f"✅ Batch upload complete: {case_id} "
            f"({len(valid_images)} valid, {len(reference_images)} reference) "
            f"({processing_time:.1f}ms)"
        )
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch upload failed: {str(e)}"
        )


@router.post("/found/batch", response_model=MultiImageUploadResponse)
async def upload_found_person_batch(
    settings: SettingsDep,
    face_detector: FaceDetectorDep,
    embedding_extractor: EmbeddingExtractorDep,
    vector_db: VectorDBDep,
    bilateral_search: BilateralSearchDep,
    confidence_scoring: ConfidenceScoringDep,
    # Multiple images (1-10)
    images: List[UploadFile] = File(..., description="1-10 images of the found person"),
    # Shared person metadata
    current_age_estimate: int = Form(..., ge=0, le=120),
    gender: str = Form(...),
    current_location: str = Form(...),
    finder_contact: str = Form(...),
    # Optional metadata
    name: Optional[str] = Form(None),
    visible_marks: Optional[str] = Form(None),
    current_condition: Optional[str] = Form(None),
    additional_info: Optional[str] = Form(None),
    # Per-image metadata (JSON string)
    image_metadata_json: Optional[str] = Form(None, description='JSON array like [{"photo_year": 2020}, ...]')
):
    """
    Upload multiple images for a found person (batch/multi-image upload).
    
    This endpoint supports 1-10 images per person for improved matching across age gaps.
    Images are processed in parallel for optimal performance.
    
    Args:
        images: List of 1-10 image files
        current_age_estimate: Estimated current age
        gender: Gender (male/female)
        current_location: Current location
        finder_contact: Finder contact information
        name: Optional name if known
        visible_marks: Optional comma-separated marks/scars
        current_condition: Optional current condition
        additional_info: Optional additional information
        image_metadata_json: Optional JSON array with per-image metadata
        
    Returns:
        MultiImageUploadResponse with upload results and potential matches
    """
    start_time = time.time()
    
    try:
        # Validate image count
        if len(images) == 0:
            raise HTTPException(400, "At least 1 image is required")
        if len(images) > 10:
            raise HTTPException(400, f"Maximum 10 images allowed, got {len(images)}")
        
        logger.info(f"Batch upload (found): {len(images)} images for age {current_age_estimate}")
        
        # Parse image_metadata_json
        per_image_metadata = []
        if image_metadata_json:
            try:
                per_image_metadata = json.loads(image_metadata_json)
                if len(per_image_metadata) != len(images):
                    raise HTTPException(
                        400, 
                        f"image_metadata_json length ({len(per_image_metadata)}) must match images count ({len(images)})"
                    )
            except json.JSONDecodeError as e:
                raise HTTPException(400, f"Invalid JSON in image_metadata_json: {e}")
        
        # Generate found_id
        found_id = generate_found_id()
        
        # Build shared params (use current_age_estimate as both age_at_disappearance and year_disappeared dummy)
        # For found persons, we don't have disappearance data, so age calculation is simpler
        import datetime
        current_year = datetime.datetime.now().year
        
        shared_params = {
            "case_id": found_id,  # Use found_id as case_id for consistency
            "type": "found",
            "year_disappeared": current_year,  # Dummy value for age calculation
            "age_at_disappearance": current_age_estimate  # Will be used as fallback
        }
        
        # Process all images CONCURRENTLY
        logger.info(f"Processing {len(images)} images in parallel...")
        tasks = [
            process_single_image(
                idx, img,
                per_image_metadata[idx] if idx < len(per_image_metadata) else None,
                shared_params,
                face_detector,
                embedding_extractor,
                settings
            )
            for idx, img in enumerate(images)
        ]
        results = await asyncio.gather(*tasks)
        
        # ═══════════════════════════════════════════════════════════════════
        # NEW: Separate by validation status (valid/reference/failed)
        # ═══════════════════════════════════════════════════════════════════
        valid_images = []
        reference_images = []
        failed_images = []
        
        for result in results:
            if not result["success"]:
                failed_images.append(
                    FailedImageInfo(
                        filename=result["filename"],
                        index=result["idx"],
                        reason=result["error"]
                    )
                )
            elif result.get("is_valid_for_matching", True):
                valid_images.append(result)
            else:
                reference_images.append(result)
        
        logger.info(
            f"Upload results: {len(valid_images)} valid, "
            f"{len(reference_images)} reference, {len(failed_images)} failed"
        )
        
        # Allow upload with reference-only images
        if not valid_images and not reference_images:
            raise HTTPException(
                400,
                f"No processable images. All {len(images)} images failed."
            )
        
        if not valid_images and reference_images:
            logger.warning(
                f"Found {found_id}: No valid images for matching, "
                f"but {len(reference_images)} reference images saved."
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # Build payloads for ALL images (valid + reference)
        # ═══════════════════════════════════════════════════════════════════
        all_images = valid_images + reference_images
        embeddings = []
        payloads = []
        
        for img_result in all_images:
            payload = {
                # Grouping & identification
                "found_id": found_id,
                "case_id": found_id,  # For compatibility
                "image_id": f"{found_id}_img_{img_result['idx']}",
                "image_index": img_result['idx'],
                "total_images": len(all_images),
                
                # Validation fields
                "is_valid_for_matching": img_result.get("is_valid_for_matching", True),
                "validation_status": img_result.get("validation_status", "valid"),
                "validation_details": img_result.get("validation_details", {}),
                
                # Per-image metadata
                "age_at_photo": img_result['age_at_photo'],
                "photo_year": img_result.get('photo_year'),
                "photo_quality_score": img_result.get('quality_score', 0.0),
                "image_url": img_result.get('image_url'),
                
                # Shared person metadata
                "name": name,
                "current_age_estimate": current_age_estimate,
                "gender": gender.lower(),
                "current_location": current_location,
                "finder_contact": finder_contact,
                "visible_marks": [m.strip() for m in visible_marks.split(",")] if visible_marks else [],
                "current_condition": current_condition,
                "additional_info": additional_info
            }
            
            embeddings.append(img_result.get('embedding'))  # May be None!
            payloads.append(payload)
        
        # Batch insert to Qdrant
        logger.info(f"Inserting {len(embeddings)} records to Qdrant ({len(valid_images)} valid, {len(reference_images)} reference)...")
        point_ids = vector_db.insert_batch(
            collection_name="found_persons",
            embeddings=embeddings,
            payloads=payloads
        )
        logger.info(f"✅ Inserted {len(point_ids)} records for found {found_id}")
        
        # ═══════════════════════════════════════════════════════════════════
        # Search ONLY with valid images
        # ═══════════════════════════════════════════════════════════════════
        potential_matches = []
        if valid_images:
            try:
                # Build query_embeddings for multi-image search
                query_embeddings = [
                    {
                        "embedding": r['embedding'],
                        "age_at_photo": r['age_at_photo'],
                        "quality": r['quality_score']
                    }
                    for r in valid_images  # Only valid images
                ]
                
                # Use multi-image search for better matching
                logger.info(f"Searching with {len(query_embeddings)} images...")
                matches = bilateral_search.search_for_missing_multi_image(
                    query_embeddings=query_embeddings,
                    query_metadata=payloads[0],  # Use first payload as representative
                    limit=settings.top_k_matches
                )
                
                potential_matches = format_match_results(matches, confidence_scoring)
                logger.info(f"Found {len(potential_matches)} potential matches (multi-image)")
                
            except Exception as e:
                logger.error(f"Multi-image search failed: {e}", exc_info=True)
                # Don't fail upload if search fails
        
        # ═══════════════════════════════════════════════════════════════════
        # Build response with valid + reference + failed images
        # ═══════════════════════════════════════════════════════════════════
        processing_time = (time.time() - start_time) * 1000
        
        response = MultiImageUploadResponse(
            success=True,
            message=f"Uploaded {len(valid_images)} valid image(s)" + 
                   (f", {len(reference_images)} reference image(s)" if reference_images else "") +
                   " for found person" +
                   (f" ({len(failed_images)} failed)" if failed_images else ""),
            case_id=found_id,
            total_images_uploaded=len(all_images),
            total_images_failed=len(failed_images),
            
            # Valid images (used for matching)
            valid_images=[
                UploadedImageInfo(
                    image_id=f"{found_id}_img_{r['idx']}",
                    image_index=r['idx'],
                    image_url=r.get('image_url'),
                    age_at_photo=r['age_at_photo'],
                    photo_year=r.get('photo_year'),
                    quality_score=r.get('quality_score', 0.0),
                    validation_status="valid"
                )
                for r in valid_images
            ],
            
            # Reference images (not used for matching)
            reference_images=[
                ReferenceImageInfo(
                    image_id=f"{found_id}_img_{r['idx']}",
                    image_index=r['idx'],
                    image_url=r.get('image_url'),
                    age_at_photo=r['age_at_photo'],
                    photo_year=r.get('photo_year'),
                    validation_status=r.get('validation_status', 'no_face_detected'),
                    reason=r.get('validation_details', {}).get('reason', 'Unknown')
                )
                for r in reference_images
            ],
            
            failed_images=failed_images,
            
            matching_images_count=len(valid_images),
            reference_images_count=len(reference_images),
            
            potential_matches=potential_matches,
            processing_time_ms=processing_time
        )
        
        logger.info(
            f"✅ Batch upload complete: {found_id} "
            f"({len(valid_images)} valid, {len(reference_images)} reference) "
            f"({processing_time:.1f}ms)"
        )
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch upload (found) failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch upload failed: {str(e)}"
        )