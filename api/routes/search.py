"""
Search routes for Missing Person AI API.

This module handles search operations for finding specific
missing or found persons by their IDs.
"""

import time
import numpy as np
from typing import Optional, List
from fastapi import APIRouter, HTTPException, status, Depends, Query
from loguru import logger

from ..dependencies import VectorDBDep, ConfidenceScoringDep, SettingsDep, BilateralSearchDep
from ..schemas.models import (
    SearchResponse,
    SearchParameters,
    MatchResult,
    PersonRecord,
    PersonImage,
    ConfidenceExplanation,
    ConfidenceFactor,
    AllCasesResponse,
    CaseRecord
)
from utils.validation import validate_search_parameters
from .upload import format_match_results


router = APIRouter()


@router.get("/cases/all", response_model=AllCasesResponse)
async def get_all_cases(
    vector_db: VectorDBDep,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of cases to return"),
    type: Optional[str] = Query(None, description="Filter by type: 'missing' or 'found'")
):
    """
    Get all cases (missing and found persons).
    
    Args:
        vector_db: Vector database service
        limit: Maximum number of cases to return per collection
        type: Optional filter by type ('missing' or 'found')
    
    Returns:
        AllCasesResponse with list of all cases
    """
    try:
        start_time = time.time()
        cases = []
        missing_count = 0
        found_count = 0
        
        # Fetch missing persons if not filtered for found only
        if type is None or type == "missing":
            try:
                missing_points = vector_db.client.scroll(
                    collection_name="missing_persons",
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )[0]
                
                for point in missing_points:
                    payload = point.payload
                    cases.append(CaseRecord(
                        id=str(point.id),
                        type="missing",
                        name=payload.get("name"),
                        age=payload.get("age"),
                        gender=payload.get("gender"),
                        last_seen_location=payload.get("last_seen_location"),
                        contact=payload.get("contact"),
                        image_url=payload.get("image_url"),
                        upload_timestamp=payload.get("upload_timestamp"),
                        metadata=payload
                    ))
                    missing_count += 1
            except Exception as e:
                logger.warning(f"Failed to fetch missing persons: {str(e)}")
        
        # Fetch found persons if not filtered for missing only
        if type is None or type == "found":
            try:
                found_points = vector_db.client.scroll(
                    collection_name="found_persons",
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )[0]
                
                for point in found_points:
                    payload = point.payload
                    cases.append(CaseRecord(
                        id=str(point.id),
                        type="found",
                        name=payload.get("name"),
                        age=payload.get("age"),
                        gender=payload.get("gender"),
                        last_seen_location=payload.get("last_seen_location") or payload.get("found_location"),
                        contact=payload.get("finder_contact") or payload.get("contact"),
                        image_url=payload.get("image_url"),
                        upload_timestamp=payload.get("upload_timestamp"),
                        metadata=payload
                    ))
                    found_count += 1
            except Exception as e:
                logger.warning(f"Failed to fetch found persons: {str(e)}")
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"[STEP 7] search_missing_person completed | duration_ms={processing_time:.2f} | matches={len(matches)}")
        
        logger.info(f"Retrieved {len(cases)} cases ({missing_count} missing, {found_count} found) in {processing_time:.2f}ms")
        
        return AllCasesResponse(
            success=True,
            message=f"Retrieved {len(cases)} case(s)",
            cases=cases,
            total_count=len(cases),
            missing_count=missing_count,
            found_count=found_count
        )
        
    except Exception as e:
        logger.error(f"Failed to get all cases: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cases"
        )


def format_person_record(point_data: dict, images: Optional[list] = None, primary_image_url: Optional[str] = None) -> PersonRecord:
    """Format a person record (the queried person, not a match)."""
    try:
        payload = point_data['payload']
        
        # Extract contact information
        contact = payload.get('contact') or payload.get('finder_contact', 'No contact available')
        
        # Extract image_url from payload (if available) or provided override
        image_url = primary_image_url or payload.get('image_url')
        
        # Create person record (no similarity/confidence scores since this is the original record)
        return PersonRecord(
            id=point_data['id'],
            contact=contact,
            metadata=payload,
            image_url=image_url,
            images=images or []
        )
        
    except Exception as e:
        logger.error(f"Failed to format person record: {str(e)}")
        raise


def build_person_images(points: list) -> List[PersonImage]:
    """Convert raw Qdrant points into PersonImage entries sorted by image_index."""
    images: list[PersonImage] = []
    for point in points:
        payload = point.get('payload', {}) or {}
        try:
            images.append(
                PersonImage(
                    point_id=str(point.get('id')),
                    image_id=payload.get('image_id'),
                    image_index=payload.get('image_index'),
                    image_url=payload.get('image_url'),
                    is_valid_for_matching=payload.get('is_valid_for_matching', True),
                    validation_status=payload.get('validation_status', 'valid'),
                    validation_details=payload.get('validation_details', {}),
                    age_at_photo=payload.get('age_at_photo'),
                    photo_year=payload.get('photo_year'),
                    quality_score=payload.get('photo_quality_score'),
                    upload_timestamp=payload.get('upload_timestamp')
                )
            )
        except Exception as e:
            logger.warning(f"Failed to parse person image payload {payload.get('image_id')}: {e}")
            continue
    
    # Sort by image_index to maintain upload order
    images.sort(key=lambda img: img.image_index if img.image_index is not None else 999)
    return images


@router.get("/missing/{case_id}", response_model=SearchResponse)
async def search_missing_person(
    case_id: str,
    vector_db: VectorDBDep,
    bilateral_search: BilateralSearchDep,
    confidence_scoring: ConfidenceScoringDep,
    settings: SettingsDep,
    limit: int = Query(default=1, ge=1, le=100, description="Maximum number of results"),
    include_similar: bool = Query(default=False, description="Include similar records")
):
    """
    Search for a specific missing person by case ID.
    
    This endpoint searches for a missing person record by their case ID
    and automatically searches for potential matches in found_persons collection.
    
    Args:
        case_id: The case ID of the missing person
        limit: Maximum number of potential matches to return (for found persons)
        include_similar: Whether to include similar missing person records
        
    Returns:
        SearchResponse with the missing person record and potential matches from found_persons
    """
    start_time = time.time()
    
    try:
        logger.info(f"[STEP 1] search_missing_person request received | case_id={case_id}")
        
        # Search for records with matching case_id using metadata search
        logger.info("[STEP 2] Searching missing_persons metadata in Qdrant")
        search_results = vector_db.search_by_metadata(
            collection_name="missing_persons",
            filters={'case_id': case_id},
            limit=1  # Only need one match for the case_id
        )
        
        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Missing person with case_id '{case_id}' not found"
            )
        
        # Get full point data for the missing person
        logger.info("[STEP 3] Fetching full point data for missing person")
        missing_person_data = None
        for result in search_results:
            try:
                point_data = vector_db.get_point_by_id("missing_persons", result['id'])
                if point_data:
                    missing_person_data = point_data
                    break
            except Exception as e:
                logger.error(f"Failed to get point data for {result['id']}: {str(e)}")
                continue
        
        if not missing_person_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Missing person with case_id '{case_id}' not found"
            )
        
        # Retrieve all images for gallery view
        logger.info("[STEP 4] Retrieving gallery images for missing person")
        person_images = []
        primary_image_url = missing_person_data['payload'].get('image_url')
        case_id_value = missing_person_data['payload'].get('case_id')
        if case_id_value:
            try:
                raw_images = vector_db.get_all_images_for_person("missing_persons", case_id_value)
                person_images = build_person_images(raw_images)
                # Prefer first image that has a URL for primary display
                for img in person_images:
                    if img.image_url:
                        primary_image_url = img.image_url
                        break
            except Exception as e:
                logger.warning(f"Failed to load gallery images for {case_id_value}: {e}")
        
        # Format the missing person record (just for reference, not as a match)
        logger.info("[STEP 5] Formatting missing person record for response")
        missing_person_result = format_person_record(
            missing_person_data,
            images=person_images,
            primary_image_url=primary_image_url
        )
        
        # Search for potential matches in found_persons collection
        logger.info("[STEP 6] Searching for potential matches in found_persons")
        matches = []
        try:
            # Get embedding and metadata from missing person
            if 'vector' in missing_person_data and missing_person_data['vector'] is not None:
                embedding = np.array(missing_person_data['vector'])
                metadata = missing_person_data['payload']
                
                logger.info(f"Searching for potential matches in found_persons for case_id: {case_id}")
                
                # Use bilateral search to find potential matches in found_persons
                found_matches = bilateral_search.search_for_found(
                    missing_embedding=embedding,
                    missing_metadata=metadata,
                    limit=limit
                )
                
                # Format the potential matches
                matches = format_match_results(found_matches, confidence_scoring)
                logger.info(f"[STEP 6] Found {len(matches)} potential matches in found_persons")
                
            else:
                logger.warning(f"Missing person {case_id} has no embedding vector, cannot search for matches")
                
        except Exception as e:
            logger.error(f"Failed to search for potential matches: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue without potential matches - don't fail the search
        
        # If include_similar is True, also find similar missing person records
        # Note: similar missing persons are added to matches but they are not "potential matches" 
        # in the traditional sense - they're just similar records in the same collection
        # For now, we'll skip this feature or handle it differently
        
        processing_time = (time.time() - start_time) * 1000
        
        search_params = SearchParameters(
            limit=limit,
            threshold=settings.similarity_threshold,
            filters=None
        )
        
        # Return the missing person record + potential matches from found_persons
        # matches now only contains potential matches from found_persons collection
        return SearchResponse(
            success=True,
            message=f"Found missing person with case_id '{case_id}' and {len(matches)} potential match(es) in found_persons",
            missing_person=missing_person_result,  # The missing person record itself
            found_person=None,  # Not applicable when searching for missing
            matches=matches,  # Potential matches from found_persons collection only
            total_found=len(matches),
            search_parameters=search_params,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search for missing person failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during search"
        )


@router.get("/found/{found_id}", response_model=SearchResponse)
async def search_found_person(
    found_id: str,
    vector_db: VectorDBDep,
    bilateral_search: BilateralSearchDep,
    confidence_scoring: ConfidenceScoringDep,
    settings: SettingsDep,
    limit: int = Query(default=1, ge=1, le=100, description="Maximum number of results"),
    include_similar: bool = Query(default=False, description="Include similar records")
):
    """
    Search for a specific found person by found ID.
    
    This endpoint searches for a found person record by their found ID
    and automatically searches for potential matches in missing_persons collection.
    
    Args:
        found_id: The found ID of the person
        limit: Maximum number of potential matches to return (for missing persons)
        include_similar: Whether to include similar found person records
        
    Returns:
        SearchResponse with the found person record and potential matches from missing_persons
    """
    start_time = time.time()
    
    try:
        logger.info(f"Searching for found person with found_id: {found_id}")
        
        # Search for records with matching found_id using metadata search
        search_results = vector_db.search_by_metadata(
            collection_name="found_persons",
            filters={'found_id': found_id},
            limit=1  # Only need one match for the found_id
        )
        
        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Found person with found_id '{found_id}' not found"
            )
        
        # Get full point data for the found person
        found_person_data = None
        for result in search_results:
            try:
                point_data = vector_db.get_point_by_id("found_persons", result['id'])
                if point_data:
                    found_person_data = point_data
                    break
            except Exception as e:
                logger.error(f"Failed to get point data for {result['id']}: {str(e)}")
                continue
        
        if not found_person_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Found person with found_id '{found_id}' not found"
            )
        
        # Retrieve all images for the found person gallery
        person_images = []
        primary_image_url = found_person_data['payload'].get('image_url')
        found_case_id = found_person_data['payload'].get('found_id') or found_person_data['payload'].get('case_id')
        if found_case_id:
            try:
                raw_images = vector_db.get_all_images_for_person("found_persons", found_case_id)
                person_images = build_person_images(raw_images)
                for img in person_images:
                    if img.image_url:
                        primary_image_url = img.image_url
                        break
            except Exception as e:
                logger.warning(f"Failed to load gallery images for found {found_case_id}: {e}")
        
        # Format the found person record (just for reference, not as a match)
        found_person_result = format_person_record(
            found_person_data,
            images=person_images,
            primary_image_url=primary_image_url
        )
        
        # Search for potential matches in missing_persons collection
        matches = []
        try:
            # Get embedding and metadata from found person
            if 'vector' in found_person_data and found_person_data['vector'] is not None:
                embedding = np.array(found_person_data['vector'])
                metadata = found_person_data['payload']
                
                logger.info(f"Searching for potential matches in missing_persons for found_id: {found_id}")
                
                # Use bilateral search to find potential matches in missing_persons
                missing_matches = bilateral_search.search_for_missing(
                    found_embedding=embedding,
                    found_metadata=metadata,
                    limit=limit
                )
                
                # Format the potential matches
                matches = format_match_results(missing_matches, confidence_scoring)
                logger.info(f"Found {len(matches)} potential matches in missing_persons")
                
            else:
                logger.warning(f"Found person {found_id} has no embedding vector, cannot search for matches")
                
        except Exception as e:
            logger.error(f"Failed to search for potential matches: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue without potential matches - don't fail the search
        
        processing_time = (time.time() - start_time) * 1000
        
        search_params = SearchParameters(
            limit=limit,
            threshold=settings.similarity_threshold,
            filters=None
        )
        
        # Return the found person record + potential matches from missing_persons
        # matches now only contains potential matches from missing_persons collection
        return SearchResponse(
            success=True,
            message=f"Found found person with found_id '{found_id}' and {len(matches)} potential match(es) in missing_persons",
            missing_person=None,  # Not applicable when searching for found
            found_person=found_person_result,  # The found person record itself
            matches=matches,  # Potential matches from missing_persons collection only
            total_found=len(matches),
            search_parameters=search_params,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search for found person failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during search"
        )


