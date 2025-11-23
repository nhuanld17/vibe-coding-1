"""
Search routes for Missing Person AI API.

This module handles search operations for finding specific
missing or found persons by their IDs.
"""

import time
import numpy as np
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Depends, Query
from loguru import logger

from ..dependencies import VectorDBDep, ConfidenceScoringDep, SettingsDep, BilateralSearchDep
from ..schemas.models import SearchResponse, SearchParameters, MatchResult, PersonRecord, ConfidenceExplanation, ConfidenceFactor
from utils.validation import validate_search_parameters
from .upload import format_match_results


router = APIRouter()


def format_person_record(point_data: dict) -> PersonRecord:
    """Format a person record (the queried person, not a match)."""
    try:
        payload = point_data['payload']
        
        # Extract contact information
        contact = payload.get('contact') or payload.get('finder_contact', 'No contact available')
        
        # Extract image_url from payload (if available)
        image_url = payload.get('image_url')
        
        # Create person record (no similarity/confidence scores since this is the original record)
        return PersonRecord(
            id=point_data['id'],
            contact=contact,
            metadata=payload,
            image_url=image_url
        )
        
    except Exception as e:
        logger.error(f"Failed to format person record: {str(e)}")
        raise


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
        logger.info(f"Searching for missing person with case_id: {case_id}")
        
        # Search for records with matching case_id using metadata search
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
        
        # Format the missing person record (just for reference, not as a match)
        missing_person_result = format_person_record(missing_person_data)
        
        # Search for potential matches in found_persons collection
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
                logger.info(f"Found {len(matches)} potential matches in found_persons")
                
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
        
        # Format the found person record (just for reference, not as a match)
        found_person_result = format_person_record(found_person_data)
        
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


