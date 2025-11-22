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

from ..dependencies import VectorDBDep, ConfidenceScoringDep, SettingsDep
from ..schemas.models import SearchResponse, SearchParameters, MatchResult, ConfidenceExplanation, ConfidenceFactor
from utils.validation import validate_search_parameters


router = APIRouter()


def format_search_result(point_data: dict, confidence_scoring) -> MatchResult:
    """Format a single search result with confidence scoring."""
    try:
        # Create a mock match result for confidence scoring
        mock_match = {
            'id': point_data['id'],
            'face_similarity': 1.0,  # Perfect match since it's the same record
            'metadata_similarity': 1.0,
            'combined_score': 1.0,
            'payload': point_data['payload'],
            'match_details': {
                'gender_match': 1.0,
                'age_consistency': 1.0,
                'marks_similarity': 1.0,
                'location_plausibility': 1.0
            }
        }
        
        # Calculate confidence (should be very high for exact matches)
        confidence_level, confidence_score, explanation = confidence_scoring.calculate_confidence(mock_match)
        
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
        payload = point_data['payload']
        contact = payload.get('contact') or payload.get('finder_contact', 'No contact available')
        
        # Extract image_url from payload (if available)
        image_url = payload.get('image_url')
        
        # Create match result
        return MatchResult(
            id=point_data['id'],
            face_similarity=1.0,
            metadata_similarity=1.0,
            combined_score=1.0,
            confidence_level=confidence_level.value,
            confidence_score=confidence_score,
            explanation=confidence_explanation,
            contact=contact,
            metadata=payload,
            image_url=image_url
        )
        
    except Exception as e:
        logger.error(f"Failed to format search result: {str(e)}")
        raise


@router.get("/missing/{case_id}", response_model=SearchResponse)
async def search_missing_person(
    case_id: str,
    vector_db: VectorDBDep,
    confidence_scoring: ConfidenceScoringDep,
    settings: SettingsDep,
    limit: int = Query(default=1, ge=1, le=100, description="Maximum number of results"),
    include_similar: bool = Query(default=False, description="Include similar records")
):
    """
    Search for a specific missing person by case ID.
    
    This endpoint searches for a missing person record by their case ID
    and optionally includes similar records based on face similarity.
    
    Args:
        case_id: The case ID of the missing person
        limit: Maximum number of results to return
        include_similar: Whether to include similar records
        
    Returns:
        SearchResponse with the found record(s)
    """
    start_time = time.time()
    
    try:
        logger.info(f"Searching for missing person with case_id: {case_id}")
        
        # Search for records with matching case_id using metadata search
        search_results = vector_db.search_by_metadata(
            collection_name="missing_persons",
            filters={'case_id': case_id},
            limit=limit
        )
        
        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Missing person with case_id '{case_id}' not found"
            )
        
        # Format results
        matches = []
        for result in search_results:
            try:
                # Get full point data
                point_data = vector_db.get_point_by_id("missing_persons", result['id'])
                if point_data:
                    match_result = format_search_result(point_data, confidence_scoring)
                    matches.append(match_result)
            except Exception as e:
                logger.error(f"Failed to process search result {result['id']}: {str(e)}")
                continue
        
        # If include_similar is True and we have the embedding, find similar records
        if include_similar and matches and limit > 1:
            try:
                # Get the embedding of the first match from search results
                if search_results and 'vector' in search_results[0]:
                    embedding = np.array(search_results[0]['vector'])
                    
                    # Search for similar faces
                    similar_results = vector_db.search_similar_faces(
                        query_embedding=embedding,
                        collection_name="missing_persons",
                        limit=limit,
                        score_threshold=settings.similarity_threshold,
                        filters=None  # No filters for similarity search
                    )
                    
                    # Add similar results (excluding the original)
                    for result in similar_results:
                        if result['id'] != matches[0].id:  # Skip the original match
                            try:
                                point_data = vector_db.get_point_by_id("missing_persons", result['id'])
                                if point_data:
                                    # Create a proper match result with actual similarity scores
                                    mock_match = {
                                        'id': result['id'],
                                        'face_similarity': result['score'],
                                        'metadata_similarity': 0.5,  # Default metadata similarity
                                        'combined_score': result['score'] * 0.7 + 0.5 * 0.3,
                                        'payload': point_data['payload'],
                                        'match_details': {
                                            'gender_match': 0.5,
                                            'age_consistency': 0.5,
                                            'marks_similarity': 0.5,
                                            'location_plausibility': 0.5
                                        }
                                    }
                                    
                                    # Calculate confidence
                                    confidence_level, confidence_score, explanation = confidence_scoring.calculate_confidence(mock_match)
                                    
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
                                    contact = point_data['payload'].get('contact', 'No contact available')
                                    
                                    # Extract image_url from payload (if available)
                                    image_url = point_data['payload'].get('image_url')
                                    
                                    similar_match = MatchResult(
                                        id=result['id'],
                                        face_similarity=result['score'],
                                        metadata_similarity=0.5,
                                        combined_score=mock_match['combined_score'],
                                        confidence_level=confidence_level.value,
                                        confidence_score=confidence_score,
                                        explanation=confidence_explanation,
                                        contact=contact,
                                        metadata=point_data['payload'],
                                        image_url=image_url
                                    )
                                    
                                    matches.append(similar_match)
                                    
                                    if len(matches) >= limit:
                                        break
                                        
                            except Exception as e:
                                logger.error(f"Failed to process similar result {result['id']}: {str(e)}")
                                continue
                                
            except Exception as e:
                logger.error(f"Failed to find similar records: {str(e)}")
                # Continue without similar records
        
        processing_time = (time.time() - start_time) * 1000
        
        search_params = SearchParameters(
            limit=limit,
            threshold=settings.similarity_threshold,
            filters=None
        )
        
        return SearchResponse(
            success=True,
            message=f"Found {len(matches)} record(s) for case_id '{case_id}'",
            matches=matches,
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
    confidence_scoring: ConfidenceScoringDep,
    settings: SettingsDep,
    limit: int = Query(default=1, ge=1, le=100, description="Maximum number of results"),
    include_similar: bool = Query(default=False, description="Include similar records")
):
    """
    Search for a specific found person by found ID.
    
    This endpoint searches for a found person record by their found ID
    and optionally includes similar records based on face similarity.
    
    Args:
        found_id: The found ID of the person
        limit: Maximum number of results to return
        include_similar: Whether to include similar records
        
    Returns:
        SearchResponse with the found record(s)
    """
    start_time = time.time()
    
    try:
        logger.info(f"Searching for found person with found_id: {found_id}")
        
        # Search for records with matching found_id using metadata search
        search_results = vector_db.search_by_metadata(
            collection_name="found_persons",
            filters={'found_id': found_id},
            limit=limit
        )
        
        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Found person with found_id '{found_id}' not found"
            )
        
        # Format results
        matches = []
        for result in search_results:
            try:
                # Get full point data
                point_data = vector_db.get_point_by_id("found_persons", result['id'])
                if point_data:
                    match_result = format_search_result(point_data, confidence_scoring)
                    matches.append(match_result)
            except Exception as e:
                logger.error(f"Failed to process search result {result['id']}: {str(e)}")
                continue
        
        # If include_similar is True and we have the embedding, find similar records
        if include_similar and matches and limit > 1:
            try:
                # Get the embedding of the first match from search results
                if search_results and 'vector' in search_results[0]:
                    embedding = np.array(search_results[0]['vector'])
                    
                    # Search for similar faces
                    similar_results = vector_db.search_similar_faces(
                        query_embedding=embedding,
                        collection_name="found_persons",
                        limit=limit,
                        score_threshold=settings.similarity_threshold,
                        filters=None  # No filters for similarity search
                    )
                    
                    # Add similar results (excluding the original)
                    for result in similar_results:
                        if result['id'] != matches[0].id:  # Skip the original match
                            try:
                                point_data = vector_db.get_point_by_id("found_persons", result['id'])
                                if point_data:
                                    # Create a proper match result with actual similarity scores
                                    mock_match = {
                                        'id': result['id'],
                                        'face_similarity': result['score'],
                                        'metadata_similarity': 0.5,  # Default metadata similarity
                                        'combined_score': result['score'] * 0.7 + 0.5 * 0.3,
                                        'payload': point_data['payload'],
                                        'match_details': {
                                            'gender_match': 0.5,
                                            'age_consistency': 0.5,
                                            'marks_similarity': 0.5,
                                            'location_plausibility': 0.5
                                        }
                                    }
                                    
                                    # Calculate confidence
                                    confidence_level, confidence_score, explanation = confidence_scoring.calculate_confidence(mock_match)
                                    
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
                                    contact = point_data['payload'].get('finder_contact', 'No contact available')
                                    
                                    # Extract image_url from payload (if available)
                                    image_url = point_data['payload'].get('image_url')
                                    
                                    similar_match = MatchResult(
                                        id=result['id'],
                                        face_similarity=result['score'],
                                        metadata_similarity=0.5,
                                        combined_score=mock_match['combined_score'],
                                        confidence_level=confidence_level.value,
                                        confidence_score=confidence_score,
                                        explanation=confidence_explanation,
                                        contact=contact,
                                        metadata=point_data['payload'],
                                        image_url=image_url
                                    )
                                    
                                    matches.append(similar_match)
                                    
                                    if len(matches) >= limit:
                                        break
                                        
                            except Exception as e:
                                logger.error(f"Failed to process similar result {result['id']}: {str(e)}")
                                continue
                                
            except Exception as e:
                logger.error(f"Failed to find similar records: {str(e)}")
                # Continue without similar records
        
        processing_time = (time.time() - start_time) * 1000
        
        search_params = SearchParameters(
            limit=limit,
            threshold=settings.similarity_threshold,
            filters=None
        )
        
        return SearchResponse(
            success=True,
            message=f"Found {len(matches)} record(s) for found_id '{found_id}'",
            matches=matches,
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


@router.get("/cases/all")
async def list_all_cases(
    vector_db: VectorDBDep,
    confidence_scoring: ConfidenceScoringDep,
    limit: int = Query(default=100, ge=1, le=500, description="Maximum number of results per collection"),
    case_type: Optional[str] = Query(default=None, alias="type", description="Filter by type: 'missing', 'found', or None for both")
):
    """
    List all missing and found person cases.
    
    This endpoint retrieves all cases from both collections, optionally filtered by type.
    
    Args:
        limit: Maximum number of results to return per collection
        case_type: Optional filter by type ('missing', 'found', or None for both)
        
    Returns:
        Dictionary containing lists of missing and found cases
    """
    start_time = time.time()
    
    try:
        logger.info(f"Listing all cases with limit={limit}, type={case_type}")
        
        all_cases = {
            'missing': [],
            'found': []
        }
        
        # Get missing persons if case_type is None or 'missing'
        if case_type is None or case_type == 'missing':
            try:
                missing_points, _ = vector_db.list_all_points(
                    collection_name="missing_persons",
                    limit=limit
                )
                
                for point_data in missing_points:
                    try:
                        match_result = format_search_result(
                            {
                                'id': point_data['id'],
                                'payload': point_data['payload'],
                                'vector': None  # Not needed for listing
                            },
                            confidence_scoring
                        )
                        all_cases['missing'].append(match_result)
                    except Exception as e:
                        logger.error(f"Failed to format missing person {point_data['id']}: {str(e)}")
                        continue
            except Exception as e:
                logger.error(f"Failed to list missing persons: {str(e)}")
                # Continue with found persons even if missing fails
        
        # Get found persons if case_type is None or 'found'
        if case_type is None or case_type == 'found':
            try:
                found_points, _ = vector_db.list_all_points(
                    collection_name="found_persons",
                    limit=limit
                )
                
                for point_data in found_points:
                    try:
                        match_result = format_search_result(
                            {
                                'id': point_data['id'],
                                'payload': point_data['payload'],
                                'vector': None  # Not needed for listing
                            },
                            confidence_scoring
                        )
                        all_cases['found'].append(match_result)
                    except Exception as e:
                        logger.error(f"Failed to format found person {point_data['id']}: {str(e)}")
                        continue
            except Exception as e:
                logger.error(f"Failed to list found persons: {str(e)}")
                # Continue even if found fails
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate statistics
        total_missing = len(all_cases['missing'])
        total_found = len(all_cases['found'])
        total_cases = total_missing + total_found
        
        return {
            'success': True,
            'message': f'Retrieved {total_cases} cases ({total_missing} missing, {total_found} found)',
            'cases': all_cases,
            'statistics': {
                'total_missing': total_missing,
                'total_found': total_found,
                'total_cases': total_cases
            },
            'processing_time_ms': processing_time
        }
        
    except Exception as e:
        logger.error(f"List all cases failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
