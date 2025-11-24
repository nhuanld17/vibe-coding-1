"""
Bilateral Search service for Missing Person AI system.

This module provides two-way matching logic between missing and found persons,
combining face similarity with intelligent metadata filtering and scoring.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from loguru import logger

from .vector_db import VectorDatabaseService


class BilateralSearchService:
    """
    Bilateral search service for matching missing and found persons.
    
    This class provides methods for:
    - Two-way matching between missing and found persons
    - Intelligent metadata filtering based on age, gender, location
    - Weighted scoring combining face similarity and metadata matching
    - Age consistency checking across time periods
    - Location plausibility assessment
    """
    
    def __init__(
        self,
        vector_db: VectorDatabaseService,
        face_threshold: float = 0.65,
        face_threshold_adult: Optional[float] = None,
        face_threshold_child: Optional[float] = None,
        metadata_weight: float = 0.3,
        age_tolerance: int = 5,
        initial_search_threshold: float = 0.60,
        combined_score_threshold: float = 0.55,
        face_metadata_fallback_threshold: float = 0.50
    ) -> None:
        """
        Initialize the bilateral search service.
        
        Args:
            vector_db: Vector database service instance
            face_threshold: Minimum face similarity threshold for final filtering (0-1) - backward compatibility
            face_threshold_adult: Face similarity threshold for adults (0-1). If None, uses face_threshold.
            face_threshold_child: Face similarity threshold for children (0-1). If None, uses face_threshold.
            metadata_weight: Weight for metadata similarity in final score (0-1)
            age_tolerance: Age tolerance in years for matching
            initial_search_threshold: Initial Qdrant search threshold (0-1)
            combined_score_threshold: Minimum combined score threshold (0-1)
            face_metadata_fallback_threshold: Minimum face similarity when metadata is high (0-1)
        """
        self.vector_db = vector_db
        # Use separate thresholds for adults and children if provided, otherwise use face_threshold for both
        self.face_threshold_adult = face_threshold_adult if face_threshold_adult is not None else face_threshold
        self.face_threshold_child = face_threshold_child if face_threshold_child is not None else face_threshold
        self.face_threshold = face_threshold  # Keep for backward compatibility
        self.metadata_weight = metadata_weight
        self.face_weight = 1.0 - metadata_weight
        self.age_tolerance = age_tolerance
        self.initial_search_threshold = initial_search_threshold
        self.combined_score_threshold = combined_score_threshold
        self.face_metadata_fallback_threshold = face_metadata_fallback_threshold
        
        logger.info(f"Bilateral search initialized with:")
        logger.info(f"  face_threshold_adult={self.face_threshold_adult}, face_threshold_child={self.face_threshold_child}")
        logger.info(f"  initial_search_threshold={initial_search_threshold}, "
                   f"combined_score_threshold={combined_score_threshold}, "
                   f"metadata_weight={metadata_weight}")
    
    def search_for_missing(
        self,
        found_embedding: np.ndarray,
        found_metadata: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for missing persons that might match a found person.
        
        Args:
            found_embedding: Face embedding of the found person
            found_metadata: Metadata of the found person
            limit: Maximum number of results to return
            
        Returns:
            List of potential matches with scores and explanations
            
        Example:
            >>> found_metadata = {
            ...     'current_age_estimate': 30,
            ...     'gender': 'male',
            ...     'current_location': 'Los Angeles, CA'
            ... }
            >>> matches = search.search_for_missing(embedding, found_metadata)
        """
        try:
            # Extract search filters from found person metadata
            filters = self._extract_search_filters(found_metadata, 'missing')
            
            # Search in missing persons collection
            # Use configurable threshold for initial search, filters will be applied in reranking
            vector_matches = self.vector_db.search_similar_faces(
                query_embedding=found_embedding,
                collection_name="missing_persons",
                limit=limit * 5,  # Get more results for reranking
                score_threshold=self.initial_search_threshold,  # Configurable initial search threshold
                filters=None  # Don't apply filters in vector search, apply in reranking
            )
            
            logger.info(f"Vector search returned {len(vector_matches)} candidates for missing persons")
            
            # Rerank with metadata similarity
            reranked_matches = self._rerank_with_metadata(
                vector_matches, found_metadata, 'missing'
            )
            
            # Filter by minimum combined score or face similarity (more lenient)
            # Accept matches with good face similarity OR good combined score OR decent face similarity with good metadata
            # BUT reject suspicious false positives (high face similarity but very low metadata similarity)
            # NOTE: This is a human-in-the-loop system - we return ranked candidates for review, not hard yes/no decisions
            # Use age-appropriate threshold for each candidate
            filtered_matches = []
            for m in reranked_matches:
                # Validate match first (rejects suspicious false positives, especially for children)
                if self._validate_match(m):
                    # Get age-appropriate threshold for the matched person
                    match_metadata = m.get('payload', {})
                    threshold_for_match = self._get_face_threshold_for_person(match_metadata)
                    
                    if ((m['face_similarity'] >= threshold_for_match) or 
                        (m['combined_score'] >= self.combined_score_threshold) or
                        (m['face_similarity'] >= self.face_metadata_fallback_threshold and m['metadata_similarity'] >= 0.60)):
                        filtered_matches.append(m)
            
            logger.info(f"After filtering: {len(filtered_matches)} matches (from {len(reranked_matches)} reranked)")
            
            # Limit results to top-k for human review
            # This is intentionally NOT a hard yes/no decision - all candidates need human verification
            final_matches = filtered_matches[:limit]
            
            logger.info(f"Found {len(final_matches)} potential missing person matches")
            return final_matches
            
        except Exception as e:
            logger.error(f"Search for missing persons failed: {str(e)}")
            raise RuntimeError(f"Search failed: {str(e)}")
    
    def search_for_found(
        self,
        missing_embedding: np.ndarray,
        missing_metadata: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for found persons that might match a missing person.
        
        Args:
            missing_embedding: Face embedding of the missing person
            missing_metadata: Metadata of the missing person
            limit: Maximum number of results to return
            
        Returns:
            List of potential matches with scores and explanations
            
        Example:
            >>> missing_metadata = {
            ...     'age_at_disappearance': 25,
            ...     'year_disappeared': 2020,
            ...     'gender': 'male',
            ...     'location_last_seen': 'New York, NY'
            ... }
            >>> matches = search.search_for_found(embedding, missing_metadata)
        """
        try:
            # Extract search filters from missing person metadata
            filters = self._extract_search_filters(missing_metadata, 'found')
            
            # Search in found persons collection
            # Use configurable threshold for initial search, filters will be applied in reranking
            vector_matches = self.vector_db.search_similar_faces(
                query_embedding=missing_embedding,
                collection_name="found_persons",
                limit=limit * 5,  # Get more results for reranking
                score_threshold=self.initial_search_threshold,  # Configurable initial search threshold
                filters=None  # Don't apply filters in vector search, apply in reranking
            )
            
            logger.info(f"Vector search returned {len(vector_matches)} candidates for found persons")
            
            # Rerank with metadata similarity
            reranked_matches = self._rerank_with_metadata(
                vector_matches, missing_metadata, 'found'
            )
            
            # Filter by minimum combined score or face similarity (more lenient)
            # Accept matches with good face similarity OR good combined score OR decent face similarity with good metadata
            # BUT reject suspicious false positives (high face similarity but very low metadata similarity)
            # NOTE: This is a human-in-the-loop system - we return ranked candidates for review, not hard yes/no decisions
            # Use age-appropriate threshold for each candidate
            filtered_matches = []
            for m in reranked_matches:
                if self._validate_match(m):
                    # Get age-appropriate threshold for the matched person
                    match_metadata = m.get('payload', {})
                    threshold_for_match = self._get_face_threshold_for_person(match_metadata)
                    
                    if (m['face_similarity'] >= threshold_for_match) or \
                       (m['combined_score'] >= self.combined_score_threshold) or \
                       (m['face_similarity'] >= self.face_metadata_fallback_threshold and m['metadata_similarity'] >= 0.60):
                        filtered_matches.append(m)
            
            logger.info(f"After filtering: {len(filtered_matches)} matches (from {len(reranked_matches)} reranked)")
            
            # Limit results to top-k for human review
            # This is intentionally NOT a hard yes/no decision - all candidates need human verification
            final_matches = filtered_matches[:limit]
            
            logger.info(f"Found {len(final_matches)} potential found person matches")
            return final_matches
            
        except Exception as e:
            logger.error(f"Search for found persons failed: {str(e)}")
            raise RuntimeError(f"Search failed: {str(e)}")
    
    def _extract_search_filters(
        self, 
        metadata: Dict[str, Any], 
        search_type: str
    ) -> Dict[str, Any]:
        """
        Extract search filters from metadata based on search type.
        
        Args:
            metadata: Input metadata
            search_type: 'missing' or 'found'
            
        Returns:
            Dictionary of filters for vector search
        """
        filters = {}
        
        try:
            # Gender filter (exact match)
            # Only apply filter for valid gender values ('male' or 'female')
            # Legacy data with 'other' or 'unknown' will skip gender filtering (logged as warning)
            if 'gender' in metadata:
                gender = metadata['gender']
                if gender in ['male', 'female']:
                    filters['gender'] = gender
                elif gender in ['other', 'unknown']:
                    # Legacy data: log warning and skip gender filter to avoid false positives
                    # This ensures backward compatibility but warns about potential issues
                    logger.warning(
                        f"Legacy gender value '{gender}' found in metadata. "
                        f"Gender filter will be skipped. Consider migrating to 'male' or 'female'. "
                        f"Metadata: {metadata.get('case_id') or metadata.get('found_id', 'unknown_id')}"
                    )
                elif gender:
                    # Invalid gender value (not empty, but not recognized)
                    logger.warning(
                        f"Invalid gender value '{gender}' found in metadata. "
                        f"Gender filter will be skipped. "
                        f"Metadata: {metadata.get('case_id') or metadata.get('found_id', 'unknown_id')}"
                    )
            
            # Age range filter (more flexible)
            if search_type == 'missing':
                # Searching missing persons with found person data
                if 'current_age_estimate' in metadata:
                    current_age = metadata['current_age_estimate']
                    # Calculate possible age range at disappearance (more flexible)
                    # Allow wider range to catch more potential matches
                    min_age_at_disappearance = max(0, current_age - 50)  # Max 50 years missing
                    max_age_at_disappearance = min(120, current_age + 5)  # Allow some tolerance
                    # Only apply filter if range is reasonable
                    if min_age_at_disappearance <= max_age_at_disappearance:
                        filters['age_range'] = [min_age_at_disappearance, max_age_at_disappearance]
            
            elif search_type == 'found':
                # Searching found persons with missing person data
                if 'age_at_disappearance' in metadata and 'year_disappeared' in metadata:
                    age_at_disappearance = metadata['age_at_disappearance']
                    years_missing = datetime.now().year - metadata['year_disappeared']
                    estimated_current_age = age_at_disappearance + years_missing
                    
                    # Age range with tolerance
                    min_age = estimated_current_age - self.age_tolerance
                    max_age = estimated_current_age + self.age_tolerance
                    filters['age_range'] = [max(0, min_age), max_age]
            
            logger.debug(f"Extracted filters for {search_type}: {filters}")
            return filters
            
        except Exception as e:
            logger.warning(f"Failed to extract filters: {str(e)}")
            return {}
    
    def _rerank_with_metadata(
        self,
        vector_matches: List[Dict[str, Any]],
        query_metadata: Dict[str, Any],
        search_type: str
    ) -> List[Dict[str, Any]]:
        """
        Rerank vector search results using metadata similarity.
        
        Args:
            vector_matches: Results from vector search
            query_metadata: Query metadata for comparison
            search_type: 'missing' or 'found'
            
        Returns:
            Reranked list of matches with combined scores
        """
        reranked_matches = []
        
        for match in vector_matches:
            try:
                face_similarity = match['score']
                match_metadata = match['payload']
                
                # Calculate metadata similarity
                metadata_similarity = self._calculate_metadata_similarity(
                    query_metadata, match_metadata, search_type
                )
                
                # Combined score
                combined_score = (
                    self.face_weight * face_similarity +
                    self.metadata_weight * metadata_similarity
                )
                
                # Enhanced match information
                enhanced_match = {
                    'id': match['id'],
                    'face_similarity': face_similarity,
                    'metadata_similarity': metadata_similarity,
                    'combined_score': combined_score,
                    'payload': match_metadata,
                    'match_details': self._get_match_details(
                        query_metadata, match_metadata, search_type
                    )
                }
                
                reranked_matches.append(enhanced_match)
                
            except Exception as e:
                logger.warning(f"Failed to rerank match {match.get('id', 'unknown')}: {str(e)}")
                continue
        
        # Sort by combined score (descending)
        reranked_matches.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return reranked_matches
    
    def _calculate_metadata_similarity(
        self,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any],
        search_type: str
    ) -> float:
        """
        Calculate similarity between two metadata dictionaries.
        
        Args:
            metadata1: First metadata dictionary
            metadata2: Second metadata dictionary
            search_type: 'missing' or 'found'
            
        Returns:
            Metadata similarity score (0-1)
        """
        try:
            similarity_scores = []
            weights = []
            
            # Gender similarity (30% weight)
            gender_sim = self._compare_gender(metadata1, metadata2)
            similarity_scores.append(gender_sim)
            weights.append(0.30)
            
            # Age consistency (30% weight)
            age_sim = self._check_age_consistency(metadata1, metadata2, search_type)
            similarity_scores.append(age_sim)
            weights.append(0.30)
            
            # Marks/features similarity (25% weight)
            marks_sim = self._compare_marks(metadata1, metadata2)
            similarity_scores.append(marks_sim)
            weights.append(0.25)
            
            # Location plausibility (15% weight)
            location_sim = self._check_location_plausibility(metadata1, metadata2)
            similarity_scores.append(location_sim)
            weights.append(0.15)
            
            # Weighted average
            total_weight = sum(weights)
            weighted_sum = sum(score * weight for score, weight in zip(similarity_scores, weights))
            
            overall_similarity = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            # Apply HEAVY penalty if gender doesn't match (reduce score by 60%)
            # Gender mismatch is a critical indicator - should severely penalize the match
            if gender_sim == 0.0:
                original_sim = overall_similarity
                overall_similarity = overall_similarity * 0.4  # Reduce by 60%
                logger.debug(f"Gender mismatch HEAVY penalty applied: metadata_similarity reduced from {original_sim:.3f} to {overall_similarity:.3f}")
            
            return min(1.0, max(0.0, overall_similarity))
            
        except Exception as e:
            logger.warning(f"Metadata similarity calculation failed: {str(e)}")
            return 0.0
    
    def _compare_gender(self, metadata1: Dict[str, Any], metadata2: Dict[str, Any]) -> float:
        """Compare gender fields between metadata."""
        gender1 = metadata1.get('gender', '').lower()
        gender2 = metadata2.get('gender', '').lower()
        
        if not gender1 or not gender2:
            return 0.5  # Neutral score if gender is missing
        
        return 1.0 if gender1 == gender2 else 0.0
    
    def _is_child(self, metadata: Dict[str, Any]) -> bool:
        """
        Determine if a person is a child based on metadata.
        
        Args:
            metadata: Person metadata dictionary
            
        Returns:
            True if person is a child (age < 18), False if adult or age unknown
        """
        # Try to get age from various fields
        age = None
        
        # Check age_at_disappearance (for missing persons)
        if 'age_at_disappearance' in metadata and metadata['age_at_disappearance'] is not None:
            try:
                age = int(metadata['age_at_disappearance'])
            except (ValueError, TypeError):
                pass
        
        # Check current_age_estimate (for found persons or current age)
        if age is None and 'current_age_estimate' in metadata and metadata['current_age_estimate'] is not None:
            try:
                age = int(metadata['current_age_estimate'])
            except (ValueError, TypeError):
                pass
        
        # Check estimated_current_age (alternative field name)
        if age is None and 'estimated_current_age' in metadata and metadata['estimated_current_age'] is not None:
            try:
                age = int(metadata['estimated_current_age'])
            except (ValueError, TypeError):
                pass
        
        # Check age field (generic)
        if age is None and 'age' in metadata and metadata['age'] is not None:
            try:
                age = int(metadata['age'])
            except (ValueError, TypeError):
                pass
        
        # If age is found, check if < 18
        if age is not None:
            return age < 18
        
        # If age is unknown, default to adult (more lenient)
        # This is a conservative choice: unknown age = use adult threshold
        return False
    
    def _get_face_threshold_for_person(self, metadata: Dict[str, Any]) -> float:
        """
        Get the appropriate face threshold based on person's age.
        
        Args:
            metadata: Person metadata dictionary
            
        Returns:
            face_threshold_child if person is a child, face_threshold_adult otherwise
        """
        if self._is_child(metadata):
            return self.face_threshold_child
        else:
            return self.face_threshold_adult
    
    def _check_age_consistency(
        self,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any],
        search_type: str
    ) -> float:
        """Check age consistency between missing and found person records."""
        try:
            if search_type == 'missing':
                # metadata1 is found person, metadata2 is missing person
                current_age = metadata1.get('current_age_estimate')
                age_at_disappearance = metadata2.get('age_at_disappearance')
                year_disappeared = metadata2.get('year_disappeared')
                
                if not all([current_age, age_at_disappearance, year_disappeared]):
                    return 0.5
                
                years_missing = datetime.now().year - year_disappeared
                expected_current_age = age_at_disappearance + years_missing
                
            else:  # search_type == 'found'
                # metadata1 is missing person, metadata2 is found person
                age_at_disappearance = metadata1.get('age_at_disappearance')
                year_disappeared = metadata1.get('year_disappeared')
                current_age = metadata2.get('current_age_estimate')
                
                if not all([age_at_disappearance, year_disappeared, current_age]):
                    return 0.5
                
                years_missing = datetime.now().year - year_disappeared
                expected_current_age = age_at_disappearance + years_missing
            
            # Calculate age difference
            age_diff = abs(current_age - expected_current_age)
            
            # Score based on age difference (within tolerance gets high score)
            if age_diff <= self.age_tolerance:
                return 1.0 - (age_diff / self.age_tolerance) * 0.3  # 0.7-1.0 range
            else:
                # Exponential decay for ages outside tolerance
                return max(0.0, 0.7 * np.exp(-(age_diff - self.age_tolerance) / 5.0))
            
        except Exception as e:
            logger.warning(f"Age consistency check failed: {str(e)}")
            return 0.5
    
    def _compare_marks(self, metadata1: Dict[str, Any], metadata2: Dict[str, Any]) -> float:
        """Compare birthmarks/visible marks between records."""
        try:
            marks1 = set()
            marks2 = set()
            
            # Extract marks from metadata1
            if 'birthmarks' in metadata1:
                marks1.update([mark.lower().strip() for mark in metadata1['birthmarks']])
            if 'visible_marks' in metadata1:
                marks1.update([mark.lower().strip() for mark in metadata1['visible_marks']])
            
            # Extract marks from metadata2
            if 'birthmarks' in metadata2:
                marks2.update([mark.lower().strip() for mark in metadata2['birthmarks']])
            if 'visible_marks' in metadata2:
                marks2.update([mark.lower().strip() for mark in metadata2['visible_marks']])
            
            if not marks1 and not marks2:
                return 0.5  # Neutral if no marks mentioned
            
            if not marks1 or not marks2:
                return 0.3  # Low score if only one has marks
            
            # Calculate Jaccard similarity
            intersection = len(marks1.intersection(marks2))
            union = len(marks1.union(marks2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Marks comparison failed: {str(e)}")
            return 0.5
    
    def _check_location_plausibility(
        self, 
        metadata1: Dict[str, Any], 
        metadata2: Dict[str, Any]
    ) -> float:
        """Check location plausibility between records."""
        try:
            # Extract locations
            location1 = metadata1.get('location_last_seen', '') or metadata1.get('current_location', '')
            location2 = metadata2.get('location_last_seen', '') or metadata2.get('current_location', '')
            
            if not location1 or not location2:
                return 0.5  # Neutral if location missing
            
            location1 = location1.lower().strip()
            location2 = location2.lower().strip()
            
            # Simple location matching (can be enhanced with geocoding)
            if location1 == location2:
                return 1.0
            
            # Check for common city/state names
            location1_parts = set(location1.replace(',', ' ').split())
            location2_parts = set(location2.replace(',', ' ').split())
            
            common_parts = location1_parts.intersection(location2_parts)
            total_parts = location1_parts.union(location2_parts)
            
            if total_parts:
                similarity = len(common_parts) / len(total_parts)
                return similarity
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Location plausibility check failed: {str(e)}")
            return 0.5
    
    def _validate_match(self, match: Dict[str, Any]) -> bool:
        """
        Validate a match to reject suspicious false positives.
        
        Rejects matches when:
        1. Gender mismatch with face similarity > 0.85 (STRICT - reject immediately)
        2. Face similarity > 0.92 but metadata_similarity < 0.35 (suspicious false positive)
        3. Face similarity > 0.90 but age_plausibility < 0.1 (very unlikely age progression)
        4. Face similarity > 0.88 but age_plausibility < 0.05 and metadata_similarity < 0.4
        5. VERY HIGH face similarity (>0.95) for children without strong metadata support (STRICT for children)
        
        Args:
            match: Match dictionary with face_similarity, metadata_similarity, and match_details
            
        Returns:
            True if match is valid, False if it should be rejected
        """
        try:
            face_sim = match.get('face_similarity', 0.0)
            metadata_sim = match.get('metadata_similarity', 0.0)
            match_details = match.get('match_details', {})
            gender_match = match_details.get('gender_match', 1.0)
            age_consistency = match_details.get('age_consistency', 1.0)
            
            # Check if both are children (age < 18, consistent with _is_child method)
            # Note: match['payload'] contains the matched person's metadata
            match_metadata = match.get('payload', {})
            
            # Try to get ages from match metadata
            # Use explicit None check to handle age=0 correctly (0 is falsy but valid)
            match_age = match_metadata.get('age_at_disappearance')
            if match_age is None:
                match_age = match_metadata.get('current_age_estimate')
            
            # For children detection, check if the matched person is a child
            # Also check query age from match_details if available
            both_children = False
            if match_age is not None and match_age < 18:
                # If matched person is a child, check query age from match_details
                query_age_info = match_details.get('query_current_age') or match_details.get('query_age_at_disappearance')
                if query_age_info is not None and query_age_info < 18:
                    both_children = True
                # If we can't determine query age but match is a child, be conservative
                # This helps prevent false positives for children (who have less distinctive faces)
                elif query_age_info is None:
                    both_children = True  # Conservative: assume both children if match is child
            
            # NOTE: No additional assignment to both_children after this point
            # Any code that overwrites both_children here would be a bug and should be removed
            
            # STRICT REJECTION: Gender mismatch with high face similarity (>0.85)
            # Gender is a critical distinguishing factor - different genders should NOT match
            # Even if faces look similar, gender mismatch is a strong indicator of false positive
            if gender_match == 0.0 and face_sim > 0.85:
                logger.warning(
                    f"STRICT REJECTION: Gender mismatch detected with high face similarity. "
                    f"face_sim={face_sim:.3f} - This is likely a false positive. Rejecting match."
                )
                return False
            
            # STRICT REJECTION FOR CHILDREN: Very high face similarity (>0.95) for children
            # Children's faces are less distinctive and models can produce false positives
            # Require VERY STRONG metadata support (metadata_sim > 0.7) for very high face similarity
            if both_children and face_sim > 0.95:
                if metadata_sim < 0.7:
                    logger.warning(
                        f"STRICT REJECTION FOR CHILDREN: Very high face similarity ({face_sim:.3f}) "
                        f"for children but weak metadata support (metadata_sim={metadata_sim:.3f}). "
                        f"Children's faces are less distinctive - requiring VERY STRONG metadata confirmation (>0.7). Rejecting match."
                    )
                    return False
            
            # STRICT REJECTION FOR CHILDREN: High face similarity (>0.90) for children without good metadata
            # Children's faces can be confused more easily - need better metadata support
            # For >0.90 similarity, require at least 0.6 metadata similarity
            if both_children and face_sim > 0.90:
                if metadata_sim < 0.6:
                    logger.warning(
                        f"STRICT REJECTION FOR CHILDREN: High face similarity ({face_sim:.3f}) "
                        f"for children but insufficient metadata support (metadata_sim={metadata_sim:.3f}, required >=0.6). "
                        f"Rejecting match to avoid false positives."
                    )
                    return False
            
            # ADDITIONAL STRICT REJECTION FOR CHILDREN: Extremely high similarity (>0.98) 
            # For children, even 98%+ similarity can be false positive - require EXCELLENT metadata (>0.75)
            if both_children and face_sim > 0.98:
                if metadata_sim < 0.75:
                    logger.warning(
                        f"STRICT REJECTION FOR CHILDREN: Extremely high face similarity ({face_sim:.3f}) "
                        f"for children but metadata support not excellent (metadata_sim={metadata_sim:.3f}, required >=0.75). "
                        f"Even 98%+ similarity for children can be false positive. Rejecting match."
                    )
                    return False
            
            # Reject: High face similarity (>0.92) but very low metadata similarity (<0.35)
            # This is a strong indicator of false positive (e.g., 2 different people with similar faces)
            if face_sim > 0.92 and metadata_sim < 0.35:
                logger.warning(
                    f"Rejecting suspicious match: face_sim={face_sim:.3f} but metadata_sim={metadata_sim:.3f} "
                    f"(likely false positive)"
                )
                return False
            
            # Reject: High face similarity (>0.90) but very low age plausibility (<0.1)
            # Age progression is a critical factor - if it doesn't make sense, likely false positive
            if face_sim > 0.90 and age_consistency < 0.1:
                logger.warning(
                    f"Rejecting match: face_sim={face_sim:.3f} but age_consistency={age_consistency:.3f} "
                    f"(age progression doesn't make sense, likely false positive)"
                )
                return False
            
            # Reject: Moderate-high face similarity (>0.88) with very poor age AND metadata
            # This combination strongly suggests false positive
            if face_sim > 0.88 and age_consistency < 0.05 and metadata_sim < 0.4:
                logger.warning(
                    f"Rejecting match: face_sim={face_sim:.3f} but age_consistency={age_consistency:.3f} "
                    f"and metadata_sim={metadata_sim:.3f} (multiple red flags, likely false positive)"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Match validation failed: {str(e)}, allowing match")
            return True  # On error, allow the match (fail open)
    
    def _get_match_details(
        self,
        query_metadata: Dict[str, Any],
        match_metadata: Dict[str, Any],
        search_type: str
    ) -> Dict[str, Any]:
        """Get detailed match information for explanation."""
        details = {
            'search_type': search_type,
            'gender_match': self._compare_gender(query_metadata, match_metadata),
            'age_consistency': self._check_age_consistency(query_metadata, match_metadata, search_type),
            'marks_similarity': self._compare_marks(query_metadata, match_metadata),
            'location_plausibility': self._check_location_plausibility(query_metadata, match_metadata)
        }
        
        # Add specific age information
        if search_type == 'missing':
            details['query_current_age'] = query_metadata.get('current_age_estimate')
            details['match_age_at_disappearance'] = match_metadata.get('age_at_disappearance')
            details['match_year_disappeared'] = match_metadata.get('year_disappeared')
        else:
            details['query_age_at_disappearance'] = query_metadata.get('age_at_disappearance')
            details['query_year_disappeared'] = query_metadata.get('year_disappeared')
            details['match_current_age'] = match_metadata.get('current_age_estimate')
        
        return details
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search service and database."""
        try:
            missing_stats = self.vector_db.get_collection_stats("missing_persons")
            found_stats = self.vector_db.get_collection_stats("found_persons")
            
            return {
                'missing_persons_count': missing_stats['points_count'],
                'found_persons_count': found_stats['points_count'],
                'face_threshold_adult': self.face_threshold_adult,
                'face_threshold_child': self.face_threshold_child,
                'face_threshold': self.face_threshold,  # Backward compatibility
                'metadata_weight': self.metadata_weight,
                'age_tolerance': self.age_tolerance,
                'total_records': missing_stats['points_count'] + found_stats['points_count']
            }
            
        except Exception as e:
            logger.error(f"Failed to get search statistics: {str(e)}")
            return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    try:
        from .vector_db import VectorDatabaseService
        
        # Initialize services
        vector_db = VectorDatabaseService(host="localhost", port=6333)
        search_service = BilateralSearchService(vector_db)
        
        # Create dummy data for testing
        dummy_embedding = np.random.rand(512).astype(np.float32)
        
        # Test search for missing persons
        found_metadata = {
            'current_age_estimate': 30,
            'gender': 'male',
            'current_location': 'Los Angeles, CA',
            'visible_marks': ['scar on left arm'],
            'finder_contact': 'finder@example.com'
        }
        
        missing_matches = search_service.search_for_missing(
            dummy_embedding, found_metadata, limit=5
        )
        print(f"Found {len(missing_matches)} potential missing person matches")
        
        # Test search for found persons
        missing_metadata = {
            'age_at_disappearance': 25,
            'year_disappeared': 2020,
            'gender': 'male',
            'location_last_seen': 'New York, NY',
            'birthmarks': ['scar on left arm'],
            'contact': 'family@example.com'
        }
        
        found_matches = search_service.search_for_found(
            dummy_embedding, missing_metadata, limit=5
        )
        print(f"Found {len(found_matches)} potential found person matches")
        
        # Get statistics
        stats = search_service.get_search_statistics()
        print(f"Search statistics: {stats}")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
