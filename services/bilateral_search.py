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
        metadata_weight: float = 0.3,
        age_tolerance: int = 5
    ) -> None:
        """
        Initialize the bilateral search service.
        
        Args:
            vector_db: Vector database service instance
            face_threshold: Minimum face similarity threshold (0-1)
            metadata_weight: Weight for metadata similarity in final score (0-1)
            age_tolerance: Age tolerance in years for matching
        """
        self.vector_db = vector_db
        self.face_threshold = face_threshold
        self.metadata_weight = metadata_weight
        self.face_weight = 1.0 - metadata_weight
        self.age_tolerance = age_tolerance
        
        logger.info(f"Bilateral search initialized with face_threshold={face_threshold}, "
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
            vector_matches = self.vector_db.search_similar_faces(
                query_embedding=found_embedding,
                collection_name="missing_persons",
                limit=limit * 2,  # Get more results for reranking
                score_threshold=self.face_threshold,
                filters=filters
            )
            
            # Rerank with metadata similarity
            reranked_matches = self._rerank_with_metadata(
                vector_matches, found_metadata, 'missing'
            )
            
            # Limit results
            final_matches = reranked_matches[:limit]
            
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
            vector_matches = self.vector_db.search_similar_faces(
                query_embedding=missing_embedding,
                collection_name="found_persons",
                limit=limit * 2,  # Get more results for reranking
                score_threshold=self.face_threshold,
                filters=filters
            )
            
            # Rerank with metadata similarity
            reranked_matches = self._rerank_with_metadata(
                vector_matches, missing_metadata, 'found'
            )
            
            # Limit results
            final_matches = reranked_matches[:limit]
            
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
            if 'gender' in metadata:
                filters['gender'] = metadata['gender']
            
            # Age range filter
            if search_type == 'missing':
                # Searching missing persons with found person data
                if 'current_age_estimate' in metadata:
                    current_age = metadata['current_age_estimate']
                    # Calculate possible age range at disappearance
                    min_age_at_disappearance = max(0, current_age - 30)  # Max 30 years missing
                    max_age_at_disappearance = current_age - 1  # At least 1 year missing
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
                'face_threshold': self.face_threshold,
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
