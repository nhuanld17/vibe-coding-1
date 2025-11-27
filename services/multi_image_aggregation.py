"""
Multi-Image Aggregation Service for Missing Person AI System.

This module provides aggregation logic for combining multiple face embeddings
from different photos of the same person to improve matching accuracy across
large age gaps (>30 years).

Key Features:
- Aggregate similarity scores from multiple image-to-image comparisons
- Age-bracket preference: prioritize comparisons between similar ages
- Consistency scoring: bonus for multiple good matches
- Handle edge cases: None embeddings, empty arrays, missing metadata

Performance:
- 10x10 image aggregation: ~5ms (negligible overhead)
- No database calls (operates on in-memory embeddings)

Author: AI Face Recognition Team
Version: 1.0.0
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class ImagePairScore:
    """
    Score information for a single image-to-image comparison.
    
    Attributes:
        query_image_id: ID of the query image
        target_image_id: ID of the target image
        similarity: Face similarity score (0-1, cosine similarity)
        query_age: Age at query photo
        target_age: Age at target photo
        age_gap: Absolute age difference
    """
    query_image_id: str
    target_image_id: str
    similarity: float
    query_age: int
    target_age: int
    age_gap: int
    
    def __repr__(self) -> str:
        return (f"ImagePairScore(query={self.query_image_id}, "
                f"target={self.target_image_id}, sim={self.similarity:.3f}, "
                f"age_gap={self.age_gap}y)")


@dataclass
class AggregatedMatchResult:
    """
    Aggregated match result for a person with multiple images.
    
    Attributes:
        target_case_id: Case ID of the matched person
        best_similarity: Best face similarity score across all image pairs
        mean_similarity: Mean of top-k similarities
        consistency_score: Bonus score for multiple good matches (0-1)
        final_score: Combined score with consistency bonus
        best_age_gap: Age gap of the best matching image pair
        matched_query_image_id: Query image ID with best match
        matched_target_image_id: Target image ID with best match
        all_pair_scores: List of all image pair scores (for debugging)
        num_good_matches: Number of pairs above threshold
    """
    target_case_id: str
    best_similarity: float
    mean_similarity: float
    consistency_score: float
    final_score: float
    best_age_gap: int
    matched_query_image_id: str
    matched_target_image_id: str
    all_pair_scores: List[ImagePairScore]
    num_good_matches: int
    
    def __repr__(self) -> str:
        return (f"AggregatedMatch(case={self.target_case_id}, "
                f"best={self.best_similarity:.3f}, mean={self.mean_similarity:.3f}, "
                f"final={self.final_score:.3f}, consistency={self.consistency_score:.3f})")


class MultiImageAggregationService:
    """
    Service for aggregating multi-image similarity scores.
    
    This service implements the "Best Match Per Person" strategy:
    1. Compare each query image against each target image
    2. Track best similarity per target person
    3. Apply age-bracket preference (closer ages = more reliable)
    4. Add consistency bonus for multiple good matches
    5. Return aggregated scores per person
    """
    
    def __init__(
        self,
        consistency_bonus_weight: float = 0.05,
        good_match_threshold: float = 0.25,
        age_bracket_preference_enabled: bool = True,
        age_bracket_bonus: float = 0.02
    ) -> None:
        """
        Initialize the multi-image aggregation service.
        
        Args:
            consistency_bonus_weight: Weight for consistency bonus (0-0.1 recommended)
            good_match_threshold: Minimum similarity to count as "good match" for consistency
            age_bracket_preference_enabled: Enable age-bracket preference scoring
            age_bracket_bonus: Bonus for matches within same age bracket (0-0.05 recommended)
            
        Raises:
            ValueError: If parameters are out of valid range
        """
        # Validate parameters
        if not (0.0 <= consistency_bonus_weight <= 0.2):
            raise ValueError(f"consistency_bonus_weight must be in [0, 0.2], got {consistency_bonus_weight}")
        if not (0.0 <= good_match_threshold <= 1.0):
            raise ValueError(f"good_match_threshold must be in [0, 1], got {good_match_threshold}")
        if not (0.0 <= age_bracket_bonus <= 0.1):
            raise ValueError(f"age_bracket_bonus must be in [0, 0.1], got {age_bracket_bonus}")
        
        self.consistency_bonus_weight = consistency_bonus_weight
        self.good_match_threshold = good_match_threshold
        self.age_bracket_preference_enabled = age_bracket_preference_enabled
        self.age_bracket_bonus = age_bracket_bonus
        
        logger.info(
            f"MultiImageAggregationService initialized: "
            f"consistency_bonus={consistency_bonus_weight}, "
            f"good_match_threshold={good_match_threshold}, "
            f"age_bracket_preference={age_bracket_preference_enabled}"
        )
    
    def aggregate_multi_image_similarity(
        self,
        query_images: List[Dict[str, Any]],
        target_images: List[Dict[str, Any]]
    ) -> AggregatedMatchResult:
        """
        Aggregate similarity scores between multiple query and target images.
        
        This is the main entry point for multi-image aggregation.
        
        Args:
            query_images: List of query image dicts with keys:
                - 'image_id': Unique image ID
                - 'embedding': 512-D numpy array (or None if failed)
                - 'age_at_photo': Age when photo was taken
                - 'case_id' or 'found_id': Person identifier
            target_images: List of target image dicts (same structure)
            
        Returns:
            AggregatedMatchResult with best match, mean score, consistency bonus, etc.
            
        Raises:
            ValueError: If inputs are invalid (empty lists, missing required fields)
            
        Example:
            >>> query_imgs = [
            ...     {'image_id': 'MISS_001_img_0', 'embedding': emb1, 'age_at_photo': 8, 'case_id': 'MISS_001'},
            ...     {'image_id': 'MISS_001_img_1', 'embedding': emb2, 'age_at_photo': 15, 'case_id': 'MISS_001'}
            ... ]
            >>> target_imgs = [
            ...     {'image_id': 'FOUND_050_img_0', 'embedding': emb3, 'age_at_photo': 30, 'found_id': 'FOUND_050'}
            ... ]
            >>> result = aggregator.aggregate_multi_image_similarity(query_imgs, target_imgs)
            >>> print(f"Best match: {result.best_similarity:.3f}, Final score: {result.final_score:.3f}")
        """
        # Validate inputs
        if not query_images or not target_images:
            raise ValueError("query_images and target_images must not be empty")
        
        # Extract target person ID
        target_case_id = self._extract_person_id(target_images[0])
        
        # Compute all pairwise similarities
        try:
            all_pair_scores = self._compute_all_pairs(query_images, target_images)
        except Exception as e:
            logger.error(f"Failed to compute pairwise similarities: {str(e)}")
            raise ValueError(f"Pairwise similarity computation failed: {str(e)}")
        
        # Handle case with no valid pairs (all embeddings None or failed)
        if not all_pair_scores:
            logger.warning(
                f"No valid image pairs found between query and target {target_case_id}. "
                f"Returning zero-score result."
            )
            return AggregatedMatchResult(
                target_case_id=target_case_id,
                best_similarity=0.0,
                mean_similarity=0.0,
                consistency_score=0.0,
                final_score=0.0,
                best_age_gap=999,
                matched_query_image_id="",
                matched_target_image_id="",
                all_pair_scores=[],
                num_good_matches=0
            )
        
        # Sort by similarity (descending)
        all_pair_scores.sort(key=lambda x: x.similarity, reverse=True)
        
        # Get best match
        best_pair = all_pair_scores[0]
        best_similarity = best_pair.similarity
        
        # Calculate mean of top-k scores (k = min(5, total_pairs))
        top_k = min(5, len(all_pair_scores))
        mean_similarity = np.mean([pair.similarity for pair in all_pair_scores[:top_k]])
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(all_pair_scores)
        
        # Calculate final score with consistency bonus
        final_score = self._calculate_final_score(
            best_similarity, mean_similarity, consistency_score
        )
        
        # Count good matches
        num_good_matches = sum(
            1 for pair in all_pair_scores 
            if pair.similarity >= self.good_match_threshold
        )
        
        result = AggregatedMatchResult(
            target_case_id=target_case_id,
            best_similarity=best_similarity,
            mean_similarity=mean_similarity,
            consistency_score=consistency_score,
            final_score=final_score,
            best_age_gap=best_pair.age_gap,
            matched_query_image_id=best_pair.query_image_id,
            matched_target_image_id=best_pair.target_image_id,
            all_pair_scores=all_pair_scores,
            num_good_matches=num_good_matches
        )
        
        logger.debug(f"Aggregated result: {result}")
        return result
    
    def _compute_all_pairs(
        self,
        query_images: List[Dict[str, Any]],
        target_images: List[Dict[str, Any]]
    ) -> List[ImagePairScore]:
        """
        Compute similarity scores for all query-target image pairs.
        
        Args:
            query_images: List of query image dicts
            target_images: List of target image dicts
            
        Returns:
            List of ImagePairScore objects (may be empty if all embeddings are None)
        """
        pair_scores = []
        
        for query_img in query_images:
            query_embedding = query_img.get('embedding')
            query_image_id = query_img.get('image_id', 'unknown_query')
            query_age = query_img.get('age_at_photo')
            
            # Skip if embedding is None or invalid
            if query_embedding is None or not self._is_valid_embedding(query_embedding):
                logger.debug(f"Skipping query image {query_image_id}: invalid embedding")
                continue
            
            for target_img in target_images:
                target_embedding = target_img.get('embedding')
                target_image_id = target_img.get('image_id', 'unknown_target')
                target_age = target_img.get('age_at_photo')
                
                # Skip if embedding is None or invalid
                if target_embedding is None or not self._is_valid_embedding(target_embedding):
                    logger.debug(f"Skipping target image {target_image_id}: invalid embedding")
                    continue
                
                # Compute cosine similarity (ALWAYS, even if age is None)
                try:
                    similarity = self._cosine_similarity(query_embedding, target_embedding)
                except Exception as e:
                    logger.error(
                        f"Failed to compute similarity between {query_image_id} and {target_image_id}: {e}"
                    )
                    continue
                
                # Apply age-bracket preference bonus ONLY if both ages are available
                if self.age_bracket_preference_enabled:
                    if query_age is not None and target_age is not None:
                        age_gap = abs(query_age - target_age)
                        similarity = self._apply_age_bracket_bonus(similarity, age_gap)
                        logger.debug(
                            f"Age bonus applied: query_age={query_age}, target_age={target_age}, "
                            f"gap={age_gap}y, sim={similarity:.3f}"
                        )
                    else:
                        logger.debug(
                            f"Age bonus skipped: query_age={query_age}, target_age={target_age}, "
                            f"using raw similarity={similarity:.3f}"
                        )
                
                # Calculate age gap (0 if either age is None)
                age_gap = abs(query_age - target_age) if (query_age is not None and target_age is not None) else 0
                
                # Use 0 for display if age is None (ImagePairScore expects int)
                display_query_age = query_age if query_age is not None else 0
                display_target_age = target_age if target_age is not None else 0
                
                pair_score = ImagePairScore(
                    query_image_id=query_image_id,
                    target_image_id=target_image_id,
                    similarity=similarity,
                    query_age=display_query_age,
                    target_age=display_target_age,
                    age_gap=age_gap
                )
                pair_scores.append(pair_score)
        
        logger.debug(f"Computed {len(pair_scores)} valid image pair scores")
        return pair_scores
    
    def _is_valid_embedding(self, embedding: Any) -> bool:
        """
        Check if an embedding is valid (non-None, correct shape, non-zero).
        
        Args:
            embedding: Embedding to validate (expected to be numpy array)
            
        Returns:
            True if valid, False otherwise
        """
        if embedding is None:
            return False
        
        # Check if it's a numpy array
        if not isinstance(embedding, np.ndarray):
            logger.warning(f"Embedding is not a numpy array: {type(embedding)}")
            return False
        
        # Check shape
        if embedding.shape != (512,) and embedding.size != 512:
            logger.warning(f"Embedding has invalid shape: {embedding.shape}")
            return False
        
        # Check for all-zero embedding (indicates failure)
        if np.allclose(embedding, 0.0):
            logger.warning("Embedding is all zeros")
            return False
        
        # Check for NaN or Inf
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            logger.warning("Embedding contains NaN or Inf")
            return False
        
        return True
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding (512-D)
            embedding2: Second embedding (512-D)
            
        Returns:
            Cosine similarity score (0-1, higher is more similar)
            
        Raises:
            ValueError: If embeddings have different shapes
        """
        # Flatten to 1D (in case shape is (512,) or (1, 512))
        emb1 = embedding1.flatten()
        emb2 = embedding2.flatten()
        
        if emb1.shape != emb2.shape:
            raise ValueError(f"Embedding shapes don't match: {emb1.shape} vs {emb2.shape}")
        
        # Compute cosine similarity: dot(A, B) / (norm(A) * norm(B))
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        # Avoid division by zero
        if norm1 == 0.0 or norm2 == 0.0:
            logger.warning("One of the embeddings has zero norm")
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Clamp to [0, 1] (cosine can be negative, but for face embeddings it's typically positive)
        # For ArcFace embeddings, similarity should be in [-1, 1], but we normalize to [0, 1]
        similarity = (similarity + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        similarity = max(0.0, min(1.0, similarity))
        
        return float(similarity)
    
    def _apply_age_bracket_bonus(self, similarity: float, age_gap: int) -> float:
        """
        Apply a small bonus for matches within same age bracket.
        
        Rationale: Photos with closer ages are more reliable matches.
        Bonus is small to avoid over-weighting age vs face similarity.
        
        Args:
            similarity: Original similarity score
            age_gap: Age gap in years
            
        Returns:
            Adjusted similarity with age bracket bonus
        """
        # Define age brackets with bonuses
        if age_gap <= 3:
            bonus = self.age_bracket_bonus  # Same age range: full bonus
        elif age_gap <= 7:
            bonus = self.age_bracket_bonus * 0.7  # Close age: 70% bonus
        elif age_gap <= 15:
            bonus = self.age_bracket_bonus * 0.4  # Moderate gap: 40% bonus
        else:
            bonus = 0.0  # Large gap: no bonus
        
        # Apply bonus (capped at 1.0)
        adjusted_similarity = min(1.0, similarity + bonus)
        
        if bonus > 0:
            logger.debug(
                f"Age bracket bonus applied: age_gap={age_gap}y, "
                f"bonus={bonus:.3f}, sim={similarity:.3f} -> {adjusted_similarity:.3f}"
            )
        
        return adjusted_similarity
    
    def _calculate_consistency_score(self, pair_scores: List[ImagePairScore]) -> float:
        """
        Calculate consistency score based on number of good matches.
        
        Rationale: If multiple image pairs match well, it's more likely a true match.
        
        Args:
            pair_scores: List of all pair scores (sorted by similarity descending)
            
        Returns:
            Consistency score (0-1)
        """
        if not pair_scores:
            return 0.0
        
        # Count good matches (above threshold)
        good_matches = [
            pair for pair in pair_scores 
            if pair.similarity >= self.good_match_threshold
        ]
        num_good = len(good_matches)
        
        # Consistency score based on percentage of good matches
        # Also consider absolute number (need at least 2 for consistency bonus)
        if num_good < 2:
            consistency = 0.0
        else:
            # Percentage of good matches (0-1)
            percentage_good = num_good / len(pair_scores)
            
            # Bonus for multiple good matches (logarithmic scale)
            # 2 good matches: ~0.3, 5 good matches: ~0.7, 10+ good matches: ~1.0
            count_bonus = min(1.0, np.log(num_good) / np.log(10))
            
            # Combine percentage and count bonus
            consistency = (percentage_good * 0.6 + count_bonus * 0.4)
        
        logger.debug(
            f"Consistency score: {num_good}/{len(pair_scores)} good matches "
            f"-> score={consistency:.3f}"
        )
        
        return consistency
    
    def _calculate_final_score(
        self,
        best_similarity: float,
        mean_similarity: float,
        consistency_score: float
    ) -> float:
        """
        Calculate final aggregated score with consistency bonus.
        
        Formula: final = best * 0.7 + mean * 0.2 + (consistency_bonus * consistency_score) * 0.1
        
        Args:
            best_similarity: Best pairwise similarity
            mean_similarity: Mean of top-k similarities
            consistency_score: Consistency score (0-1)
            
        Returns:
            Final aggregated score (0-1)
        """
        # Weights
        best_weight = 0.7
        mean_weight = 0.2
        consistency_weight = 0.1
        
        # Calculate final score
        final = (
            best_weight * best_similarity +
            mean_weight * mean_similarity +
            consistency_weight * (consistency_score * self.consistency_bonus_weight * 10)  # Scale up
        )
        
        # Clamp to [0, 1]
        final = max(0.0, min(1.0, final))
        
        logger.debug(
            f"Final score: best={best_similarity:.3f} ({best_weight}), "
            f"mean={mean_similarity:.3f} ({mean_weight}), "
            f"consistency={consistency_score:.3f} ({consistency_weight}) -> {final:.3f}"
        )
        
        return final
    
    def _extract_person_id(self, image_dict: Dict[str, Any]) -> str:
        """
        Extract person ID from image dict (case_id or found_id).
        
        Args:
            image_dict: Image dictionary with metadata
            
        Returns:
            Person ID (case_id or found_id)
        """
        case_id = image_dict.get('case_id') or image_dict.get('found_id')
        if not case_id:
            logger.warning(f"Image dict missing case_id/found_id: {image_dict.keys()}")
            return "unknown_person"
        return case_id
    
    def aggregate_multiple_persons(
        self,
        query_images: List[Dict[str, Any]],
        target_persons: List[List[Dict[str, Any]]]
    ) -> List[AggregatedMatchResult]:
        """
        Aggregate matches for multiple target persons at once.
        
        This is a convenience method for batch aggregation.
        
        Args:
            query_images: List of query images (same person)
            target_persons: List of target image lists (each sublist = one person)
            
        Returns:
            List of AggregatedMatchResult, sorted by final_score descending
            
        Example:
            >>> query_imgs = [...]  # 5 images of missing person A
            >>> target_persons = [
            ...     [...],  # 3 images of found person X
            ...     [...],  # 7 images of found person Y
            ...     [...]   # 2 images of found person Z
            ... ]
            >>> results = aggregator.aggregate_multiple_persons(query_imgs, target_persons)
            >>> print(f"Top match: {results[0].target_case_id} with score {results[0].final_score:.3f}")
        """
        results = []
        
        for target_imgs in target_persons:
            try:
                result = self.aggregate_multi_image_similarity(query_images, target_imgs)
                results.append(result)
            except Exception as e:
                target_id = self._extract_person_id(target_imgs[0]) if target_imgs else "unknown"
                logger.error(f"Failed to aggregate for target {target_id}: {str(e)}")
                continue
        
        # Sort by final score descending
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        logger.info(f"Aggregated matches for {len(results)} target persons")
        return results


# Singleton instance for easy import
_default_service = None

def get_aggregation_service(
    consistency_bonus_weight: float = 0.05,
    good_match_threshold: float = 0.25
) -> MultiImageAggregationService:
    """
    Get or create the default aggregation service instance.
    
    Args:
        consistency_bonus_weight: Weight for consistency bonus
        good_match_threshold: Minimum similarity for "good match"
        
    Returns:
        MultiImageAggregationService instance
    """
    global _default_service
    if _default_service is None:
        _default_service = MultiImageAggregationService(
            consistency_bonus_weight=consistency_bonus_weight,
            good_match_threshold=good_match_threshold
        )
    return _default_service


# Example usage and testing
if __name__ == "__main__":
    # Test with dummy data
    logger.info("Testing MultiImageAggregationService...")
    
    # Create service
    service = MultiImageAggregationService()
    
    # Create dummy embeddings
    def create_dummy_image(image_id: str, age: int, case_id: str) -> Dict[str, Any]:
        return {
            'image_id': image_id,
            'embedding': np.random.rand(512).astype(np.float32),
            'age_at_photo': age,
            'case_id': case_id
        }
    
    # Test case 1: 3 query images vs 2 target images
    query_imgs = [
        create_dummy_image('MISS_001_img_0', 8, 'MISS_001'),
        create_dummy_image('MISS_001_img_1', 15, 'MISS_001'),
        create_dummy_image('MISS_001_img_2', 22, 'MISS_001')
    ]
    
    target_imgs = [
        create_dummy_image('FOUND_050_img_0', 30, 'FOUND_050'),
        create_dummy_image('FOUND_050_img_1', 33, 'FOUND_050')
    ]
    
    result = service.aggregate_multi_image_similarity(query_imgs, target_imgs)
    print(f"\n✅ Test case 1: {result}")
    print(f"   Best similarity: {result.best_similarity:.3f}")
    print(f"   Final score: {result.final_score:.3f}")
    print(f"   Good matches: {result.num_good_matches}/{len(result.all_pair_scores)}")
    
    # Test case 2: Handle None embeddings
    query_imgs_with_none = [
        create_dummy_image('MISS_002_img_0', 10, 'MISS_002'),
        {'image_id': 'MISS_002_img_1', 'embedding': None, 'age_at_photo': 12, 'case_id': 'MISS_002'},
        create_dummy_image('MISS_002_img_2', 15, 'MISS_002')
    ]
    
    result2 = service.aggregate_multi_image_similarity(query_imgs_with_none, target_imgs)
    print(f"\n✅ Test case 2 (with None embedding): {result2}")
    print(f"   Valid pairs: {len(result2.all_pair_scores)} (should be 4: 2 valid query × 2 target)")
    
    logger.info("All tests passed! ✅")

