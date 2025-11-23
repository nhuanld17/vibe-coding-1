"""
Confidence Scoring service for Missing Person AI system.

This module provides explainable confidence scoring for face matches,
combining multiple factors to generate human-readable explanations.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class ConfidenceLevel(Enum):
    """Confidence level enumeration."""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


@dataclass
class ConfidenceFactors:
    """Data class for confidence scoring factors."""
    face_similarity: float
    metadata_consistency: float
    age_plausibility: float
    location_proximity: float
    distinctive_features: float
    temporal_consistency: float


class ConfidenceScoringService:
    """
    Confidence scoring service for explainable match assessment.
    
    This class provides methods for:
    - Calculating multi-factor confidence scores
    - Generating human-readable explanations
    - Classifying confidence levels
    - Providing detailed factor analysis
    """
    
    def __init__(
        self,
        face_weight: float = 0.5,
        metadata_weight: float = 0.2,
        age_weight: float = 0.15,
        location_weight: float = 0.1,
        features_weight: float = 0.05
    ) -> None:
        """
        Initialize the confidence scoring service.
        
        Args:
            face_weight: Weight for face similarity factor (0-1)
            metadata_weight: Weight for metadata consistency factor (0-1)
            age_weight: Weight for age plausibility factor (0-1)
            location_weight: Weight for location proximity factor (0-1)
            features_weight: Weight for distinctive features factor (0-1)
        """
        # Normalize weights to sum to 1.0
        total_weight = face_weight + metadata_weight + age_weight + location_weight + features_weight
        
        self.face_weight = face_weight / total_weight
        self.metadata_weight = metadata_weight / total_weight
        self.age_weight = age_weight / total_weight
        self.location_weight = location_weight / total_weight
        self.features_weight = features_weight / total_weight
        
        # Confidence thresholds
        self.thresholds = {
            ConfidenceLevel.VERY_HIGH: 0.90,
            ConfidenceLevel.HIGH: 0.75,
            ConfidenceLevel.MEDIUM: 0.60,
            ConfidenceLevel.LOW: 0.40,
            ConfidenceLevel.VERY_LOW: 0.0
        }
        
        logger.info(f"Confidence scoring initialized with weights: "
                   f"face={self.face_weight:.2f}, metadata={self.metadata_weight:.2f}, "
                   f"age={self.age_weight:.2f}, location={self.location_weight:.2f}, "
                   f"features={self.features_weight:.2f}")
    
    def calculate_confidence(
        self, 
        match_result: Dict[str, Any]
    ) -> Tuple[ConfidenceLevel, float, Dict[str, Any]]:
        """
        Calculate comprehensive confidence score for a match result.
        
        Args:
            match_result: Match result from bilateral search containing:
                - face_similarity: Face similarity score (0-1)
                - metadata_similarity: Metadata similarity score (0-1)
                - match_details: Detailed match information
                - payload: Metadata of the matched record
                
        Returns:
            Tuple of (confidence_level, confidence_score, explanation_dict)
            
        Example:
            >>> confidence_service = ConfidenceScoringService()
            >>> level, score, explanation = confidence_service.calculate_confidence(match_result)
            >>> print(f"Confidence: {level.value} ({score:.2f})")
        """
        try:
            # Extract factors from match result
            factors = self._extract_confidence_factors(match_result)
            
            # Check for gender mismatch (get from match_details)
            match_details = match_result.get('match_details', {})
            gender_match = match_details.get('gender_match', 1.0)
            
            # Calculate weighted confidence score
            confidence_score = (
                self.face_weight * factors.face_similarity +
                self.metadata_weight * factors.metadata_consistency +
                self.age_weight * factors.age_plausibility +
                self.location_weight * factors.location_proximity +
                self.features_weight * factors.distinctive_features
            )
            
            # Apply VERY HEAVY penalty if gender doesn't match (reduce confidence score by 50%)
            # Gender mismatch is a critical factor - different genders should not match
            if gender_match == 0.0:
                original_score = confidence_score
                confidence_score = confidence_score * 0.5  # Reduce by 50%
                logger.debug(f"Gender mismatch VERY HEAVY penalty applied: confidence_score reduced from {original_score:.3f} to {confidence_score:.3f}")
            
            # Ensure score is in valid range
            confidence_score = max(0.0, min(1.0, confidence_score))
            
            # Determine confidence level
            confidence_level = self._classify_confidence_level(confidence_score)
            
            # Generate explanation
            explanation = self._build_explanation(factors, confidence_score, confidence_level, match_result)
            
            logger.debug(f"Calculated confidence: {confidence_level.value} ({confidence_score:.3f})")
            
            return confidence_level, confidence_score, explanation
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return ConfidenceLevel.VERY_LOW, 0.0, {'error': str(e)}
    
    def _extract_confidence_factors(self, match_result: Dict[str, Any]) -> ConfidenceFactors:
        """Extract confidence factors from match result."""
        try:
            # Face similarity factor
            face_similarity = match_result.get('face_similarity', 0.0)
            
            # Metadata consistency factor
            metadata_consistency = match_result.get('metadata_similarity', 0.0)
            
            # Extract detailed match information
            match_details = match_result.get('match_details', {})
            
            # Age plausibility factor
            age_plausibility = self._calculate_age_factor(match_details)
            
            # Location proximity factor
            location_proximity = self._calculate_location_factor(match_details)
            
            # Distinctive features factor
            distinctive_features = self._calculate_features_factor(match_details)
            
            # Temporal consistency factor (for future use)
            temporal_consistency = self._calculate_temporal_factor(match_details)
            
            return ConfidenceFactors(
                face_similarity=face_similarity,
                metadata_consistency=metadata_consistency,
                age_plausibility=age_plausibility,
                location_proximity=location_proximity,
                distinctive_features=distinctive_features,
                temporal_consistency=temporal_consistency
            )
            
        except Exception as e:
            logger.warning(f"Factor extraction failed: {str(e)}")
            return ConfidenceFactors(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _calculate_age_factor(self, details: Dict[str, Any]) -> float:
        """Calculate age plausibility factor."""
        try:
            age_consistency = details.get('age_consistency', 0.0)
            
            # Age factor is directly the age consistency score
            # with some adjustments for edge cases
            if age_consistency >= 0.9:
                return 1.0  # Perfect age match
            elif age_consistency >= 0.7:
                return 0.8 + (age_consistency - 0.7) * 1.0  # 0.8-1.0 range
            elif age_consistency >= 0.5:
                return 0.5 + (age_consistency - 0.5) * 1.5  # 0.5-0.8 range
            else:
                return age_consistency  # 0.0-0.5 range
                
        except Exception as e:
            logger.warning(f"Age factor calculation failed: {str(e)}")
            return 0.5
    
    def _calculate_location_factor(self, details: Dict[str, Any]) -> float:
        """Calculate location proximity factor."""
        try:
            location_plausibility = details.get('location_plausibility', 0.0)
            
            # Location factor with geographical considerations
            if location_plausibility >= 0.8:
                return 1.0  # Same or very close locations
            elif location_plausibility >= 0.5:
                return 0.7 + (location_plausibility - 0.5) * 1.0  # 0.7-1.0 range
            elif location_plausibility >= 0.2:
                return 0.4 + (location_plausibility - 0.2) * 1.0  # 0.4-0.7 range
            else:
                return location_plausibility * 2.0  # 0.0-0.4 range
                
        except Exception as e:
            logger.warning(f"Location factor calculation failed: {str(e)}")
            return 0.5
    
    def _calculate_features_factor(self, details: Dict[str, Any]) -> float:
        """Calculate distinctive features factor."""
        try:
            marks_similarity = details.get('marks_similarity', 0.0)
            
            # Features factor based on distinctive marks/scars matching
            if marks_similarity >= 0.8:
                return 1.0  # Strong distinctive feature match
            elif marks_similarity >= 0.5:
                return 0.8 + (marks_similarity - 0.5) * 0.67  # 0.8-1.0 range
            elif marks_similarity >= 0.3:
                return 0.5 + (marks_similarity - 0.3) * 1.5  # 0.5-0.8 range
            else:
                return marks_similarity * 1.67  # 0.0-0.5 range
                
        except Exception as e:
            logger.warning(f"Features factor calculation failed: {str(e)}")
            return 0.5
    
    def _calculate_temporal_factor(self, details: Dict[str, Any]) -> float:
        """Calculate temporal consistency factor."""
        try:
            # For future implementation - consider time patterns, seasonal factors, etc.
            # Currently returns neutral score
            return 0.5
            
        except Exception as e:
            logger.warning(f"Temporal factor calculation failed: {str(e)}")
            return 0.5
    
    def _classify_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Classify confidence score into discrete levels."""
        if confidence_score >= self.thresholds[ConfidenceLevel.VERY_HIGH]:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= self.thresholds[ConfidenceLevel.HIGH]:
            return ConfidenceLevel.HIGH
        elif confidence_score >= self.thresholds[ConfidenceLevel.MEDIUM]:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= self.thresholds[ConfidenceLevel.LOW]:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _build_explanation(
        self,
        factors: ConfidenceFactors,
        confidence_score: float,
        confidence_level: ConfidenceLevel,
        match_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build human-readable explanation for the confidence score."""
        try:
            # Factor contributions
            factor_contributions = {
                'face_similarity': {
                    'score': factors.face_similarity,
                    'weight': self.face_weight,
                    'contribution': factors.face_similarity * self.face_weight,
                    'description': self._describe_face_similarity(factors.face_similarity)
                },
                'metadata_consistency': {
                    'score': factors.metadata_consistency,
                    'weight': self.metadata_weight,
                    'contribution': factors.metadata_consistency * self.metadata_weight,
                    'description': self._describe_metadata_consistency(factors.metadata_consistency)
                },
                'age_plausibility': {
                    'score': factors.age_plausibility,
                    'weight': self.age_weight,
                    'contribution': factors.age_plausibility * self.age_weight,
                    'description': self._describe_age_plausibility(factors.age_plausibility)
                },
                'location_proximity': {
                    'score': factors.location_proximity,
                    'weight': self.location_weight,
                    'contribution': factors.location_proximity * self.location_weight,
                    'description': self._describe_location_proximity(factors.location_proximity)
                },
                'distinctive_features': {
                    'score': factors.distinctive_features,
                    'weight': self.features_weight,
                    'contribution': factors.distinctive_features * self.features_weight,
                    'description': self._describe_distinctive_features(factors.distinctive_features)
                }
            }
            
            # Generate reasons list
            reasons = self._generate_reasons(factors, match_result)
            
            # Generate summary
            summary = self._generate_summary(confidence_level, confidence_score, factors)
            
            # Recommendations
            recommendations = self._generate_recommendations(confidence_level, factors)
            
            explanation = {
                'confidence_level': confidence_level.value,
                'confidence_score': round(confidence_score, 3),
                'factors': factor_contributions,
                'reasons': reasons,
                'summary': summary,
                'recommendations': recommendations,
                'threshold_info': {
                    'very_high': self.thresholds[ConfidenceLevel.VERY_HIGH],
                    'high': self.thresholds[ConfidenceLevel.HIGH],
                    'medium': self.thresholds[ConfidenceLevel.MEDIUM],
                    'low': self.thresholds[ConfidenceLevel.LOW]
                }
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation building failed: {str(e)}")
            return {'error': str(e)}
    
    def _describe_face_similarity(self, score: float) -> str:
        """Generate description for face similarity score."""
        if score >= 0.9:
            return "Excellent facial similarity - very strong match"
        elif score >= 0.8:
            return "High facial similarity - strong match"
        elif score >= 0.7:
            return "Good facial similarity - moderate match"
        elif score >= 0.6:
            return "Fair facial similarity - weak match"
        else:
            return "Low facial similarity - poor match"
    
    def _describe_metadata_consistency(self, score: float) -> str:
        """Generate description for metadata consistency score."""
        if score >= 0.9:
            return "Excellent metadata alignment - all details match well"
        elif score >= 0.8:
            return "High metadata consistency - most details align"
        elif score >= 0.7:
            return "Good metadata consistency - key details match"
        elif score >= 0.6:
            return "Fair metadata consistency - some details match"
        else:
            return "Low metadata consistency - few details align"
    
    def _describe_age_plausibility(self, score: float) -> str:
        """Generate description for age plausibility score."""
        if score >= 0.9:
            return "Age progression is highly plausible"
        elif score >= 0.8:
            return "Age progression is very plausible"
        elif score >= 0.7:
            return "Age progression is plausible"
        elif score >= 0.6:
            return "Age progression is somewhat plausible"
        else:
            return "Age progression is questionable"
    
    def _describe_location_proximity(self, score: float) -> str:
        """Generate description for location proximity score."""
        if score >= 0.9:
            return "Locations are very close or identical"
        elif score >= 0.8:
            return "Locations are in close proximity"
        elif score >= 0.7:
            return "Locations are reasonably close"
        elif score >= 0.6:
            return "Locations are somewhat distant"
        else:
            return "Locations are far apart"
    
    def _describe_distinctive_features(self, score: float) -> str:
        """Generate description for distinctive features score."""
        if score >= 0.9:
            return "Distinctive features match excellently"
        elif score >= 0.8:
            return "Distinctive features match well"
        elif score >= 0.7:
            return "Some distinctive features match"
        elif score >= 0.6:
            return "Few distinctive features match"
        else:
            return "No distinctive features match or insufficient data"
    
    def _generate_reasons(self, factors: ConfidenceFactors, match_result: Dict[str, Any]) -> List[str]:
        """Generate list of reasons supporting or questioning the match."""
        reasons = []
        
        # Face similarity reasons
        if factors.face_similarity >= 0.8:
            reasons.append(f"Strong facial similarity ({factors.face_similarity:.2f})")
        elif factors.face_similarity >= 0.6:
            reasons.append(f"Moderate facial similarity ({factors.face_similarity:.2f})")
        else:
            reasons.append(f"Weak facial similarity ({factors.face_similarity:.2f})")
        
        # Age reasons
        if factors.age_plausibility >= 0.8:
            reasons.append("Age progression is highly consistent")
        elif factors.age_plausibility < 0.5:
            reasons.append("Age progression raises concerns")
        
        # Location reasons
        if factors.location_proximity >= 0.7:
            reasons.append("Locations are geographically consistent")
        elif factors.location_proximity < 0.3:
            reasons.append("Locations are geographically distant")
        
        # Features reasons
        if factors.distinctive_features >= 0.7:
            reasons.append("Distinctive features support the match")
        elif factors.distinctive_features < 0.3:
            reasons.append("Distinctive features do not align")
        
        # Gender consistency
        match_details = match_result.get('match_details', {})
        gender_match = match_details.get('gender_match', 0.0)
        if gender_match == 1.0:
            reasons.append("Gender information matches")
        elif gender_match == 0.0:
            reasons.append("Gender information conflicts")
        
        # Warning for suspicious high face similarity with low metadata
        if factors.face_similarity > 0.95 and factors.metadata_consistency < 0.3:
            reasons.append("⚠️ WARNING: Very high face similarity but very low metadata consistency - possible false positive")
        elif factors.face_similarity > 0.90 and factors.metadata_consistency < 0.4:
            reasons.append("⚠️ CAUTION: High face similarity but low metadata consistency - verify carefully")
        
        return reasons
    
    def _generate_summary(
        self, 
        confidence_level: ConfidenceLevel, 
        confidence_score: float, 
        factors: ConfidenceFactors
    ) -> str:
        """Generate overall summary of the match confidence."""
        level_descriptions = {
            ConfidenceLevel.VERY_HIGH: "This is a very strong potential match with excellent alignment across multiple factors.",
            ConfidenceLevel.HIGH: "This is a strong potential match with good alignment across most factors.",
            ConfidenceLevel.MEDIUM: "This is a moderate potential match with reasonable alignment in key factors.",
            ConfidenceLevel.LOW: "This is a weak potential match with limited alignment across factors.",
            ConfidenceLevel.VERY_LOW: "This is a very weak potential match with poor alignment across factors."
        }
        
        base_summary = level_descriptions[confidence_level]
        
        # Add specific insights
        if factors.face_similarity >= 0.8 and factors.age_plausibility >= 0.8:
            base_summary += " Both facial features and age progression strongly support this match."
        elif factors.face_similarity >= 0.8:
            base_summary += " Facial features strongly support this match."
        elif factors.age_plausibility >= 0.8:
            base_summary += " Age progression strongly supports this match."
        
        # Add warning for suspicious matches
        if factors.face_similarity > 0.95 and factors.metadata_consistency < 0.3:
            base_summary += " ⚠️ WARNING: Despite high face similarity, metadata does not align well - verify carefully to avoid false positive."
        elif factors.face_similarity > 0.90 and factors.metadata_consistency < 0.4:
            base_summary += " ⚠️ CAUTION: High face similarity but metadata inconsistencies detected - additional verification recommended."
        
        return base_summary
    
    def _generate_recommendations(
        self, 
        confidence_level: ConfidenceLevel, 
        factors: ConfidenceFactors
    ) -> List[str]:
        """Generate recommendations based on confidence level and factors."""
        recommendations = []
        
        if confidence_level == ConfidenceLevel.VERY_HIGH:
            recommendations.extend([
                "Immediately contact the relevant authorities",
                "Prepare for potential family notification",
                "Gather additional verification materials"
            ])
        elif confidence_level == ConfidenceLevel.HIGH:
            recommendations.extend([
                "Contact authorities for further investigation",
                "Collect additional identifying information",
                "Consider DNA testing if possible"
            ])
        elif confidence_level == ConfidenceLevel.MEDIUM:
            recommendations.extend([
                "Conduct additional verification steps",
                "Gather more recent photos if available",
                "Cross-reference with other databases"
            ])
        elif confidence_level == ConfidenceLevel.LOW:
            recommendations.extend([
                "Treat as preliminary lead requiring verification",
                "Collect more detailed information",
                "Consider expanding search parameters"
            ])
        else:  # VERY_LOW
            recommendations.extend([
                "Consider this a weak lead",
                "Review search parameters",
                "Focus on higher-confidence matches"
            ])
        
        # Factor-specific recommendations
        if factors.face_similarity < 0.6:
            recommendations.append("Consider using additional photos for comparison")
        
        if factors.age_plausibility < 0.5:
            recommendations.append("Verify age information and timeline details")
        
        if factors.location_proximity < 0.3:
            recommendations.append("Investigate possible travel or relocation patterns")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize confidence scoring service
        confidence_service = ConfidenceScoringService()
        
        # Create dummy match result for testing
        dummy_match_result = {
            'face_similarity': 0.85,
            'metadata_similarity': 0.75,
            'match_details': {
                'gender_match': 1.0,
                'age_consistency': 0.9,
                'marks_similarity': 0.6,
                'location_plausibility': 0.4
            },
            'payload': {
                'name': 'Test Person',
                'age_at_disappearance': 25,
                'year_disappeared': 2020
            }
        }
        
        # Calculate confidence
        level, score, explanation = confidence_service.calculate_confidence(dummy_match_result)
        
        print(f"Confidence Level: {level.value}")
        print(f"Confidence Score: {score:.3f}")
        print(f"Summary: {explanation['summary']}")
        print(f"Reasons: {explanation['reasons']}")
        print(f"Recommendations: {explanation['recommendations']}")
        
        # Print factor details
        print("\nFactor Analysis:")
        for factor_name, factor_data in explanation['factors'].items():
            print(f"  {factor_name}: {factor_data['score']:.2f} "
                  f"(weight: {factor_data['weight']:.2f}, "
                  f"contribution: {factor_data['contribution']:.3f})")
            print(f"    {factor_data['description']}")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
