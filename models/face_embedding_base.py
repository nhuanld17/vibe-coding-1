"""
Base abstraction for face embedding backends.

This module provides a pluggable interface for different face embedding
implementations (InsightFace, ONNX, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np
from loguru import logger


class BaseFaceEmbedder(ABC):
    """
    Abstract base class for face embedding extractors.
    
    All embedding backends must implement this interface to ensure
    compatibility with the rest of the system.
    """
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings (e.g., 512)."""
        pass
    
    @abstractmethod
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from an aligned face image.
        
        Args:
            face_image: Aligned face image as numpy array (BGR format, 112x112)
            
        Returns:
            L2-normalized embedding vector of shape (embedding_dim,)
            
        Raises:
            ValueError: If input image is invalid
            RuntimeError: If embedding extraction fails
        """
        pass
    
    def extract_batch_embeddings(self, face_images: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from multiple face images in batch.
        
        Default implementation processes images sequentially.
        Subclasses can override for optimized batch processing.
        
        Args:
            face_images: List of aligned face images (BGR format, 112x112)
            
        Returns:
            Array of L2-normalized embeddings with shape (N, embedding_dim)
        """
        embeddings = [self.extract_embedding(img) for img in face_images]
        return np.array(embeddings)
    
    @staticmethod
    def normalize(embedding: np.ndarray) -> np.ndarray:
        """
        L2 normalize an embedding vector.
        
        Args:
            embedding: Embedding vector
            
        Returns:
            L2-normalized embedding vector
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(similarity)
    
    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Euclidean distance (lower values indicate higher similarity)
        """
        distance = np.linalg.norm(emb1 - emb2)
        return float(distance)
    
    def compare_faces(
        self, 
        face1: np.ndarray, 
        face2: np.ndarray,
        threshold: float = 0.6
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Compare two faces and determine if they are the same person.
        
        Args:
            face1: First face image
            face2: Second face image
            threshold: Similarity threshold for matching (0-1)
            
        Returns:
            Tuple of (is_match, similarity_score, metrics_dict)
        """
        try:
            emb1 = self.extract_embedding(face1)
            emb2 = self.extract_embedding(face2)
            
            cosine_sim = self.cosine_similarity(emb1, emb2)
            euclidean_dist = self.euclidean_distance(emb1, emb2)
            
            is_match = cosine_sim >= threshold
            
            metrics = {
                'cosine_similarity': cosine_sim,
                'euclidean_distance': euclidean_dist,
                'threshold': threshold
            }
            
            return is_match, cosine_sim, metrics
            
        except Exception as e:
            logger.error(f"Face comparison failed: {str(e)}")
            raise RuntimeError(f"Face comparison failed: {str(e)}")
    
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding backend.
        
        Returns:
            Dictionary containing backend information
        """
        return {
            'backend_type': self.__class__.__name__,
            'embedding_dim': self.embedding_dim
        }



