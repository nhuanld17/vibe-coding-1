"""
Face Embedding module - Factory and compatibility layer.

This module provides a factory function to create the appropriate embedding backend
based on configuration, and maintains backward compatibility with the old
FaceEmbeddingExtractor interface.
"""

import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any
from pathlib import Path
import os
from loguru import logger

from .face_embedding_base import BaseFaceEmbedder
from .face_embedding_insightface import InsightFaceEmbedder
from .face_embedding_onnx import OnnxArcFaceEmbedder


def create_embedding_backend(
    backend_type: str = "insightface",
    model_path: Optional[str] = None,
    use_gpu: bool = False,
    **kwargs
) -> BaseFaceEmbedder:
    """
    Factory function to create the appropriate embedding backend.
    
    Args:
        backend_type: Type of backend ("insightface" or "onnx")
        model_path: Path to ONNX model (required for "onnx" backend)
        use_gpu: Whether to use GPU for inference
        **kwargs: Additional backend-specific parameters
        
    Returns:
        BaseFaceEmbedder instance
        
    Raises:
        ValueError: If backend_type is invalid or required parameters are missing
        RuntimeError: If backend initialization fails
        
    Example:
        >>> # Use InsightFace (recommended)
        >>> embedder = create_embedding_backend("insightface", use_gpu=False)
        >>> 
        >>> # Use ONNX (deprecated)
        >>> embedder = create_embedding_backend("onnx", model_path="model.onnx")
    """
    backend_type = backend_type.lower()
    
    if backend_type == "insightface":
        model_name = kwargs.get("model_name", "antelopev2")
        det_size = kwargs.get("det_size", (640, 640))
        return InsightFaceEmbedder(
            model_name=model_name,
            use_gpu=use_gpu,
            det_size=det_size
        )
    
    elif backend_type == "onnx":
        if not model_path:
            raise ValueError("model_path is required for ONNX backend")
        deprecated = kwargs.get("deprecated", True)
        return OnnxArcFaceEmbedder(
            model_path=model_path,
            use_gpu=use_gpu,
            deprecated=deprecated
        )
    
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Supported backends: 'insightface', 'onnx'"
        )


# Backward compatibility: Keep FaceEmbeddingExtractor as an alias
# This ensures existing code continues to work
class FaceEmbeddingExtractor:
    """
    Face embedding extractor (backward compatibility wrapper).
    
    This class maintains backward compatibility with the old API.
    Internally, it uses the new pluggable backend system.
    
    DEPRECATED: Use create_embedding_backend() directly for new code.
    """
    
    def __init__(self, model_path: str, use_gpu: bool = False) -> None:
        """
        Initialize the face embedding extractor.
        
        Args:
            model_path: Path to the ArcFace ONNX model file (deprecated)
            use_gpu: Whether to use GPU for inference
            
        Note:
            This constructor defaults to InsightFace backend for better reliability.
            To use ONNX backend explicitly, use create_embedding_backend() instead.
        """
        logger.warning(
            "FaceEmbeddingExtractor is deprecated. "
            "Use create_embedding_backend() instead. "
            "Defaulting to InsightFace backend for better reliability."
        )
        
        # Default to InsightFace backend (more reliable)
        # If user explicitly wants ONNX, they should use create_embedding_backend()
        self._backend = create_embedding_backend(
            backend_type="insightface",
            use_gpu=use_gpu
        )
        
        # Store original parameters for compatibility
        self.model_path = model_path
        self.use_gpu = use_gpu
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract embedding (delegates to backend)."""
        return self._backend.extract_embedding(face_image)
    
    def extract_batch_embeddings(self, face_images: List[np.ndarray]) -> np.ndarray:
        """Extract batch embeddings (delegates to backend)."""
        return self._backend.extract_batch_embeddings(face_images)
    
    @staticmethod
    def normalize(embedding: np.ndarray) -> np.ndarray:
        """L2 normalize (delegates to base class)."""
        return BaseFaceEmbedder.normalize(embedding)
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity (delegates to base class)."""
        return BaseFaceEmbedder.cosine_similarity(emb1, emb2)
    
    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate Euclidean distance (delegates to base class)."""
        return BaseFaceEmbedder.euclidean_distance(emb1, emb2)
    
    def compare_faces(
        self, 
        face1: np.ndarray, 
        face2: np.ndarray,
        threshold: float = 0.6
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Compare faces (delegates to backend)."""
        return self._backend.compare_faces(face1, face2, threshold)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model info (delegates to backend)."""
        info = self._backend.get_backend_info()
        info['model_path'] = self.model_path
        info['use_gpu'] = self.use_gpu
        return info
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._backend.embedding_dim


# Export the factory function and base class
__all__ = [
    'BaseFaceEmbedder',
    'InsightFaceEmbedder',
    'OnnxArcFaceEmbedder',
    'FaceEmbeddingExtractor',  # For backward compatibility
    'create_embedding_backend'
]
