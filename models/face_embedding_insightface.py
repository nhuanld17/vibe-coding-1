"""
InsightFace-based face embedding extractor.

This module provides face embedding extraction using the InsightFace library,
which automatically handles model downloading and preprocessing.
"""

import cv2
import numpy as np
from typing import Optional
from pathlib import Path
from loguru import logger

from .face_embedding_base import BaseFaceEmbedder


class InsightFaceEmbedder(BaseFaceEmbedder):
    """
    Face embedding extractor using InsightFace library.
    
    This backend uses the InsightFace Python library which:
    - Automatically downloads the correct ArcFace model
    - Handles preprocessing internally
    - Provides reliable 512-D embeddings
    
    The InsightFace library uses the 'antelopev2' model by default,
    which includes ArcFace ResNet100 for recognition.
    """
    
    def __init__(
        self, 
        model_name: str = "antelopev2",
        use_gpu: bool = False,
        det_size: tuple = (640, 640)
    ) -> None:
        """
        Initialize the InsightFace embedder.
        
        Args:
            model_name: InsightFace model name (default: "antelopev2")
            use_gpu: Whether to use GPU for inference
            det_size: Detection size for face detection (not used for embedding-only mode)
            
        Raises:
            ImportError: If insightface library is not installed
            RuntimeError: If model initialization fails
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface library is not installed. "
                "Please install it with: pip install insightface"
            )
        
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._embedding_dim = 512  # ArcFace produces 512-D embeddings
        
        try:
            # Initialize FaceAnalysis
            # Note: We use FaceAnalysis but will only use the recognition model
            # Our pipeline already handles detection and alignment
            ctx_id = 0 if use_gpu else -1  # -1 for CPU, 0+ for GPU
            
            logger.info(f"Initializing InsightFace model: {model_name}")
            
            # For insightface 0.7.3, FaceAnalysis requires models to be loaded before __init__
            # We need to handle the AssertionError that occurs if models aren't loaded
            try:
                # Initialize FaceAnalysis - version 0.7.3
                # The models should be auto-loaded when FaceAnalysis is instantiated
                self.app = FaceAnalysis(name=model_name)
            except AssertionError as ae:
                # This happens if the models dictionary doesn't have 'detection' key
                # This can occur if models weren't properly downloaded or extracted
                error_msg = str(ae)
                logger.error(f"FaceAnalysis AssertionError: {error_msg}")
                logger.info("This usually means models weren't properly loaded.")
                logger.info("Model files should be in ~/.insightface/models/antelopev2/")
                logger.info("Please ensure models are downloaded (they should auto-download).")
                raise RuntimeError(
                    f"FaceAnalysis initialization failed: {error_msg}. "
                    "Models may not be properly downloaded. Try deleting ~/.insightface/models/antelopev2/ "
                    "and running again to re-download."
                )
            
            # Prepare the model
            # We set det_size but won't use detection since we already have aligned faces
            try:
                self.app.prepare(ctx_id=ctx_id, det_size=det_size)
            except TypeError:
                # Some versions might not support det_size parameter
                try:
                    self.app.prepare(ctx_id=ctx_id)
                except Exception as e2:
                    logger.warning(f"prepare() failed: {e2}, trying with no parameters")
                    self.app.prepare()
            
            # Get the recognition model (ArcFace)
            # InsightFace stores models in app.models dictionary
            if hasattr(self.app, 'models') and 'recognition' in self.app.models:
                self.recognition_model = self.app.models['recognition']
            else:
                # Fallback: try to get from app directly
                self.recognition_model = self.app
            
            logger.info(f"InsightFace model initialized successfully")
            logger.info(f"Using device: {'GPU' if use_gpu else 'CPU'}")
            
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace model: {str(e)}")
            raise RuntimeError(f"InsightFace model initialization failed: {str(e)}")
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (512 for ArcFace)."""
        return self._embedding_dim
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from an aligned face image.
        
        Args:
            face_image: Aligned face image as numpy array (BGR format, 112x112)
            
        Returns:
            L2-normalized embedding vector of shape (512,)
            
        Raises:
            ValueError: If input image is invalid
            RuntimeError: If embedding extraction fails
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Input face image is None or empty")
        
        try:
            # InsightFace recognition model expects RGB format
            # Our pipeline provides BGR, so convert
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Ensure image is 112x112 (InsightFace requirement)
            if face_rgb.shape[:2] != (112, 112):
                face_rgb = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_CUBIC)
            
            # Get recognition model (ArcFaceONNX)
            if hasattr(self.app, 'models') and 'recognition' in self.app.models:
                rec_model = self.app.models['recognition']
            else:
                rec_model = self.recognition_model
            
            # Use recognition model's get_feat() method directly with aligned face
            # This is the recommended way for pre-aligned faces
            if hasattr(rec_model, 'get_feat'):
                # get_feat() expects RGB image(s) and handles preprocessing internally
                # Returns (1, 512) or (batch, 512) shape - embeddings are NOT normalized
                embedding = rec_model.get_feat(face_rgb)
                
                # Remove batch dimension if present
                if isinstance(embedding, np.ndarray):
                    if embedding.ndim == 2:
                        embedding = embedding[0]  # Remove batch dimension
                    elif embedding.ndim > 2:
                        embedding = embedding.flatten()
                elif isinstance(embedding, list):
                    embedding = np.array(embedding[0])
                
                # Ensure it's a numpy array
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                
                # L2 normalize (get_feat returns unnormalized embeddings)
                embedding = self.normalize(embedding)
                
            elif hasattr(rec_model, 'forward'):
                # Manual preprocessing and forward pass
                # Recognition model expects: (1, 3, 112, 112) input in RGB format
                # Preprocessing: (pixel - input_mean) / input_std
                # For antelopev2: input_mean=127.5, input_std=127.5
                face_normalized = face_rgb.astype(np.float32)
                input_mean = getattr(rec_model, 'input_mean', 127.5)
                input_std = getattr(rec_model, 'input_std', 127.5)
                face_normalized = (face_normalized - input_mean) / input_std
                face_batch = np.transpose(face_normalized, (2, 0, 1))
                face_batch = np.expand_dims(face_batch, axis=0)
                embedding = rec_model.forward(face_batch)
                if embedding.ndim > 1:
                    embedding = embedding[0]  # Remove batch dimension
                embedding = self.normalize(embedding)
            else:
                # Fallback: use app.get() with full image context
                # This is less ideal but works when direct model access fails
                logger.warning("Direct recognition model access failed, using app.get() fallback")
                # Create a larger image with the aligned face centered
                # InsightFace works better with some context around the face
                full_image = np.zeros((640, 640, 3), dtype=np.uint8)
                y_offset = (640 - 112) // 2
                x_offset = (640 - 112) // 2
                full_image[y_offset:y_offset+112, x_offset:x_offset+112] = face_rgb
                faces = self.app.get(full_image)
                if not faces or len(faces) == 0:
                    raise RuntimeError("InsightFace failed to extract embedding")
                embedding = faces[0].normed_embedding
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                embedding = self.normalize(embedding)
                logger.debug(f"Extracted embedding with shape {embedding.shape} (via app.get fallback)")
                return embedding
            
            # Verify dimension
            if embedding.shape[0] != self.embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                    f"got {embedding.shape[0]}. Resizing..."
                )
                if embedding.shape[0] > self.embedding_dim:
                    embedding = embedding[:self.embedding_dim]
                else:
                    padding = np.zeros(self.embedding_dim - embedding.shape[0])
                    embedding = np.concatenate([embedding, padding])
            
            logger.debug(f"Extracted embedding with shape {embedding.shape}")
            return embedding
                    
        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Embedding extraction failed: {str(e)}")
    
    def get_backend_info(self) -> dict:
        """Get information about the InsightFace backend."""
        info = super().get_backend_info()
        info.update({
            'model_name': self.model_name,
            'use_gpu': self.use_gpu,
            'backend': 'insightface'
        })
        return info

