"""
ONNX-based face embedding extractor (deprecated - kept for compatibility).

This module provides face embedding extraction using ONNX models.
NOTE: The model from ONNX Model Zoo (arcfaceresnet100-8.onnx) is known to be broken.
This backend is kept for compatibility but should NOT be used in production.
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import Optional
from pathlib import Path
import os
from loguru import logger

from .face_embedding_base import BaseFaceEmbedder


class OnnxArcFaceEmbedder(BaseFaceEmbedder):
    """
    Face embedding extractor using ONNX ArcFace model.
    
    WARNING: This backend uses ONNX models which may not work correctly.
    The model from ONNX Model Zoo (arcfaceresnet100-8.onnx) is known to produce
    embeddings that are too similar for different faces.
    
    This backend is kept for compatibility but is DEPRECATED.
    Use InsightFaceEmbedder instead.
    """
    
    def __init__(
        self, 
        model_path: str, 
        use_gpu: bool = False,
        deprecated: bool = True
    ) -> None:
        """
        Initialize the ONNX embedder.
        
        Args:
            model_path: Path to the ArcFace ONNX model file
            use_gpu: Whether to use GPU for inference
            deprecated: Whether to show deprecation warning (default: True)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if deprecated:
            logger.warning(
                "OnnxArcFaceEmbedder is DEPRECATED. "
                "The ONNX Model Zoo model is known to be broken. "
                "Please use InsightFaceEmbedder instead."
            )
        
        self.model_path = model_path
        self.use_gpu = use_gpu
        self._embedding_dim = 512
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ArcFace model not found at {model_path}. "
                f"Please download the model first."
            )
        
        try:
            # Configure ONNX Runtime providers
            providers = ['CPUExecutionProvider']
            if use_gpu and ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            # Load ONNX model
            self.session = ort.InferenceSession(
                model_path,
                providers=providers
            )
            
            # Get model input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"ONNX ArcFace model loaded from {model_path}")
            logger.warning("NOTE: This ONNX model may produce incorrect embeddings. Use InsightFaceEmbedder for production.")
            logger.info(f"Using providers: {self.session.get_providers()}")
            logger.info(f"Input shape: {self.session.get_inputs()[0].shape}")
            logger.info(f"Output shape: {self.session.get_outputs()[0].shape}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (512 for ArcFace)."""
        return self._embedding_dim
    
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for ArcFace ONNX model.
        
        Preprocessing steps:
        1. Resize to 112x112
        2. Convert BGR to RGB
        3. Normalize to [-1, 1]: (pixel - 127.5) / 128.0
        4. Transpose to CHW format (channels first)
        5. Add batch dimension
        
        Args:
            face_image: Face image as numpy array (BGR format)
            
        Returns:
            Preprocessed image tensor ready for inference
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Input face image is None or empty")
        
        try:
            # Resize to model input size
            if face_image.shape[:2] != (112, 112):
                face_resized = cv2.resize(
                    face_image, 
                    (112, 112), 
                    interpolation=cv2.INTER_CUBIC
                )
            else:
                face_resized = face_image.copy()
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to float32 and normalize to [-1, 1]
            face_normalized = face_rgb.astype(np.float32)
            face_normalized = (face_normalized - 127.5) / 128.0
            
            # Transpose from HWC to CHW (channels first)
            face_transposed = np.transpose(face_normalized, (2, 0, 1))
            
            # Add batch dimension
            face_batch = np.expand_dims(face_transposed, axis=0)
            
            return face_batch
            
        except Exception as e:
            logger.error(f"Face preprocessing failed: {str(e)}")
            raise ValueError(f"Face preprocessing failed: {str(e)}")
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from an aligned face image.
        
        Args:
            face_image: Face image as numpy array (BGR format, 112x112)
            
        Returns:
            L2-normalized embedding vector of shape (512,)
            
        Raises:
            ValueError: If input image is invalid
            RuntimeError: If embedding extraction fails
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess(face_image)
            
            # Run inference
            outputs = self.session.run(
                [self.output_name], 
                {self.input_name: input_tensor}
            )
            
            # Extract embedding from output
            embedding = outputs[0][0]  # Remove batch dimension
            
            # L2 normalize
            embedding_normalized = self.normalize(embedding)
            
            logger.debug(f"Extracted embedding with shape {embedding_normalized.shape}")
            return embedding_normalized
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            raise RuntimeError(f"Embedding extraction failed: {str(e)}")
    
    def extract_batch_embeddings(self, face_images: list) -> np.ndarray:
        """
        Extract embeddings from multiple face images in batch.
        
        Args:
            face_images: List of face images as numpy arrays (BGR format)
            
        Returns:
            Array of L2-normalized embeddings with shape (N, 512)
        """
        if not face_images:
            raise ValueError("Input face_images list is empty")
        
        try:
            # Preprocess all images
            batch_tensors = []
            for i, face_image in enumerate(face_images):
                try:
                    tensor = self.preprocess(face_image)
                    batch_tensors.append(tensor[0])  # Remove batch dimension
                except Exception as e:
                    logger.warning(f"Failed to preprocess face {i}: {str(e)}")
                    continue
            
            if not batch_tensors:
                raise ValueError("No valid face images after preprocessing")
            
            # Stack into batch
            batch_input = np.stack(batch_tensors, axis=0)
            
            # Run batch inference
            outputs = self.session.run(
                [self.output_name], 
                {self.input_name: batch_input}
            )
            
            # Extract and normalize embeddings
            embeddings = outputs[0]
            embeddings_normalized = np.array([
                self.normalize(embedding) for embedding in embeddings
            ])
            
            logger.debug(f"Extracted {len(embeddings_normalized)} embeddings in batch")
            return embeddings_normalized
            
        except Exception as e:
            logger.error(f"Batch embedding extraction failed: {str(e)}")
            raise RuntimeError(f"Batch embedding extraction failed: {str(e)}")
    
    def get_backend_info(self) -> dict:
        """Get information about the ONNX backend."""
        info = super().get_backend_info()
        info.update({
            'model_path': self.model_path,
            'use_gpu': self.use_gpu,
            'backend': 'onnx',
            'deprecated': True,
            'warning': 'This ONNX model may produce incorrect embeddings'
        })
        return info

