"""
Face Embedding module using InsightFace ArcFace for Missing Person AI system.

This module provides face embedding extraction using the ArcFace model
with ONNX runtime for efficient inference.
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Optional, Union, Tuple, Dict, Any
from pathlib import Path
import os
from loguru import logger


class FaceEmbeddingExtractor:
    """
    Face embedding extractor using InsightFace ArcFace model.
    
    This class provides methods for:
    - Loading pre-trained ArcFace ONNX model
    - Preprocessing face images for embedding extraction
    - Extracting 512-dimensional L2-normalized embeddings
    - Batch processing for multiple faces
    - Computing similarity metrics between embeddings
    """
    
    def __init__(self, model_path: str, use_gpu: bool = False) -> None:
        """
        Initialize the face embedding extractor.
        
        Args:
            model_path: Path to the ArcFace ONNX model file
            use_gpu: Whether to use GPU for inference
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.input_size = (112, 112)  # Standard ArcFace input size
        self.embedding_dim = 512
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ArcFace model not found at {model_path}. "
                f"Please download arcface_r100_v1.onnx from "
                f"https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100_v1.onnx"
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
            
            logger.info(f"ArcFace model loaded from {model_path}")
            logger.info(f"Using providers: {self.session.get_providers()}")
            logger.info(f"Input shape: {self.session.get_inputs()[0].shape}")
            logger.info(f"Output shape: {self.session.get_outputs()[0].shape}")
            
        except Exception as e:
            logger.error(f"Failed to load ArcFace model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for ArcFace model.
        
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
            
        Raises:
            ValueError: If input image is invalid
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Input face image is None or empty")
        
        try:
            # Resize to model input size
            if face_image.shape[:2] != self.input_size:
                face_resized = cv2.resize(
                    face_image, 
                    self.input_size, 
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
        Extract 512-dimensional embedding from a face image.
        
        Args:
            face_image: Face image as numpy array (BGR format)
            
        Returns:
            L2-normalized embedding vector of shape (512,)
            
        Raises:
            ValueError: If input image is invalid
            RuntimeError: If embedding extraction fails
            
        Example:
            >>> extractor = FaceEmbeddingExtractor("arcface_r100_v1.onnx")
            >>> embedding = extractor.extract_embedding(face_image)
            >>> print(f"Embedding shape: {embedding.shape}")
            >>> print(f"Embedding norm: {np.linalg.norm(embedding)}")
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
    
    def extract_batch_embeddings(self, face_images: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from multiple face images in batch.
        
        Args:
            face_images: List of face images as numpy arrays (BGR format)
            
        Returns:
            Array of L2-normalized embeddings with shape (N, 512)
            
        Raises:
            ValueError: If input list is empty or contains invalid images
            RuntimeError: If batch extraction fails
            
        Example:
            >>> extractor = FaceEmbeddingExtractor("arcface_r100_v1.onnx")
            >>> embeddings = extractor.extract_batch_embeddings([face1, face2, face3])
            >>> print(f"Batch embeddings shape: {embeddings.shape}")
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
    
    @staticmethod
    def normalize(embedding: np.ndarray) -> np.ndarray:
        """
        L2 normalize an embedding vector.
        
        Args:
            embedding: Embedding vector
            
        Returns:
            L2-normalized embedding vector
            
        Example:
            >>> normalized = FaceEmbeddingExtractor.normalize(embedding)
            >>> print(f"Norm: {np.linalg.norm(normalized)}")  # Should be ~1.0
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
            
        Example:
            >>> similarity = FaceEmbeddingExtractor.cosine_similarity(emb1, emb2)
            >>> print(f"Cosine similarity: {similarity}")
        """
        # Ensure embeddings are normalized
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        # Calculate cosine similarity
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
            
        Example:
            >>> distance = FaceEmbeddingExtractor.euclidean_distance(emb1, emb2)
            >>> print(f"Euclidean distance: {distance}")
        """
        distance = np.linalg.norm(emb1 - emb2)
        return float(distance)
    
    @staticmethod
    def similarity_to_distance(similarity: float) -> float:
        """
        Convert cosine similarity to distance metric.
        
        Args:
            similarity: Cosine similarity score (-1 to 1)
            
        Returns:
            Distance metric (0 to 2, lower is more similar)
        """
        return 1.0 - similarity
    
    @staticmethod
    def distance_to_similarity(distance: float) -> float:
        """
        Convert distance metric to cosine similarity.
        
        Args:
            distance: Distance metric (0 to 2)
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        return 1.0 - distance
    
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
            
        Example:
            >>> extractor = FaceEmbeddingExtractor("arcface_r100_v1.onnx")
            >>> is_match, score, metrics = extractor.compare_faces(face1, face2)
            >>> print(f"Match: {is_match}, Score: {score:.3f}")
        """
        try:
            # Extract embeddings
            emb1 = self.extract_embedding(face1)
            emb2 = self.extract_embedding(face2)
            
            # Calculate similarity metrics
            cosine_sim = self.cosine_similarity(emb1, emb2)
            euclidean_dist = self.euclidean_distance(emb1, emb2)
            
            # Determine if faces match
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
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_path': self.model_path,
            'use_gpu': self.use_gpu,
            'providers': self.session.get_providers(),
            'input_shape': self.session.get_inputs()[0].shape,
            'output_shape': self.session.get_outputs()[0].shape,
            'input_name': self.input_name,
            'output_name': self.output_name,
            'embedding_dim': self.embedding_dim
        }


def download_arcface_model(model_dir: str = "./models/weights") -> str:
    """
    Download ArcFace model if it doesn't exist.
    
    Args:
        model_dir: Directory to save the model
        
    Returns:
        Path to the downloaded model file
        
    Note:
        This function provides instructions for manual download.
        Automatic download would require additional dependencies.
    """
    model_path = os.path.join(model_dir, "arcface_r100_v1.onnx")
    
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        
        download_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100_v1.onnx"
        
        logger.warning(f"ArcFace model not found at {model_path}")
        logger.info(f"Please download the model manually:")
        logger.info(f"wget {download_url} -O {model_path}")
        logger.info("Or use curl:")
        logger.info(f"curl -L {download_url} -o {model_path}")
        
        raise FileNotFoundError(
            f"Please download ArcFace model to {model_path} from {download_url}"
        )
    
    return model_path


# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize extractor (model must be downloaded first)
        model_path = "./models/weights/arcface_r100_v1.onnx"
        
        if not os.path.exists(model_path):
            print("Model not found. Please download it first:")
            download_arcface_model()
        else:
            extractor = FaceEmbeddingExtractor(model_path, use_gpu=False)
            
            # Print model info
            model_info = extractor.get_model_info()
            print("Model Info:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
            
            # Create dummy face images for testing
            dummy_face1 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            dummy_face2 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            
            # Extract embeddings
            embedding1 = extractor.extract_embedding(dummy_face1)
            embedding2 = extractor.extract_embedding(dummy_face2)
            
            print(f"Embedding 1 shape: {embedding1.shape}")
            print(f"Embedding 1 norm: {np.linalg.norm(embedding1):.3f}")
            print(f"Embedding 2 shape: {embedding2.shape}")
            print(f"Embedding 2 norm: {np.linalg.norm(embedding2):.3f}")
            
            # Calculate similarity
            similarity = extractor.cosine_similarity(embedding1, embedding2)
            distance = extractor.euclidean_distance(embedding1, embedding2)
            
            print(f"Cosine similarity: {similarity:.3f}")
            print(f"Euclidean distance: {distance:.3f}")
            
            # Compare faces
            is_match, score, metrics = extractor.compare_faces(
                dummy_face1, dummy_face2, threshold=0.6
            )
            print(f"Face match: {is_match}, Score: {score:.3f}")
            print(f"Metrics: {metrics}")
            
    except Exception as e:
        print(f"Error in example: {str(e)}")
