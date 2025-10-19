"""
Face Detection module using MTCNN for Missing Person AI system.

This module provides face detection, alignment, and quality assessment
using the MTCNN (Multi-task CNN) algorithm.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from mtcnn import MTCNN
from loguru import logger


class FaceDetector:
    """
    Face detector using MTCNN for detecting and aligning faces in images.
    
    This class provides methods for:
    - Detecting faces with confidence scores
    - Extracting facial landmarks (5 points: eyes, nose, mouth)
    - Aligning faces using eye positions
    - Assessing face quality (blur, brightness, contrast)
    """
    
    def __init__(self, min_face_size: int = 40, device: str = "CPU:0") -> None:
        """
        Initialize the face detector.
        
        Args:
            min_face_size: Minimum face size in pixels for detection
            device: Device to use for inference (CPU:0 or GPU:0)
        """
        self.min_face_size = min_face_size
        self.device = device
        
        try:
            self.detector = MTCNN()
            logger.info(f"MTCNN face detector initialized with min_face_size={min_face_size}")
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN detector: {str(e)}")
            raise RuntimeError(f"Face detector initialization failed: {str(e)}")
    
    def detect_faces(
        self, 
        image: np.ndarray, 
        confidence_threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Detect faces in an image with confidence scores and landmarks.
        
        Args:
            image: Input image as numpy array (BGR format)
            confidence_threshold: Minimum confidence score for face detection
            
        Returns:
            List of detected faces with bounding boxes, confidence, and landmarks
            
        Raises:
            ValueError: If image is invalid or empty
            RuntimeError: If detection fails
            
        Example:
            >>> detector = FaceDetector()
            >>> faces = detector.detect_faces(image, confidence_threshold=0.9)
            >>> print(f"Found {len(faces)} faces")
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel BGR image")
        
        try:
            # Convert BGR to RGB for MTCNN
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = self.detector.detect_faces(rgb_image)
            
            # Filter by confidence threshold
            valid_faces = []
            for detection in detections:
                confidence = detection['confidence']
                if confidence >= confidence_threshold:
                    # Extract bounding box
                    bbox = detection['box']
                    x, y, w, h = bbox
                    
                    # Extract keypoints (landmarks)
                    keypoints = detection['keypoints']
                    landmarks = np.array([
                        [keypoints['left_eye'][0], keypoints['left_eye'][1]],
                        [keypoints['right_eye'][0], keypoints['right_eye'][1]],
                        [keypoints['nose'][0], keypoints['nose'][1]],
                        [keypoints['mouth_left'][0], keypoints['mouth_left'][1]],
                        [keypoints['mouth_right'][0], keypoints['mouth_right'][1]]
                    ], dtype=np.float32)
                    
                    face_info = {
                        'bbox': [x, y, x + w, y + h],  # [x1, y1, x2, y2] format
                        'confidence': confidence,
                        'keypoints': landmarks,
                        'area': w * h
                    }
                    valid_faces.append(face_info)
            
            # Sort by confidence (highest first)
            valid_faces.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.debug(f"Detected {len(valid_faces)} faces with confidence >= {confidence_threshold}")
            return valid_faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            raise RuntimeError(f"Face detection failed: {str(e)}")
    
    def align_face(
        self, 
        image: np.ndarray, 
        landmarks: np.ndarray, 
        output_size: Tuple[int, int] = (112, 112)
    ) -> np.ndarray:
        """
        Align face using eye positions and resize to standard size.
        
        Args:
            image: Input image as numpy array (BGR format)
            landmarks: Facial landmarks array of shape (5, 2)
            output_size: Output image size (width, height)
            
        Returns:
            Aligned face image as numpy array
            
        Raises:
            ValueError: If landmarks are invalid
            RuntimeError: If alignment fails
        """
        if landmarks.shape != (5, 2):
            raise ValueError("Landmarks must be array of shape (5, 2)")
        
        try:
            # Get eye positions (first two landmarks)
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            # Calculate angle between eyes
            eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Calculate scale to fit desired eye distance
            eye_distance = np.sqrt((dx ** 2) + (dy ** 2))
            desired_eye_distance = output_size[0] * 0.35  # 35% of face width
            scale = desired_eye_distance / eye_distance
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(eye_center, angle, scale)
            
            # Adjust translation to center the face
            tx = output_size[0] * 0.5
            ty = output_size[1] * 0.35  # Eyes at 35% from top
            M[0, 2] += (tx - eye_center[0])
            M[1, 2] += (ty - eye_center[1])
            
            # Apply transformation
            aligned_face = cv2.warpAffine(
                image, M, output_size, 
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT
            )
            
            return aligned_face
            
        except Exception as e:
            logger.error(f"Face alignment failed: {str(e)}")
            raise RuntimeError(f"Face alignment failed: {str(e)}")
    
    def extract_largest_face(
        self, 
        image: np.ndarray, 
        align: bool = True,
        confidence_threshold: float = 0.9
    ) -> Optional[np.ndarray]:
        """
        Extract the largest face from an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            align: Whether to align the face
            confidence_threshold: Minimum confidence for face detection
            
        Returns:
            Extracted face image or None if no face found
            
        Raises:
            ValueError: If image is invalid
            RuntimeError: If extraction fails
            
        Example:
            >>> detector = FaceDetector()
            >>> face = detector.extract_largest_face(image, align=True)
            >>> if face is not None:
            ...     print(f"Extracted face shape: {face.shape}")
        """
        try:
            faces = self.detect_faces(image, confidence_threshold)
            
            if not faces:
                logger.warning("No faces detected in image")
                return None
            
            # Get the largest face (first one after sorting by confidence)
            largest_face = faces[0]
            
            if align and 'keypoints' in largest_face:
                # Extract and align face
                landmarks = largest_face['keypoints']
                aligned_face = self.align_face(image, landmarks)
                return aligned_face
            else:
                # Extract face using bounding box
                bbox = largest_face['bbox']
                x1, y1, x2, y2 = bbox
                
                # Add padding
                padding = 0.2
                h, w = image.shape[:2]
                pad_x = int((x2 - x1) * padding)
                pad_y = int((y2 - y1) * padding)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                face_crop = image[y1:y2, x1:x2]
                
                # Resize to standard size
                face_resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_CUBIC)
                return face_resized
                
        except Exception as e:
            logger.error(f"Face extraction failed: {str(e)}")
            raise RuntimeError(f"Face extraction failed: {str(e)}")
    
    def check_face_quality(
        self, 
        face_image: np.ndarray, 
        blur_threshold: float = 100.0
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check the quality of a face image.
        
        Args:
            face_image: Face image as numpy array
            blur_threshold: Minimum sharpness threshold (Laplacian variance)
            
        Returns:
            Tuple of (is_good_quality, quality_metrics)
            
        Quality Metrics:
            - Sharpness: Laplacian variance >= 100
            - Brightness: 40 <= mean_brightness <= 220
            - Contrast: std_deviation >= 30
            
        Example:
            >>> detector = FaceDetector()
            >>> is_good, metrics = detector.check_face_quality(face_image)
            >>> print(f"Quality: {is_good}, Metrics: {metrics}")
        """
        try:
            # Convert to grayscale for quality checks
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Calculate sharpness using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Calculate brightness (mean pixel value)
            brightness = gray.mean()
            
            # Calculate contrast (standard deviation)
            contrast = gray.std()
            
            # Quality thresholds
            is_sharp = sharpness >= blur_threshold
            is_bright_enough = 40 <= brightness <= 220
            is_contrasted = contrast >= 30
            
            # Overall quality assessment
            is_good_quality = is_sharp and is_bright_enough and is_contrasted
            
            quality_metrics = {
                'sharpness': float(sharpness),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'is_sharp': is_sharp,
                'is_bright_enough': is_bright_enough,
                'is_contrasted': is_contrasted
            }
            
            logger.debug(f"Face quality: {is_good_quality}, metrics: {quality_metrics}")
            return is_good_quality, quality_metrics
            
        except Exception as e:
            logger.error(f"Face quality check failed: {str(e)}")
            raise RuntimeError(f"Face quality check failed: {str(e)}")


def load_image_from_path(image_path: str) -> np.ndarray:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array in BGR format
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    except Exception as e:
        logger.error(f"Failed to load image from {image_path}: {str(e)}")
        raise


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes.
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        Image as numpy array in BGR format
        
    Raises:
        ValueError: If image cannot be decoded
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from bytes")
        return image
    except Exception as e:
        logger.error(f"Failed to load image from bytes: {str(e)}")
        raise ValueError(f"Failed to load image from bytes: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = FaceDetector(min_face_size=40)
    
    # Example with dummy image (replace with actual image path for testing)
    try:
        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Detect faces
        faces = detector.detect_faces(dummy_image, confidence_threshold=0.9)
        print(f"Detected {len(faces)} faces")
        
        # Extract largest face
        largest_face = detector.extract_largest_face(dummy_image, align=True)
        if largest_face is not None:
            print(f"Extracted face shape: {largest_face.shape}")
            
            # Check quality
            is_good, metrics = detector.check_face_quality(largest_face)
            print(f"Face quality: {is_good}")
            print(f"Quality metrics: {metrics}")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
