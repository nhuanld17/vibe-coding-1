"""
Tests for face detection module.
"""

import pytest
import numpy as np
import cv2
from models.face_detection import FaceDetector, load_image_from_bytes


class TestFaceDetector:
    """Test cases for FaceDetector class."""
    
    def test_face_detector_initialization(self):
        """Test face detector initialization."""
        detector = FaceDetector(min_face_size=40)
        assert detector.min_face_size == 40
        assert detector.device == "CPU:0"
        assert detector.detector is not None
    
    def test_detect_faces_with_valid_image(self, test_image):
        """Test face detection with valid image."""
        detector = FaceDetector()
        
        # Mock the MTCNN detector to return a face
        detector.detector.detect_faces = lambda x: [
            {
                'box': [100, 100, 100, 100],
                'confidence': 0.95,
                'keypoints': {
                    'left_eye': (120, 130),
                    'right_eye': (180, 130),
                    'nose': (150, 150),
                    'mouth_left': (130, 180),
                    'mouth_right': (170, 180)
                }
            }
        ]
        
        faces = detector.detect_faces(test_image, confidence_threshold=0.9)
        
        assert len(faces) == 1
        assert faces[0]['confidence'] == 0.95
        assert 'bbox' in faces[0]
        assert 'keypoints' in faces[0]
        assert faces[0]['keypoints'].shape == (5, 2)
    
    def test_detect_faces_with_invalid_image(self):
        """Test face detection with invalid image."""
        detector = FaceDetector()
        
        # Test with None image
        with pytest.raises(ValueError, match="Input image is None or empty"):
            detector.detect_faces(None)
        
        # Test with empty image
        with pytest.raises(ValueError, match="Input image is None or empty"):
            detector.detect_faces(np.array([]))
        
        # Test with wrong dimensions
        with pytest.raises(ValueError, match="Input image must be a 3-channel BGR image"):
            detector.detect_faces(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
    
    def test_align_face(self, test_image):
        """Test face alignment."""
        detector = FaceDetector()
        
        # Create sample landmarks
        landmarks = np.array([
            [120, 130],  # left eye
            [180, 130],  # right eye
            [150, 150],  # nose
            [130, 180],  # mouth left
            [170, 180]   # mouth right
        ], dtype=np.float32)
        
        aligned_face = detector.align_face(test_image, landmarks, output_size=(112, 112))
        
        assert aligned_face.shape == (112, 112, 3)
        assert aligned_face.dtype == np.uint8
    
    def test_align_face_with_invalid_landmarks(self, test_image):
        """Test face alignment with invalid landmarks."""
        detector = FaceDetector()
        
        # Test with wrong shape
        with pytest.raises(ValueError, match="Landmarks must be array of shape"):
            detector.align_face(test_image, np.array([[1, 2]]))
    
    def test_extract_largest_face(self, test_image):
        """Test extracting largest face."""
        detector = FaceDetector()
        
        # Mock detect_faces to return a face
        detector.detect_faces = lambda img, conf: [
            {
                'bbox': [100, 100, 200, 200],
                'confidence': 0.95,
                'keypoints': np.array([
                    [120, 130], [180, 130], [150, 150], [130, 180], [170, 180]
                ], dtype=np.float32),
                'area': 10000
            }
        ]
        
        face = detector.extract_largest_face(test_image, align=True)
        
        assert face is not None
        assert face.shape == (112, 112, 3)
    
    def test_extract_largest_face_no_face_found(self, test_image):
        """Test extracting largest face when no face is found."""
        detector = FaceDetector()
        
        # Mock detect_faces to return no faces
        detector.detect_faces = lambda img, conf: []
        
        face = detector.extract_largest_face(test_image)
        
        assert face is None
    
    def test_check_face_quality(self, test_face_image):
        """Test face quality assessment."""
        detector = FaceDetector()
        
        is_good, metrics = detector.check_face_quality(test_face_image)
        
        assert isinstance(is_good, bool)
        assert 'sharpness' in metrics
        assert 'brightness' in metrics
        assert 'contrast' in metrics
        assert 'is_sharp' in metrics
        assert 'is_bright_enough' in metrics
        assert 'is_contrasted' in metrics
        
        # Check metric ranges
        assert metrics['brightness'] >= 0
        assert metrics['contrast'] >= 0
        assert metrics['sharpness'] >= 0


class TestImageLoading:
    """Test cases for image loading functions."""
    
    def test_load_image_from_bytes(self, sample_image_bytes):
        """Test loading image from bytes."""
        image = load_image_from_bytes(sample_image_bytes)
        
        assert image is not None
        assert len(image.shape) == 3
        assert image.shape[2] == 3  # BGR channels
        assert image.dtype == np.uint8
    
    def test_load_image_from_invalid_bytes(self):
        """Test loading image from invalid bytes."""
        with pytest.raises(ValueError, match="Failed to load image from bytes"):
            load_image_from_bytes(b"invalid image data")
    
    def test_load_image_from_empty_bytes(self):
        """Test loading image from empty bytes."""
        with pytest.raises(ValueError, match="Failed to load image from bytes"):
            load_image_from_bytes(b"")
