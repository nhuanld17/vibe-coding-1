"""
Pytest configuration and fixtures for Missing Person AI tests.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, MagicMock
from fastapi.testclient import TestClient

# Mock the dependencies to avoid requiring actual services during testing
@pytest.fixture(autouse=True)
def mock_services():
    """Mock all external services for testing."""
    # Mock face detector
    mock_face_detector = Mock()
    mock_face_detector.detect_faces.return_value = [
        {
            'bbox': [100, 100, 200, 200],
            'confidence': 0.95,
            'keypoints': np.array([[120, 130], [180, 130], [150, 150], [130, 180], [170, 180]]),
            'area': 10000
        }
    ]
    mock_face_detector.extract_largest_face.return_value = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    mock_face_detector.check_face_quality.return_value = (True, {
        'sharpness': 150.0,
        'brightness': 128.0,
        'contrast': 45.0,
        'is_sharp': True,
        'is_bright_enough': True,
        'is_contrasted': True
    })
    
    # Mock embedding extractor
    mock_embedding_extractor = Mock()
    mock_embedding_extractor.extract_embedding.return_value = np.random.rand(512).astype(np.float32)
    mock_embedding_extractor.get_model_info.return_value = {
        'model_path': 'test_model.onnx',
        'embedding_dim': 512
    }
    
    # Mock vector database
    mock_vector_db = Mock()
    mock_vector_db.insert_missing_person.return_value = "test-point-id-123"
    mock_vector_db.insert_found_person.return_value = "test-point-id-456"
    mock_vector_db.search_similar_faces.return_value = []
    mock_vector_db.get_collection_stats.return_value = {
        'collection_name': 'test_collection',
        'points_count': 100,
        'vector_size': 512,
        'status': 'green'
    }
    mock_vector_db.health_check.return_value = {'status': 'healthy'}
    
    # Mock bilateral search
    mock_bilateral_search = Mock()
    mock_bilateral_search.search_for_missing.return_value = []
    mock_bilateral_search.search_for_found.return_value = []
    
    # Mock confidence scoring
    mock_confidence_scoring = Mock()
    mock_confidence_scoring.calculate_confidence.return_value = (
        Mock(value='HIGH'),
        0.85,
        {
            'factors': {
                'face_similarity': {
                    'score': 0.9,
                    'weight': 0.5,
                    'contribution': 0.45,
                    'description': 'High facial similarity'
                }
            },
            'reasons': ['Strong facial match'],
            'summary': 'High confidence match',
            'recommendations': ['Contact authorities'],
            'threshold_info': {'high': 0.75}
        }
    )
    
    return {
        'face_detector': mock_face_detector,
        'embedding_extractor': mock_embedding_extractor,
        'vector_db': mock_vector_db,
        'bilateral_search': mock_bilateral_search,
        'confidence_scoring': mock_confidence_scoring
    }


@pytest.fixture
def test_image():
    """Create a test image for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_face_image():
    """Create a test face image for testing."""
    return np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)


@pytest.fixture
def test_embedding():
    """Create a test embedding vector."""
    embedding = np.random.rand(512).astype(np.float32)
    # Normalize to unit length
    return embedding / np.linalg.norm(embedding)


@pytest.fixture
def missing_person_metadata():
    """Sample missing person metadata."""
    return {
        'case_id': 'MISS_2023_001',
        'name': 'John Doe',
        'age_at_disappearance': 25,
        'year_disappeared': 2020,
        'gender': 'male',
        'location_last_seen': 'New York, NY',
        'contact': 'family@example.com',
        'height_cm': 175,
        'birthmarks': ['scar on left arm']
    }


@pytest.fixture
def found_person_metadata():
    """Sample found person metadata."""
    return {
        'found_id': 'FOUND_2023_001',
        'current_age_estimate': 30,
        'gender': 'male',
        'current_location': 'Los Angeles, CA',
        'finder_contact': 'finder@example.com',
        'visible_marks': ['scar on left arm'],
        'current_condition': 'Good health'
    }


@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        # Write some dummy data
        f.write(b'dummy model data')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def api_client(mock_services):
    """Create a test client for the API."""
    # Import here to avoid circular imports
    from api.main import app
    from api.dependencies import (
        _face_detector, _embedding_extractor, _vector_db, 
        _bilateral_search, _confidence_scoring
    )
    
    # Override the global service instances with mocks
    import api.dependencies
    api.dependencies._face_detector = mock_services['face_detector']
    api.dependencies._embedding_extractor = mock_services['embedding_extractor']
    api.dependencies._vector_db = mock_services['vector_db']
    api.dependencies._bilateral_search = mock_services['bilateral_search']
    api.dependencies._confidence_scoring = mock_services['confidence_scoring']
    
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing file uploads."""
    # Create a simple test image
    import io
    from PIL import Image
    
    # Create a simple RGB image
    img = Image.new('RGB', (640, 480), color='red')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()


@pytest.fixture
def match_result_sample():
    """Sample match result for testing."""
    return {
        'id': 'test-match-id',
        'face_similarity': 0.85,
        'metadata_similarity': 0.75,
        'combined_score': 0.82,
        'payload': {
            'name': 'Test Person',
            'contact': 'test@example.com',
            'age_at_disappearance': 25
        },
        'match_details': {
            'gender_match': 1.0,
            'age_consistency': 0.9,
            'marks_similarity': 0.6,
            'location_plausibility': 0.4
        }
    }
