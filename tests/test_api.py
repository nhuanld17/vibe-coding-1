"""
Tests for API endpoints.
"""

import pytest
import json
import io
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint."""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Missing Person AI API"
        assert "endpoints" in data
        assert "documentation_url" in data
    
    def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "version" in data
    
    def test_upload_missing_person_success(self, api_client, sample_image_bytes, missing_person_metadata):
        """Test successful missing person upload."""
        # Prepare form data
        files = {"image": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        data = {"metadata": json.dumps(missing_person_metadata)}
        
        response = api_client.post("/api/v1/upload/missing", files=files, data=data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "point_id" in result
        assert "face_quality" in result
        assert "processing_time_ms" in result
    
    def test_upload_missing_person_invalid_metadata(self, api_client, sample_image_bytes):
        """Test missing person upload with invalid metadata."""
        # Invalid metadata (missing required fields)
        invalid_metadata = {"name": "John Doe"}
        
        files = {"image": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        data = {"metadata": json.dumps(invalid_metadata)}
        
        response = api_client.post("/api/v1/upload/missing", files=files, data=data)
        
        assert response.status_code == 400
        result = response.json()
        assert result["success"] is False
    
    def test_upload_missing_person_invalid_json(self, api_client, sample_image_bytes):
        """Test missing person upload with invalid JSON."""
        files = {"image": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        data = {"metadata": "invalid json"}
        
        response = api_client.post("/api/v1/upload/missing", files=files, data=data)
        
        assert response.status_code == 400
        result = response.json()
        assert "Invalid JSON" in result["message"]
    
    def test_upload_found_person_success(self, api_client, sample_image_bytes, found_person_metadata):
        """Test successful found person upload."""
        files = {"image": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        data = {"metadata": json.dumps(found_person_metadata)}
        
        response = api_client.post("/api/v1/upload/found", files=files, data=data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "point_id" in result
        assert "face_quality" in result
    
    def test_upload_found_person_invalid_metadata(self, api_client, sample_image_bytes):
        """Test found person upload with invalid metadata."""
        invalid_metadata = {"found_id": "FOUND_2023_001"}  # Missing required fields
        
        files = {"image": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        data = {"metadata": json.dumps(invalid_metadata)}
        
        response = api_client.post("/api/v1/upload/found", files=files, data=data)
        
        assert response.status_code == 400
        result = response.json()
        assert result["success"] is False
    
    def test_upload_invalid_file_type(self, api_client, missing_person_metadata):
        """Test upload with invalid file type."""
        # Create a text file instead of image
        text_content = b"This is not an image"
        files = {"image": ("test.txt", io.BytesIO(text_content), "text/plain")}
        data = {"metadata": json.dumps(missing_person_metadata)}
        
        response = api_client.post("/api/v1/upload/missing", files=files, data=data)
        
        assert response.status_code == 400
        result = response.json()
        assert "Invalid file" in result["message"]
    
    def test_search_missing_person_not_found(self, api_client, mock_services):
        """Test search for missing person that doesn't exist."""
        # Mock vector_db to return empty results
        mock_services['vector_db'].search_similar_faces.return_value = []
        
        response = api_client.get("/api/v1/search/missing/NONEXISTENT_CASE")
        
        assert response.status_code == 404
        result = response.json()
        assert "not found" in result["message"]
    
    def test_search_found_person_not_found(self, api_client, mock_services):
        """Test search for found person that doesn't exist."""
        # Mock vector_db to return empty results
        mock_services['vector_db'].search_similar_faces.return_value = []
        
        response = api_client.get("/api/v1/search/found/NONEXISTENT_ID")
        
        assert response.status_code == 404
        result = response.json()
        assert "not found" in result["message"]
    
    def test_upload_missing_person_no_face_detected(self, api_client, sample_image_bytes, missing_person_metadata, mock_services):
        """Test upload when no face is detected."""
        # Mock face detector to return None (no face found)
        mock_services['face_detector'].extract_largest_face.return_value = None
        
        files = {"image": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        data = {"metadata": json.dumps(missing_person_metadata)}
        
        response = api_client.post("/api/v1/upload/missing", files=files, data=data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is False
        assert "No face detected" in result["message"]
    
    def test_api_documentation_accessible(self, api_client):
        """Test that API documentation is accessible."""
        response = api_client.get("/docs")
        assert response.status_code == 200
        
        response = api_client.get("/redoc")
        assert response.status_code == 200
        
        response = api_client.get("/openapi.json")
        assert response.status_code == 200
        
        # Check that openapi.json contains expected structure
        openapi_data = response.json()
        assert "openapi" in openapi_data
        assert "info" in openapi_data
        assert "paths" in openapi_data


class TestAPIValidation:
    """Test cases for API input validation."""
    
    def test_missing_person_metadata_validation(self, api_client, sample_image_bytes):
        """Test missing person metadata validation."""
        # Test various invalid metadata scenarios
        invalid_cases = [
            # Missing case_id
            {
                "name": "John Doe",
                "age_at_disappearance": 25,
                "year_disappeared": 2020,
                "gender": "male",
                "location_last_seen": "New York, NY",
                "contact": "family@example.com"
            },
            # Invalid age
            {
                "case_id": "MISS_2023_001",
                "name": "John Doe",
                "age_at_disappearance": -5,
                "year_disappeared": 2020,
                "gender": "male",
                "location_last_seen": "New York, NY",
                "contact": "family@example.com"
            },
            # Invalid year
            {
                "case_id": "MISS_2023_001",
                "name": "John Doe",
                "age_at_disappearance": 25,
                "year_disappeared": 2050,
                "gender": "male",
                "location_last_seen": "New York, NY",
                "contact": "family@example.com"
            },
            # Invalid gender
            {
                "case_id": "MISS_2023_001",
                "name": "John Doe",
                "age_at_disappearance": 25,
                "year_disappeared": 2020,
                "gender": "invalid_gender",
                "location_last_seen": "New York, NY",
                "contact": "family@example.com"
            }
        ]
        
        for invalid_metadata in invalid_cases:
            files = {"image": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
            data = {"metadata": json.dumps(invalid_metadata)}
            
            response = api_client.post("/api/v1/upload/missing", files=files, data=data)
            assert response.status_code == 400
    
    def test_found_person_metadata_validation(self, api_client, sample_image_bytes):
        """Test found person metadata validation."""
        # Test various invalid metadata scenarios
        invalid_cases = [
            # Missing found_id
            {
                "current_age_estimate": 30,
                "gender": "male",
                "current_location": "Los Angeles, CA",
                "finder_contact": "finder@example.com"
            },
            # Invalid age
            {
                "found_id": "FOUND_2023_001",
                "current_age_estimate": 150,
                "gender": "male",
                "current_location": "Los Angeles, CA",
                "finder_contact": "finder@example.com"
            },
            # Invalid contact
            {
                "found_id": "FOUND_2023_001",
                "current_age_estimate": 30,
                "gender": "male",
                "current_location": "Los Angeles, CA",
                "finder_contact": "invalid_contact"
            }
        ]
        
        for invalid_metadata in invalid_cases:
            files = {"image": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
            data = {"metadata": json.dumps(invalid_metadata)}
            
            response = api_client.post("/api/v1/upload/found", files=files, data=data)
            assert response.status_code == 400
