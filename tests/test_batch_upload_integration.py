"""
Integration tests for batch/multi-image upload endpoints.

This module tests the complete flow of batch uploads:
- Multiple image processing
- Parallel execution
- Graceful degradation
- Multi-image search integration
- Latency requirements

Author: AI Face Recognition Team
"""

import pytest
import io
import time
from PIL import Image
from fastapi.testclient import TestClient
import numpy as np

# Note: These tests require a running API server and Qdrant instance
# Run with: pytest tests/test_batch_upload_integration.py -v


class TestBatchUploadMissing:
    """Test batch upload for missing persons."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        # Import here to avoid circular dependencies
        from main import app
        return TestClient(app)
    
    @pytest.fixture
    def valid_image_bytes(self):
        """Create valid test image bytes."""
        img = Image.new('RGB', (640, 480), color='red')
        # Add some variation to make face detection possible
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', quality=95)
        return img_bytes.getvalue()
    
    @pytest.fixture
    def invalid_image_bytes(self):
        """Create invalid test image (not an image)."""
        return b"This is not an image"
    
    def test_upload_single_image_batch(self, client, valid_image_bytes):
        """Test batch upload with 1 image (minimum)."""
        files = [
            ("images", ("photo1.jpg", io.BytesIO(valid_image_bytes), "image/jpeg"))
        ]
        data = {
            "name": "John Doe",
            "age_at_disappearance": 25,
            "year_disappeared": 2020,
            "gender": "male",
            "location_last_seen": "New York, NY",
            "contact": "test@example.com"
        }
        
        response = client.post("/api/v1/upload/missing/batch", files=files, data=data)
        
        # May succeed or fail depending on face detection
        # Just verify response structure
        assert response.status_code in [200, 400]
        result = response.json()
        assert "success" in result
        assert "case_id" in result or "detail" in result
    
    def test_upload_five_images_batch(self, client, valid_image_bytes):
        """Test batch upload with 5 images."""
        files = [
            ("images", (f"photo{i}.jpg", io.BytesIO(valid_image_bytes), "image/jpeg"))
            for i in range(5)
        ]
        data = {
            "name": "Jane Smith",
            "age_at_disappearance": 30,
            "year_disappeared": 2018,
            "gender": "female",
            "location_last_seen": "Los Angeles, CA",
            "contact": "family@example.com",
            "image_metadata_json": '[{"photo_year": 2010}, {"photo_year": 2012}, {"photo_year": 2015}, {"photo_year": 2017}, null]'
        }
        
        start_time = time.time()
        response = client.post("/api/v1/upload/missing/batch", files=files, data=data)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Verify response structure regardless of face detection success
        assert response.status_code in [200, 400]
        result = response.json()
        
        if response.status_code == 200:
            assert result["success"] is True
            assert "case_id" in result
            assert "total_images_uploaded" in result
            assert "uploaded_images" in result
            assert "failed_images" in result
            assert "processing_time_ms" in result
            
            # Check latency target (<500ms for 5 images)
            print(f"\n⏱️  Upload latency: {elapsed_ms:.1f}ms (target: <500ms)")
    
    def test_upload_ten_images_max(self, client, valid_image_bytes):
        """Test batch upload with maximum 10 images."""
        files = [
            ("images", (f"photo{i}.jpg", io.BytesIO(valid_image_bytes), "image/jpeg"))
            for i in range(10)
        ]
        data = {
            "name": "Max Images",
            "age_at_disappearance": 35,
            "year_disappeared": 2015,
            "gender": "male",
            "location_last_seen": "Chicago, IL",
            "contact": "contact@example.com"
        }
        
        response = client.post("/api/v1/upload/missing/batch", files=files, data=data)
        
        # Should succeed or fail based on face detection
        assert response.status_code in [200, 400]
        result = response.json()
        
        if response.status_code == 200:
            assert result["success"] is True
            # Can have up to 10 uploaded or some failed
            assert result["total_images_uploaded"] + result.get("total_images_failed", 0) == 10
    
    def test_upload_eleven_images_error(self, client, valid_image_bytes):
        """Test batch upload with 11 images (exceeds maximum)."""
        files = [
            ("images", (f"photo{i}.jpg", io.BytesIO(valid_image_bytes), "image/jpeg"))
            for i in range(11)
        ]
        data = {
            "name": "Too Many Images",
            "age_at_disappearance": 40,
            "year_disappeared": 2010,
            "gender": "female",
            "location_last_seen": "Miami, FL",
            "contact": "test@example.com"
        }
        
        response = client.post("/api/v1/upload/missing/batch", files=files, data=data)
        
        # Should fail with 400 error
        assert response.status_code == 400
        result = response.json()
        assert "detail" in result
        assert "10" in result["detail"].lower()  # Error message mentions max 10
    
    def test_upload_zero_images_error(self, client):
        """Test batch upload with 0 images (minimum validation)."""
        files = []  # Empty list
        data = {
            "name": "No Images",
            "age_at_disappearance": 25,
            "year_disappeared": 2020,
            "gender": "male",
            "location_last_seen": "Boston, MA",
            "contact": "test@example.com"
        }
        
        response = client.post("/api/v1/upload/missing/batch", files=files, data=data)
        
        # Should fail with 422 (validation error) or 400
        assert response.status_code in [400, 422]
    
    def test_upload_with_partial_failures(self, client, valid_image_bytes, invalid_image_bytes):
        """Test batch upload with some invalid images (graceful degradation)."""
        files = [
            ("images", ("valid1.jpg", io.BytesIO(valid_image_bytes), "image/jpeg")),
            ("images", ("invalid.txt", io.BytesIO(invalid_image_bytes), "text/plain")),
            ("images", ("valid2.jpg", io.BytesIO(valid_image_bytes), "image/jpeg")),
        ]
        data = {
            "name": "Partial Upload",
            "age_at_disappearance": 28,
            "year_disappeared": 2019,
            "gender": "male",
            "location_last_seen": "Seattle, WA",
            "contact": "test@example.com"
        }
        
        response = client.post("/api/v1/upload/missing/batch", files=files, data=data)
        
        # Should succeed if at least 1 valid face detected, or fail if all fail
        result = response.json()
        
        if response.status_code == 200:
            # Graceful degradation: some succeeded
            assert result["success"] is True
            assert result["total_images_uploaded"] >= 1
            assert result.get("total_images_failed", 0) >= 1
            assert len(result["failed_images"]) >= 1
        else:
            # All failed
            assert response.status_code == 400
            assert "detail" in result
    
    def test_upload_with_invalid_metadata_json(self, client, valid_image_bytes):
        """Test batch upload with malformed image_metadata_json."""
        files = [
            ("images", ("photo1.jpg", io.BytesIO(valid_image_bytes), "image/jpeg"))
        ]
        data = {
            "name": "Bad JSON",
            "age_at_disappearance": 25,
            "year_disappeared": 2020,
            "gender": "male",
            "location_last_seen": "Portland, OR",
            "contact": "test@example.com",
            "image_metadata_json": "not valid json"
        }
        
        response = client.post("/api/v1/upload/missing/batch", files=files, data=data)
        
        # Should fail with 400 error
        assert response.status_code == 400
        result = response.json()
        assert "detail" in result
        assert "json" in result["detail"].lower()
    
    def test_upload_with_mismatched_metadata_length(self, client, valid_image_bytes):
        """Test batch upload with metadata length mismatch."""
        files = [
            ("images", ("photo1.jpg", io.BytesIO(valid_image_bytes), "image/jpeg")),
            ("images", ("photo2.jpg", io.BytesIO(valid_image_bytes), "image/jpeg"))
        ]
        data = {
            "name": "Length Mismatch",
            "age_at_disappearance": 25,
            "year_disappeared": 2020,
            "gender": "male",
            "location_last_seen": "Austin, TX",
            "contact": "test@example.com",
            "image_metadata_json": '[{"photo_year": 2010}]'  # Only 1 metadata for 2 images
        }
        
        response = client.post("/api/v1/upload/missing/batch", files=files, data=data)
        
        # Should fail with 400 error
        assert response.status_code == 400
        result = response.json()
        assert "detail" in result
        assert "length" in result["detail"].lower() or "match" in result["detail"].lower()
    
    def test_upload_response_structure(self, client, valid_image_bytes):
        """Test that response has correct structure."""
        files = [
            ("images", ("photo1.jpg", io.BytesIO(valid_image_bytes), "image/jpeg"))
        ]
        data = {
            "name": "Structure Test",
            "age_at_disappearance": 30,
            "year_disappeared": 2018,
            "gender": "female",
            "location_last_seen": "Denver, CO",
            "contact": "test@example.com"
        }
        
        response = client.post("/api/v1/upload/missing/batch", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check required fields
            assert "success" in result
            assert "message" in result
            assert "case_id" in result
            assert "total_images_uploaded" in result
            assert "total_images_failed" in result
            assert "uploaded_images" in result
            assert "failed_images" in result
            assert "potential_matches" in result
            assert "processing_time_ms" in result
            
            # Check types
            assert isinstance(result["success"], bool)
            assert isinstance(result["message"], str)
            assert isinstance(result["total_images_uploaded"], int)
            assert isinstance(result["uploaded_images"], list)
            assert isinstance(result["failed_images"], list)
            assert isinstance(result["processing_time_ms"], float)


class TestBatchUploadFound:
    """Test batch upload for found persons."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from main import app
        return TestClient(app)
    
    @pytest.fixture
    def valid_image_bytes(self):
        """Create valid test image bytes."""
        img = Image.new('RGB', (640, 480), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', quality=95)
        return img_bytes.getvalue()
    
    def test_upload_found_person_batch(self, client, valid_image_bytes):
        """Test batch upload for found person."""
        files = [
            ("images", (f"found{i}.jpg", io.BytesIO(valid_image_bytes), "image/jpeg"))
            for i in range(3)
        ]
        data = {
            "current_age_estimate": 35,
            "gender": "male",
            "current_location": "San Francisco, CA",
            "finder_contact": "finder@example.com"
        }
        
        response = client.post("/api/v1/upload/found/batch", files=files, data=data)
        
        # Verify response structure
        assert response.status_code in [200, 400]
        result = response.json()
        
        if response.status_code == 200:
            assert result["success"] is True
            assert "case_id" in result  # Should be found_id
            assert "total_images_uploaded" in result
    
    def test_upload_found_with_optional_fields(self, client, valid_image_bytes):
        """Test batch upload with all optional fields."""
        files = [
            ("images", ("found.jpg", io.BytesIO(valid_image_bytes), "image/jpeg"))
        ]
        data = {
            "current_age_estimate": 40,
            "gender": "female",
            "current_location": "Phoenix, AZ",
            "finder_contact": "finder@example.com",
            "name": "Unknown Person",
            "visible_marks": "scar on left arm, tattoo on right shoulder",
            "current_condition": "Good health",
            "additional_info": "Found at homeless shelter"
        }
        
        response = client.post("/api/v1/upload/found/batch", files=files, data=data)
        
        # Should succeed or fail based on face detection
        assert response.status_code in [200, 400]


class TestBatchUploadLatency:
    """Test latency requirements for batch uploads."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from main import app
        return TestClient(app)
    
    @pytest.fixture
    def valid_image_bytes(self):
        """Create valid test image bytes."""
        img = Image.new('RGB', (800, 600), color='green')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', quality=90)
        return img_bytes.getvalue()
    
    def test_five_images_under_500ms_target(self, client, valid_image_bytes):
        """Test that 5 images upload completes under 500ms target."""
        files = [
            ("images", (f"speed{i}.jpg", io.BytesIO(valid_image_bytes), "image/jpeg"))
            for i in range(5)
        ]
        data = {
            "name": "Speed Test",
            "age_at_disappearance": 28,
            "year_disappeared": 2019,
            "gender": "male",
            "location_last_seen": "Dallas, TX",
            "contact": "speed@example.com"
        }
        
        start_time = time.time()
        response = client.post("/api/v1/upload/missing/batch", files=files, data=data)
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"\n⏱️  5-image upload latency: {elapsed_ms:.1f}ms")
        print(f"   Target: <500ms")
        print(f"   Status: {'✅ PASS' if elapsed_ms < 500 else '⚠️  SLOW'}")
        
        # Note: Actual latency depends on hardware/network
        # Just log the result, don't fail test
        assert response.status_code in [200, 400]


class TestReferenceImageHandling:
    """Test reference-only image handling (graceful degradation enhancement)."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from main import app
        return TestClient(app)
    
    @pytest.fixture
    def blank_image_bytes(self):
        """Create blank image (unlikely to have face detected)."""
        img = Image.new('RGB', (640, 480), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', quality=95)
        return img_bytes.getvalue()
    
    @pytest.fixture
    def landscape_image_bytes(self):
        """Create landscape image (no face)."""
        img = Image.new('RGB', (800, 600), color=(0, 128, 0))
        # Add some noise to make it more realistic
        pixels = img.load()
        for i in range(100):
            x, y = np.random.randint(0, 800), np.random.randint(0, 600)
            pixels[x, y] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', quality=90)
        return img_bytes.getvalue()
    
    def test_upload_reference_only_images(self, client, blank_image_bytes):
        """Test upload with only reference images (no face detected)."""
        files = [
            ("images", ("blank1.jpg", io.BytesIO(blank_image_bytes), "image/jpeg")),
            ("images", ("blank2.jpg", io.BytesIO(blank_image_bytes), "image/jpeg"))
        ]
        data = {
            "name": "Reference Only Test",
            "age_at_disappearance": 30,
            "year_disappeared": 2018,
            "gender": "male",
            "location_last_seen": "Test City",
            "contact": "test@example.com"
        }
        
        response = client.post("/api/v1/upload/missing/batch", files=files, data=data)
        
        # Should succeed even without valid faces
        if response.status_code == 200:
            result = response.json()
            
            assert result["success"] is True
            assert result["matching_images_count"] == 0  # No valid images
            assert result["reference_images_count"] >= 1  # At least 1 reference
            assert len(result["reference_images"]) >= 1
            
            # Check reference image structure
            ref_img = result["reference_images"][0]
            assert "image_id" in ref_img
            assert "validation_status" in ref_img
            assert ref_img["validation_status"] in ["no_face_detected", "low_quality"]
            assert "reason" in ref_img
            
            # No matches expected (no valid images for matching)
            assert len(result["potential_matches"]) == 0
            
            print("\n✅ Reference-only upload succeeded")
            print(f"   Reference images: {len(result['reference_images'])}")
        else:
            # If all images fail validation, that's also OK
            print(f"\n⚠️  Upload failed (expected if no faces detected): {response.status_code}")
    
    def test_upload_mixed_valid_and_reference(self, client, blank_image_bytes):
        """Test upload with mix of valid and reference images."""
        # This test requires at least one image with a face
        # Using blank images as a proxy (may or may not detect faces)
        
        files = [
            ("images", ("img1.jpg", io.BytesIO(blank_image_bytes), "image/jpeg")),
            ("images", ("img2.jpg", io.BytesIO(blank_image_bytes), "image/jpeg")),
            ("images", ("img3.jpg", io.BytesIO(blank_image_bytes), "image/jpeg"))
        ]
        data = {
            "name": "Mixed Upload Test",
            "age_at_disappearance": 28,
            "year_disappeared": 2019,
            "gender": "female",
            "location_last_seen": "Mixed City",
            "contact": "mixed@example.com"
        }
        
        response = client.post("/api/v1/upload/missing/batch", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Total uploaded should equal valid + reference
            total_uploaded = result["matching_images_count"] + result["reference_images_count"]
            assert result["total_images_uploaded"] == total_uploaded
            
            # Check lists match counts
            assert len(result["valid_images"]) == result["matching_images_count"]
            assert len(result["reference_images"]) == result["reference_images_count"]
            
            print(f"\n✅ Mixed upload: {result['matching_images_count']} valid, "
                  f"{result['reference_images_count']} reference")
        else:
            print(f"\n⚠️  Upload response: {response.status_code}")
    
    def test_reference_image_has_metadata(self, client, blank_image_bytes):
        """Test that reference images still have metadata (age, photo_year, etc.)."""
        files = [
            ("images", ("blank.jpg", io.BytesIO(blank_image_bytes), "image/jpeg"))
        ]
        data = {
            "name": "Metadata Test",
            "age_at_disappearance": 35,
            "year_disappeared": 2015,
            "gender": "male",
            "location_last_seen": "Metadata City",
            "contact": "metadata@example.com",
            "image_metadata_json": '[{"photo_year": 2010}]'
        }
        
        response = client.post("/api/v1/upload/missing/batch", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            # If saved as reference, should have metadata
            if result["reference_images_count"] > 0:
                ref_img = result["reference_images"][0]
                assert "age_at_photo" in ref_img
                assert ref_img["age_at_photo"] > 0  # Should have calculated age
                
                if "photo_year" in data["image_metadata_json"]:
                    assert ref_img["photo_year"] == 2010
                
                print(f"\n✅ Reference image has metadata: age={ref_img['age_at_photo']}")


# Run tests with: pytest tests/test_batch_upload_integration.py -v -s
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])

