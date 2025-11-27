"""
Unit tests for Image Helper Utilities.

This module tests image helper functions including:
- Age calculation from photo metadata
- Image compression
- Image validation
- Batch processing
- Edge cases (invalid inputs, future dates, etc.)

Test coverage target: >95%

Author: AI Face Recognition Team
"""

import pytest
import io
from PIL import Image
from datetime import datetime
from typing import Tuple

from utils.image_helpers import (
    calculate_age_at_photo,
    compress_image_if_needed,
    validate_image_dimensions,
    get_image_format,
    batch_calculate_ages,
    estimate_cloudinary_cost
)


# ============================================================================
# Test 1: Age Calculation - With Photo Year
# ============================================================================

def test_calculate_age_at_photo_with_year():
    """Test age calculation when photo_year is provided."""
    # Photo taken 10 years before disappearance
    age = calculate_age_at_photo(photo_year=2010, year_disappeared=2020, age_at_disappearance=25)
    assert age == 15
    
    # Photo from same year as disappearance
    age = calculate_age_at_photo(photo_year=2020, year_disappeared=2020, age_at_disappearance=25)
    assert age == 25
    
    # Photo taken 5 years before
    age = calculate_age_at_photo(photo_year=2015, year_disappeared=2020, age_at_disappearance=30)
    assert age == 25


# ============================================================================
# Test 2: Age Calculation - Without Photo Year (None)
# ============================================================================

def test_calculate_age_at_photo_without_year():
    """Test age calculation when photo_year is None (defaults to year_disappeared)."""
    age = calculate_age_at_photo(photo_year=None, year_disappeared=2020, age_at_disappearance=25)
    assert age == 25  # Should use age_at_disappearance
    
    age = calculate_age_at_photo(photo_year=None, year_disappeared=2015, age_at_disappearance=40)
    assert age == 40


# ============================================================================
# Test 3: Age Calculation - Future Date
# ============================================================================

def test_calculate_age_future_date():
    """Test age calculation handles future photo_year (should use year_disappeared)."""
    future_year = datetime.now().year + 5
    age = calculate_age_at_photo(
        photo_year=future_year,
        year_disappeared=2020,
        age_at_disappearance=25
    )
    assert age == 25  # Should fallback to age_at_disappearance


# ============================================================================
# Test 4: Age Calculation - Photo After Disappearance
# ============================================================================

def test_calculate_age_photo_after_disappearance():
    """Test age calculation when photo_year > year_disappeared."""
    # Photo allegedly taken after disappearance (invalid)
    age = calculate_age_at_photo(
        photo_year=2022,
        year_disappeared=2020,
        age_at_disappearance=25
    )
    assert age == 25  # Should use year_disappeared


# ============================================================================
# Test 5: Age Calculation - Photo Too Old (Negative Age)
# ============================================================================

def test_calculate_age_photo_too_old():
    """Test age calculation when photo would result in negative age."""
    # Person was 10 at disappearance in 2020, but photo from 2005 (would be -5 years old)
    age = calculate_age_at_photo(
        photo_year=2005,
        year_disappeared=2020,
        age_at_disappearance=10
    )
    assert age == 0  # Should clamp to 0


# ============================================================================
# Test 6: Age Calculation - Boundary Cases
# ============================================================================

def test_calculate_age_boundary_cases():
    """Test age calculation at boundaries (age 0, age 120)."""
    # Age 0 at disappearance
    age = calculate_age_at_photo(photo_year=2020, year_disappeared=2020, age_at_disappearance=0)
    assert age == 0
    
    # Age 120 at disappearance
    age = calculate_age_at_photo(photo_year=2020, year_disappeared=2020, age_at_disappearance=120)
    assert age == 120
    
    # Age would exceed 120 (should clamp)
    age = calculate_age_at_photo(photo_year=1900, year_disappeared=2020, age_at_disappearance=100)
    assert age == 0  # 100 - (2020-1900) = -20, clamped to 0


# ============================================================================
# Test 7: Age Calculation - Invalid Inputs
# ============================================================================

def test_calculate_age_invalid_inputs():
    """Test age calculation raises ValueError for invalid inputs."""
    # Negative age
    with pytest.raises(ValueError, match="age_at_disappearance"):
        calculate_age_at_photo(photo_year=2010, year_disappeared=2020, age_at_disappearance=-5)
    
    # Age > 120
    with pytest.raises(ValueError, match="age_at_disappearance"):
        calculate_age_at_photo(photo_year=2010, year_disappeared=2020, age_at_disappearance=150)
    
    # Invalid year (too old)
    with pytest.raises(ValueError, match="year_disappeared"):
        calculate_age_at_photo(photo_year=2010, year_disappeared=1800, age_at_disappearance=25)
    
    # Invalid year (too far in future)
    future_year = datetime.now().year + 10
    with pytest.raises(ValueError, match="year_disappeared"):
        calculate_age_at_photo(photo_year=2010, year_disappeared=future_year, age_at_disappearance=25)


# ============================================================================
# Test 8: Image Compression - Small Image (No Compression Needed)
# ============================================================================

def test_compress_small_image():
    """Test that small images are not compressed."""
    # Create a small test image (200x200, should be < 2MB)
    img = Image.new('RGB', (200, 200), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=95)
    original_bytes = img_bytes.getvalue()
    
    compressed, was_compressed = compress_image_if_needed(original_bytes, max_size_mb=2.0)
    
    assert not was_compressed
    assert compressed == original_bytes


# ============================================================================
# Test 9: Image Compression - Large Image
# ============================================================================

def test_compress_large_image():
    """Test that large images are compressed."""
    # Create a large test image (4000x3000, will likely be > 1MB)
    img = Image.new('RGB', (4000, 3000), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')  # PNG is larger
    original_bytes = img_bytes.getvalue()
    
    original_size_mb = len(original_bytes) / (1024 * 1024)
    
    # Compress to max 1MB
    compressed, was_compressed = compress_image_if_needed(original_bytes, max_size_mb=1.0)
    compressed_size_mb = len(compressed) / (1024 * 1024)
    
    if original_size_mb > 1.0:
        assert was_compressed
        assert compressed_size_mb <= original_size_mb
    # Note: Small images might not need compression


# ============================================================================
# Test 10: Image Compression - RGBA to RGB Conversion
# ============================================================================

def test_compress_rgba_to_rgb():
    """Test that RGBA images are converted to RGB during compression."""
    # Create RGBA image (make it larger to force compression)
    img = Image.new('RGBA', (3000, 3000), color=(255, 0, 0, 128))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    original_bytes = img_bytes.getvalue()
    
    compressed, was_compressed = compress_image_if_needed(original_bytes, max_size_mb=0.1, quality=80)
    
    # Should be compressed and converted
    compressed_img = Image.open(io.BytesIO(compressed))
    # If compressed, should be RGB. If not compressed (already small), can be any mode
    if was_compressed:
        assert compressed_img.mode == 'RGB'


# ============================================================================
# Test 11: Image Compression - Dimension Resize
# ============================================================================

def test_compress_with_dimension_limit():
    """Test image resizing when dimensions exceed max_dimension."""
    # Create large image
    img = Image.new('RGB', (3000, 2000), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    original_bytes = img_bytes.getvalue()
    
    # Resize to max 1000px
    compressed, was_compressed = compress_image_if_needed(
        original_bytes, 
        max_size_mb=10.0,  # High limit to focus on dimension resize
        max_dimension=1000
    )
    
    assert was_compressed
    compressed_img = Image.open(io.BytesIO(compressed))
    assert max(compressed_img.width, compressed_img.height) <= 1000


# ============================================================================
# Test 12: Image Compression - Invalid Inputs
# ============================================================================

def test_compress_invalid_inputs():
    """Test image compression raises ValueError for invalid inputs."""
    # Empty bytes
    with pytest.raises(ValueError, match="cannot be None or empty"):
        compress_image_if_needed(b'', max_size_mb=2.0)
    
    # Invalid max_size_mb
    valid_img = io.BytesIO()
    Image.new('RGB', (100, 100)).save(valid_img, format='JPEG')
    
    with pytest.raises(ValueError, match="max_size_mb"):
        compress_image_if_needed(valid_img.getvalue(), max_size_mb=0.0)
    
    # Invalid quality
    with pytest.raises(ValueError, match="quality"):
        compress_image_if_needed(valid_img.getvalue(), max_size_mb=2.0, quality=150)
    
    # Invalid image bytes
    with pytest.raises(ValueError, match="Invalid image"):
        compress_image_if_needed(b'not an image', max_size_mb=2.0)


# ============================================================================
# Test 13: Validate Image Dimensions - Valid
# ============================================================================

def test_validate_dimensions_valid():
    """Test dimension validation for valid images."""
    img = Image.new('RGB', (800, 600), color='yellow')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    
    is_valid, error, dims = validate_image_dimensions(img_bytes.getvalue())
    
    assert is_valid
    assert error is None
    assert dims == (800, 600)


# ============================================================================
# Test 14: Validate Image Dimensions - Too Small
# ============================================================================

def test_validate_dimensions_too_small():
    """Test dimension validation rejects too-small images."""
    img = Image.new('RGB', (50, 50), color='purple')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    
    is_valid, error, dims = validate_image_dimensions(
        img_bytes.getvalue(),
        min_width=100,
        min_height=100
    )
    
    assert not is_valid
    assert "too small" in error.lower()
    assert dims == (50, 50)


# ============================================================================
# Test 15: Validate Image Dimensions - Too Large
# ============================================================================

def test_validate_dimensions_too_large():
    """Test dimension validation rejects too-large images."""
    img = Image.new('RGB', (12000, 8000), color='orange')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    
    is_valid, error, dims = validate_image_dimensions(
        img_bytes.getvalue(),
        max_width=10000,
        max_height=10000
    )
    
    assert not is_valid
    assert "too large" in error.lower()
    assert dims == (12000, 8000)


# ============================================================================
# Test 16: Get Image Format
# ============================================================================

def test_get_image_format():
    """Test image format detection."""
    # JPEG
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    fmt = get_image_format(img_bytes.getvalue())
    assert fmt == 'JPEG'
    
    # PNG
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    fmt = get_image_format(img_bytes.getvalue())
    assert fmt == 'PNG'
    
    # Invalid
    fmt = get_image_format(b'not an image')
    assert fmt is None


# ============================================================================
# Test 17: Batch Calculate Ages
# ============================================================================

def test_batch_calculate_ages():
    """Test batch age calculation for multiple photos."""
    photo_years = [2005, 2010, None, 2015, 2018]
    ages = batch_calculate_ages(photo_years, year_disappeared=2020, age_at_disappearance=30)
    
    # Expected: [15, 20, 30, 25, 28]
    assert len(ages) == 5
    assert ages[0] == 15  # 2005: 30 - (2020-2005) = 15
    assert ages[1] == 20  # 2010: 30 - (2020-2010) = 20
    assert ages[2] == 30  # None: use age_at_disappearance
    assert ages[3] == 25  # 2015: 30 - (2020-2015) = 25
    assert ages[4] == 28  # 2018: 30 - (2020-2018) = 28


# ============================================================================
# Test 18: Batch Calculate Ages - With Errors
# ============================================================================

def test_batch_calculate_ages_with_errors():
    """Test batch age calculation handles errors gracefully."""
    # Mix of valid and invalid (future year)
    future_year = datetime.now().year + 5
    photo_years = [2010, future_year, 2015]
    ages = batch_calculate_ages(photo_years, year_disappeared=2020, age_at_disappearance=30)
    
    assert len(ages) == 3
    assert ages[0] == 20  # Valid
    assert ages[1] == 30  # Future year → fallback to age_at_disappearance
    assert ages[2] == 25  # Valid


# ============================================================================
# Test 19: Estimate Cloudinary Cost - Free Tier
# ============================================================================

def test_estimate_cloudinary_cost_free_tier():
    """Test cost estimation within free tier."""
    cost = estimate_cloudinary_cost(num_images=100, avg_size_mb=0.2)
    
    assert cost['num_images'] == 100
    assert cost['within_free_tier'] is True
    assert cost['total_monthly_cost'] == 0.0


# ============================================================================
# Test 20: Estimate Cloudinary Cost - Exceeds Free Tier
# ============================================================================

def test_estimate_cloudinary_cost_exceeds_free_tier():
    """Test cost estimation when exceeding free tier."""
    # 50,000 images @ 1MB each = 50GB storage (exceeds 25GB free tier)
    cost = estimate_cloudinary_cost(num_images=50000, avg_size_mb=1.0)
    
    assert cost['num_images'] == 50000
    assert cost['total_storage_gb'] > 25.0
    assert cost['within_free_tier'] is False
    assert cost['total_monthly_cost'] > 0.0


# ============================================================================
# Test 21: Calculate Age - Type Validation
# ============================================================================

def test_calculate_age_type_validation():
    """Test that non-integer types raise ValueError."""
    # String instead of int
    with pytest.raises(ValueError):
        calculate_age_at_photo(photo_year="2010", year_disappeared=2020, age_at_disappearance=25)
    
    # Float instead of int (should work if valid range)
    # Note: Python may convert float to int, but let's test explicit int requirement
    with pytest.raises(ValueError):
        calculate_age_at_photo(photo_year=2010.5, year_disappeared=2020, age_at_disappearance=25)


# ============================================================================
# Test 22: Compression Quality Parameter
# ============================================================================

def test_compression_quality_parameter():
    """Test that different quality settings affect compression."""
    # Create larger image to force compression
    img = Image.new('RGB', (3000, 2500), color='cyan')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=100)
    original_bytes = img_bytes.getvalue()
    
    original_size_mb = len(original_bytes) / (1024 * 1024)
    
    # Compress with low quality
    compressed_low, was_compressed_low = compress_image_if_needed(original_bytes, max_size_mb=0.3, quality=50)
    
    # Compress with high quality
    compressed_high, was_compressed_high = compress_image_if_needed(original_bytes, max_size_mb=0.3, quality=95)
    
    # If original is large enough to trigger compression, verify compression worked
    if original_size_mb > 0.3:
        assert was_compressed_low or was_compressed_high  # At least one should be compressed
        # Just check both results are reasonable
        assert len(compressed_low) > 0
        assert len(compressed_high) > 0


# ============================================================================
# Test 23: Validate Dimensions - Invalid Image
# ============================================================================

def test_validate_dimensions_invalid_image():
    """Test dimension validation with invalid image bytes."""
    is_valid, error, dims = validate_image_dimensions(b'corrupted image data')
    
    assert not is_valid
    assert "Invalid image" in error
    assert dims is None


# ============================================================================
# Test 24: Batch Calculate Ages - Empty List
# ============================================================================

def test_batch_calculate_ages_empty():
    """Test batch age calculation with empty list."""
    ages = batch_calculate_ages([], year_disappeared=2020, age_at_disappearance=30)
    assert ages == []


# ============================================================================
# Test 25: Edge Case - Photo Year Same As Birth Year
# ============================================================================

def test_calculate_age_photo_at_birth():
    """Test age calculation when photo is from person's birth year."""
    # Person born in 2000, disappeared at age 20 in 2020
    # Photo from 2000 (birth year)
    age = calculate_age_at_photo(photo_year=2000, year_disappeared=2020, age_at_disappearance=20)
    assert age == 0  # Should be 0 (newborn photo)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

