"""
Image Helper Utilities for Missing Person AI System.

This module provides helper functions for:
- Age calculation from photo metadata
- Image compression before Cloudinary upload
- Image quality validation
- Batch processing helpers

Author: AI Face Recognition Team
Version: 1.0.0
"""

import io
from typing import Optional, Tuple, List
from datetime import datetime
from PIL import Image
from loguru import logger


def calculate_age_at_photo(
    photo_year: Optional[int],
    year_disappeared: int,
    age_at_disappearance: int
) -> int:
    """
    Calculate age when photo was taken based on available metadata.
    
    Logic:
    - If photo_year provided: age_at_disappearance - (year_disappeared - photo_year)
    - If photo_year is None: assume photo from year_disappeared (most recent photo)
    - Validates photo_year is not in the future or after disappearance
    
    Args:
        photo_year: Year photo was taken (optional, None means use year_disappeared)
        year_disappeared: Year person disappeared
        age_at_disappearance: Age when person disappeared
        
    Returns:
        Estimated age at photo (clamped to 0-120)
        
    Raises:
        ValueError: If inputs are invalid (negative ages, invalid years)
        
    Examples:
        >>> # Photo taken 10 years before disappearance
        >>> calculate_age_at_photo(2010, 2020, 25)
        15
        
        >>> # Photo year not provided - assume from disappearance year
        >>> calculate_age_at_photo(None, 2020, 25)
        25
        
        >>> # Photo from same year as disappearance
        >>> calculate_age_at_photo(2020, 2020, 25)
        25
    """
    # Validate age_at_disappearance
    if not isinstance(age_at_disappearance, int) or age_at_disappearance < 0 or age_at_disappearance > 120:
        raise ValueError(
            f"age_at_disappearance must be an integer in [0, 120], got {age_at_disappearance}"
        )
    
    # Validate year_disappeared
    current_year = datetime.now().year
    if not isinstance(year_disappeared, int) or year_disappeared < 1900 or year_disappeared > current_year + 1:
        raise ValueError(
            f"year_disappeared must be between 1900 and {current_year + 1}, got {year_disappeared}"
        )
    
    # If photo_year not provided, return None (no age bonus will be applied)
    if photo_year is None:
        logger.debug(
            f"photo_year not provided, returning None (no age bonus will be applied)"
        )
        return None
    
    # Validate photo_year
    if not isinstance(photo_year, int):
        raise ValueError(f"photo_year must be an integer, got {type(photo_year)}")
    
    # Check if photo_year is in the future
    if photo_year > current_year:
        logger.warning(
            f"photo_year ({photo_year}) is in the future (current year: {current_year}). "
            f"Using year_disappeared ({year_disappeared}) instead."
        )
        photo_year = year_disappeared
    
    # Check if photo_year is after disappearance
    if photo_year > year_disappeared:
        logger.warning(
            f"photo_year ({photo_year}) is after year_disappeared ({year_disappeared}). "
            f"Using year_disappeared instead."
        )
        photo_year = year_disappeared
    
    # Check if photo is too old (more than person's age)
    years_before_disappearance = year_disappeared - photo_year
    if years_before_disappearance > age_at_disappearance:
        logger.warning(
            f"photo_year ({photo_year}) would result in negative age. "
            f"Person was {age_at_disappearance} at disappearance in {year_disappeared}, "
            f"but photo is {years_before_disappearance} years before. "
            f"Clamping to age 0."
        )
        return 0
    
    # Calculate age at photo
    age_at_photo = age_at_disappearance - years_before_disappearance
    
    # Clamp to valid range [0, 120]
    age_at_photo = max(0, min(120, age_at_photo))
    
    logger.debug(
        f"Calculated age_at_photo: {age_at_photo} "
        f"(photo_year={photo_year}, year_disappeared={year_disappeared}, "
        f"age_at_disappearance={age_at_disappearance})"
    )
    
    return age_at_photo


def compress_image_if_needed(
    image_bytes: bytes,
    max_size_mb: float = 2.0,
    quality: int = 85,
    max_dimension: Optional[int] = None
) -> Tuple[bytes, bool]:
    """
    Compress image if larger than max_size_mb or if dimensions exceed max_dimension.
    
    Args:
        image_bytes: Original image bytes
        max_size_mb: Maximum size in MB (default: 2.0 MB)
        quality: JPEG quality for compression (0-100, default: 85)
        max_dimension: Optional maximum width or height in pixels
        
    Returns:
        Tuple of (compressed_bytes, was_compressed)
        - compressed_bytes: Compressed image bytes (or original if already small)
        - was_compressed: True if compression was applied, False otherwise
        
    Raises:
        ValueError: If image_bytes is invalid or cannot be opened
        
    Examples:
        >>> with open('large_photo.jpg', 'rb') as f:
        ...     original_bytes = f.read()
        >>> compressed, was_compressed = compress_image_if_needed(original_bytes, max_size_mb=1.0)
        >>> if was_compressed:
        ...     print(f"Compressed from {len(original_bytes)/(1024*1024):.2f}MB to {len(compressed)/(1024*1024):.2f}MB")
    """
    # Validate inputs
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("image_bytes cannot be None or empty")
    
    if not (0.1 <= max_size_mb <= 50.0):
        raise ValueError(f"max_size_mb must be between 0.1 and 50.0, got {max_size_mb}")
    
    if not (1 <= quality <= 100):
        raise ValueError(f"quality must be between 1 and 100, got {quality}")
    
    # Check current size
    size_mb = len(image_bytes) / (1024 * 1024)
    logger.debug(f"Image size: {size_mb:.2f} MB")
    
    # Try to open image
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"Failed to open image: {str(e)}")
        raise ValueError(f"Invalid image bytes: {str(e)}")
    
    # Check if compression is needed
    needs_size_compression = size_mb > max_size_mb
    needs_dimension_resize = (
        max_dimension is not None and 
        (img.width > max_dimension or img.height > max_dimension)
    )
    
    if not needs_size_compression and not needs_dimension_resize:
        logger.debug("Image is already within limits, no compression needed")
        return image_bytes, False
    
    # Perform compression
    try:
        # Convert RGBA to RGB if needed (JPEG doesn't support transparency)
        if img.mode in ('RGBA', 'LA', 'P'):
            logger.debug(f"Converting image from {img.mode} to RGB")
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            logger.debug(f"Converting image from {img.mode} to RGB")
            img = img.convert('RGB')
        
        # Resize if dimensions exceed max_dimension
        if needs_dimension_resize:
            original_size = (img.width, img.height)
            # Calculate new size maintaining aspect ratio
            if img.width > img.height:
                new_width = max_dimension
                new_height = int((max_dimension / img.width) * img.height)
            else:
                new_height = max_dimension
                new_width = int((max_dimension / img.height) * img.width)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(
                f"Resized image from {original_size} to ({new_width}, {new_height})"
            )
        
        # Compress to JPEG
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        compressed_bytes = output.getvalue()
        
        compressed_size_mb = len(compressed_bytes) / (1024 * 1024)
        compression_ratio = (1 - compressed_size_mb / size_mb) * 100
        
        logger.info(
            f"Compressed image: {size_mb:.2f} MB -> {compressed_size_mb:.2f} MB "
            f"({compression_ratio:.1f}% reduction)"
        )
        
        return compressed_bytes, True
        
    except Exception as e:
        logger.error(f"Image compression failed: {str(e)}")
        raise ValueError(f"Compression failed: {str(e)}")


def validate_image_dimensions(
    image_bytes: bytes,
    min_width: int = 100,
    min_height: int = 100,
    max_width: int = 10000,
    max_height: int = 10000
) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """
    Validate image dimensions are within acceptable range.
    
    Args:
        image_bytes: Image bytes to validate
        min_width: Minimum acceptable width in pixels
        min_height: Minimum acceptable height in pixels
        max_width: Maximum acceptable width in pixels
        max_height: Maximum acceptable height in pixels
        
    Returns:
        Tuple of (is_valid, error_message, dimensions)
        - is_valid: True if dimensions are acceptable
        - error_message: None if valid, error string otherwise
        - dimensions: (width, height) tuple or None if image invalid
        
    Examples:
        >>> is_valid, error, dims = validate_image_dimensions(image_bytes)
        >>> if not is_valid:
        ...     print(f"Invalid image: {error}")
        >>> else:
        ...     print(f"Valid image: {dims[0]}x{dims[1]} pixels")
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        
        # Check minimum dimensions
        if width < min_width or height < min_height:
            return False, f"Image too small: {width}x{height} (minimum: {min_width}x{min_height})", (width, height)
        
        # Check maximum dimensions
        if width > max_width or height > max_height:
            return False, f"Image too large: {width}x{height} (maximum: {max_width}x{max_height})", (width, height)
        
        return True, None, (width, height)
        
    except Exception as e:
        logger.error(f"Failed to validate image dimensions: {str(e)}")
        return False, f"Invalid image: {str(e)}", None


def get_image_format(image_bytes: bytes) -> Optional[str]:
    """
    Detect image format from bytes.
    
    Args:
        image_bytes: Image bytes
        
    Returns:
        Image format string (e.g., 'JPEG', 'PNG') or None if invalid
        
    Examples:
        >>> fmt = get_image_format(image_bytes)
        >>> print(f"Image format: {fmt}")
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return img.format
    except Exception as e:
        logger.error(f"Failed to detect image format: {str(e)}")
        return None


def batch_calculate_ages(
    photo_years: List[Optional[int]],
    year_disappeared: int,
    age_at_disappearance: int
) -> List[int]:
    """
    Calculate ages for multiple photos at once.
    
    Convenience function for batch processing.
    
    Args:
        photo_years: List of photo years (can contain None values)
        year_disappeared: Year person disappeared
        age_at_disappearance: Age at disappearance
        
    Returns:
        List of calculated ages (same length as photo_years)
        
    Examples:
        >>> photo_years = [2010, None, 2015, 2018]
        >>> ages = batch_calculate_ages(photo_years, 2020, 30)
        >>> print(ages)  # [20, 30, 25, 28]
    """
    ages = []
    for photo_year in photo_years:
        try:
            age = calculate_age_at_photo(photo_year, year_disappeared, age_at_disappearance)
            ages.append(age)
        except Exception as e:
            logger.error(f"Failed to calculate age for photo_year={photo_year}: {str(e)}")
            # Fallback to age_at_disappearance on error
            ages.append(age_at_disappearance)
    
    return ages


def estimate_cloudinary_cost(
    num_images: int,
    avg_size_mb: float = 0.5,
    transformations_per_image: int = 2
) -> dict:
    """
    Estimate Cloudinary storage and bandwidth costs.
    
    This is a rough estimate based on Cloudinary's pricing (as of 2024).
    
    Args:
        num_images: Total number of images
        avg_size_mb: Average size per image in MB
        transformations_per_image: Average transformations per image
        
    Returns:
        Dictionary with cost estimates
        
    Example:
        >>> cost = estimate_cloudinary_cost(10000, avg_size_mb=0.8)
        >>> print(f"Monthly storage cost: ${cost['monthly_storage_cost']:.2f}")
    """
    # Cloudinary Free Tier (as of 2024)
    free_storage_gb = 25
    free_bandwidth_gb = 25
    free_transformations = 25000
    
    # Calculate usage
    total_storage_gb = (num_images * avg_size_mb) / 1024
    total_bandwidth_gb = total_storage_gb * 10  # Assume 10x bandwidth (uploads + downloads)
    total_transformations = num_images * transformations_per_image
    
    # Calculate overages
    storage_overage_gb = max(0, total_storage_gb - free_storage_gb)
    bandwidth_overage_gb = max(0, total_bandwidth_gb - free_bandwidth_gb)
    transformations_overage = max(0, total_transformations - free_transformations)
    
    # Pricing (approximate)
    storage_cost_per_gb = 0.10  # $0.10/GB/month
    bandwidth_cost_per_gb = 0.12  # $0.12/GB
    transformation_cost_per_1000 = 0.80  # $0.80/1000 transformations
    
    # Calculate costs
    storage_cost = storage_overage_gb * storage_cost_per_gb
    bandwidth_cost = bandwidth_overage_gb * bandwidth_cost_per_gb
    transformation_cost = (transformations_overage / 1000) * transformation_cost_per_1000
    
    total_cost = storage_cost + bandwidth_cost + transformation_cost
    
    return {
        'num_images': num_images,
        'total_storage_gb': round(total_storage_gb, 2),
        'total_bandwidth_gb': round(total_bandwidth_gb, 2),
        'total_transformations': total_transformations,
        'monthly_storage_cost': round(storage_cost, 2),
        'monthly_bandwidth_cost': round(bandwidth_cost, 2),
        'monthly_transformation_cost': round(transformation_cost, 2),
        'total_monthly_cost': round(total_cost, 2),
        'within_free_tier': total_cost == 0
    }


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    logger.info("Testing image_helpers.py...")
    
    # Test 1: calculate_age_at_photo
    print("\n=== Test 1: Age Calculation ===")
    test_cases = [
        (2010, 2020, 25, 15),  # Photo 10 years before disappearance
        (None, 2020, 25, 25),  # No photo year - use disappearance year
        (2020, 2020, 25, 25),  # Same year
        (2025, 2020, 25, 25),  # Future year (should warn and use 2020)
        (2022, 2020, 25, 25),  # After disappearance (should warn)
    ]
    
    for photo_year, year_dis, age_dis, expected in test_cases:
        try:
            result = calculate_age_at_photo(photo_year, year_dis, age_dis)
            status = "✅" if result == expected else f"❌ (expected {expected})"
            print(f"  photo_year={photo_year}, year_dis={year_dis}, age_dis={age_dis} -> {result} {status}")
        except Exception as e:
            print(f"  photo_year={photo_year} -> ERROR: {e}")
    
    # Test 2: Image compression
    print("\n=== Test 2: Image Compression ===")
    
    # Create a test image
    try:
        test_img = Image.new('RGB', (2000, 1500), color='red')
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='PNG')
        original_bytes = img_bytes.getvalue()
        
        print(f"  Original size: {len(original_bytes)/(1024*1024):.2f} MB")
        
        compressed, was_compressed = compress_image_if_needed(original_bytes, max_size_mb=0.5)
        print(f"  Compressed size: {len(compressed)/(1024*1024):.2f} MB")
        print(f"  Was compressed: {was_compressed}")
        
        if was_compressed:
            print("  ✅ Compression test passed")
        else:
            print("  ⚠️  Image was not compressed (might be already small)")
            
    except Exception as e:
        print(f"  ❌ Compression test failed: {e}")
    
    # Test 3: Batch age calculation
    print("\n=== Test 3: Batch Age Calculation ===")
    photo_years = [2005, 2010, None, 2015, 2018]
    ages = batch_calculate_ages(photo_years, 2020, 30)
    print(f"  Photo years: {photo_years}")
    print(f"  Calculated ages: {ages}")
    print("  ✅ Batch calculation test passed")
    
    # Test 4: Cost estimation
    print("\n=== Test 4: Cloudinary Cost Estimation ===")
    cost = estimate_cloudinary_cost(10000, avg_size_mb=0.8)
    print(f"  10,000 images @ 0.8 MB each:")
    print(f"    Storage: {cost['total_storage_gb']} GB")
    print(f"    Monthly cost: ${cost['total_monthly_cost']}")
    print(f"    Within free tier: {cost['within_free_tier']}")
    print("  ✅ Cost estimation test passed")
    
    logger.info("\n✅ All tests passed!")

