"""
Image processing utilities for Missing Person AI system.

This module provides image loading, validation, and preprocessing utilities.
"""

import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Optional, Union
import io
from loguru import logger


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


def validate_image(image: np.ndarray, min_size: Tuple[int, int] = (50, 50)) -> bool:
    """
    Validate image dimensions and format.
    
    Args:
        image: Image as numpy array
        min_size: Minimum required size (width, height)
        
    Returns:
        True if image is valid
    """
    if image is None or image.size == 0:
        return False
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False
    
    height, width = image.shape[:2]
    if width < min_size[0] or height < min_size[1]:
        return False
    
    return True


def resize_image(
    image: np.ndarray, 
    target_size: Tuple[int, int], 
    maintain_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if maintain_aspect_ratio:
        # Calculate scaling factor
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Create canvas and center image
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)


def normalize_image_orientation(image_bytes: bytes) -> bytes:
    """
    Normalize image orientation based on EXIF data.
    
    Args:
        image_bytes: Original image bytes
        
    Returns:
        Normalized image bytes
    """
    try:
        # Open image with PIL to handle EXIF orientation
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Apply EXIF orientation
        pil_image = ImageOps.exif_transpose(pil_image)
        
        # Convert back to bytes
        output_buffer = io.BytesIO()
        pil_image.save(output_buffer, format='JPEG', quality=95)
        return output_buffer.getvalue()
        
    except Exception as e:
        logger.warning(f"Failed to normalize image orientation: {str(e)}")
        return image_bytes


def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality for better face detection.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
        
    except Exception as e:
        logger.warning(f"Image enhancement failed: {str(e)}")
        return image


def calculate_image_hash(image: np.ndarray) -> str:
    """
    Calculate perceptual hash of image for duplicate detection.
    
    Args:
        image: Input image
        
    Returns:
        Image hash as string
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to 8x8
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_CUBIC)
        
        # Calculate average
        avg = resized.mean()
        
        # Create hash
        hash_bits = []
        for row in resized:
            for pixel in row:
                hash_bits.append('1' if pixel > avg else '0')
        
        # Convert to hex
        hash_string = ''.join(hash_bits)
        hash_hex = hex(int(hash_string, 2))[2:]
        
        return hash_hex
        
    except Exception as e:
        logger.error(f"Image hash calculation failed: {str(e)}")
        return ""


def is_image_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Check if image is blurry using Laplacian variance.
    
    Args:
        image: Input image
        threshold: Blur threshold
        
    Returns:
        True if image is blurry
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance < threshold
    except Exception as e:
        logger.error(f"Blur detection failed: {str(e)}")
        return False


def get_image_info(image: np.ndarray) -> dict:
    """
    Get comprehensive image information.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image information
    """
    try:
        height, width, channels = image.shape
        
        # Calculate file size estimate
        size_bytes = image.nbytes
        
        # Calculate brightness and contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        contrast = gray.std()
        
        # Check if blurry
        is_blurry = is_image_blurry(image)
        
        # Calculate hash
        img_hash = calculate_image_hash(image)
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'size_bytes': size_bytes,
            'brightness': float(brightness),
            'contrast': float(contrast),
            'is_blurry': is_blurry,
            'hash': img_hash
        }
        
    except Exception as e:
        logger.error(f"Failed to get image info: {str(e)}")
        return {}
