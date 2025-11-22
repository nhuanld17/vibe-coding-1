"""
Cloudinary service for uploading images.

This module provides functionality to upload images to Cloudinary.
"""

import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
from typing import Optional, Dict, Any
from loguru import logger
import io
from PIL import Image


def configure_cloudinary(
    cloud_name: str,
    api_key: str,
    api_secret: str
) -> None:
    """
    Configure Cloudinary with credentials.
    
    Args:
        cloud_name: Cloudinary cloud name
        api_key: Cloudinary API key
        api_secret: Cloudinary API secret
    """
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True  # Use HTTPS
    )
    logger.info("Cloudinary configured successfully")


def upload_image_to_cloudinary(
    image_bytes: bytes,
    folder: str,
    public_id: Optional[str] = None,
    resource_type: str = "image",
    overwrite: bool = True,
    invalidate: bool = True
) -> Dict[str, Any]:
    """
    Upload image to Cloudinary.
    
    Args:
        image_bytes: Image data as bytes
        folder: Cloudinary folder path
        public_id: Optional public ID for the image
        resource_type: Resource type (default: "image")
        overwrite: Whether to overwrite if exists
        invalidate: Whether to invalidate CDN cache
        
    Returns:
        Dictionary containing upload result with:
        - public_id: Public ID of uploaded image
        - secure_url: HTTPS URL of the image
        - url: HTTP URL of the image
        - format: Image format
        - width: Image width
        - height: Image height
        - bytes: Image size in bytes
        
    Raises:
        Exception: If upload fails
    """
    try:
        # Prepare upload options
        upload_options = {
            "folder": folder,
            "resource_type": resource_type,
            "overwrite": overwrite,
            "invalidate": invalidate,
            "quality": "auto:good",  # Auto quality optimization
            "fetch_format": "auto",  # Auto format (WebP when supported)
        }
        
        if public_id:
            upload_options["public_id"] = public_id
        
        # Upload image
        result = cloudinary.uploader.upload(
            image_bytes,
            **upload_options
        )
        
        logger.info(f"Image uploaded to Cloudinary: {result.get('public_id')}")
        
        return {
            "public_id": result.get("public_id"),
            "secure_url": result.get("secure_url"),
            "url": result.get("url"),
            "format": result.get("format"),
            "width": result.get("width"),
            "height": result.get("height"),
            "bytes": result.get("bytes"),
            "created_at": result.get("created_at"),
        }
        
    except Exception as e:
        logger.error(f"Failed to upload image to Cloudinary: {str(e)}")
        raise Exception(f"Cloudinary upload failed: {str(e)}")


def upload_image_from_pil(
    pil_image: Image.Image,
    folder: str,
    public_id: Optional[str] = None,
    format: str = "JPEG",
    quality: int = 90
) -> Dict[str, Any]:
    """
    Upload PIL Image to Cloudinary.
    
    Args:
        pil_image: PIL Image object
        folder: Cloudinary folder path
        public_id: Optional public ID for the image
        format: Image format (JPEG, PNG, etc.)
        quality: Image quality (1-100)
        
    Returns:
        Dictionary containing upload result
    """
    # Convert PIL Image to bytes
    output_buffer = io.BytesIO()
    pil_image.save(output_buffer, format=format, quality=quality)
    image_bytes = output_buffer.getvalue()
    
    return upload_image_to_cloudinary(
        image_bytes=image_bytes,
        folder=folder,
        public_id=public_id
    )


def delete_image_from_cloudinary(public_id: str) -> Dict[str, Any]:
    """
    Delete image from Cloudinary.
    
    Args:
        public_id: Public ID of the image to delete
        
    Returns:
        Dictionary containing deletion result
    """
    try:
        result = cloudinary.uploader.destroy(public_id, invalidate=True)
        logger.info(f"Image deleted from Cloudinary: {public_id}")
        return result
    except Exception as e:
        logger.error(f"Failed to delete image from Cloudinary: {str(e)}")
        raise Exception(f"Cloudinary deletion failed: {str(e)}")


def get_image_url(public_id: str, transformation: Optional[Dict] = None) -> str:
    """
    Get Cloudinary URL for an image.
    
    Args:
        public_id: Public ID of the image
        transformation: Optional transformation parameters
        
    Returns:
        Cloudinary URL
    """
    try:
        url, _ = cloudinary_url(
            public_id,
            transformation=transformation,
            secure=True
        )
        return url
    except Exception as e:
        logger.error(f"Failed to generate Cloudinary URL: {str(e)}")
        raise Exception(f"Failed to generate URL: {str(e)}")

