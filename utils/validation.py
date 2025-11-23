"""
Input validation utilities for Missing Person AI system.

This module provides validation functions for API inputs,
file uploads, and data integrity checks.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger

# Try to import magic (optional - may not be available on Windows)
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logger.warning("python-magic not available. File type validation will use filename extension fallback.")


# Supported image MIME types
SUPPORTED_IMAGE_TYPES = {
    'image/jpeg',
    'image/jpg', 
    'image/png',
    'image/gif',
    'image/webp'
}

# File size limits (in bytes)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_FILE_SIZE = 1024  # 1KB

# Validation patterns
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
PHONE_PATTERN = re.compile(r'^\+?[\d\s\-\(\)]{10,}$')
# More flexible case_id pattern - allow any alphanumeric with underscores/dashes
CASE_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]{3,50}$')


def validate_file_upload(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    """
    Validate uploaded file.
    
    Args:
        file_bytes: File content as bytes
        filename: Original filename
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check file size
        if len(file_bytes) < MIN_FILE_SIZE:
            return False, f"File too small (minimum {MIN_FILE_SIZE} bytes)"
        
        if len(file_bytes) > MAX_FILE_SIZE:
            return False, f"File too large (maximum {MAX_FILE_SIZE // (1024*1024)}MB)"
        
        # Check MIME type
        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_buffer(file_bytes, mime=True)
            except Exception:
                # Fallback to filename extension
                ext = filename.lower().split('.')[-1] if '.' in filename else ''
                mime_type = f"image/{ext}" if ext in ['jpg', 'jpeg', 'png', 'gif', 'webp'] else 'unknown'
        else:
            # Fallback to filename extension when magic is not available
            ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if ext in ['jpg', 'jpeg']:
                mime_type = 'image/jpeg'
            elif ext == 'png':
                mime_type = 'image/png'
            elif ext == 'gif':
                mime_type = 'image/gif'
            elif ext == 'webp':
                mime_type = 'image/webp'
            else:
                mime_type = 'unknown'
        
        if mime_type not in SUPPORTED_IMAGE_TYPES:
            return False, f"Unsupported file type: {mime_type}"
        
        return True, ""
        
    except Exception as e:
        logger.error(f"File validation failed: {str(e)}")
        return False, f"File validation error: {str(e)}"


def validate_missing_person_metadata(metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate missing person metadata.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields (case_id is auto-generated, so not required)
    required_fields = {
        'name': str,
        'age_at_disappearance': int,
        'year_disappeared': int,
        'gender': str,
        'location_last_seen': str,
        'contact': str
    }
    
    # Check required fields
    for field, expected_type in required_fields.items():
        if field not in metadata:
            errors.append(f"Missing required field: {field}")
            continue
        
        if not isinstance(metadata[field], expected_type):
            errors.append(f"Field '{field}' must be of type {expected_type.__name__}")
            continue
        
        # Field-specific validation
        if field == 'case_id' and metadata.get('case_id'):
            if not CASE_ID_PATTERN.match(metadata[field]):
                errors.append("Invalid case_id format (3-50 alphanumeric characters, underscores, or dashes allowed)")
        
        elif field == 'name':
            if len(metadata[field].strip()) < 2:
                errors.append("Name must be at least 2 characters")
        
        elif field == 'age_at_disappearance':
            if not 0 <= metadata[field] <= 120:
                errors.append("Age at disappearance must be between 0 and 120")
        
        elif field == 'year_disappeared':
            current_year = datetime.now().year
            if not 1900 <= metadata[field] <= current_year:
                errors.append(f"Year disappeared must be between 1900 and {current_year}")
        
        elif field == 'gender':
            valid_genders = {'male', 'female', 'other', 'unknown'}
            if metadata[field].lower() not in valid_genders:
                errors.append(f"Gender must be one of: {', '.join(valid_genders)}")
        
        elif field == 'location_last_seen':
            if len(metadata[field].strip()) < 3:
                errors.append("Location must be at least 3 characters")
        
        elif field == 'contact':
            contact = metadata[field].strip()
            if not (EMAIL_PATTERN.match(contact) or PHONE_PATTERN.match(contact)):
                errors.append("Contact must be a valid email or phone number")
    
    # Optional fields validation
    if 'height_cm' in metadata:
        if not isinstance(metadata['height_cm'], int) or not 50 <= metadata['height_cm'] <= 250:
            errors.append("Height must be between 50 and 250 cm")
    
    if 'birthmarks' in metadata:
        if not isinstance(metadata['birthmarks'], list):
            errors.append("Birthmarks must be a list")
        elif len(metadata['birthmarks']) > 10:
            errors.append("Maximum 10 birthmarks allowed")
    
    return len(errors) == 0, errors


def validate_found_person_metadata(metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate found person metadata.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields (found_id is auto-generated, so not required)
    required_fields = {
        'current_age_estimate': int,
        'gender': str,
        'current_location': str,
        'finder_contact': str
    }
    
    # Check required fields
    for field, expected_type in required_fields.items():
        if field not in metadata:
            errors.append(f"Missing required field: {field}")
            continue
        
        if not isinstance(metadata[field], expected_type):
            errors.append(f"Field '{field}' must be of type {expected_type.__name__}")
            continue
        
        # Field-specific validation
        if field == 'found_id' and metadata.get('found_id'):
            # More flexible pattern
            if not re.match(r'^[A-Za-z0-9_-]{3,50}$', metadata[field]):
                errors.append("Invalid found_id format (3-50 alphanumeric characters, underscores, or dashes allowed)")
        
        elif field == 'current_age_estimate':
            if not 0 <= metadata[field] <= 120:
                errors.append("Current age estimate must be between 0 and 120")
        
        elif field == 'gender':
            valid_genders = {'male', 'female', 'other', 'unknown'}
            if metadata[field].lower() not in valid_genders:
                errors.append(f"Gender must be one of: {', '.join(valid_genders)}")
        
        elif field == 'current_location':
            if len(metadata[field].strip()) < 3:
                errors.append("Current location must be at least 3 characters")
        
        elif field == 'finder_contact':
            contact = metadata[field].strip()
            if not (EMAIL_PATTERN.match(contact) or PHONE_PATTERN.match(contact)):
                errors.append("Finder contact must be a valid email or phone number")
    
    # Optional fields validation
    if 'visible_marks' in metadata:
        if not isinstance(metadata['visible_marks'], list):
            errors.append("Visible marks must be a list")
        elif len(metadata['visible_marks']) > 10:
            errors.append("Maximum 10 visible marks allowed")
    
    if 'current_condition' in metadata:
        if not isinstance(metadata['current_condition'], str):
            errors.append("Current condition must be a string")
        elif len(metadata['current_condition']) > 500:
            errors.append("Current condition must be less than 500 characters")
    
    return len(errors) == 0, errors


def validate_search_parameters(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate search parameters.
    
    Args:
        params: Search parameters dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Validate limit
    if 'limit' in params:
        if not isinstance(params['limit'], int) or not 1 <= params['limit'] <= 100:
            errors.append("Limit must be an integer between 1 and 100")
    
    # Validate threshold
    if 'threshold' in params:
        if not isinstance(params['threshold'], (int, float)) or not 0.0 <= params['threshold'] <= 1.0:
            errors.append("Threshold must be a number between 0.0 and 1.0")
    
    # Validate filters
    if 'filters' in params:
        if not isinstance(params['filters'], dict):
            errors.append("Filters must be a dictionary")
        else:
            # Validate filter values
            for key, value in params['filters'].items():
                if key == 'gender':
                    valid_genders = {'male', 'female', 'other', 'unknown'}
                    if value not in valid_genders:
                        errors.append(f"Filter gender must be one of: {', '.join(valid_genders)}")
                
                elif key == 'age_range':
                    if not isinstance(value, list) or len(value) != 2:
                        errors.append("Age range filter must be a list of two integers")
                    elif not all(isinstance(x, int) for x in value):
                        errors.append("Age range values must be integers")
                    elif value[0] > value[1]:
                        errors.append("Age range minimum must be less than maximum")
    
    return len(errors) == 0, errors


def sanitize_string(text: str, max_length: int = 1000) -> str:
    """
    Sanitize string input.
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove control characters and excessive whitespace
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    sanitized = re.sub(r'\s+', ' ', sanitized)
    sanitized = sanitized.strip()
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def validate_confidence_threshold(threshold: float) -> bool:
    """
    Validate confidence threshold value.
    
    Args:
        threshold: Confidence threshold
        
    Returns:
        True if valid
    """
    return isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0


def validate_age_range(age_range: List[int]) -> bool:
    """
    Validate age range.
    
    Args:
        age_range: List of [min_age, max_age]
        
    Returns:
        True if valid
    """
    if not isinstance(age_range, list) or len(age_range) != 2:
        return False
    
    min_age, max_age = age_range
    
    if not all(isinstance(x, int) for x in age_range):
        return False
    
    if not 0 <= min_age <= max_age <= 120:
        return False
    
    return True


# Example usage and testing
if __name__ == "__main__":
    # Test missing person metadata validation
    missing_metadata = {
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
    
    is_valid, errors = validate_missing_person_metadata(missing_metadata)
    print(f"Missing person metadata valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Test found person metadata validation
    found_metadata = {
        'found_id': 'FOUND_2023_001',
        'current_age_estimate': 30,
        'gender': 'male',
        'current_location': 'Los Angeles, CA',
        'finder_contact': 'finder@example.com',
        'visible_marks': ['scar on left arm'],
        'current_condition': 'Good health'
    }
    
    is_valid, errors = validate_found_person_metadata(found_metadata)
    print(f"Found person metadata valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
