"""
Identifier generation utilities for Family Finder AI.
"""

from datetime import datetime
import uuid


def _generate_identifier(prefix: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    random_segment = uuid.uuid4().hex[:6].upper()
    return f"{prefix}_{timestamp}_{random_segment}"


def generate_case_id() -> str:
    """Generate a unique case_id for missing persons."""
    return _generate_identifier("MISS")


def generate_found_id() -> str:
    """Generate a unique found_id for found persons."""
    return _generate_identifier("FOUND")

