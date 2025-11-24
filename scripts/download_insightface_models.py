"""
Helper script to pre-download InsightFace models.

This script downloads InsightFace models before running the API,
so you can verify the download works and models are ready.

Usage:
    python scripts/download_insightface_models.py
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from loguru import logger

def download_models():
    """Download InsightFace models."""
    try:
        logger.info("Downloading InsightFace models...")
        logger.info("This may take a few minutes depending on your internet connection...")
        
        # Import insightface
        from insightface.app import FaceAnalysis
        
        # Initialize FaceAnalysis - this will trigger model download
        # Models will be saved to ~/.insightface/models/antelopev2/
        logger.info("Initializing FaceAnalysis (this will download models if not present)...")
        app = FaceAnalysis(name="antelopev2")
        app.prepare(ctx_id=-1)  # Use CPU
        
        logger.info("✅ Models downloaded successfully!")
        logger.info(f"Models location: {Path.home() / '.insightface' / 'models' / 'antelopev2'}")
        logger.info("You can now start the API server.")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download models: {str(e)}")
        logger.error("Please check your internet connection and try again.")
        return False

if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)

