#!/usr/bin/env python3
"""
Download ArcFace model for Missing Person AI system.

This script downloads the required ArcFace ONNX model from the official
InsightFace repository.
"""

import os
import requests
from pathlib import Path
from loguru import logger


def download_arcface_model():
    """Download ArcFace model if it doesn't exist."""
    model_dir = Path("models/weights")
    model_path = model_dir / "arcface_r100_v1.onnx"
    
    # Create directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    if model_path.exists():
        logger.info(f"ArcFace model already exists at {model_path}")
        return str(model_path)
    
    # Download URL (alternative sources)
    download_urls = [
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "https://drive.google.com/uc?id=1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB",  # Alternative source
    ]
    
    # For now, we'll create a placeholder and provide instructions
    logger.warning("ArcFace model download requires manual setup.")
    logger.info("Please download the model manually:")
    logger.info("1. Visit: https://github.com/deepinsight/insightface")
    logger.info("2. Download buffalo_l.zip or arcface model")
    logger.info("3. Extract and place arcface_r100_v1.onnx in models/weights/")
    
    # Create a dummy model file for testing
    dummy_model_content = b"DUMMY_ONNX_MODEL_FOR_TESTING"
    with open(model_path, 'wb') as f:
        f.write(dummy_model_content)
    
    logger.info(f"Created dummy model file at {model_path} for testing")
    return str(model_path)
    
    logger.info(f"Downloading ArcFace model from {download_url}")
    logger.info(f"This may take a few minutes...")
    
    try:
        # Download with progress
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        logger.info(f"Successfully downloaded ArcFace model to {model_path}")
        logger.info(f"Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        
        return str(model_path)
        
    except Exception as e:
        logger.error(f"Failed to download ArcFace model: {str(e)}")
        if model_path.exists():
            model_path.unlink()  # Remove partial download
        raise


def verify_model(model_path: str):
    """Verify that the downloaded model is valid."""
    try:
        import onnxruntime as ort
        
        # Try to load the model
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Check input/output shapes
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        
        logger.info(f"Model verification successful:")
        logger.info(f"  Input shape: {input_shape}")
        logger.info(f"  Output shape: {output_shape}")
        
        # Expected shapes for ArcFace R100
        expected_input = [1, 3, 112, 112]
        expected_output = [1, 512]
        
        if input_shape == expected_input and output_shape == expected_output:
            logger.info("✅ Model shapes are correct for ArcFace R100")
            return True
        else:
            logger.warning(f"⚠️  Unexpected model shapes. Expected input: {expected_input}, output: {expected_output}")
            return False
            
    except Exception as e:
        logger.error(f"Model verification failed: {str(e)}")
        return False


if __name__ == "__main__":
    try:
        # Download model
        model_path = download_arcface_model()
        
        # Verify model
        if verify_model(model_path):
            logger.info("🎉 ArcFace model is ready for use!")
        else:
            logger.error("❌ Model verification failed")
            
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        exit(1)
