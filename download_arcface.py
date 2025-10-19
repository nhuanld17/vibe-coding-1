#!/usr/bin/env python3
"""
Download ArcFace model from direct URL.
"""
import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url: str, destination: Path):
    """Download file with progress bar."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {destination}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    print(f"✅ Download complete! File size: {destination.stat().st_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    # Direct download URL for ArcFace model
    MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"
    
    # Alternative URL (backup)
    BACKUP_URL = "https://huggingface.co/datasets/Xenova/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
    
    model_path = Path("models/weights/arcface_r100_v1.onnx")
    
    try:
        print("Attempting to download ArcFace model...")
        download_file(MODEL_URL, model_path)
    except Exception as e:
        print(f"Primary download failed: {e}")
        print("\nTrying backup URL...")
        try:
            download_file(BACKUP_URL, model_path)
        except Exception as e2:
            print(f"❌ Backup download also failed: {e2}")
            print("\n📋 Manual download instructions:")
            print("1. Visit: https://github.com/deepinsight/insightface")
            print("2. Or: https://huggingface.co/datasets/Xenova/insightface")
            print("3. Download buffalo_l model")
            print(f"4. Place the .onnx file at: {model_path.absolute()}")
            exit(1)

