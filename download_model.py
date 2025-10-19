#!/usr/bin/env python3
"""Simple ArcFace model downloader."""
import requests
from pathlib import Path

def download_model():
    """Download ArcFace model."""
    url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"
    model_path = Path("models/weights/arcface_r100_v1.onnx")
    
    print(f"Downloading ArcFace model...")
    print(f"URL: {url}")
    print(f"Destination: {model_path}")
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(model_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB)", end='')
    
    print(f"\nDownload complete! Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
    return model_path

if __name__ == "__main__":
    try:
        model_path = download_model()
        print(f"SUCCESS! Model saved to: {model_path}")
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)

