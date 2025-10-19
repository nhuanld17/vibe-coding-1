#!/usr/bin/env python3
"""Test upload API with FGNET images."""
import requests
import json
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000/api/v1/upload/missing"

# Image path from FGNET dataset
image_path = Path("datasets/FGNET_organized/person_001/age_02.jpg")

if not image_path.exists():
    print(f"ERROR: Image not found at {image_path}")
    print("Please run organize_fgnet.py first!")
    exit(1)

print(f"Testing upload with: {image_path}")

# Metadata
metadata = {
    "case_id": "TEST_001",
    "name": "Test Person 001",
    "age_at_disappearance": 2,
    "year_disappeared": 2023,
    "gender": "male",
    "location_last_seen": "Test Location",
    "contact": "test@example.com"
}

print(f"\nMetadata:")
print(json.dumps(metadata, indent=2))

# Upload
print(f"\nUploading to {API_URL}...")

try:
    with open(image_path, 'rb') as img_file:
        files = {
            'image': (image_path.name, img_file, 'image/jpeg')
        }
        data = {
            'metadata': json.dumps(metadata)
        }
        
        response = requests.post(API_URL, files=files, data=data, timeout=30)
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        print("\nSUCCESS!")
        result = response.json()
        print(json.dumps(result, indent=2))
    else:
        print("\nERROR!")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\nEXCEPTION: {e}")
    import traceback
    traceback.print_exc()

