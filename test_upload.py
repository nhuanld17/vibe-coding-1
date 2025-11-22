#!/usr/bin/env python3
"""Test upload missing person API with FGNET images."""
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

# Metadata theo schema mới
metadata = {
    "case_id": "TEST_001",  # Optional - có thể để None để tự động generate
    "name": "Test Person 001",
    "age_at_disappearance": 2,
    "year_disappeared": 2023,
    "gender": "male",
    "location_last_seen": "Test Location",
    "contact": "test@example.com",
    "height_cm": None,  # Optional
    "birthmarks": None,  # Optional - có thể là list như ["scar on left arm"]
    "additional_info": "Test description"  # Optional
}

print(f"\nMetadata:")
print(json.dumps(metadata, indent=2, ensure_ascii=False))

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
        print("\n✅ SUCCESS!")
        result = response.json()
        
        print(f"\nUpload Details:")
        print(f"  Case ID: {result.get('case_id', 'N/A')}")
        print(f"  Point ID: {result.get('point_id', 'N/A')}")
        print(f"  Message: {result.get('message', 'N/A')}")
        print(f"  Processing Time: {result.get('processing_time_ms', 0):.2f} ms")
        
        # Face quality
        if 'face_quality' in result:
            quality = result['face_quality']
            print(f"\nFace Quality:")
            print(f"  Sharpness: {quality.get('sharpness', 0):.3f}")
            print(f"  Brightness: {quality.get('brightness', 0):.3f}")
            print(f"  Contrast: {quality.get('contrast', 0):.3f}")
        
        # Potential matches
        matches = result.get('potential_matches', [])
        if matches:
            print(f"\nPotential Matches Found: {len(matches)}")
            for i, match in enumerate(matches[:3], 1):  # Show top 3
                print(f"\n  Match {i}:")
                print(f"    Face Similarity: {match.get('face_similarity', 0):.4f}")
                print(f"    Confidence Level: {match.get('confidence_level', 'N/A')}")
                print(f"    Confidence Score: {match.get('confidence_score', 0):.4f}")
                print(f"    Contact: {match.get('contact', 'N/A')}")
        else:
            print(f"\nNo potential matches found.")
        
        print(f"\nFull Response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\n❌ ERROR!")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\n❌ EXCEPTION: {e}")
    import traceback
    traceback.print_exc()