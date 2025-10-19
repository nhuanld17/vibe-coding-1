#!/usr/bin/env python3
"""Test upload found person và xem matching."""
import requests
import json
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000/api/v1/upload/found"

# Upload ảnh cùng người nhưng lớn tuổi hơn để test matching
image_path = Path("datasets/FGNET_organized/person_001/age_22.jpg")

if not image_path.exists():
    print(f"ERROR: Image not found at {image_path}")
    exit(1)

print(f"Testing found person upload with: {image_path}")
print("(This should match with TEST_001 uploaded earlier)")

# Metadata
metadata = {
    "found_id": "FOUND_001",
    "current_age_estimate": 22,
    "gender": "male",
    "current_location": "Found Location",
    "finder_contact": "finder@example.com"
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
        
        print(f"\nUpload successful!")
        print(f"Point ID: {result.get('point_id')}")
        print(f"Processing time: {result.get('processing_time_ms'):.1f} ms")
        
        # Show matches
        matches = result.get('potential_matches', [])
        print(f"\nFound {len(matches)} potential matches:")
        
        for i, match in enumerate(matches, 1):
            print(f"\n--- Match {i} ---")
            print(f"Face Similarity: {match['face_similarity']:.3f}")
            print(f"Confidence Level: {match['confidence_level']}")
            print(f"Confidence Score: {match['confidence_score']:.3f}")
            print(f"Contact: {match['contact']}")
            print(f"Metadata: {match['metadata'].get('name')} (Case: {match['metadata'].get('case_id')})")
            print(f"Summary: {match['explanation']['summary']}")
    else:
        print("\nERROR!")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\nEXCEPTION: {e}")
    import traceback
    traceback.print_exc()

