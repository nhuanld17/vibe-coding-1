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
    print("Please make sure FGNET dataset is available!")
    exit(1)

print(f"Testing found person upload with: {image_path}")
print("(This should match with TEST_001 uploaded earlier)")

# Metadata theo schema mới
metadata = {
    "found_id": "FOUND_001",  # Optional - có thể để None để tự động generate
    "name": None,  # Optional
    "current_age_estimate": 22,
    "gender": "male",
    "current_location": "Found Location",
    "finder_contact": "finder@example.com",
    "visible_marks": None,  # Optional - có thể là list như ["tattoo on arm"]
    "current_condition": "Good health",  # Optional
    "additional_info": "Adult male found wandering"  # Optional
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
        print(f"  Found ID: {result.get('found_id', 'N/A')}")
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
            print(f"\n🎯 MATCHES FOUND: {len(matches)}")
            
            for i, match in enumerate(matches, 1):
                print(f"\n--- Match {i} ---")
                print(f"  Face Similarity: {match.get('face_similarity', 0):.4f}")
                print(f"  Metadata Similarity: {match.get('metadata_similarity', 0):.4f}")
                print(f"  Combined Score: {match.get('combined_score', 0):.4f}")
                print(f"  Confidence Level: {match.get('confidence_level', 'N/A')}")
                print(f"  Confidence Score: {match.get('confidence_score', 0):.4f}")
                print(f"  Contact: {match.get('contact', 'N/A')}")
                
                # Metadata info
                metadata_info = match.get('metadata', {})
                print(f"  Case ID: {metadata_info.get('case_id', 'N/A')}")
                print(f"  Name: {metadata_info.get('name', 'N/A')}")
                print(f"  Age at disappearance: {metadata_info.get('age_at_disappearance', 'N/A')}")
                
                # Explanation
                if 'explanation' in match:
                    explanation = match['explanation']
                    print(f"  Summary: {explanation.get('summary', 'N/A')}")
                    if 'reasons' in explanation:
                        print(f"  Reasons: {', '.join(explanation['reasons'][:3])}")
                    if 'recommendations' in explanation:
                        print(f"  Recommendations: {', '.join(explanation['recommendations'][:2])}")
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