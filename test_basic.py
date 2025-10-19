#!/usr/bin/env python3
"""
Basic test script for Missing Person AI.
"""

import sys

def test_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print("OK: NumPy imported")
        
        import cv2
        print("OK: OpenCV imported")
        
        import fastapi
        print("OK: FastAPI imported")
        
        from models.face_detection import FaceDetector
        print("OK: FaceDetector imported")
        
        # Test creating detector
        detector = FaceDetector()
        print("OK: FaceDetector created")
        
        print("SUCCESS: All basic imports working!")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def main():
    """Main test."""
    print("Missing Person AI - Basic Test")
    print("=" * 40)
    
    success = test_imports()
    
    print("=" * 40)
    if success:
        print("RESULT: Basic functionality is working!")
        print("\nNext steps:")
        print("1. Start Docker Desktop")
        print("2. Run: docker-compose up -d")
        print("3. Test API: curl http://localhost:8000/health")
    else:
        print("RESULT: Some components need fixing")
    
    return success

if __name__ == "__main__":
    main()
