#!/usr/bin/env python3
"""
Simple test script để kiểm tra các module cơ bản.
"""

import sys
import os

def test_basic_imports():
    """Test basic imports."""
    print("🧪 Testing basic imports...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import cv2
        print("✅ OpenCV imported successfully")
        
        from models.face_detection import FaceDetector
        print("✅ FaceDetector imported successfully")
        
        # Test tạo detector
        detector = FaceDetector()
        print("✅ FaceDetector created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

def test_simple_api():
    """Test simple FastAPI without dependencies."""
    print("\n🧪 Testing simple FastAPI...")
    
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        
        app = FastAPI(title="Simple Test API")
        
        @app.get("/")
        def root():
            return {"message": "Hello World", "status": "working"}
        
        @app.get("/test")
        def test():
            return {"test": "success", "python_version": sys.version}
        
        print("✅ FastAPI app created successfully")
        print("💡 You can run: uvicorn simple_test:app --port 8001")
        
        return True
        
    except Exception as e:
        print(f"❌ FastAPI test failed: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🚀 Missing Person AI - Simple Test")
    print("=" * 50)
    
    # Test basic imports
    imports_ok = test_basic_imports()
    
    # Test simple API
    api_ok = test_simple_api()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Basic Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Simple API: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if imports_ok and api_ok:
        print("\n🎉 Basic functionality is working!")
        print("\n📋 Next steps:")
        print("1. Install Docker Desktop and start it")
        print("2. Run: docker-compose up -d")
        print("3. Or run simple API: uvicorn simple_test:app --port 8001")
        print("4. Test with: curl http://localhost:8001/")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
    
    return imports_ok and api_ok

# Simple FastAPI app for testing
from fastapi import FastAPI

app = FastAPI(title="Simple Test API")

@app.get("/")
def root():
    return {
        "message": "Missing Person AI - Simple Test", 
        "status": "working",
        "python_version": sys.version,
        "endpoints": ["/", "/test", "/health"]
    }

@app.get("/test")
def test():
    return {"test": "success", "timestamp": "2025-10-19"}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "simple-test"}

if __name__ == "__main__":
    main()
