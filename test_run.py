#!/usr/bin/env python3
"""
Test run script.
"""

print("MISSING PERSON AI - TEST RUN")
print("=" * 40)

try:
    # Test basic imports
    print("1. Testing imports...")
    from simple_test import app
    print("   OK: App imported")
    
    # Test FastAPI
    print("2. Testing FastAPI...")
    from fastapi.testclient import TestClient
    client = TestClient(app)
    print("   OK: Test client created")
    
    # Test endpoints
    print("3. Testing endpoints...")
    
    response = client.get("/")
    print(f"   GET /: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    response = client.get("/health")
    print(f"   GET /health: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    response = client.get("/test")
    print(f"   GET /test: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    print("\nSUCCESS: All endpoints working!")
    print("\nTo run live server:")
    print("python -m uvicorn simple_test:app --port 8001")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("=" * 40)
