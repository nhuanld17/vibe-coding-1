#!/usr/bin/env python3
"""
Demo script để chạy Missing Person AI system.
"""

import subprocess
import time
import requests
import sys
import threading
from pathlib import Path

def run_api_server():
    """Chạy API server trong thread riêng."""
    try:
        print("🚀 Starting API server...")
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "simple_test:app", 
            "--host", "0.0.0.0", 
            "--port", "8001",
            "--log-level", "info"
        ], check=True)
    except Exception as e:
        print(f"❌ API server failed: {e}")

def test_api():
    """Test API endpoints."""
    base_url = "http://localhost:8001"
    
    print("⏳ Waiting for API to start...")
    for i in range(10):
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            if response.status_code == 200:
                print("✅ API is running!")
                break
        except:
            print(f"   Attempt {i+1}/10...")
            time.sleep(2)
    else:
        print("❌ API failed to start")
        return False
    
    # Test endpoints
    endpoints = ["/", "/test", "/health"]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            print(f"✅ {endpoint}: {response.status_code} - {response.json()}")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")
    
    return True

def show_demo_info():
    """Hiển thị thông tin demo."""
    print("=" * 60)
    print("🎉 MISSING PERSON AI - DEMO RUNNING")
    print("=" * 60)
    print("📍 API Endpoints:")
    print("   • Main: http://localhost:8001/")
    print("   • Test: http://localhost:8001/test")
    print("   • Health: http://localhost:8001/health")
    print("   • Docs: http://localhost:8001/docs")
    print()
    print("🧪 Test Commands:")
    print('   curl "http://localhost:8001/"')
    print('   curl "http://localhost:8001/health"')
    print()
    print("🌐 Open in Browser:")
    print("   http://localhost:8001/docs")
    print("=" * 60)

def main():
    """Main demo function."""
    print("🚀 Missing Person AI - Demo Launcher")
    print("=" * 50)
    
    # Check if we can import the app
    try:
        from simple_test import app
        print("✅ Simple test app imported successfully")
    except Exception as e:
        print(f"❌ Failed to import app: {e}")
        return
    
    # Start API server in background thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Test API
    if test_api():
        show_demo_info()
        
        print("\n🎯 Demo is running! Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Demo stopped!")
    else:
        print("❌ Demo failed to start")

if __name__ == "__main__":
    main()
