#!/usr/bin/env python3
"""
Start Missing Person AI demo.
"""

import subprocess
import time
import sys
import webbrowser
from pathlib import Path

def start_simple_api():
    """Start simple API server."""
    print("Starting Missing Person AI Demo...")
    print("=" * 50)
    
    try:
        # Test import first
        from simple_test import app
        print("OK: Simple test app imported")
        
        print("Starting API server on http://localhost:8001")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        # Start server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "simple_test:app", 
            "--host", "0.0.0.0", 
            "--port", "8001",
            "--reload"
        ])
        
    except KeyboardInterrupt:
        print("\nDemo stopped!")
    except Exception as e:
        print(f"Error: {e}")

def show_info():
    """Show demo information."""
    print("\nMISSING PERSON AI - DEMO")
    print("=" * 40)
    print("API Endpoints:")
    print("  http://localhost:8001/")
    print("  http://localhost:8001/health")
    print("  http://localhost:8001/docs")
    print("=" * 40)

if __name__ == "__main__":
    show_info()
    start_simple_api()
