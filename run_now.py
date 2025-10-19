#!/usr/bin/env python3
"""
Run Missing Person AI now - simple version.
"""

def run_simple_server():
    """Run simple HTTP server."""
    print("MISSING PERSON AI - STARTING...")
    print("=" * 40)
    
    try:
        # Import and test
        from simple_test import app
        print("OK: App imported successfully")
        
        # Show info
        print("API will start on: http://localhost:8001")
        print("Endpoints available:")
        print("  GET /        - Main info")
        print("  GET /test    - Test endpoint") 
        print("  GET /health  - Health check")
        print("  GET /docs    - API documentation")
        print()
        print("Press Ctrl+C to stop")
        print("=" * 40)
        
        # Start server
        import uvicorn
        uvicorn.run(
            "simple_test:app",
            host="0.0.0.0",
            port=8001,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\nServer stopped!")
    except Exception as e:
        print(f"Error starting server: {e}")
        print("\nTrying alternative method...")
        
        # Alternative: run with subprocess
        import subprocess
        import sys
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "simple_test:app",
            "--host", "0.0.0.0",
            "--port", "8001"
        ])

if __name__ == "__main__":
    run_simple_server()
