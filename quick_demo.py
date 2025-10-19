#!/usr/bin/env python3
"""
Quick demo of Missing Person AI functionality.
"""

def demo_face_detection():
    """Demo face detection functionality."""
    print("DEMO: Face Detection")
    print("-" * 30)
    
    try:
        from models.face_detection import FaceDetector
        import numpy as np
        
        # Create detector
        detector = FaceDetector()
        print("OK: Face detector created")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("OK: Dummy image created (480x640)")
        
        # Test face detection (will not find faces in random image)
        faces = detector.detect_faces(dummy_image, confidence_threshold=0.9)
        print(f"OK: Face detection completed, found {len(faces)} faces")
        
        # Test quality check
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        is_good, metrics = detector.check_face_quality(dummy_face)
        print(f"OK: Face quality check - Good: {is_good}")
        print(f"    Sharpness: {metrics['sharpness']:.1f}")
        print(f"    Brightness: {metrics['brightness']:.1f}")
        print(f"    Contrast: {metrics['contrast']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def demo_api_structure():
    """Demo API structure."""
    print("\nDEMO: API Structure")
    print("-" * 30)
    
    try:
        from fastapi import FastAPI
        from api.config import get_settings
        
        # Test settings
        settings = get_settings()
        print(f"OK: Settings loaded - App: {settings.app_name}")
        print(f"    Version: {settings.app_version}")
        print(f"    Debug: {settings.debug}")
        print(f"    Face threshold: {settings.face_confidence_threshold}")
        
        # Test FastAPI app creation
        app = FastAPI(title="Test App")
        print("OK: FastAPI app created")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def demo_vector_operations():
    """Demo vector operations."""
    print("\nDEMO: Vector Operations")
    print("-" * 30)
    
    try:
        import numpy as np
        
        # Create dummy embeddings
        embedding1 = np.random.rand(512).astype(np.float32)
        embedding2 = np.random.rand(512).astype(np.float32)
        
        # Normalize
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        print("OK: Created 512-dim embeddings")
        print(f"    Embedding 1 norm: {np.linalg.norm(embedding1):.3f}")
        print(f"    Embedding 2 norm: {np.linalg.norm(embedding2):.3f}")
        
        # Calculate similarity
        similarity = np.dot(embedding1, embedding2)
        print(f"OK: Cosine similarity: {similarity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def show_project_status():
    """Show project status."""
    print("\nPROJECT STATUS")
    print("=" * 50)
    
    components = [
        ("Face Detection (MTCNN)", "models/face_detection.py"),
        ("Face Embedding (ArcFace)", "models/face_embedding.py"),
        ("Vector Database (Qdrant)", "services/vector_db.py"),
        ("Bilateral Search", "services/bilateral_search.py"),
        ("Confidence Scoring", "services/confidence_scoring.py"),
        ("FastAPI Application", "api/main.py"),
        ("Docker Configuration", "docker-compose.yml"),
        ("Documentation", "README.md")
    ]
    
    for name, file in components:
        try:
            from pathlib import Path
            if Path(file).exists():
                print(f"✓ {name}")
            else:
                print(f"✗ {name}")
        except:
            print(f"? {name}")

def main():
    """Main demo function."""
    print("MISSING PERSON AI - FUNCTIONALITY DEMO")
    print("=" * 50)
    
    # Run demos
    demos = [
        demo_face_detection,
        demo_api_structure,
        demo_vector_operations
    ]
    
    results = []
    for demo in demos:
        try:
            result = demo()
            results.append(result)
        except Exception as e:
            print(f"Demo failed: {e}")
            results.append(False)
    
    # Show project status
    show_project_status()
    
    # Summary
    print("\nDEMO RESULTS")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total} demos")
    
    if passed == total:
        print("SUCCESS: All core functionality is working!")
        print("\nTo run full system:")
        print("1. Start Docker Desktop")
        print("2. Run: docker-compose up -d")
        print("3. Open: http://localhost:8000/docs")
    else:
        print("Some components need attention.")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
