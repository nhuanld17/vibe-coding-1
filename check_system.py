#!/usr/bin/env python3
"""
System check script for Missing Person AI.

This script verifies that all components are properly installed
and configured.
"""

import sys
import os
import importlib
from pathlib import Path
from loguru import logger


def check_python_version():
    """Check Python version."""
    logger.info("Checking Python version...")
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        logger.info("✅ Python version is compatible")
        return True
    else:
        logger.error("❌ Python 3.8+ is required")
        return False


def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'numpy',
        'opencv-python',
        'onnxruntime',
        'qdrant-client',
        'loguru',
        'mtcnn',
        'tensorflow',
        'pytest'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Handle package name differences
            import_name = package
            if package == 'opencv-python':
                import_name = 'cv2'
            elif package == 'pydantic-settings':
                import_name = 'pydantic_settings'
            
            importlib.import_module(import_name)
            logger.info(f"✅ {package}")
        except ImportError:
            logger.error(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ All dependencies are installed")
    return True


def check_project_structure():
    """Check if project structure is correct."""
    logger.info("Checking project structure...")
    
    required_dirs = [
        'models',
        'models/weights',
        'services',
        'api',
        'api/routes',
        'api/schemas',
        'utils',
        'database',
        'tests',
        'logs'
    ]
    
    required_files = [
        'models/__init__.py',
        'models/face_detection.py',
        'models/face_embedding.py',
        'services/__init__.py',
        'services/vector_db.py',
        'services/bilateral_search.py',
        'services/confidence_scoring.py',
        'api/__init__.py',
        'api/main.py',
        'api/config.py',
        'api/dependencies.py',
        'api/routes/__init__.py',
        'api/routes/upload.py',
        'api/routes/search.py',
        'api/schemas/__init__.py',
        'api/schemas/models.py',
        'utils/__init__.py',
        'utils/logger.py',
        'utils/image_processing.py',
        'utils/validation.py',
        'requirements.txt',
        'docker-compose.yml',
        'Dockerfile',
        'README.md'
    ]
    
    missing_items = []
    
    # Check directories
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created directory: {dir_path}")
        else:
            logger.info(f"✅ {dir_path}/")
    
    # Check files
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"❌ {file_path}")
            missing_items.append(file_path)
        else:
            logger.info(f"✅ {file_path}")
    
    if missing_items:
        logger.error(f"Missing files: {', '.join(missing_items)}")
        return False
    
    logger.info("✅ Project structure is complete")
    return True


def check_model_files():
    """Check if model files exist."""
    logger.info("Checking model files...")
    
    model_path = Path("models/weights/arcface_r100_v1.onnx")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ ArcFace model found ({size_mb:.1f} MB)")
        
        # Check if it's a dummy file
        if size_mb < 1:
            logger.warning("⚠️  Model appears to be a dummy file for testing")
            logger.info("For production use, download the real model:")
            logger.info("1. Visit: https://github.com/deepinsight/insightface")
            logger.info("2. Download buffalo_l.zip")
            logger.info("3. Extract arcface_r100_v1.onnx to models/weights/")
        
        return True
    else:
        logger.error("❌ ArcFace model not found")
        logger.info("Run: python download_model.py")
        return False


def check_configuration():
    """Check configuration files."""
    logger.info("Checking configuration...")
    
    # Check .env.example
    env_example = Path(".env.example")
    if env_example.exists():
        logger.info("✅ .env.example found")
    else:
        logger.error("❌ .env.example not found")
        return False
    
    # Check if .env exists (optional)
    env_file = Path(".env")
    if env_file.exists():
        logger.info("✅ .env file found")
    else:
        logger.info("ℹ️  .env file not found (optional)")
        logger.info("Copy .env.example to .env and customize if needed")
    
    return True


def test_imports():
    """Test importing main modules."""
    logger.info("Testing module imports...")
    
    modules_to_test = [
        'models.face_detection',
        'models.face_embedding',
        'services.vector_db',
        'services.bilateral_search',
        'services.confidence_scoring',
        'api.main',
        'api.config',
        'utils.logger',
        'utils.validation'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            logger.info(f"✅ {module}")
        except Exception as e:
            logger.error(f"❌ {module}: {str(e)}")
            failed_imports.append(module)
    
    if failed_imports:
        logger.error(f"Failed imports: {', '.join(failed_imports)}")
        return False
    
    logger.info("✅ All modules import successfully")
    return True


def check_docker():
    """Check Docker configuration."""
    logger.info("Checking Docker configuration...")
    
    docker_files = ['Dockerfile', 'docker-compose.yml', '.dockerignore']
    
    for file in docker_files:
        if Path(file).exists():
            logger.info(f"✅ {file}")
        else:
            logger.error(f"❌ {file}")
            return False
    
    logger.info("✅ Docker configuration is complete")
    return True


def main():
    """Run all system checks."""
    logger.info("🚀 Starting Missing Person AI System Check")
    logger.info("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Model Files", check_model_files),
        ("Configuration", check_configuration),
        ("Module Imports", test_imports),
        ("Docker Config", check_docker)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        logger.info(f"\n🔍 {check_name}")
        logger.info("-" * 30)
        
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            logger.error(f"Check failed with exception: {str(e)}")
            results.append((check_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 SYSTEM CHECK SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{check_name:<20} {status}")
        if result:
            passed += 1
    
    logger.info("-" * 50)
    logger.info(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("🎉 System is ready!")
        logger.info("\nNext steps:")
        logger.info("1. Start services: docker-compose up -d")
        logger.info("2. Check API: curl http://localhost:8000/health")
        logger.info("3. View docs: http://localhost:8000/docs")
        return True
    else:
        logger.error("❌ System has issues that need to be resolved")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
