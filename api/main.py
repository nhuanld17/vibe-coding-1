"""
Main FastAPI application for Missing Person AI system.

This module creates and configures the FastAPI application with
all routes, middleware, and lifecycle events.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import uvicorn

from .config import get_settings
from .dependencies import initialize_services, check_services_health
from .schemas.models import APIInfo, HealthResponse, ErrorResponse
from .routes import upload, search
from utils.logger import setup_logger


# Initialize settings
settings = get_settings()

# Setup logging
setup_logger(
    log_level=settings.log_level,
    log_file=settings.log_file
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting Missing Person AI API...")
    logger.info(f"Version: {settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    
    try:
        # Initialize all services
        initialize_services(settings)
        logger.info("All services initialized successfully")
        
        # Log configuration
        logger.info(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
        logger.info(f"Model path: {settings.arcface_model_path}")
        logger.info(f"GPU enabled: {settings.use_gpu}")
        logger.info(f"Face threshold: {settings.face_confidence_threshold}")
        logger.info(f"Similarity threshold: {settings.similarity_threshold}")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise RuntimeError(f"Application startup failed: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Missing Person AI API...")
    logger.info("Shutdown completed")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered system for matching missing and found persons using facial recognition",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=settings.get_cors_methods(),
    allow_headers=settings.get_cors_headers(),
)

# Add trusted host middleware (security)
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", settings.host]
    )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal server error occurred",
            details={"path": str(request.url)} if settings.debug else None
        ).dict()
    )


# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler for API errors."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
            details={"path": str(request.url)} if settings.debug else None
        ).dict()
    )


# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Root endpoint
@app.get("/", response_model=APIInfo)
async def root():
    """
    Get API information.
    
    Returns basic information about the Missing Person AI API.
    """
    return APIInfo(
        name=settings.app_name,
        version=settings.app_version,
        description="AI-powered system for matching missing and found persons using facial recognition",
        endpoints={
            "health": "/health",
            "upload_missing": "/api/v1/upload/missing",
            "upload_found": "/api/v1/upload/found",
            "search_missing": "/api/v1/search/missing/{case_id}",
            "search_found": "/api/v1/search/found/{found_id}",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        documentation_url="/docs"
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the API and all services.
    """
    try:
        # Check service health
        services_health = check_services_health()
        
        # Get database stats if available
        database_stats = None
        try:
            from .dependencies import get_vector_db
            vector_db = get_vector_db()
            database_stats = {
                "missing_persons": vector_db.get_collection_stats("missing_persons"),
                "found_persons": vector_db.get_collection_stats("found_persons")
            }
        except Exception as e:
            logger.warning(f"Could not get database stats: {str(e)}")
        
        # Determine overall status
        overall_status = "healthy" if services_health.get("overall", False) else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            timestamp=time.time(),
            services=services_health,
            database_stats=database_stats,
            version=settings.app_version
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )


# Include routers
app.include_router(
    upload.router,
    prefix="/api/v1/upload",
    tags=["upload"]
)

app.include_router(
    search.router,
    prefix="/api/v1/search",
    tags=["search"]
)


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
