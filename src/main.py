import time
import uuid
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_client import make_asgi_app
import structlog

from src.core.config import settings
from src.core.logging import configure_logging, set_request_id, get_logger
from src.api.routes import router as api_router
from src.services.pipeline import get_pipeline_manager
from src.core.metrics import REQUEST_COUNT, REQUEST_LATENCY

# 1. Initialize Structured Logging
configure_logging(settings.LOG_LEVEL)
logger = get_logger("main")

# 2. Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-Grade Document Layout Analysis and OCR Pipeline"
)

# 3. Mount Prometheus ASGI exporter to /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# 4. Request Logging & Correlation ID Middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    # Retrieve or generate request correlation ID
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    set_request_id(request_id)
    
    # Bind structlog contextvars for current async task
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    
    start_time = time.perf_counter()
    
    # Log incoming request metadata (PII-scrubbed)
    logger.info(
        "Incoming HTTP request",
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else None
    )
    
    try:
        response = await call_next(request)
    except Exception as e:
        # Catch unhandled exceptions to prevent stack trace leaks
        duration = time.perf_counter() - start_time
        logger.exception("Unhandled server exception during request", path=request.url.path)
        
        # Track metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=500
        ).inc()
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred. Please contact support.",
                "request_id": request_id
            }
        )

    duration = time.perf_counter() - start_time
    
    # Append Request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    # Record metrics for non-system endpoints
    if not request.url.path.endswith(("/health", "/ready", "/metrics")):
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
        
    logger.info(
        "HTTP request completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_sec=duration
    )
    
    return response

# 5. Handle Request Validation Errors gracefully
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = request.headers.get("X-Request-ID", "")
    logger.warning("Request schema validation failed", errors=exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Unprocessable Entity",
            "message": "Input validation failed.",
            "details": exc.errors(),
            "request_id": request_id
        }
    )

# 6. Include API routes
app.include_router(api_router, prefix=settings.API_PREFIX)

@app.on_event("startup")
def startup_event():
    logger.info("Initializing models on service startup...")
    # Pre-load models in memory to avoid cold-start latencies on first API request
    try:
        get_pipeline_manager()
        logger.info("All pipeline models loaded and ready.")
    except Exception as e:
        logger.exception("Failed to initialize models on startup")
        # Do not block startup, allow container to run so ready/health probes report failure
