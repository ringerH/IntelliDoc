import io
import time
from typing import Optional, Literal
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, status, Response
from PIL import Image
import numpy as np
from pydantic import BaseModel

from src.core.config import settings, get_active_config, save_active_config
from src.core.logging import get_logger
from src.services.pipeline import get_pipeline_manager
from src.core.metrics import REQUEST_COUNT, PIPELINE_LATENCY, MODEL_CONFIDENCE

router = APIRouter()
logger = get_logger("routes")

# Request/Response schemas for configuration management
class ConfigUpdateSchema(BaseModel):
    active_backend: Optional[Literal["pytorch", "onnx"]] = None
    classifier_version: Optional[str] = None
    detector_version: Optional[str] = None

@router.get("/health", tags=["Infrastructure"])
def liveness():
    """Liveness probe. Checks if the service is up."""
    return {"status": "healthy", "timestamp": time.time()}

@router.get("/ready", tags=["Infrastructure"])
def readiness():
    """Readiness probe. Checks if ML models are initialized and cached in memory."""
    try:
        manager = get_pipeline_manager()
        active_config = get_active_config()
        # Verify classification and detection pipelines are loaded for default settings
        is_ready = manager.is_ready(
            backend=active_config["active_backend"],
            classifier_version=active_config["classifier_version"],
            detector_version=active_config["detector_version"]
        )
        if not is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not fully initialized"
            )
        return {"status": "ready", "backend": active_config["active_backend"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Readiness check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Readiness check failed: {str(e)}"
        )

@router.post("/process", tags=["Processing"])
async def process_document(
    file: UploadFile = File(...),
    backend: Optional[Literal["pytorch", "onnx"]] = None,
    classifier_version: Optional[str] = None,
    detector_version: Optional[str] = None,
    conf_threshold: float = Query(0.25, ge=0.0, le=1.0)
):
    """Processes uploaded document to classify type, detect layout, and perform region & full OCR."""
    log = get_logger("document_processing")
    
    # 1. Validate File type
    if not file.content_type.startswith("image/"):
        log.warning("Invalid file type uploaded", content_type=file.content_type)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image."
        )

    # Resolve active settings
    active_config = get_active_config()
    active_backend = backend or active_config["active_backend"]
    active_classifier = classifier_version or active_config["classifier_version"]
    active_detector = detector_version or active_config["detector_version"]

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Enforce reasonable size limits
        if image.width > 4000 or image.height > 4000:
            log.warning("Image dimensions exceeded bounds", width=image.width, height=image.height)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image size exceeds 4000x4000 pixels limit."
            )

        manager = get_pipeline_manager()
        
        # Run pipeline with duration tracking per step
        start_time = time.perf_counter()
        
        # Fetch models to ensure they are loaded and isolate step timings
        classifier, detector, ocr = manager.get_pipeline(
            backend=active_backend,
            classifier_version=active_classifier,
            detector_version=active_detector
        )
        
        # 1. Classification
        t0 = time.perf_counter()
        class_res = classifier.predict(image)
        t_class = time.perf_counter() - t0
        PIPELINE_LATENCY.labels(step="classification", backend=active_backend).observe(t_class)
        MODEL_CONFIDENCE.labels(model_type="classifier").set(class_res["confidence"])
        
        # 2. Layout Detection
        t0 = time.perf_counter()
        regions = detector.detect(image, conf_threshold=conf_threshold)
        t_detect = time.perf_counter() - t0
        PIPELINE_LATENCY.labels(step="detection", backend=active_backend).observe(t_detect)
        if regions:
            mean_conf = float(np.mean([r["confidence"] for r in regions]))
            MODEL_CONFIDENCE.labels(model_type="detector").set(mean_conf)
        
        # 3. Crop OCR and Global OCR (Single-Pass)
        t0 = time.perf_counter()
        ocr_regions, full_text = ocr.process_ocr(image, regions)
        t_ocr = time.perf_counter() - t0
        PIPELINE_LATENCY.labels(step="ocr", backend=active_backend).observe(t_ocr)
        
        total_time = time.perf_counter() - start_time
        PIPELINE_LATENCY.labels(step="total", backend=active_backend).observe(total_time)
        
        # Log telemetry details (PII stripped via logger scrubber)
        log.info(
            "Document processing completed successfully",
            backend=active_backend,
            classifier_version=active_classifier,
            detector_version=active_detector,
            duration_total=total_time,
            duration_classification=t_class,
            duration_detection=t_detect,
            duration_ocr=t_ocr,
            classification=class_res["predicted_class"],
            classification_confidence=class_res["confidence"],
            regions_detected=len(regions)
        )
        
        return {
            "metadata": {
                "backend": active_backend,
                "classifier_version": active_classifier,
                "detector_version": active_detector,
                "image_width": image.width,
                "image_height": image.height,
                "processing_time_sec": total_time
            },
            "classification": class_res,
            "regions": ocr_regions,
            "full_text": full_text
        }

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Pipeline execution failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal pipeline failure: {str(e)}"
        )

@router.get("/config", tags=["Configuration"])
def get_current_config():
    """Retrieves current serving backend settings and active model versions."""
    active_config = get_active_config()
    return {
        "active_backend": active_config["active_backend"],
        "classifier_version": active_config["classifier_version"],
        "detector_version": active_config["detector_version"],
        "settings_defaults": {
            "ACTIVE_BACKEND": settings.ACTIVE_BACKEND,
            "CLASSIFIER_VERSION": settings.CLASSIFIER_VERSION,
            "DETECTOR_VERSION": settings.DETECTOR_VERSION
        }
    }

@router.post("/config", tags=["Configuration"])
def update_current_config(config: ConfigUpdateSchema):
    """Hot-swaps serving backend or model versions in memory for subsequent requests."""
    manager = get_pipeline_manager()
    
    active_config = get_active_config()
    new_backend = config.active_backend or active_config["active_backend"]
    new_classifier = config.classifier_version or active_config["classifier_version"]
    new_detector = config.detector_version or active_config["detector_version"]
    
    try:
        # Load pipeline with new settings to cache them
        manager.get_pipeline(
            backend=new_backend,
            classifier_version=new_classifier,
            detector_version=new_detector
        )
        
        # Save to the shared JSON configuration file
        save_active_config(
            backend=new_backend,
            classifier_version=new_classifier,
            detector_version=new_detector
        )
            
        logger.info(
            "Serving configuration updated successfully",
            backend=new_backend,
            classifier_version=new_classifier,
            detector_version=new_detector
        )
        
        updated_config = get_active_config()
        return {
            "status": "success",
            "message": "Serving configuration updated",
            "config": {
                "active_backend": updated_config["active_backend"],
                "classifier_version": updated_config["classifier_version"],
                "detector_version": updated_config["detector_version"]
            }
        }
    except Exception as e:
        logger.exception("Serving configuration hot-swap failed")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to hot-swap config: {str(e)}"
        )
