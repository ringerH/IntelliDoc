import os
import threading
from pathlib import Path
from PIL import Image
from src.core.config import settings, get_active_config
from src.services.classification import DocumentClassifier
from src.services.detection import DocumentDetector
from src.services.ocr import DocumentOCR

class PipelineManager:
    def __init__(self):
        # In-memory model caches (keys are (backend, version))
        self._classifiers = {}
        self._detectors = {}
        self._ocr = None
        self._lock = threading.Lock()
        
        # Pre-initialize models on manager boot, but fail gracefully if not found
        try:
            active_config = get_active_config()
            self.get_pipeline(
                backend=active_config["active_backend"],
                classifier_version=active_config["classifier_version"],
                detector_version=active_config["detector_version"]
            )
        except Exception as e:
            # Import logger locally if needed, or print fallback to stdout.
            # We do not raise so that singleton instantiation succeeds and doesn't loop.
            print(f"Failed to pre-load default models on startup: {str(e)}")

    def is_ready(self, backend: str, classifier_version: str, detector_version: str) -> bool:
        """Checks if the requested models are already loaded and cached in memory."""
        with self._lock:
            classifier_key = (backend, classifier_version)
            detector_key = (backend, detector_version)
            return (
                classifier_key in self._classifiers and
                detector_key in self._detectors and
                self._ocr is not None
            )

    def get_pipeline(
        self,
        backend: str,
        classifier_version: str,
        detector_version: str
    ):
        """Thread-safely loads and returns model instances, caching them in memory."""
        with self._lock:
            # 1. Initialize OCR if not loaded (OCR versioning is simplified to easyocr singleton)
            if self._ocr is None:
                self._ocr = DocumentOCR()
                
            classifier_key = (backend, classifier_version)
            detector_key = (backend, detector_version)
            
            # 2. Check cache or load classification model locally
            classifier = self._classifiers.get(classifier_key)
            if classifier is None:
                classifier_dir = settings.MODEL_DIR / "classifier" / classifier_version
                print(f"Loading Classifier: backend={backend}, version={classifier_version} from {classifier_dir}")
                classifier = DocumentClassifier(classifier_dir, backend=backend)
                
            # 3. Check cache or load detection model locally
            detector = self._detectors.get(detector_key)
            if detector is None:
                detector_dir = settings.MODEL_DIR / "detector" / detector_version
                print(f"Loading Detector: backend={backend}, version={detector_version} from {detector_dir}")
                detector = DocumentDetector(detector_dir, backend=backend)
                
            # If both succeeded without raising exception, commit to cache
            self._classifiers[classifier_key] = classifier
            self._detectors[detector_key] = detector
            
            return classifier, detector, self._ocr

    def process_document(
        self,
        image: Image.Image,
        backend: str = None,
        classifier_version: str = None,
        detector_version: str = None,
        conf_threshold: float = 0.25
    ) -> dict:
        """Runs the entire pipeline on the document image:
        1. Classify Document Type
        2. Detect Regions (tables, text blocks, signatures)
        3. Extract Text from Detected Regions (OCR)
        4. Extract Full Text (global OCR)
        """
        # Resolve configurations
        active_config = get_active_config()
        backend = backend or active_config["active_backend"]
        classifier_version = classifier_version or active_config["classifier_version"]
        detector_version = detector_version or active_config["detector_version"]
        
        # Fetch cached/reloaded model instances
        classifier, detector, ocr = self.get_pipeline(
            backend=backend,
            classifier_version=classifier_version,
            detector_version=detector_version
        )
        
        # 1. Run Classification
        classification_res = classifier.predict(image)
        
        # 2. Run Object Detection
        regions = detector.detect(image, conf_threshold=conf_threshold)
        
        # 3. Run OCR in a single-pass and map layout regions
        ocr_regions, full_text = ocr.process_ocr(image, regions)
        
        return {
            "metadata": {
                "backend": backend,
                "classifier_version": classifier_version,
                "detector_version": detector_version,
                "image_width": image.width,
                "image_height": image.height
            },
            "classification": classification_res,
            "regions": ocr_regions,
            "full_text": full_text
        }

# Global Singleton Manager
pipeline_manager = None

def get_pipeline_manager() -> PipelineManager:
    global pipeline_manager
    if pipeline_manager is None:
        pipeline_manager = PipelineManager()
    return pipeline_manager
