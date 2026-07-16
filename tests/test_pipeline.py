import pytest
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from src.core.config import settings
from src.services.pipeline import get_pipeline_manager
from src.services.classification import DocumentClassifier
from src.services.detection import DocumentDetector
from src.services.ocr import DocumentOCR

@pytest.fixture
def dummy_image():
    """Generates a dummy 640x640 RGB Pillow image with some text/shapes for pipeline validation."""
    image = Image.new("RGB", (640, 640), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    # Draw simple shapes representing a table (dining table class)
    draw.rectangle([100, 100, 500, 400], outline="black", fill="grey")
    # Draw a line/signature representation (tie class)
    draw.line([200, 500, 400, 500], fill="black", width=5)
    return image

def test_classifier_pytorch(dummy_image):
    classifier_dir = settings.MODEL_DIR / "classifier" / "v1"
    classifier = DocumentClassifier(classifier_dir, backend="pytorch")
    res = classifier.predict(dummy_image)
    assert "predicted_class" in res
    assert res["predicted_class"] in ["invoice", "receipt", "id", "form"]
    assert 0.0 <= res["confidence"] <= 1.0

def test_classifier_onnx(dummy_image):
    classifier_dir = settings.MODEL_DIR / "classifier" / "v1"
    classifier = DocumentClassifier(classifier_dir, backend="onnx")
    res = classifier.predict(dummy_image)
    assert "predicted_class" in res
    assert res["predicted_class"] in ["invoice", "receipt", "id", "form"]
    assert 0.0 <= res["confidence"] <= 1.0

def test_detector_pytorch(dummy_image):
    detector_dir = settings.MODEL_DIR / "detector" / "v1"
    detector = DocumentDetector(detector_dir, backend="pytorch")
    res = detector.detect(dummy_image, conf_threshold=0.1)
    assert isinstance(res, list)
    if len(res) > 0:
        for detection in res:
            assert "box" in detection
            assert "class" in detection
            assert len(detection["box"]) == 4

def test_detector_onnx(dummy_image):
    detector_dir = settings.MODEL_DIR / "detector" / "v1"
    detector = DocumentDetector(detector_dir, backend="onnx")
    res = detector.detect(dummy_image, conf_threshold=0.1)
    assert isinstance(res, list)
    if len(res) > 0:
        for detection in res:
            assert "box" in detection
            assert "class" in detection
            assert len(detection["box"]) == 4

def test_ocr_extraction(dummy_image):
    ocr = DocumentOCR()
    text = ocr.extract_text(dummy_image)
    assert isinstance(text, str)
    
    # Test sub-region extraction
    regions = [{"box": [100, 100, 500, 400], "class": "table"}]
    extracted = ocr.extract_regions(dummy_image, regions)
    assert len(extracted) == 1
    assert "text" in extracted[0]

def test_pipeline_integration(dummy_image):
    manager = get_pipeline_manager()
    
    # Process using ONNX backend (default)
    res_onnx = manager.process_document(dummy_image, backend="onnx")
    assert res_onnx["metadata"]["backend"] == "onnx"
    assert "classification" in res_onnx
    assert "regions" in res_onnx
    assert "full_text" in res_onnx
    
    # Process using PyTorch backend
    res_pt = manager.process_document(dummy_image, backend="pytorch")
    assert res_pt["metadata"]["backend"] == "pytorch"
    assert "classification" in res_pt
    assert "regions" in res_pt
    assert "full_text" in res_pt
