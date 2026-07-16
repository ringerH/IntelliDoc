import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from src.main import app
from src.core.config import settings

@pytest.fixture
def client():
    # Use TestClient with FastAPI application context
    with TestClient(app) as c:
        yield c

@pytest.fixture
def dummy_image_file():
    """Generates a dummy 640x640 JPEG image in memory for API testing."""
    image = Image.new("RGB", (640, 640), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([150, 150, 450, 450], outline="blue", fill="lightgrey")
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_liveness_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_readiness_endpoint(client):
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text or "http_request_duration_seconds" in response.text

def test_config_endpoints(client):
    # Test GET config
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "active_backend" in data
    
    # Test POST config update (swap to PyTorch)
    update_res = client.post("/api/v1/config", json={"active_backend": "pytorch"})
    assert update_res.status_code == 200
    assert update_res.json()["config"]["active_backend"] == "pytorch"
    
    # Re-fetch config to check change
    response = client.get("/api/v1/config")
    assert response.json()["active_backend"] == "pytorch"
    
    # Reset config back to ONNX
    update_res = client.post("/api/v1/config", json={"active_backend": "onnx"})
    assert update_res.status_code == 200
    assert update_res.json()["config"]["active_backend"] == "onnx"

def test_process_document_endpoint(client, dummy_image_file):
    # Upload image
    files = {"file": ("test_doc.jpg", dummy_image_file, "image/jpeg")}
    response = client.post(
        "/api/v1/process",
        files=files,
        params={"backend": "onnx", "conf_threshold": 0.2}
    )
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data
    assert "classification" in data
    assert "regions" in data
    assert "full_text" in data
    
    # Verify response schema fields
    assert data["metadata"]["backend"] == "onnx"
    assert "predicted_class" in data["classification"]
    assert "confidence" in data["classification"]

def test_process_invalid_file_type(client):
    # Upload text file instead of image
    files = {"file": ("test_doc.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/api/v1/process", files=files)
    assert response.status_code == 400
    assert "must be a valid image" in response.json()["detail"]
