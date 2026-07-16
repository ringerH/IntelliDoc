import io
from locust import HttpUser, task, between
from PIL import Image, ImageDraw

class DocumentIntelligenceUser(HttpUser):
    # Simulate a user thinking for 1 to 3 seconds between requests
    wait_time = between(1, 3)
    
    def on_start(self):
        """Generates a dummy document image file to reuse during load testing tasks."""
        image = Image.new("RGB", (640, 640), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle([150, 150, 450, 450], outline="blue", fill="lightgrey")
        
        # Save image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        self.image_bytes = img_byte_arr.read()

    @task(3)
    def process_document_onnx(self):
        """Simulates uploading a document using the ONNX backend (optimized)."""
        files = {
            "file": ("load_test_doc.jpg", self.image_bytes, "image/jpeg")
        }
        params = {
            "backend": "onnx",
            "conf_threshold": 0.25
        }
        with self.client.post("/api/v1/process", files=files, params=params, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # 429 is expected if rate limit is hit, do not log as a failure
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(1)
    def process_document_pytorch(self):
        """Simulates uploading a document using the Native PyTorch backend."""
        files = {
            "file": ("load_test_doc.jpg", self.image_bytes, "image/jpeg")
        }
        params = {
            "backend": "pytorch",
            "conf_threshold": 0.25
        }
        with self.client.post("/api/v1/process", files=files, params=params, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(2)
    def check_liveness(self):
        """Simulates liveness probe checks."""
        self.client.get("/api/v1/health")

    @task(2)
    def check_readiness(self):
        """Simulates readiness probe checks."""
        self.client.get("/api/v1/ready")
