from prometheus_client import Counter, Histogram, Gauge

# 1. Total HTTP Requests Counter
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP Requests",
    ["method", "endpoint", "status_code"]
)

# 2. HTTP Request Duration Histogram
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP Request Latency in seconds",
    ["endpoint"]
)

# 3. Model Inference Pipeline Latencies
PIPELINE_LATENCY = Histogram(
    "pipeline_step_duration_seconds",
    "Inference latency per pipeline component in seconds",
    ["step", "backend"]
)

# 4. Model Confidence Distribution Gauge (drift signal)
# Track confidence of classification and region detections
MODEL_CONFIDENCE = Gauge(
    "model_confidence_score",
    "Track model prediction confidence scores (drift indicator)",
    ["model_type"]
)
