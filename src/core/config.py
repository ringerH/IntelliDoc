import os
import json
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base workspace directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Document Intelligence Service"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    ENV: str = "development"
    
    # Model Serving Settings
    MODEL_DIR: Path = BASE_DIR / "models"
    ACTIVE_BACKEND: Literal["pytorch", "onnx"] = "onnx"
    
    # Model Versions
    CLASSIFIER_VERSION: str = "v1"
    DETECTOR_VERSION: str = "v1"
    OCR_VERSION: str = "v1"
    
    # Observability
    LOG_LEVEL: str = "INFO"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()

def get_active_config() -> dict:
    """Reads the active configuration from the shared JSON file, falling back to settings defaults."""
    config_path = settings.MODEL_DIR / "active_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
                return {
                    "active_backend": data.get("active_backend") or settings.ACTIVE_BACKEND,
                    "classifier_version": data.get("classifier_version") or settings.CLASSIFIER_VERSION,
                    "detector_version": data.get("detector_version") or settings.DETECTOR_VERSION,
                }
        except Exception:
            pass # Fall back to settings on read error or corrupted json
            
    return {
        "active_backend": settings.ACTIVE_BACKEND,
        "classifier_version": settings.CLASSIFIER_VERSION,
        "detector_version": settings.DETECTOR_VERSION,
    }

def save_active_config(backend: str, classifier_version: str, detector_version: str):
    """Saves the configuration parameters to the shared JSON file so all workers pick them up."""
    config_path = settings.MODEL_DIR / "active_config.json"
    data = {
        "active_backend": backend,
        "classifier_version": classifier_version,
        "detector_version": detector_version,
    }
    with open(config_path, "w") as f:
        json.dump(data, f, indent=4)
