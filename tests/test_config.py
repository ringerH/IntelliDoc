from src.core.config import settings

def test_settings_load():
    assert settings.APP_NAME == "Document Intelligence Service"
    assert settings.APP_VERSION == "1.0.0"
    assert settings.API_PREFIX == "/api/v1"
    assert settings.ACTIVE_BACKEND in ("pytorch", "onnx")
