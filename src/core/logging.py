import logging
import sys
import uuid
from contextvars import ContextVar
import structlog

# Context variable to store the request ID for the current async task context
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

def get_request_id() -> str:
    return request_id_var.get()

def set_request_id(request_id: str) -> None:
    request_id_var.set(request_id)

def pii_scrubber(logger, method_name, event_dict):
    """Processor to scrub PII from logs. Ensure no raw text, filenames, or payloads are logged."""
    # List of keys we want to completely omit or scrub if they appear in logs
    pii_keys = ["text", "ocr_text", "raw_text", "payload", "image", "file", "filename"]
    for key in list(event_dict.keys()):
        if any(pii_word in key.lower() for pii_word in pii_keys):
            event_dict[key] = "[REDACTED_PII]"
    return event_dict

def configure_logging(log_level: str = "INFO"):
    # Configure root standard logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO)
    )

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        # Custom PII scrubbing processor
        pii_scrubber,
        structlog.processors.JSONRenderer()
    ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str = None) -> structlog.stdlib.BoundLogger:
    """Returns a logger bound with the current request_id if available."""
    logger = structlog.get_logger(name)
    request_id = get_request_id()
    if request_id:
        logger = logger.bind(request_id=request_id)
    return logger
