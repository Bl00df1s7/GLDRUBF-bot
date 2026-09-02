"""Structured per-run logging."""

import json
import logging
import os
import uuid
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """Format log records as compact JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
        }
        return json.dumps(payload, ensure_ascii=True)


def configure_logging() -> tuple:
    """Configure daily file logging and return logger plus correlation ID."""
    correlation_id = str(uuid.uuid4())
    log_dir = os.environ.get("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"bot_{datetime.utcnow():%Y%m%d}.log")
    handler = logging.FileHandler(filename)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("gldrubf")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info("run_started", extra={"correlation_id": correlation_id})
    return logger, correlation_id
