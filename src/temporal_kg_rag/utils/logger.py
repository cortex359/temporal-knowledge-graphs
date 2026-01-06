"""Logging configuration and utilities."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler

from temporal_kg_rag.config.settings import get_settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> None:
    """
    Set up logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ('json' or 'text')
        log_file: Optional path to log file
    """
    settings = get_settings()

    # Use provided values or fall back to settings
    level = log_level or settings.log_level
    format_type = log_format or settings.log_format

    # Convert log level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)

    # Set up console handler
    if format_type.lower() == "json":
        # JSON formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(console_handler)
    else:
        # Rich text formatting (default)
        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        console_handler.setFormatter(
            logging.Formatter("%(message)s", datefmt="[%X]")
        )
        root_logger.addHandler(console_handler)

    # Set up file handler if log_file is provided
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)

        if format_type.lower() == "json":
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        root_logger.addHandler(file_handler)

    # Suppress verbose logs from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured logger instance
    """
    # Ensure logging is set up
    if not logging.getLogger().handlers:
        setup_logging()

    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for adding context to log messages."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add extra context to log messages."""
        extra = kwargs.get("extra", {})
        if self.extra:
            extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def get_contextual_logger(name: str, **context: Any) -> LoggerAdapter:
    """
    Get a logger with additional context that will be included in all log messages.

    Args:
        name: Logger name
        **context: Additional context key-value pairs

    Returns:
        Logger adapter with context

    Example:
        >>> logger = get_contextual_logger(__name__, user_id="123", session="abc")
        >>> logger.info("User action")  # Will include user_id and session in logs
    """
    base_logger = get_logger(name)
    return LoggerAdapter(base_logger, context)
