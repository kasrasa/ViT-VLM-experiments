from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys


LOGGER_NAME = "vit_vlm_experiments.vla"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(config: dict | None = None) -> logging.Logger:
    """Configure this package's console and optional rotating file logger."""
    options = (config or {}).get("logging", {})
    if not isinstance(options, dict):
        raise ValueError("logging must be a YAML mapping.")

    level_name = str(options.get("level", "INFO")).upper()
    if level_name not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(f"Unsupported logging level: {level_name}")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level_name))
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_vla_managed", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console._vla_managed = True
    logger.addHandler(console)

    default_file = (
        Path(config.get("output_dir", "outputs/vla")) / "run.log"
        if config is not None
        else None
    )
    log_file = options.get("file", default_file)
    if log_file:
        path = Path(log_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                path, maxBytes=5 * 1024 * 1024, backupCount=2,
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Cannot open log file %s: %s", path, exc)
        else:
            file_handler.setFormatter(formatter)
            file_handler._vla_managed = True
            logger.addHandler(file_handler)

    return logger
