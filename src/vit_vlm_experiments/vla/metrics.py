from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .storage import atomic_write_text


LOGGER = logging.getLogger(__name__)
WRITE_METRIC_ATTEMPTS = 3


def regression_metrics(
    y_true,
    y_pred,
) -> dict:
    y_true = np.asarray(
        y_true,
        dtype=np.float32,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=np.float32,
    )

    if y_true.shape != y_pred.shape:
        LOGGER.error(
            "Prediction/target shape mismatch: "
            "%s vs %s",
            y_pred.shape,
            y_true.shape,
        )
        raise ValueError(
            "Prediction/target shape mismatch: "
            f"{y_pred.shape} vs {y_true.shape}"
        )

    error = y_true - y_pred

    per_dim_mae = np.mean(
        np.abs(error),
        axis=0,
    )
    target_scale = np.mean(
        np.abs(y_true),
        axis=0,
    )
    relative_mae = (
        per_dim_mae
        / np.maximum(target_scale, 1e-8)
    )

    return {
        "mae": float(
            np.mean(np.abs(error))
        ),
        "mse": float(
            np.mean(error ** 2)
        ),
        "per_dim": [
            {
                "action_dim": int(i),
                "mae": float(per_dim_mae[i]),
                "relative_mae":
                    float(relative_mae[i]),
            }
            for i in range(y_true.shape[1])
        ],
    }


def save_metrics(
    metrics: dict,
    output_path: str | Path,
    metadata: dict | None = None,
) -> bool:
    """Save metrics atomically; log disk failures and let the run continue."""
    output_path = Path(output_path)
    payload = {**metrics, "metadata": metadata} if metadata is not None else metrics
    for attempt in range(1, WRITE_METRIC_ATTEMPTS + 1):
        try:
            serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
            atomic_write_text(output_path, serialized)
        except ValueError as exc:
            LOGGER.exception(
                "Could not serialize metrics to JSON (attempt %d/%d): %s",
                attempt,
                WRITE_METRIC_ATTEMPTS,
                exc,
            )
            return False
        except OSError:
            LOGGER.exception("Could not save metrics to %s; continuing", output_path)
            return False

    LOGGER.info("Saved metrics to %s", output_path)
    return True
