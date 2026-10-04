from __future__ import annotations

import json
from pathlib import Path

import numpy as np


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
) -> None:
    output_path = Path(output_path)
    # should log if an error occurs and move on without raising an exception
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # should log if an error occurs and move on without raising an exception
    # there should be some traceablity of the model and settings like the dataset information
    # version of the model, hyperparameters, and any other relevant information should be included in the metrics file
    output_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )
