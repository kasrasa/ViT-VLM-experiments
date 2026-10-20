from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import logging
from pathlib import Path
import random

import numpy as np
import torch

from .config import resolve_device
from .data import (
    build_non_image_cache,
    create_or_load_split,
)
from .experiments import EXPERIMENTS
from .logging_config import configure_logging
from .metrics import save_metrics


LOGGER = logging.getLogger(__name__)


@dataclass
class ExperimentContext:
    config: dict
    dataset: object
    cache: dict
    split: dict
    device: str


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def build_context(
    config: dict,
) -> ExperimentContext:
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
    )

    seed = int(
        config.get("seed", 42)
    )
    seed_everything(seed)

    dataset_config = config["dataset"]

    dataset = LeRobotDataset(
        dataset_config["repo_id"]
    )
    LOGGER.info("Loaded dataset %s (%d frames)", dataset_config["repo_id"], len(dataset))

    cache = build_non_image_cache(
        dataset,
        state_key=dataset_config[
            "state_key"
        ],
        action_key=dataset_config[
            "action_key"
        ],
        episode_key=dataset_config[
            "episode_key"
        ],
        task_key=dataset_config[
            "task_key"
        ],
    )
    LOGGER.info("Prepared non-image data for %d frames", len(cache["episode_ids"]))

    split = create_or_load_split(
        cache,
        config["split"],
    )

    return ExperimentContext(
        config=config,
        dataset=dataset,
        cache=cache,
        split=split,
        device=resolve_device(
            config.get(
                "device",
                "auto",
            )
        ),
    )


def experiment_config(
    config: dict,
    experiment_name: str,
) -> dict:
    return (
        config
        .get("experiments", {})
        .get(
            experiment_name,
            {},
        )
    )


def experiment_metadata(context: ExperimentContext, name: str, params: dict) -> dict:
    try:
        package_version = version("vit-vlm-experiments")
    except PackageNotFoundError:
        package_version = None

    split_config = context.config["split"]
    return {
        "experiment": name,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": package_version,
        "model": {
            "name": name,
            "implementation_version": package_version,
            "checkpoint": params.get("checkpoint"),
        },
        "dataset": context.config["dataset"],
        "loader": context.config.get("loader", {}),
        "split": {
            "strategy": split_config.get("strategy", "task_stratified_episode"),
            "train_fraction": float(split_config.get("train_fraction", 0.8)),
            "seed": int(split_config.get("seed", 42)),
            "train_episodes": context.split["train_episodes"].tolist(),
            "val_episodes": context.split["val_episodes"].tolist(),
        },
        "seed": int(context.config.get("seed", 42)),
        "device": context.device,
        "hyperparameters": params,
    }


def run_from_config(
    config: dict,
    only=None,
):
    configure_logging(config)
    selected = (
        only
        if only
        else config.get("run", [])
    )

    if not selected:
        raise ValueError(
            "No experiments selected."
        )

    unknown = [
        name
        for name in selected
        if name not in EXPERIMENTS
    ]
    if unknown:
        raise ValueError(
            "Unknown experiment(s): "
            + ", ".join(unknown)
        )

    context = build_context(config)

    output_dir = Path(
        config.get(
            "output_dir",
            "outputs/vla",
        )
    )
    LOGGER.info("Selected experiments: %s", ", ".join(selected))
    LOGGER.info(
        "Device: %s; train: %d episodes / %d frames; "
        "validation: %d episodes / %d frames",
        context.device,
        len(context.split["train_episodes"]),
        len(context.split["train_indices"]),
        len(context.split["val_episodes"]),
        len(context.split["val_indices"]),
    )

    results = {}

    for name in selected:
        LOGGER.info("Starting experiment %s", name)

        params = experiment_config(config, name)

        result = EXPERIMENTS[name](
            context,
            params,
        )
        results[name] = result

        saved = save_metrics(
            result,
            output_dir / f"{name}.json",
            metadata=experiment_metadata(context, name, params),
        )

        LOGGER.info(
            "Completed %s: MAE=%.6f MSE=%.6f%s",
            name, result["mae"], result["mse"],
            " (metrics file unavailable)" if not saved else "",
        )

    return results
