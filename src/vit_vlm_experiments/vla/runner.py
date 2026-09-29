from __future__ import annotations

from dataclasses import dataclass
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
from .metrics import save_metrics


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


def run_from_config(
    config: dict,
    only=None,
):
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
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Device: {context.device}"
    )
    print(
        "Train episodes:",
        len(
            context.split[
                "train_episodes"
            ]
        ),
    )
    print(
        "Validation episodes:",
        len(
            context.split[
                "val_episodes"
            ]
        ),
    )
    print(
        "Train frames:",
        len(
            context.split[
                "train_indices"
            ]
        ),
    )
    print(
        "Validation frames:",
        len(
            context.split[
                "val_indices"
            ]
        ),
    )

    results = {}

    for name in selected:
        print(
            f"\n=== {name} ==="
        )

        result = EXPERIMENTS[name](
            context,
            experiment_config(
                config,
                name,
            ),
        )
        results[name] = result

        save_metrics(
            result,
            output_dir
            / f"{name}.json",
        )

        print(
            f"MAE: {result['mae']:.6f}"
        )
        print(
            f"MSE: {result['mse']:.6f}"
        )

    return results
