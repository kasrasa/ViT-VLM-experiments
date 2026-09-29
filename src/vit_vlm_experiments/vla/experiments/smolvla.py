import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data import (
    SmolVLAImageDataset,
    build_non_image_cache,
)
from ..metrics import regression_metrics


def run_smolvla_eval(
    context,
    config,
):
    checkpoint = config.get(
        "checkpoint"
    )

    if not checkpoint:
        raise ValueError(
            "Set experiments.smolvla_eval."
            "checkpoint in the YAML config."
        )

    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
    )
    from lerobot.policies.factory import (
        make_pre_post_processors,
    )
    from lerobot.policies.smolvla.modeling_smolvla import (
        SmolVLAPolicy,
    )

    dataset_config = (
        context.config["dataset"]
    )

    val_dataset = LeRobotDataset(
        dataset_config["repo_id"],
        episodes=(
            context.split[
                "val_episodes"
            ].tolist()
        ),
    )

    cache = build_non_image_cache(
        val_dataset,
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

    image_keys = dataset_config[
        "image_keys"
    ]
    model_image_keys = config.get(
        "model_image_keys",
        image_keys,
    )

    if (
        len(image_keys)
        != len(model_image_keys)
    ):
        raise ValueError(
            "dataset.image_keys and "
            "smolvla_eval.model_image_keys "
            "must have the same length."
        )

    image_dataset = (
        SmolVLAImageDataset(
            val_dataset,
            image_keys,
        )
    )

    num_workers = int(
        config.get(
            "num_workers",
            2,
        )
    )

    loader = DataLoader(
        image_dataset,
        batch_size=int(
            config.get(
                "batch_size",
                8,
            )
        ),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(
            num_workers > 0
        ),
    )

    policy = (
        SmolVLAPolicy
        .from_pretrained(
            checkpoint
        )
        .to(context.device)
    )
    policy.eval()

    (
        preprocessor,
        postprocessor,
    ) = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {
                "device":
                    context.device
            }
        },
    )

    y_true = cache["actions"]
    y_pred = np.empty_like(
        y_true,
        dtype=np.float32,
    )

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=(
                "SmolVLA batched "
                "offline evaluation"
            ),
        ):
            indices = batch[
                "dataset_index"
            ].long()
            index_list = (
                indices
                .cpu()
                .tolist()
            )

            raw_batch = {
                "observation.state":
                    cache["states"][
                        indices
                    ],
                "task": [
                    cache["tasks"][i]
                    for i
                    in index_list
                ],
            }

            for (
                source_key,
                model_key,
            ) in zip(
                image_keys,
                model_image_keys,
            ):
                raw_batch[model_key] = (
                    batch[source_key]
                )

            processed_batch = (
                preprocessor(
                    raw_batch
                )
            )

            action_chunk = (
                policy
                .predict_action_chunk(
                    processed_batch
                )
            )

            prediction = (
                postprocessor(
                    action_chunk[
                        :,
                        0,
                        :,
                    ]
                )
            )

            y_pred[index_list] = (
                prediction
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    return regression_metrics(
        y_true,
        y_pred,
    )
