import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data import SmolVLAImageDataset
from ..metrics import regression_metrics


LOGGER = logging.getLogger(__name__)


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

    from lerobot.policies.factory import (
        make_pre_post_processors,
    )
    from lerobot.policies.smolvla.modeling_smolvla import (
        SmolVLAPolicy,
    )

    dataset_config = (
        context.config["dataset"]
    )

    cache = context.cache
    val_indices = context.split["evaluation_indices"]
    if checkpoint == "lerobot/smolvla_libero" and dataset_config["repo_id"] == "lerobot/libero":
        LOGGER.warning(
            "lerobot/smolvla_libero was trained on LIBERO without this project's "
            "episode split. The same validation frames will be scored, but its "
            "result is not a held-out measurement."
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
            context.dataset,
            image_keys,
            indices=val_indices,
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

    state_feature = policy.config.input_features.get("observation.state")
    action_feature = policy.config.output_features.get("action")
    state_shape = None if state_feature is None else tuple(state_feature.shape)
    action_shape = None if action_feature is None else tuple(action_feature.shape)
    dataset_state_shape = (cache["states"].shape[1],)
    dataset_action_shape = (cache["actions"].shape[1],)
    if (
        state_shape != dataset_state_shape
        or action_shape != dataset_action_shape
    ):
        raise ValueError(
            f"SmolVLA checkpoint {checkpoint!r} expects state {state_shape} "
            f"and action {action_shape}, but the dataset has state "
            f"{dataset_state_shape} and action {dataset_action_shape}. "
            "Choose a checkpoint trained for these LIBERO features "
            "(for example, lerobot/smolvla_libero)."
        )

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

    y_true = cache["actions"][val_indices]
    y_pred = np.empty_like(
        y_true,
        dtype=np.float32,
    )

    LOGGER.info("Evaluating SmolVLA checkpoint %s on %d frames", checkpoint, len(y_true))

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    processed_frames = 0
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
            batch_end = processed_frames + len(index_list)
            if not np.array_equal(index_list, val_indices[processed_frames:batch_end]):
                raise ValueError("SmolVLA images do not match the shared validation frame order.")

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

            if prediction.shape != (len(index_list), y_true.shape[1]):
                raise ValueError(
                    f"SmolVLA checkpoint {checkpoint!r} returned actions with "
                    f"shape {tuple(prediction.shape)}; expected "
                    f"({len(index_list)}, {y_true.shape[1]})."
                )
            y_pred[processed_frames:batch_end] = (
                prediction
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            processed_frames = batch_end

    if processed_frames != len(y_true):
        raise ValueError("SmolVLA did not predict every shared validation frame.")

    return regression_metrics(
        y_true,
        y_pred,
    )

