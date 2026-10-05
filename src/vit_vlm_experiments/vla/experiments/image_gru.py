import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import (
    ImageTemporalDataset,
    make_normalization_stats,
)
from ..metrics import regression_metrics
from ..models import ImageGRUActionPolicy


LOGGER = logging.getLogger(__name__)


def _collate(batch):
    num_cameras = len(
        batch[0]["images"]
    )

    return {
        "sequence": torch.stack(
            [
                item["sequence"]
                for item in batch
            ]
        ),
        "action": torch.stack(
            [
                item["action"]
                for item in batch
            ]
        ),
        "action_norm": torch.stack(
            [
                item["action_norm"]
                for item in batch
            ]
        ),
        "images": [
            torch.stack(
                [
                    item["images"][
                        camera_index
                    ]
                    for item in batch
                ]
            )
            for camera_index in range(
                num_cameras
            )
        ],
    }


def run_image_gru(
    context,
    config,
):
    history_k = int(
        config.get(
            "history_k",
            2,
        )
    )
    image_keys = (
        context.config["dataset"][
            "image_keys"
        ]
    )

    stats = make_normalization_stats(
        context.cache,
        context.split["train_indices"],
    )

    train_dataset = ImageTemporalDataset(
        context.dataset,
        context.cache,
        context.split["train_indices"],
        stats,
        history_k,
        image_keys,
    )
    val_dataset = ImageTemporalDataset(
        context.dataset,
        context.cache,
        context.split["val_indices"],
        stats,
        history_k,
        image_keys,
    )

    batch_size = int(
        config.get(
            "batch_size",
            32,
        )
    )
    num_workers = int(
        context.config
        .get("loader", {})
        .get("num_workers", 2)
    )
    pin_memory = bool(
        context.config
        .get("loader", {})
        .get("pin_memory", True)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
    )

    sample = train_dataset[0]

    model = ImageGRUActionPolicy(
        sequence_input_dim=
            sample["sequence"].shape[-1],
        action_dim=
            sample["action"].shape[-1],
        num_cameras=len(image_keys),
        image_embedding_dim=int(
            config.get(
                "image_embedding_dim",
                256,
            )
        ),
        hidden_dim=int(
            config.get(
                "hidden_dim",
                256,
            )
        ),
        num_layers=int(
            config.get(
                "num_layers",
                3,
            )
        ),
        dropout=float(
            config.get(
                "dropout",
                0.2,
            )
        ),
    ).to(context.device)

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter
            in model.parameters()
            if parameter.requires_grad
        ],
        lr=float(
            config.get("lr", 1e-3)
        ),
        weight_decay=float(
            config.get(
                "weight_decay",
                1e-4,
            )
        ),
    )

    epochs = int(config.get("epochs", 10))
    LOGGER.info("Image GRU: %d training windows, %d validation windows", len(train_dataset), len(val_dataset))
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        trained_windows = 0

        for batch in train_loader:
            sequence = batch[
                "sequence"
            ].to(context.device)
            images = [
                image.to(context.device)
                for image
                in batch["images"]
            ]
            target = batch[
                "action_norm"
            ].to(context.device)

            prediction = model(
                sequence,
                images,
            )
            loss = (
                torch.nn.functional
                .mse_loss(
                    prediction,
                    target,
                )
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(sequence)
            trained_windows += len(sequence)

        LOGGER.info("Image GRU epoch %d/%d: training MSE=%.6f", epoch, epochs, total_loss / trained_windows)

    y_true = []
    y_pred = []

    model.eval()
    with torch.inference_mode():
        for batch in val_loader:
            sequence = batch[
                "sequence"
            ].to(context.device)
            images = [
                image.to(context.device)
                for image
                in batch["images"]
            ]

            pred_norm = model(
                sequence,
                images,
            ).cpu()
            pred = (
                pred_norm
                * stats.action_std
                + stats.action_mean
            )

            y_true.append(
                batch["action"].numpy()
            )
            y_pred.append(
                pred.numpy()
            )

    return regression_metrics(
        np.concatenate(
            y_true,
            axis=0,
        ),
        np.concatenate(
            y_pred,
            axis=0,
        ),
    )
