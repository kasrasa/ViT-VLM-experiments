import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import (
    TemporalDataset,
    make_normalization_stats,
)
from ..metrics import regression_metrics
from ..models import (
    GRUActionPolicy,
    TaskConditionedGRUActionPolicy,
)


def _train_and_evaluate(
    context,
    config,
    task_conditioned: bool,
):
    history_k = int(
        config.get(
            "history_k",
            2,
        )
    )

    stats = make_normalization_stats(
        context.cache,
        context.split["train_indices"],
    )

    train_dataset = TemporalDataset(
        context.cache,
        context.split["train_indices"],
        stats,
        history_k,
        include_task=task_conditioned,
    )
    val_dataset = TemporalDataset(
        context.cache,
        context.split["val_indices"],
        stats,
        history_k,
        include_task=task_conditioned,
    )

    batch_size = int(
        config.get(
            "batch_size",
            256,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    sample = train_dataset[0]

    model_kwargs = {
        "input_dim":
            sample["sequence"].shape[-1],
        "action_dim":
            sample["action"].shape[-1],
        "hidden_dim":
            int(
                config.get(
                    "hidden_dim",
                    256,
                )
            ),
        "num_layers":
            int(
                config.get(
                    "num_layers",
                    3,
                )
            ),
        "dropout":
            float(
                config.get(
                    "dropout",
                    0.2,
                )
            ),
    }

    if task_conditioned:
        model = (
            TaskConditionedGRUActionPolicy(
                **model_kwargs,
                num_tasks=int(
                    context.cache[
                        "task_ids"
                    ].max()
                ) + 1,
                task_embedding_dim=int(
                    config.get(
                        "task_embedding_dim",
                        64,
                    )
                ),
            )
        )
    else:
        model = GRUActionPolicy(
            **model_kwargs
        )

    model = model.to(context.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
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

    for _ in range(
        int(config.get("epochs", 10))
    ):
        model.train()

        for batch in train_loader:
            sequence = batch[
                "sequence"
            ].to(context.device)
            target = batch[
                "action_norm"
            ].to(context.device)

            if task_conditioned:
                prediction = model(
                    sequence,
                    batch["task_id"].to(
                        context.device
                    ),
                )
            else:
                prediction = model(
                    sequence
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

    y_true = []
    y_pred = []

    model.eval()
    with torch.inference_mode():
        for batch in val_loader:
            sequence = batch[
                "sequence"
            ].to(context.device)

            if task_conditioned:
                pred_norm = model(
                    sequence,
                    batch["task_id"].to(
                        context.device
                    ),
                )
            else:
                pred_norm = model(
                    sequence
                )

            pred = (
                pred_norm.cpu()
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


def run_gru(
    context,
    config,
):
    return _train_and_evaluate(
        context,
        config,
        task_conditioned=False,
    )


def run_task_gru(
    context,
    config,
):
    return _train_and_evaluate(
        context,
        config,
        task_conditioned=True,
    )
