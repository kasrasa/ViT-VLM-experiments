import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from ..data import (
    make_normalization_stats,
)
from ..metrics import regression_metrics
from ..models import StateMLP


def run_state_mlp(
    context,
    config,
):
    train_indices = (
        context.split["train_indices"]
    )
    val_indices = (
        context.split["val_indices"]
    )

    stats = make_normalization_stats(
        context.cache,
        train_indices,
    )

    train_states = (
        context.cache["states"][
            train_indices
        ]
    )
    train_actions = torch.as_tensor(
        context.cache["actions"][
            train_indices
        ],
        dtype=torch.float32,
    )

    train_states = (
        train_states
        - stats.state_mean
    ) / stats.state_std
    train_actions_norm = (
        train_actions
        - stats.action_mean
    ) / stats.action_std

    batch_size = int(
        config.get(
            "batch_size",
            256,
        )
    )

    loader = DataLoader(
        TensorDataset(
            train_states,
            train_actions_norm,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    model = StateMLP(
        state_dim=train_states.shape[-1],
        action_dim=train_actions.shape[-1],
        hidden_dim=int(
            config.get(
                "hidden_dim",
                256,
            )
        ),
    ).to(context.device)

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

        for states, targets in loader:
            states = states.to(
                context.device
            )
            targets = targets.to(
                context.device
            )

            predictions = model(states)
            loss = (
                torch.nn.functional
                .mse_loss(
                    predictions,
                    targets,
                )
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    val_states = (
        context.cache["states"][
            val_indices
        ]
    )
    val_states = (
        val_states
        - stats.state_mean
    ) / stats.state_std

    y_pred = []

    model.eval()
    with torch.inference_mode():
        for start in range(
            0,
            len(val_states),
            batch_size,
        ):
            batch = val_states[
                start:
                start + batch_size
            ].to(context.device)

            pred_norm = model(
                batch
            ).cpu()
            pred = (
                pred_norm
                * stats.action_std
                + stats.action_mean
            )
            y_pred.append(
                pred.numpy()
            )

    return regression_metrics(
        context.cache["actions"][
            val_indices
        ],
        np.concatenate(
            y_pred,
            axis=0,
        ),
    )
