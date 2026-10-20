import numpy as np

from ..metrics import regression_metrics


def run_mean_action(
    context,
    config,
):
    train_actions = (
        context.cache["actions"][
            context.split["train_indices"]
        ]
    )
    y_true = (
        context.cache["actions"][
            context.split["val_indices"]
        ]
    )

    mean_action = train_actions.mean(
        axis=0,
        keepdims=True,
    )
    y_pred = np.repeat(
        mean_action,
        len(y_true),
        axis=0,
    )

    return regression_metrics(
        y_true,
        y_pred,
    )


def run_previous_action(
    context,
    config,
):
    actions = context.cache["actions"]
    episode_ids = (
        context.cache["episode_ids"]
    )

    y_true = []
    y_pred = []

    for idx in (
        context.split["val_indices"]
    ):
        idx = int(idx)

        if idx == 0:
            continue

        if (
            episode_ids[idx - 1]
            != episode_ids[idx]
        ):
            continue

        y_true.append(actions[idx])
        y_pred.append(actions[idx - 1])

    return regression_metrics(
        np.asarray(y_true),
        np.asarray(y_pred),
    )
