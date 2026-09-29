from .baselines import run_mean_action, run_previous_action
from .gru import run_gru, run_task_gru
from .image_gru import run_image_gru
from .smolvla import run_smolvla_eval
from .state_mlp import run_state_mlp

EXPERIMENTS = {
    "mean_action": run_mean_action,
    "previous_action": run_previous_action,
    "state_mlp": run_state_mlp,
    "gru": run_gru,
    "task_gru": run_task_gru,
    "image_gru": run_image_gru,
    "smolvla_eval": run_smolvla_eval,
}

__all__ = ["EXPERIMENTS"]
