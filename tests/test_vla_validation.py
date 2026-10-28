"""Check identical validation targets without downloading models or datasets."""

from contextlib import ExitStack
import copy
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from vit_vlm_experiments.vla import data, metrics, runner
from vit_vlm_experiments.vla.experiments import baselines, gru, image_gru, smolvla, state_mlp


class ImageDataset:
    def __init__(self, size):
        self.size = size
        self.loaded = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.loaded.append(index)
        return {"camera_a": torch.full((3, 4, 4), float(index)),
                "camera_b": torch.full((3, 4, 4), float(-index))}


def make_context():
    count = 24
    cache = {
        "episode_ids": np.repeat([10, 20, 30, 40], 6),
        "states": torch.arange(count * 8, dtype=torch.float32).reshape(count, 8) / 100,
        "actions": np.arange(count * 7, dtype=np.float32).reshape(count, 7) / 100,
        "task_ids": np.tile(np.repeat([0, 1], 6), 2),
        "tasks": [f"frame {i}" for i in range(count)],
    }
    cache["states"][:, 0] = torch.arange(count)
    split = {
        "train_indices": np.r_[0:6, 12:18],
        "val_indices": np.r_[6:12, 18:24],
        "train_episodes": np.array([10, 30]),
        "val_episodes": np.array([20, 40]),
    }
    config = {
        "dataset": {"repo_id": "lerobot/libero", "image_keys": ["camera_a", "camera_b"]},
        "split": {},
        "loader": {"num_workers": 0, "pin_memory": False},
        "evaluation": {"history_k": 3},
        "experiments": {"gru": {"history_k": 2}, "task_gru": {"history_k": 3},
                        "image_gru": {"history_k": 1}},
    }
    split["evaluation_indices"] = data.shared_validation_indices(cache, split, config)
    return runner.ExperimentContext(config, ImageDataset(count), cache, split, "cpu")


def fake_lerobot(context, seen):
    names = ["lerobot", "lerobot.policies", "lerobot.policies.factory",
             "lerobot.policies.smolvla", "lerobot.policies.smolvla.modeling_smolvla"]
    modules = {name: ModuleType(name) for name in names}

    class Processor:
        def reset(self):
            pass

        def __call__(self, batch):
            indices = batch["observation.state"][:, 0].long().tolist()
            seen.extend(indices)
            if batch["task"] != [context.cache["tasks"][i] for i in indices]:
                raise AssertionError("Task/image frame mismatch")
            for row, i in enumerate(indices):
                assert torch.all(batch["model_camera_a"][row] == i)
                assert torch.all(batch["model_camera_b"][row] == -i)
            return batch

    class Postprocessor:
        def reset(self):
            pass

        def __call__(self, predictions):
            # Ensure evaluation compares actions after postprocessing.
            return predictions * 2

    class Policy:
        config = SimpleNamespace(
            input_features={"observation.state": SimpleNamespace(shape=(8,))},
            output_features={"action": SimpleNamespace(shape=(7,))},
        )

        @classmethod
        def from_pretrained(cls, checkpoint):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            pass

        def reset(self):
            pass

        def predict_action_chunk(self, batch):
            indices = batch["observation.state"][:, 0].long().tolist()
            first = torch.from_numpy(context.cache["actions"][indices]) / 2
            return torch.stack([first, first + 10], dim=1)

    modules["lerobot.policies.factory"].make_pre_post_processors = (
        lambda *args, **kwargs: (Processor(), Postprocessor())
    )
    modules["lerobot.policies.smolvla.modeling_smolvla"].SmolVLAPolicy = Policy
    return modules


class SharedValidationTests(unittest.TestCase):
    def test_history_is_kept_and_does_not_cross_episodes_or_get_dropped_twice(self):
        context = make_context()
        np.testing.assert_array_equal(context.split["evaluation_indices"], [9, 10, 11, 21, 22, 23])
        stats = data.make_normalization_stats(context.cache, context.split["train_indices"])
        dataset = data.TemporalDataset(
            context.cache, context.split["val_indices"], stats, 3,
            target_indices=context.split["evaluation_indices"],
        )
        self.assertEqual(dataset[0]["dataset_index"], 9)
        self.assertEqual(len(dataset), 6)
        recovered = dataset[0]["sequence"][:3, :8] * stats.state_std + stats.state_mean
        torch.testing.assert_close(recovered, context.cache["states"][[6, 7, 8]])

    def test_selection_is_independent_of_only_and_rejects_insufficient_history(self):
        context = make_context()
        single = copy.deepcopy(context.config)
        single["run"] = ["smolvla_eval"]
        np.testing.assert_array_equal(
            data.shared_validation_indices(context.cache, context.split, single),
            context.split["evaluation_indices"],
        )
        single["evaluation"]["history_k"] = 2
        with self.assertRaisesRegex(ValueError, "required history"):
            data.shared_validation_indices(context.cache, context.split, single)
        single["evaluation"]["history_k"] = 10
        with self.assertRaisesRegex(ValueError, "No shared validation"):
            data.shared_validation_indices(context.cache, context.split, single)

    def test_all_seven_experiments_score_identical_frames(self):
        torch.set_num_threads(1)
        context = make_context()
        expected = context.cache["actions"][context.split["evaluation_indices"]]
        targets = []
        seen_smolvla = []

        def record(y_true, y_pred):
            targets.append(np.asarray(y_true).copy())
            return metrics.regression_metrics(y_true, y_pred)

        def tiny_encoder(embedding_dim):
            return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, embedding_dim))

        common = {"epochs": 1, "batch_size": 4, "hidden_dim": 8,
                  "num_layers": 1, "dropout": 0, "image_embedding_dim": 4}
        with ExitStack() as stack:
            for module in [baselines, state_mlp, gru, image_gru, smolvla]:
                stack.enter_context(patch.object(module, "regression_metrics", side_effect=record))
            stack.enter_context(patch(
                "vit_vlm_experiments.vla.models.image_gru.FrozenResNet50Encoder", tiny_encoder
            ))
            stack.enter_context(patch.dict("sys.modules", fake_lerobot(context, seen_smolvla)))
            results = [
                baselines.run_mean_action(context, {}),
                baselines.run_previous_action(context, {}),
                state_mlp.run_state_mlp(context, common),
                gru.run_gru(context, {**common, "history_k": 2}),
                gru.run_task_gru(context, {**common, "history_k": 3}),
                image_gru.run_image_gru(context, {**common, "history_k": 1}),
                smolvla.run_smolvla_eval(context, {
                    "checkpoint": "lerobot/smolvla_libero", "batch_size": 4, "num_workers": 0,
                    "model_image_keys": ["model_camera_a", "model_camera_b"],
                }),
            ]
        self.assertEqual(len(targets), 7)
        for actual, result in zip(targets, results):
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(result["num_samples"], 6)
        self.assertEqual(seen_smolvla, context.split["evaluation_indices"].tolist())
        self.assertEqual(results[-1]["mae"], 0)
        self.assertEqual(runner.experiment_metadata(context, "smolvla_eval", {
            "checkpoint": "lerobot/smolvla_libero"
        })["model"]["validation_exposure"], "known_training_overlap")
        self.assertEqual(
            runner.experiment_metadata(context, "gru", {})["evaluation"],
            runner.experiment_metadata(context, "smolvla_eval", {})["evaluation"],
        )

    def test_grouped_metrics_use_original_action_errors(self):
        truth = np.zeros((2, 7), dtype=np.float32)
        prediction = np.array([[1, 2, 3, 4, 5, 6, 2], [1, 2, 3, 4, 5, 6, 0]])
        result = metrics.regression_metrics(truth, prediction)
        self.assertEqual(result["motion_mae"], 3.5)
        self.assertEqual(result["gripper_mae"], 1)
        self.assertAlmostEqual(result["mae"], (6 * 3.5 + 1) / 7, places=6)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            metrics.regression_metrics(np.empty((0, 7)), np.empty((0, 7)))


if __name__ == "__main__":
    unittest.main()
