from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def build_non_image_cache(
    dataset,
    state_key: str,
    action_key: str,
    episode_key: str,
    task_key: str,
) -> dict:
    columns = [state_key, action_key, episode_key, task_key]
    available = set(dataset.hf_dataset.column_names)

    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    data = dataset.select_columns(columns)[:]

    states = torch.as_tensor(
        np.asarray(data[state_key]),
        dtype=torch.float32,
    )
    actions = np.asarray(
        data[action_key],
        dtype=np.float32,
    )
    episode_ids = np.asarray(
        data[episode_key],
        dtype=np.int64,
    )
    task_ids = np.asarray(
        data[task_key],
        dtype=np.int64,
    )

    task_table = dataset.meta.tasks.reset_index()
    task_lookup = dict(
        zip(
            task_table["task_index"].astype(int),
            task_table["task"],
        )
    )
    tasks = [
        task_lookup[int(task_id)]
        for task_id in task_ids
    ]

    return {
        "states": states,
        "actions": actions,
        "episode_ids": episode_ids,
        "task_ids": task_ids,
        "tasks": tasks,
    }


def task_stratified_episode_split(
    episode_ids: np.ndarray,
    task_ids: np.ndarray,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    episode_to_task: dict[int, int] = {}

    for episode_id, task_id in zip(episode_ids, task_ids):
        episode_to_task.setdefault(
            int(episode_id),
            int(task_id),
        )

    task_to_episodes: dict[int, list[int]] = {}
    for episode_id, task_id in episode_to_task.items():
        task_to_episodes.setdefault(
            task_id,
            [],
        ).append(episode_id)

    rng = np.random.default_rng(seed)
    train_episodes: list[int] = []
    val_episodes: list[int] = []

    for _, episodes in sorted(task_to_episodes.items()):
        episodes = np.asarray(
            sorted(episodes),
            dtype=np.int64,
        )
        rng.shuffle(episodes)

        if len(episodes) == 1:
            train_episodes.extend(episodes.tolist())
            continue

        n_train = int(train_fraction * len(episodes))
        n_train = min(
            max(n_train, 1),
            len(episodes) - 1,
        )

        train_episodes.extend(
            episodes[:n_train].tolist()
        )
        val_episodes.extend(
            episodes[n_train:].tolist()
        )

    return (
        np.asarray(sorted(train_episodes), dtype=np.int64),
        np.asarray(sorted(val_episodes), dtype=np.int64),
    )


def create_or_load_split(
    cache: dict,
    split_config: dict,
) -> dict:
    split_file = Path(split_config["file"])
    reuse_existing = bool(
        split_config.get("reuse_existing", True)
    )

    if split_file.exists() and reuse_existing:
        payload = json.loads(
            split_file.read_text(encoding="utf-8")
        )
        train_episodes = np.asarray(
            payload["train_episodes"],
            dtype=np.int64,
        )
        val_episodes = np.asarray(
            payload["val_episodes"],
            dtype=np.int64,
        )
    else:
        strategy = split_config.get(
            "strategy",
            "task_stratified_episode",
        )
        if strategy != "task_stratified_episode":
            raise ValueError(
                f"Unsupported split strategy: {strategy}"
            )

        train_episodes, val_episodes = (
            task_stratified_episode_split(
                cache["episode_ids"],
                cache["task_ids"],
                train_fraction=float(
                    split_config.get("train_fraction", 0.8)
                ),
                seed=int(split_config.get("seed", 42)),
            )
        )

        split_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        split_file.write_text(
            json.dumps(
                {
                    "train_episodes":
                        train_episodes.tolist(),
                    "val_episodes":
                        val_episodes.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    overlap = (
        set(train_episodes.tolist())
        & set(val_episodes.tolist())
    )
    if overlap:
        raise RuntimeError(
            "Train/validation episode overlap detected."
        )

    train_indices = np.flatnonzero(
        np.isin(
            cache["episode_ids"],
            train_episodes,
        )
    )
    val_indices = np.flatnonzero(
        np.isin(
            cache["episode_ids"],
            val_episodes,
        )
    )

    return {
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }


@dataclass
class NormalizationStats:
    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor


def make_normalization_stats(
    cache: dict,
    train_indices,
) -> NormalizationStats:
    train_indices = np.asarray(train_indices)

    train_states = cache["states"][train_indices]
    train_actions = torch.as_tensor(
        cache["actions"][train_indices],
        dtype=torch.float32,
    )

    return NormalizationStats(
        state_mean=train_states.mean(dim=0),
        state_std=train_states.std(dim=0) + 1e-6,
        action_mean=train_actions.mean(dim=0),
        action_std=train_actions.std(dim=0) + 1e-6,
    )


def valid_temporal_indices(
    cache: dict,
    indices,
    history_k: int,
) -> np.ndarray:
    allowed = set(
        map(
            int,
            np.asarray(indices).tolist(),
        )
    )
    episode_ids = cache["episode_ids"]

    valid = []

    for idx in sorted(allowed):
        if idx - history_k < 0:
            continue

        previous_indices = [
            idx - offset
            for offset in range(1, history_k + 1)
        ]

        if any(
            previous not in allowed
            for previous in previous_indices
        ):
            continue

        if not all(
            episode_ids[previous] == episode_ids[idx]
            for previous in previous_indices
        ):
            continue

        valid.append(idx)

    return np.asarray(
        valid,
        dtype=np.int64,
    )


class TemporalDataset(Dataset):
    def __init__(
        self,
        cache: dict,
        indices,
        stats: NormalizationStats,
        history_k: int,
        include_task: bool = False,
    ):
        self.cache = cache
        self.indices = valid_temporal_indices(
            cache,
            indices,
            history_k,
        )
        self.stats = stats
        self.history_k = history_k
        self.include_task = include_task

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])

        sequence = []

        for offset in range(
            self.history_k,
            0,
            -1,
        ):
            previous_idx = idx - offset

            state = self.cache["states"][previous_idx]
            action = torch.as_tensor(
                self.cache["actions"][previous_idx],
                dtype=torch.float32,
            )

            state = (
                state - self.stats.state_mean
            ) / self.stats.state_std
            action = (
                action - self.stats.action_mean
            ) / self.stats.action_std

            history_token = torch.cat(
                [
                    state,
                    action,
                    torch.tensor([1.0]),
                ],
                dim=0,
            )
            sequence.append(history_token)

        current_state = self.cache["states"][idx]
        current_state = (
            current_state - self.stats.state_mean
        ) / self.stats.state_std

        current_token = torch.cat(
            [
                current_state,
                torch.zeros_like(
                    self.stats.action_mean
                ),
                torch.tensor([0.0]),
            ],
            dim=0,
        )
        sequence.append(current_token)

        action = torch.as_tensor(
            self.cache["actions"][idx],
            dtype=torch.float32,
        )
        action_norm = (
            action - self.stats.action_mean
        ) / self.stats.action_std

        result = {
            "sequence": torch.stack(sequence),
            "action": action,
            "action_norm": action_norm,
            "dataset_index": idx,
        }

        if self.include_task:
            result["task_id"] = int(
                self.cache["task_ids"][idx]
            )

        return result


class ImageTemporalDataset(TemporalDataset):
    def __init__(
        self,
        dataset,
        cache,
        indices,
        stats,
        history_k,
        image_keys,
        include_task: bool = False,
    ):
        super().__init__(
            cache,
            indices,
            stats,
            history_k,
            include_task=include_task,
        )
        self.dataset = dataset
        self.image_keys = list(image_keys)

    def __getitem__(self, item):
        result = super().__getitem__(item)
        sample = self.dataset[
            result["dataset_index"]
        ]

        result["images"] = [
            sample[key]
            for key in self.image_keys
        ]

        return result


class SmolVLAImageDataset(Dataset):
    def __init__(
        self,
        dataset,
        image_keys,
    ):
        self.dataset = dataset
        self.image_keys = list(image_keys)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        result = {
            "dataset_index": idx,
        }

        for key in self.image_keys:
            result[key] = sample[key]

        return result
