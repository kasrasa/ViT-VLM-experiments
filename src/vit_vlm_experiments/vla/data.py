from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from .storage import atomic_write_text


LOGGER = logging.getLogger(__name__)
SPLIT_IO_ATTEMPTS = 3


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


def episode_stratified_episode_split(
    episode_ids: np.ndarray,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode_ids)

    if len(episodes) < 2:
        raise ValueError(
            "Cannot split a single episode into train/validation sets."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)

    n_train = int(train_fraction * len(episodes))
    n_train = min(max(n_train, 1), len(episodes) - 1)

    return episodes[:n_train], episodes[n_train:]


# Each strategy declares the cache fields passed before the common settings.
SPLIT_STRATEGIES = {
    "task_stratified_episode": (
        task_stratified_episode_split,
        ("episode_ids", "task_ids"),
    ),
    "episode_stratified_episode": (
        episode_stratified_episode_split,
        ("episode_ids",),
    ),
}


def _valid_episode_split(
    train_episodes: np.ndarray,
    val_episodes: np.ndarray,
    available_episodes: np.ndarray,
) -> bool:
    return (
        train_episodes.size > 0
        and val_episodes.size > 0
        and np.unique(train_episodes).size == train_episodes.size
        and np.unique(val_episodes).size == val_episodes.size
        and not np.intersect1d(train_episodes, val_episodes).size
        and np.array_equal(
            np.union1d(train_episodes, val_episodes),
            available_episodes,
        )
    )

class InvalidSplitFile(ValueError):
    """A saved split cannot be parsed or has an invalid structure."""


class SplitFileReadError(OSError):
    """Reading a saved split failed after repeated I/O attempts."""


def _check_split_payload(payload: object) -> None:
    if not isinstance(payload, dict):
        raise InvalidSplitFile("Expected a JSON object.")

    missing = {"train_episodes", "val_episodes"} - payload.keys()
    if missing:
        raise InvalidSplitFile(f"Missing split keys: {sorted(missing)}")

    if "settings" in payload and not isinstance(payload["settings"], dict):
        raise InvalidSplitFile("Split settings must be an object.")

    int64 = np.iinfo(np.int64)
    for name in ("train_episodes", "val_episodes"):
        values = payload[name]
        if not isinstance(values, list) or any(
            type(value) is not int or not int64.min <= value <= int64.max
            for value in values
        ):
            raise InvalidSplitFile(f"{name} must be a list of episode IDs.")


def read_split_file(split_file: Path) -> dict:
    """Read a saved split, retrying up to three times before reporting failure."""
    for attempt in range(1, SPLIT_IO_ATTEMPTS + 1):
        try:
            payload = json.loads(split_file.read_text(encoding="utf-8"))
            _check_split_payload(payload)
            return payload
        except (UnicodeDecodeError, json.JSONDecodeError, InvalidSplitFile) as exc:
            LOGGER.warning(
                "Invalid split file %s (read %d/%d): %s",
                split_file, attempt, SPLIT_IO_ATTEMPTS, exc,
            )
            if attempt == SPLIT_IO_ATTEMPTS:
                raise InvalidSplitFile(
                    f"Invalid split file after {attempt} attempts: {split_file}"
                ) from exc
        except OSError as exc:
            LOGGER.warning(
                "Cannot read split file %s (read %d/%d): %s",
                split_file, attempt, SPLIT_IO_ATTEMPTS, exc,
            )
            if attempt == SPLIT_IO_ATTEMPTS:
                raise SplitFileReadError(
                    f"Cannot read split file after {attempt} attempts: {split_file}"
                ) from exc
        time.sleep(0.05 * attempt)

    raise AssertionError("The split reader exhausted its attempts.")


def write_split_file(split_file: Path, payload: dict) -> None:
    """Atomically save a JSON-ready split, retrying I/O failures."""
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    for attempt in range(1, SPLIT_IO_ATTEMPTS + 1):
        try:
            atomic_write_text(split_file, serialized)
        except OSError as exc:
            LOGGER.warning(
                "Cannot write split file %s (write %d/%d): %s",
                split_file, attempt, SPLIT_IO_ATTEMPTS, exc,
            )
            if attempt == SPLIT_IO_ATTEMPTS:
                raise OSError(
                    f"Cannot save split file after {attempt} attempts: {split_file}"
                ) from exc
            time.sleep(0.05 * attempt)
        else:
            LOGGER.info("Saved episode split to %s", split_file)
            return


def create_or_load_split(
    cache: dict,
    split_config: dict,
) -> dict:
    strategy = split_config.get(
        "strategy", "task_stratified_episode"
    )
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(
            f"Unsupported split strategy: {strategy}. "
            f"Choose from: {', '.join(SPLIT_STRATEGIES)}"
        )

    train_fraction = float(split_config.get("train_fraction", 0.8))
    if not 0 < train_fraction < 1:
        raise ValueError("split.train_fraction must be between 0 and 1.")
    seed = int(split_config.get("seed", 42))

    split_file = Path(split_config["file"])
    reuse_existing = bool(
        split_config.get("reuse_existing", True)
    )
    episode_ids = np.asarray(cache["episode_ids"], dtype=np.int64)
    available_episodes = np.unique(episode_ids)
    settings = {
        "strategy": strategy,
        "train_fraction": train_fraction,
        "seed": seed,
    }

    cached_split = None
    if split_file.exists() and reuse_existing:
        try:
            payload = read_split_file(split_file)
        except InvalidSplitFile:
            LOGGER.warning("Regenerating malformed episode split %s", split_file)
        except SplitFileReadError as exc:
            if not isinstance(exc.__cause__, FileNotFoundError):
                raise
            LOGGER.warning("Split file disappeared; regenerating %s", split_file)
        else:
            # Existing flat-format split files remain readable.
            saved_settings = payload.get("settings", payload)
            if all(saved_settings.get(key) == value for key, value in settings.items()):
                train_episodes = np.asarray(payload["train_episodes"], dtype=np.int64)
                val_episodes = np.asarray(payload["val_episodes"], dtype=np.int64)
                if _valid_episode_split(
                    train_episodes, val_episodes, available_episodes
                ):
                    cached_split = (train_episodes, val_episodes)
                    LOGGER.info("Reusing episode split from %s", split_file)
                else:
                    LOGGER.warning(
                        "Saved split does not match current episodes; regenerating %s",
                        split_file,
                    )
            else:
                LOGGER.info("Split settings changed; regenerating %s", split_file)

    if cached_split is None:
        split_fn, cache_keys = SPLIT_STRATEGIES[strategy]
        train_episodes, val_episodes = split_fn(
            *(cache[key] for key in cache_keys),
            train_fraction=train_fraction,
            seed=seed,
        )
        if not _valid_episode_split(
            train_episodes, val_episodes, available_episodes
        ):
            raise ValueError(
                "The split must assign every episode exactly once "
                "and include both train and validation episodes."
            )

        write_split_file(
            split_file,
            {
                "settings": settings,
                "train_episodes": train_episodes.tolist(),
                "val_episodes": val_episodes.tolist(),
            },
        )
    else:
        train_episodes, val_episodes = cached_split

    train_mask = np.isin(episode_ids, train_episodes)
    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(~train_mask)

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
