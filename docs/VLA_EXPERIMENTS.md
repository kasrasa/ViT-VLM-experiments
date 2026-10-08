# VLA experiments

## Structure

```text
configs/vla/libero.yaml
scripts/run_vla_experiments.py

src/vit_vlm_experiments/vla/
  config.py
  data.py
  logging_config.py
  metrics.py
  runner.py
  storage.py

  models/
    mlp.py
    gru.py
    image_gru.py

  experiments/
    baselines.py
    state_mlp.py
    gru.py
    image_gru.py
    smolvla.py
```

The split is deliberately simple:

- `data.py`: LeRobot loading helpers, non-image caching, episode-level split strategies, normalization, temporal dataset wrappers.
- `models/`: model definitions only.
- `experiments/`: training/evaluation loops for each benchmark.
- `metrics.py`: shared MAE/MSE and per-action-dimension metrics.
- `logging_config.py`: one console and rotating file logger for the VLA package.
- `storage.py`: atomic writes for split and metrics JSON files.
- `runner.py`: loads the dataset/cache/split once and runs the selected experiments.
- YAML: controls which experiments run and their hyperparameters.

## Install

```bash
python -m pip install -e ".[vla]"
```

If your SmolVLA checkpoint was trained with a particular LeRobot revision, use that same revision in the environment.

## Run the configured suite

```bash
python scripts/run_vla_experiments.py   --config configs/vla/libero.yaml
```

## Run only selected experiments

```bash
python scripts/run_vla_experiments.py   --config configs/vla/libero.yaml   --only gru image_gru
```

or:

```bash
python scripts/run_vla_experiments.py   --config configs/vla/libero.yaml   --only smolvla_eval
```

## Current experiment names

- `mean_action`
- `previous_action`
- `state_mlp`
- `gru`
- `task_gru`
- `image_gru`
- `smolvla_eval`

## Episode splits

The LIBERO configuration uses `task_stratified_episode` for an 80/20 split
within each task. For a single-task dataset, set
`split.strategy: episode_stratified_episode` to shuffle and split the unique
episode IDs without using task IDs. Both strategies keep complete episodes
together, use `split.train_fraction` and `split.seed`, and require at least one
training and one validation episode overall.

The generated episode IDs are saved to:

```text
outputs/vla/libero/split.json
```

and reused so every policy sees the same train/validation episodes. A saved
split is regenerated when its strategy, train fraction, seed, or set of
available episode IDs differs from the current configuration and data. Both
older flat settings and newer nested `settings` files can be reused. Empty or
malformed files are retried three times and then regenerated. Persistent read
or write I/O failures stop the run after three attempts, so an unusable split
cannot silently change the experiment's train and validation assignments.

This avoids frame-level leakage and avoids accidentally giving some LIBERO tasks no validation episodes.

## Logs and results

The suite logs dataset preparation, split reuse, experiment start and completion,
and errors to stderr and to `output_dir/run.log`. Training progress and the
running training MSE appear in tqdm bars on the console, not in the log file.
The log file rotates at 5 MB with two backups. Set `logging.level` to `DEBUG`,
`INFO`, `WARNING`, `ERROR`, or `CRITICAL`;
set `logging.file` to choose another path (or `null` to disable file logging).
If the log file cannot be opened, console logging continues.

Each experiment writes `output_dir/<experiment>.json`. Alongside its error
metrics, the file records the experiment name, package version, dataset config,
split settings and episode IDs, seed, device, model checkpoint (if any), and
hyperparameters. A metrics write failure is logged and the suite continues.
These files contain measurements and run details; the training loops do not
save model weights.

## SmolVLA

`smolvla_eval` expects a fine-tuned checkpoint path:

```yaml
experiments:
  smolvla_eval:
    checkpoint: /path/to/checkpoint/pretrained_model
```

The evaluation uses a DataLoader for image decoding and batched preprocessing/inference.

It calls `predict_action_chunk()` and compares the first predicted action in the chunk with the recorded expert action. This keeps the offline metric comparable to the one-step GRU/MLP baselines.
