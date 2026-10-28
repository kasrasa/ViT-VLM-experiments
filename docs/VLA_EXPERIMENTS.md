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

## Shared validation frames

Every model now scores the same global dataset frame indices. With the default
`evaluation.history_k: 2`, the first two frames of each validation episode are
excluded as prediction targets because temporal models need their history.
Those frames remain available as context for later predictions. Training
episodes and training windows are unchanged.

The shared target set is independent of `--only`: running `smolvla_eval` alone
uses the same frames as running the full suite. `evaluation.history_k` must
cover the largest configured GRU history (and at least one preceding frame
for `previous_action`). Temporal validation datasets retain the complete
validation history, then select the shared targets; they do not drop the
history twice. SmolVLA reads its images, states, tasks, and action targets
from that same full dataset/cache using the shared global frame indices.

Each metrics file includes `num_samples` and
`metadata.evaluation.frame_indices_sha256`. Both must match across runs on
the same dataset before comparing scores. Old metrics used different target
frames, so rerun the experiments rather than combining old and new scores.
The existing loops do not save local model weights, so local learned models
must train again to obtain those new scores.

This remains an offline one-step evaluation: temporal models receive recorded
expert actions as their preceding history, not their own earlier predictions.

### Motion and gripper MAE

For LIBERO's seven-dimensional action vector, `motion_mae` averages absolute
prediction errors across frames and action dimensions 0–5 (translation and
rotation commands). `gripper_mae` averages absolute errors for dimension 6.
Metrics are calculated after predictions have been returned to the original
dataset action scale, not in the standardized training space. These are errors
in control commands, not measured robot position errors or task success rates.

Overall `mae` weights all seven dimensions equally:
`mae = (6 * motion_mae + gripper_mae) / 7`. Separating the groups prevents a
large gripper error from obscuring differences in motion prediction. The
motion dimensions can also have different scales; inspect `per_dim` and
`relative_mae` when interpreting them. The latter divides each dimension's
MAE by that dimension's mean absolute target command.

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

`smolvla_eval` loads `lerobot/smolvla_libero` directly from the Hugging Face Hub
by default. This checkpoint declares an 8-value state, a 7-value action,
and the two camera features used by `lerobot/libero`. Its saved
preprocessor and postprocessor supply the corresponding normalization.
The evaluation checks the state and action widths before inference and
reports a clear error for incompatible checkpoints such as
`lerobot/smolvla_base`, which declares 6-value state and action features.

Run it with:

```bash
python scripts/run_vla_experiments.py --config configs/vla/libero.yaml --only smolvla_eval
```

The Hub checkpoint was trained on `lerobot/libero` without restricting its
training to this project's saved training episodes. Its score here is an
offline reconstruction measurement on recorded demonstrations, **not a
held-out comparison** with the locally trained baselines. To make that
comparison, fine-tune `lerobot/smolvla_base` on the saved training
episodes and point `experiments.smolvla_eval.checkpoint` to the resulting
`pretrained_model` directory.

The known overlap is logged as a warning and recorded as
`metadata.model.validation_exposure: known_training_overlap`. Other SmolVLA
checkpoints are marked `unverified`; supplying a different checkpoint does not
automatically establish that its training excluded these episodes. Matching
validation frames does not remove information already learned during training.

The evaluation loads the saved validation episodes, decodes images with
a DataLoader, and compares the first predicted action of each chunk
with the corresponding recorded expert action. It does not measure
closed-loop task success in a simulator.

