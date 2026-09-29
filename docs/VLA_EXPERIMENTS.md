# VLA experiments

This branch extracts the robotics/VLA work from the notebook-style workflow into a small package that can be developed one experiment at a time.

## Structure

```text
configs/vla/libero.yaml
scripts/run_vla_experiments.py

src/vit_vlm_experiments/vla/
  config.py
  data.py
  metrics.py
  runner.py

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

- `data.py`: LeRobot loading helpers, non-image caching, task-stratified episode split, normalization, temporal dataset wrappers.
- `models/`: model definitions only.
- `experiments/`: training/evaluation loops for each benchmark.
- `metrics.py`: shared MAE/MSE and per-action-dimension metrics.
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

## LIBERO split

The default configuration uses a task-stratified, episode-level 80/20 split.

The generated episode IDs are saved to:

```text
outputs/vla/libero/split.json
```

and reused so every policy sees the same train/validation episodes.

This avoids frame-level leakage and avoids accidentally giving some LIBERO tasks no validation episodes.

## SmolVLA

`smolvla_eval` expects a fine-tuned checkpoint path:

```yaml
experiments:
  smolvla_eval:
    checkpoint: /path/to/checkpoint/pretrained_model
```

The evaluation uses a DataLoader for image decoding and batched preprocessing/inference.

It calls `predict_action_chunk()` and compares the first predicted action in the chunk with the recorded expert action. This keeps the offline metric comparable to the one-step GRU/MLP baselines.

## Notes

This branch is intentionally an offline benchmark pipeline. It does not include closed-loop simulation or ROS integration. That can remain a separate pipeline and connect to the trained policies later.
