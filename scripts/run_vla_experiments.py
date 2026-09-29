from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vit_vlm_experiments.vla.config import load_config
from vit_vlm_experiments.vla.runner import run_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one or more VLA/robotics experiments."
    )
    parser.add_argument(
        "--config",
        default="configs/vla/libero.yaml",
        help="Path to the YAML experiment config.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional experiment names that override config.run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    run_from_config(config, only=args.only)
