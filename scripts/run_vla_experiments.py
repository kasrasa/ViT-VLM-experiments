from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vit_vlm_experiments.vla.config import load_config
from vit_vlm_experiments.vla.logging_config import configure_logging
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


def main() -> int:
    configure_logging()
    args = parse_args()
    try:
        config = load_config(args.config)
        run_from_config(config, only=args.only)
    except Exception:
        logging.getLogger("vit_vlm_experiments.vla").exception("VLA run failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
