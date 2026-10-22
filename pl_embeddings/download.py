"""Explicitly download the authors' checkpoints and preprocessing assets."""

import argparse
from pathlib import Path

from .assets import DEFAULT_MODEL_DIR, download_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    try:
        download_models(args.model_dir)
    except Exception as error:
        parser.exit(1, f"Model download failed: {error}\nRetry the same command to resume setup.\n")


if __name__ == "__main__":
    main()
