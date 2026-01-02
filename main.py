import argparse
from pathlib import Path

from scripts.process_data import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Process electoral data with Polars")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    run_pipeline(args.config)


if __name__ == "__main__":
    main()
