import argparse

from oxford_pets_mlflow_augmentation.config import (
    EPOCHS,
    MAX_TEST_SAMPLES,
    MAX_TRAIN_SAMPLES,
    RUN_NAMES,
)


def positive_int(value):
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")

    return parsed_value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Oxford-IIIT Pets cat vs dog augmentation experiments.",
    )
    parser.add_argument(
        "--run",
        choices=["all", *RUN_NAMES],
        default="all",
        help="Run all augmentation setups or only one selected setup.",
    )
    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=EPOCHS,
        help=f"Number of training epochs. Default: {EPOCHS}.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=positive_int,
        default=MAX_TRAIN_SAMPLES,
        help=f"Maximum number of train samples. Default: {MAX_TRAIN_SAMPLES}.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=positive_int,
        default=MAX_TEST_SAMPLES,
        help=f"Maximum number of test samples. Default: {MAX_TEST_SAMPLES}.",
    )
    parser.add_argument(
        "--skip-xai",
        action="store_true",
        help="Skip explainability map generation for faster smoke runs.",
    )
    parser.add_argument(
        "--copy-docs-artifacts",
        action="store_true",
        help="Copy selected run artifacts to docs/images after training.",
    )

    return parser.parse_args()
