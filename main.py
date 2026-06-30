import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from oxford_pets_mlflow_augmentation.artifacts import copy_docs_artifacts
from oxford_pets_mlflow_augmentation.cli import parse_args
from oxford_pets_mlflow_augmentation.config import DEVICE, PROJECT_NAME, RANDOM_SEED
from oxford_pets_mlflow_augmentation.data import get_transforms
from oxford_pets_mlflow_augmentation.experiment import run_experiment
from oxford_pets_mlflow_augmentation.utils import set_seed


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)

    print(f"Device: {DEVICE}")
    print(
        f"Config: run={args.run}, epochs={args.epochs}, "
        f"max_train_samples={args.max_train_samples}, "
        f"max_test_samples={args.max_test_samples}, "
        f"skip_xai={args.skip_xai}"
    )
    print(f"Projekt MLflow: {PROJECT_NAME}")

    mlflow.set_experiment(PROJECT_NAME)

    augmentations, test_transform = get_transforms()

    if args.run == "all":
        selected_augmentations = augmentations.items()
    else:
        selected_augmentations = [(args.run, augmentations[args.run])]

    results = []

    for run_name, train_transform in selected_augmentations:
        result = run_experiment(
            run_name=run_name,
            train_transform=train_transform,
            test_transform=test_transform,
            epochs=args.epochs,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            skip_xai=args.skip_xai,
        )
        results.append(result)

    if args.copy_docs_artifacts:
        best_result = max(
            results,
            key=lambda result: (result["f1"], result["accuracy"]),
        )
        copied_paths = copy_docs_artifacts(
            run_name=best_result["run_name"],
            epochs=args.epochs,
            skip_xai=args.skip_xai,
        )

        print("Docs artifacts:")
        for path in copied_paths:
            print(path)

    print("\nWszystkie eksperymenty zostaly zakonczone.")
    print("Uruchom teraz w terminalu:")
    print("mlflow ui")


if __name__ == "__main__":
    main()
