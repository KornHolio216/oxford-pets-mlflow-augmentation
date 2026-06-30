import mlflow
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from oxford_pets_mlflow_augmentation.artifacts import (
    collect_prediction_results,
    save_confusion_matrix,
    save_metric_plots,
    save_prediction_grid,
    save_raw_grid,
    save_tensor_grid,
    save_wrong_predictions_grid,
)
from oxford_pets_mlflow_augmentation.config import (
    BATCH_SIZE,
    DEVICE,
    IMAGE_SIZE,
    LEARNING_RATE,
    OUTPUT_DIR,
    RANDOM_SEED,
)
from oxford_pets_mlflow_augmentation.data import BinaryOxfordPets, make_subset
from oxford_pets_mlflow_augmentation.model import create_model, log_model
from oxford_pets_mlflow_augmentation.train import evaluate, train_one_epoch
from oxford_pets_mlflow_augmentation.utils import set_seed
from oxford_pets_mlflow_augmentation.xai import save_explainability_grid


def run_experiment(
    run_name,
    train_transform,
    test_transform,
    epochs,
    max_train_samples,
    max_test_samples,
    skip_xai,
):
    set_seed(RANDOM_SEED)

    print("\n" + "=" * 70)
    print(f"START RUN: {run_name}")
    print("=" * 70)

    run_output_dir = OUTPUT_DIR / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    raw_train_dataset = BinaryOxfordPets(split="trainval", transform=None)
    train_dataset = BinaryOxfordPets(split="trainval", transform=train_transform)
    test_dataset = BinaryOxfordPets(split="test", transform=test_transform)

    raw_train_subset = make_subset(
        raw_train_dataset,
        max_train_samples,
        seed=RANDOM_SEED,
    )

    train_subset = make_subset(
        train_dataset,
        max_train_samples,
        seed=RANDOM_SEED,
    )

    test_subset = make_subset(
        test_dataset,
        max_test_samples,
        seed=RANDOM_SEED + 1,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = create_model()
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("dataset", "Oxford-IIIT Pets")
        mlflow.log_param("task", "cat_vs_dog")
        mlflow.log_param("model", "MobileNetV3 Small")
        mlflow.log_param("transfer_learning", True)
        mlflow.log_param("augmentation_type", run_name)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("max_train_samples", max_train_samples)
        mlflow.log_param("max_test_samples", max_test_samples)
        mlflow.log_param("device", str(DEVICE))
        mlflow.log_param("skip_xai", skip_xai)

        raw_path = run_output_dir / "samples_before_augmentation.png"
        save_raw_grid(
            raw_train_subset,
            raw_path,
            "Images before augmentation",
        )
        mlflow.log_artifact(str(raw_path), artifact_path="samples/before_augmentation")

        for epoch in range(epochs):
            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )

            test_loss, accuracy, precision, recall, f1 = evaluate(
                model,
                test_loader,
                criterion,
            )

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["loss"].append(test_loss)
            history["accuracy"].append(accuracy)
            history["precision"].append(precision)
            history["recall"].append(recall)
            history["f1"].append(f1)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("loss", test_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            mlflow.log_metric("precision", precision, step=epoch)
            mlflow.log_metric("recall", recall, step=epoch)
            mlflow.log_metric("f1", f1, step=epoch)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"test_loss={test_loss:.4f} | "
                f"acc={accuracy:.4f} | "
                f"precision={precision:.4f} | "
                f"recall={recall:.4f} | "
                f"f1={f1:.4f}"
            )

            train_images, _ = next(iter(train_loader))
            test_images, _ = next(iter(test_loader))

            train_aug_path = run_output_dir / f"epoch_{epoch + 1}" / "train_after_augmentation.png"
            test_path = run_output_dir / f"epoch_{epoch + 1}" / "test_samples.png"

            save_tensor_grid(
                train_images,
                train_aug_path,
                f"Train images after augmentation - epoch {epoch + 1}",
            )

            save_tensor_grid(
                test_images,
                test_path,
                f"Test images - epoch {epoch + 1}",
            )

            mlflow.log_artifact(
                str(train_aug_path),
                artifact_path=f"samples/epoch_{epoch + 1}/train_after_augmentation",
            )

            mlflow.log_artifact(
                str(test_path),
                artifact_path=f"samples/epoch_{epoch + 1}/test",
            )

        loss_plot_path, metrics_plot_path = save_metric_plots(
            history,
            run_output_dir / "plots",
        )

        mlflow.log_artifact(str(loss_plot_path), artifact_path="plots")
        mlflow.log_artifact(str(metrics_plot_path), artifact_path="plots")

        prediction_grid_path = run_output_dir / "predictions_grid.png"

        save_prediction_grid(
            model,
            test_loader,
            prediction_grid_path,
            f"Predictions on test images - {run_name}",
        )

        mlflow.log_artifact(
            str(prediction_grid_path),
            artifact_path="predictions",
        )

        true_labels, predicted_labels, wrong_examples = collect_prediction_results(
            model,
            test_loader,
        )

        evaluation_dir = run_output_dir / "evaluation"
        confusion_matrix_path = evaluation_dir / "confusion_matrix.png"
        wrong_predictions_path = evaluation_dir / "wrong_predictions_grid.png"

        confusion_matrix = save_confusion_matrix(
            true_labels,
            predicted_labels,
            confusion_matrix_path,
            f"Confusion matrix - {run_name}",
        )

        save_wrong_predictions_grid(
            wrong_examples,
            wrong_predictions_path,
            f"Wrong predictions - {run_name}",
        )

        mlflow.log_artifact(
            str(confusion_matrix_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(wrong_predictions_path),
            artifact_path="evaluation",
        )

        print(f"Confusion matrix for {run_name}:")
        print(confusion_matrix)
        print(f"Wrong prediction examples saved: {len(wrong_examples)}")

        if skip_xai:
            mlflow.log_param("explainability_methods", "skipped")
            print("Skipped explainability maps.")
        else:
            explainability_path = run_output_dir / "explainability" / "xai_grid.png"

            save_explainability_grid(
                model=model,
                test_loader=test_loader,
                path=explainability_path,
                title=f"Explainability maps - {run_name}",
                max_images=3,
            )

            mlflow.log_artifact(
                str(explainability_path),
                artifact_path="explainability",
            )

            mlflow.log_param(
                "explainability_methods",
                "Saliency Map, Integrated Gradients, Grad-CAM",
            )

        log_model(model)

    print(f"Zakonczono run: {run_name}")

    return {
        "run_name": run_name,
        "accuracy": history["accuracy"][-1],
        "f1": history["f1"][-1],
    }
