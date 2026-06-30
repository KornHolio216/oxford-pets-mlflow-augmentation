import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms

from oxford_pets_mlflow_augmentation.config import DEVICE, IMAGE_SIZE, OUTPUT_DIR


def denormalize_batch(images):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    images = images.cpu() * std + mean
    images = torch.clamp(images, 0, 1)

    return images


def denormalize_image(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image_tensor.detach().cpu() * std + mean
    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0).numpy()

    return image


def save_tensor_grid(images, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)

    images = denormalize_batch(images[:8])
    grid = torchvision.utils.make_grid(images, nrow=4)

    plt.figure(figsize=(8, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_prediction_grid(model, test_loader, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    images, labels = next(iter(test_loader))
    images_device = images.to(DEVICE)

    with torch.no_grad():
        outputs = model(images_device)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)

    images = denormalize_batch(images[:8])
    labels = labels[:8].cpu()
    predictions = predictions[:8].cpu()
    confidences = confidences[:8].cpu()

    class_names = ["cat", "dog"]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(8):
        image = images[i].permute(1, 2, 0).numpy()

        true_label = class_names[int(labels[i])]
        predicted_label = class_names[int(predictions[i])]
        confidence = float(confidences[i])

        is_correct = true_label == predicted_label
        title_color = "green" if is_correct else "red"

        axes[i].imshow(image)
        axes[i].set_title(
            f"true: {true_label}\n"
            f"pred: {predicted_label}\n"
            f"conf: {confidence:.2f}",
            color=title_color,
            fontsize=10,
        )
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def collect_prediction_results(model, test_loader, max_wrong_examples=8):
    model.eval()

    true_labels = []
    predicted_labels = []
    wrong_examples = []

    with torch.no_grad():
        for images, labels in test_loader:
            images_device = images.to(DEVICE)

            outputs = model(images_device)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)

            labels_cpu = labels.cpu()
            predictions_cpu = predictions.cpu()
            confidences_cpu = confidences.cpu()

            true_labels.extend(labels_cpu.tolist())
            predicted_labels.extend(predictions_cpu.tolist())

            wrong_indices = torch.where(predictions_cpu != labels_cpu)[0]

            for index in wrong_indices.tolist():
                if len(wrong_examples) >= max_wrong_examples:
                    break

                wrong_examples.append(
                    {
                        "image": images[index].cpu(),
                        "true_label": int(labels_cpu[index]),
                        "predicted_label": int(predictions_cpu[index]),
                        "confidence": float(confidences_cpu[index]),
                    }
                )

    return np.array(true_labels), np.array(predicted_labels), wrong_examples


def save_confusion_matrix(true_labels, predicted_labels, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)

    class_names = ["cat", "dog"]
    matrix = np.zeros((len(class_names), len(class_names)), dtype=int)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        matrix[int(true_label), int(predicted_label)] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            text_color = "white" if matrix[row, column] > threshold else "black"
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

    return matrix


def save_wrong_predictions_grid(wrong_examples, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(wrong_examples) == 0:
        plt.figure(figsize=(8, 3))
        plt.text(
            0.5,
            0.5,
            "No wrong predictions found in the evaluated test subset.",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        return

    class_names = ["cat", "dog"]
    columns = min(4, len(wrong_examples))
    rows = (len(wrong_examples) + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for index, example in enumerate(wrong_examples):
        image = denormalize_image(example["image"])
        true_label = class_names[example["true_label"]]
        predicted_label = class_names[example["predicted_label"]]
        confidence = example["confidence"]

        axes[index].imshow(image)
        axes[index].set_title(
            f"true: {true_label}\n"
            f"pred: {predicted_label}\n"
            f"conf: {confidence:.2f}",
            color="red",
            fontsize=10,
        )
        axes[index].axis("off")

    for axis in axes[len(wrong_examples):]:
        axis.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def save_raw_grid(dataset, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    for i in range(8):
        image, _ = dataset[i]
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        images.append(transform(image))

    batch = torch.stack(images)
    grid = torchvision.utils.make_grid(batch, nrow=4)

    plt.figure(figsize=(8, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_metric_plots(history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["loss"]) + 1)

    loss_path = output_dir / "loss_plot.png"

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["loss"], label="test_loss")
    plt.title("Loss during training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    metrics_path = output_dir / "metrics_plot.png"

    plt.figure()
    plt.plot(epochs, history["accuracy"], label="accuracy")
    plt.plot(epochs, history["f1"], label="f1")
    plt.plot(epochs, history["precision"], label="precision")
    plt.plot(epochs, history["recall"], label="recall")
    plt.title("Metrics during training")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(metrics_path)
    plt.close()

    return loss_path, metrics_path


def copy_docs_artifacts(run_name, epochs, skip_xai):
    docs_dir = Path("docs") / "images"
    docs_dir.mkdir(parents=True, exist_ok=True)

    run_output_dir = OUTPUT_DIR / run_name
    artifact_map = {
        run_output_dir / "predictions_grid.png": docs_dir / "predictions_grid.png",
        run_output_dir / "evaluation" / "confusion_matrix.png": docs_dir / "confusion_matrix.png",
        run_output_dir / "evaluation" / "wrong_predictions_grid.png": docs_dir / "wrong_predictions_grid.png",
        run_output_dir / "plots" / "metrics_plot.png": docs_dir / "metrics_plot.png",
        run_output_dir / "plots" / "loss_plot.png": docs_dir / "loss_plot.png",
        run_output_dir / f"epoch_{epochs}" / "train_after_augmentation.png": docs_dir / "train_after_augmentation.png",
    }

    if not skip_xai:
        artifact_map[
            run_output_dir / "explainability" / "xai_grid.png"
        ] = docs_dir / "explainability_xai_grid.png"

    copied_paths = []

    for source_path, target_path in artifact_map.items():
        if not source_path.exists():
            print(f"Skipping missing docs artifact: {source_path}")
            continue

        shutil.copy2(source_path, target_path)
        copied_paths.append(target_path)

    print(f"Copied {len(copied_paths)} docs artifacts from {run_name}.")

    return copied_paths
