import random
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from captum.attr import Saliency, IntegratedGradients, LayerGradCam, LayerAttribution

# main config
PROJECT_NAME = "OxfordPets_CatDog_Augmentation_Study"

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

IMAGE_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001

# uzywam czesci danych dla szybszego treningu
MAX_TRAIN_SAMPLES = 1200
MAX_TEST_SAMPLES = 600

RANDOM_SEED = 42

# ustawiam urzadzenie jako karta graficzna jeśli jest dostępna, inaczej CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# funkcje pomocnicze


def set_seed(seed):
    # ziarno losowosci
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_to_cat_dog(label, classes):
    # dataset Oxford-IIIT Pets posiada 37 ras zamieniam je na problem binarny: 0 - kot 1 - pies

    cat_breeds = {
        "abyssinian",
        "bengal",
        "birman",
        "bombay",
        "british shorthair",
        "egyptian mau",
        "maine coon",
        "persian",
        "ragdoll",
        "russian blue",
        "siamese",
        "sphynx",
    }

    class_name = classes[label].lower().replace("_", " ")

    if class_name in cat_breeds:
        return 0

    return 1


class BinaryOxfordPets(Dataset):
    def __init__(self, split, transform=None):
        self.base_dataset = datasets.OxfordIIITPet(
            root=DATA_DIR,
            split=split,
            target_types="category",
            download=True,
            transform=None,
        )

        self.transform = transform
        self.classes = self.base_dataset.classes
        self.binary_classes = ["cat", "dog"]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]

        # sprawdzam czy obraz jest w RGB.
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        binary_label = target_to_cat_dog(label, self.classes)

        if self.transform is not None:
            image = self.transform(image)

        return image, binary_label


def make_subset(dataset, max_samples, seed):
    rng = random.Random(seed)

    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:max_samples]

    return Subset(dataset, indices)


def denormalize_batch(images):
    # odwracam normalizacje ImageNet, zeby obrazki zapisane jako artefakty wyglądaly normalnie
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    images = images.cpu() * std + mean
    images = torch.clamp(images, 0, 1)

    return images


def save_tensor_grid(images, path, title):
    # zapis siatki po transformacjach
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
    # zapisuje przykładowe predykcje modelu na obrazach testowych
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
    # zbiera predykcje z calego zbioru testowego do analizy bledow
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
    # zapisuje macierz pomylek bez dodatkowych zaleznosci typu scikit-learn
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
    # zapisuje grid z przykladami blednych predykcji modelu
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
    # zapis siatki przed augmentacja
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

def denormalize_image(image_tensor):
    # odwracam normalizacje ImageNet dla pojedynczego obrazu
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image_tensor.detach().cpu() * std + mean
    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0).numpy()

    return image


def normalize_xai_map(xai_map):
    xai_map = np.maximum(xai_map, 0)
    return xai_map / (xai_map.max() + 1e-8)


def make_xai_overlay(image, heatmap, alpha=0.45):
    heatmap = normalize_xai_map(heatmap)

    colored_heatmap = plt.get_cmap("jet")(heatmap)[..., :3]
    overlay = (1 - alpha) * image + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay


def calculate_xai_maps(model, image_tensor, target_class, ig_steps=24):
    model.eval()

    # saliency map
    saliency_input = image_tensor.unsqueeze(0).to(DEVICE)
    saliency_input.requires_grad_(True)

    saliency = Saliency(model)
    saliency_attr = saliency.attribute(
        saliency_input,
        target=target_class,
    )

    saliency_map = (
        saliency_attr.abs()
        .squeeze()
        .max(dim=0)[0]
        .detach()
        .cpu()
        .numpy()
    )

    # integrated gradients
    ig_input = image_tensor.unsqueeze(0).to(DEVICE)
    ig_input.requires_grad_(True)

    baseline = torch.zeros_like(ig_input)

    integrated_gradients = IntegratedGradients(model)
    ig_attr = integrated_gradients.attribute(
        ig_input,
        baselines=baseline,
        target=target_class,
        n_steps=ig_steps,
    )

    ig_map = (
        ig_attr.abs()
        .squeeze()
        .max(dim=0)[0]
        .detach()
        .cpu()
        .numpy()
    )

    # Grad-CAM dla ostatniej warstwy cech MobileNetV3
    gradcam_input = image_tensor.unsqueeze(0).to(DEVICE)
    target_layer = model.features[-1]

    gradcam = LayerGradCam(model, target_layer)

    gradcam_attr = gradcam.attribute(
        gradcam_input,
        target=target_class,
    )

    gradcam_attr = LayerAttribution.interpolate(
        gradcam_attr,
        (IMAGE_SIZE, IMAGE_SIZE),
    )

    gradcam_map = (
        gradcam_attr.squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    return {
        "saliency": normalize_xai_map(saliency_map),
        "integrated_gradients": normalize_xai_map(ig_map),
        "gradcam": normalize_xai_map(gradcam_map),
    }


def save_explainability_grid(model, test_loader, path, title, max_images=3):
    # zapisuje mapy wyjaśnialności jako artefakt MLflow
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    images, labels = next(iter(test_loader))
    images = images[:max_images]
    labels = labels[:max_images]

    class_names = ["cat", "dog"]

    fig, axes = plt.subplots(
        max_images,
        4,
        figsize=(16, 4 * max_images),
        squeeze=False,
    )

    for row in range(max_images):
        image_tensor = images[row]
        true_label = int(labels[row])

        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

        pred_class = probabilities.argmax(dim=1).item()
        pred_score = probabilities[0, pred_class].item()

        image_np = denormalize_image(image_tensor)

        xai_maps = calculate_xai_maps(
            model=model,
            image_tensor=image_tensor,
            target_class=pred_class,
        )

        is_correct = true_label == pred_class
        title_color = "green" if is_correct else "red"

        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(
            f"true: {class_names[true_label]}\n"
            f"pred: {class_names[pred_class]}\n"
            f"conf: {pred_score:.2f}",
            color=title_color,
            fontsize=10,
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(
            make_xai_overlay(image_np, xai_maps["saliency"])
        )
        axes[row, 1].set_title("Saliency Map")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(
            make_xai_overlay(image_np, xai_maps["integrated_gradients"])
        )
        axes[row, 2].set_title("Integrated Gradients")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(
            make_xai_overlay(image_np, xai_maps["gradcam"])
        )
        axes[row, 3].set_title("Grad-CAM")
        axes[row, 3].axis("off")

    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

def save_metric_plots(history, output_dir):
    # zapisuje wykres loss oraz wykres metryk accuracy/F1

    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["loss"]) + 1)

    # wykres funkcji straty
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

    # wykres accuracy i F1
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


def calculate_metrics(correct, total, true_positive, false_positive, false_negative):
    # obliczenie metryk klasyfikacji binarnej, klasa pozytywna dog

    accuracy = correct / total if total > 0 else 0.0

    precision = true_positive / (true_positive + false_positive) \
        if (true_positive + false_positive) > 0 else 0.0

    recall = true_positive / (true_positive + false_negative) \
        if (true_positive + false_negative) > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1


def create_model():
    # tworze MobileNetV3 Small z transfer leaningiem i zamrażam warstwy ekstratora cech po czym trenuje klasyfikator binarny

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, 2)

    return model


def log_model(model):
    # zapis modelu do mlflow
    try:
        mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
        )
    except TypeError:
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            serialization_format="pickle",
        )


# augmentacje danych
def get_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    baseline_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    light_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomCrop(IMAGE_SIZE, padding=8),
        transforms.ToTensor(),
        normalize,
    ])

    strong_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.35,
            contrast=0.35,
            saturation=0.35,
            hue=0.08,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3)],
            p=0.4,
        ),
        transforms.RandomCrop(IMAGE_SIZE, padding=16),
        transforms.ToTensor(),
        normalize,
    ])

    augmentations = {
        "baseline_no_aug": baseline_transform,
        "light_aug_flip_crop": light_transform,
        "strong_aug_color_rotation_blur": strong_transform,
    }

    return augmentations, test_transform


# trening i ewaluacja
def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total if total > 0 else 0.0

    return train_loss, train_accuracy


def evaluate(model, test_loader, criterion):
    # ewaluacja modelu na zbiorze testowym
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            true_positive += ((predicted == 1) & (labels == 1)).sum().item()
            false_positive += ((predicted == 1) & (labels == 0)).sum().item()
            false_negative += ((predicted == 0) & (labels == 1)).sum().item()

    test_loss = running_loss / len(test_loader)
    accuracy, precision, recall, f1 = calculate_metrics(
        correct,
        total,
        true_positive,
        false_positive,
        false_negative,
    )

    return test_loss, accuracy, precision, recall, f1


def run_experiment(run_name, train_transform, test_transform):
    # jeden run MLflow = jedna konfiguracja augmentacji
    set_seed(RANDOM_SEED)

    print("\n" + "=" * 70)
    print(f"START RUN: {run_name}")
    print("=" * 70)

    run_output_dir = OUTPUT_DIR / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # dataset przed augmentacja do zapisania przykładowych obrazow
    raw_train_dataset = BinaryOxfordPets(split="trainval", transform=None)

    # dataset treningowy z dana augmentacja
    train_dataset = BinaryOxfordPets(split="trainval", transform=train_transform)

    # dataset testowy bez augmentacji
    test_dataset = BinaryOxfordPets(split="test", transform=test_transform)

    raw_train_subset = make_subset(
        raw_train_dataset,
        MAX_TRAIN_SAMPLES,
        seed=RANDOM_SEED,
    )

    train_subset = make_subset(
        train_dataset,
        MAX_TRAIN_SAMPLES,
        seed=RANDOM_SEED,
    )

    test_subset = make_subset(
        test_dataset,
        MAX_TEST_SAMPLES,
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
        # parametry
        mlflow.log_param("dataset", "Oxford-IIIT Pets")
        mlflow.log_param("task", "cat_vs_dog")
        mlflow.log_param("model", "MobileNetV3 Small")
        mlflow.log_param("transfer_learning", True)
        mlflow.log_param("augmentation_type", run_name)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("max_train_samples", MAX_TRAIN_SAMPLES)
        mlflow.log_param("max_test_samples", MAX_TEST_SAMPLES)
        mlflow.log_param("device", str(DEVICE))

        # obrazy przed augmentacja
        raw_path = run_output_dir / "samples_before_augmentation.png"
        save_raw_grid(
            raw_train_subset,
            raw_path,
            "Images before augmentation",
        )
        mlflow.log_artifact(str(raw_path), artifact_path="samples/before_augmentation")

        # trening
        for epoch in range(EPOCHS):
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

            # logowanie metryk do mlflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("loss", test_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            mlflow.log_metric("precision", precision, step=epoch)
            mlflow.log_metric("recall", recall, step=epoch)
            mlflow.log_metric("f1", f1, step=epoch)

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"train_loss={train_loss:.4f} | "
                f"test_loss={test_loss:.4f} | "
                f"acc={accuracy:.4f} | "
                f"precision={precision:.4f} | "
                f"recall={recall:.4f} | "
                f"f1={f1:.4f}"
            )

            # artefakty
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

        # wykresy metryk
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

    print(f"Zakończono run: {run_name}")


# uruchomienie programu
def main():
    set_seed(RANDOM_SEED)

    print(f"Device: {DEVICE}")
    print(f"Projekt MLflow: {PROJECT_NAME}")

    mlflow.set_experiment(PROJECT_NAME)

    augmentations, test_transform = get_transforms()

    for run_name, train_transform in augmentations.items():
        run_experiment(
            run_name=run_name,
            train_transform=train_transform,
            test_transform=test_transform,
        )

    print("\nWszystkie eksperymenty zostały zakończone.")
    print("Uruchom teraz w terminalu:")
    print("mlflow ui")


if __name__ == "__main__":
    main()
