import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients, LayerAttribution, LayerGradCam, Saliency

from oxford_pets_mlflow_augmentation.artifacts import denormalize_image
from oxford_pets_mlflow_augmentation.config import DEVICE, IMAGE_SIZE


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
