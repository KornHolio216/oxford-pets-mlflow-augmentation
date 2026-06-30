import mlflow.pytorch
import torch.nn as nn
from torchvision import models


def create_model():
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, 2)

    return model


def log_model(model):
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
