from pathlib import Path

import torch

PROJECT_NAME = "OxfordPets_CatDog_Augmentation_Study"

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

IMAGE_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001

MAX_TRAIN_SAMPLES = 1200
MAX_TEST_SAMPLES = 600

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_NAMES = [
    "baseline_no_aug",
    "light_aug_flip_crop",
    "strong_aug_color_rotation_blur",
]
