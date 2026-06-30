import random

from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from oxford_pets_mlflow_augmentation.config import DATA_DIR, IMAGE_SIZE


def target_to_cat_dog(label, classes):
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
