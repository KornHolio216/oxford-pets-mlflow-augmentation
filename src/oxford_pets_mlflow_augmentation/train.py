import torch

from oxford_pets_mlflow_augmentation.config import DEVICE


def calculate_metrics(correct, total, true_positive, false_positive, false_negative):
    accuracy = correct / total if total > 0 else 0.0

    precision = true_positive / (true_positive + false_positive) \
        if (true_positive + false_positive) > 0 else 0.0

    recall = true_positive / (true_positive + false_negative) \
        if (true_positive + false_negative) > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1


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
