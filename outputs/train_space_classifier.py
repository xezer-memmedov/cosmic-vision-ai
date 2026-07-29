import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = BASE_DIR
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "space_classifier_resnet18.pth")
CHART_PATH = os.path.join(OUTPUTS_DIR, "training_history.png")
MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix_space.png")

BATCH_SIZE = 8
EPOCHS = 10
RANDOM_STATE = 42

random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Istifade edilen cihaz: {DEVICE}")
print(f"Dataset yolu: {DATASET_DIR}")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

base_dataset = datasets.ImageFolder(DATASET_DIR)
class_names = base_dataset.classes

train_indices = []
validation_indices = []

for class_index in range(len(class_names)):
    class_indices = []

    for index, target in enumerate(base_dataset.targets):
        if target == class_index:
            class_indices.append(index)

    random.shuffle(class_indices)

    validation_count = max(1, int(len(class_indices) * 0.20))

    validation_indices.extend(class_indices[:validation_count])
    train_indices.extend(class_indices[validation_count:])

train_dataset_all = datasets.ImageFolder(
    DATASET_DIR,
    transform=train_transform
)

validation_dataset_all = datasets.ImageFolder(
    DATASET_DIR,
    transform=validation_transform
)

train_dataset = Subset(train_dataset_all, train_indices)
validation_dataset = Subset(validation_dataset_all, validation_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Sinifler: {class_names}")
print(f"Training sekil sayi: {len(train_dataset)}")
print(f"Validation sekil sayi: {len(validation_dataset)}")

weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

train_losses = []
validation_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    train_losses.append(average_loss)

    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    validation_accuracy = correct_predictions / total_predictions
    validation_accuracies.append(validation_accuracy)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Validation accuracy: {validation_accuracy:.2%}"
    )

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": class_names
    },
    MODEL_PATH
)

print(f"\nModel saxlanildi: {MODEL_PATH}")

model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in validation_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())

matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names)))
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(
    range(1, EPOCHS + 1),
    train_losses,
    marker="o"
)

axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

axes[1].plot(
    range(1, EPOCHS + 1),
    validation_accuracies,
    marker="o",
    color="green"
)

axes[1].set_title("Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].grid(True)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200)
plt.show()

plt.figure(figsize=(7, 6))
plt.imshow(matrix, cmap="Blues")
plt.title("Space Classifier - Confusion Matrix")
plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=20
)

plt.yticks(range(len(class_names)), class_names)

for row in range(len(class_names)):
    for col in range(len(class_names)):
        plt.text(
            col,
            row,
            str(matrix[row, col]),
            ha="center",
            va="center",
            color="white" if matrix[row, col] > matrix.max() / 2 else "black"
        )

plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.tight_layout()
plt.savefig(MATRIX_PATH, dpi=200)
plt.show()

print(f"Qrafik saxlanildi: {CHART_PATH}")
print(f"Confusion matrix saxlanildi: {MATRIX_PATH}")