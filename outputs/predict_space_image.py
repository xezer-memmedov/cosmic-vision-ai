import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "space_classifier_resnet18.pth")

TEST_IMAGE_PATH = os.path.join(
    BASE_DIR,
    "video_frames",
    "space_frame_000.jpg"
)

RESULT_PATH = os.path.join(
    BASE_DIR,
    "space_prediction_result.png"
)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

image = Image.open(TEST_IMAGE_PATH).convert("RGB")

image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]

prediction_index = torch.argmax(probabilities).item()
prediction_name = class_names[prediction_index]
confidence = probabilities[prediction_index].item() * 100

print(f"Analiz edilen sekil: {TEST_IMAGE_PATH}")
print(f"Proqnoz: {prediction_name}")
print(f"Etibar seviyesi: {confidence:.2f}%")

plt.figure(figsize=(7, 7))
plt.imshow(image)
plt.axis("off")

plt.title(
    f"Prediction: {prediction_name}\n"
    f"Confidence: {confidence:.2f}%"
)

plt.tight_layout()
plt.savefig(RESULT_PATH, dpi=200)
plt.show()

print(f"Netice sekli saxlanildi: {RESULT_PATH}")