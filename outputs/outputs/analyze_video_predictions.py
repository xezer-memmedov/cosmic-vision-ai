import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(
    OUTPUTS_DIR,
    "space_classifier_resnet18.pth"
)

FRAMES_DIR = os.path.join(
    OUTPUTS_DIR,
    "video_frames"
)

CSV_PATH = os.path.join(
    OUTPUTS_DIR,
    "video_predictions.csv"
)

CHART_PATH = os.path.join(
    OUTPUTS_DIR,
    "video_predictions_chart.png"
)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Istifade edilen cihaz: {DEVICE}")
print(f"Model yolu: {MODEL_PATH}")
print(f"Kadrlar qovlugu: {FRAMES_DIR}")

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

frame_files = []

for file_name in os.listdir(FRAMES_DIR):
    if file_name.lower().endswith(".jpg"):
        frame_files.append(file_name)

frame_files = sorted(frame_files)

results = []

for frame_index, file_name in enumerate(frame_files):
    image_path = os.path.join(FRAMES_DIR, file_name)

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    prediction_index = torch.argmax(probabilities).item()
    prediction_name = class_names[prediction_index]
    confidence = probabilities[prediction_index].item() * 100

    result = {
        "frame_index": frame_index,
        "frame_file": file_name,
        "prediction": prediction_name,
        "confidence_percent": round(confidence, 2)
    }

    for class_index, class_name in enumerate(class_names):
        column_name = f"{class_name}_probability"

        result[column_name] = round(
            probabilities[class_index].item() * 100,
            2
        )

    results.append(result)

    print(
        f"{file_name} -> "
        f"{prediction_name} "
        f"({confidence:.2f}%)"
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    CSV_PATH,
    index=False
)

plt.figure(figsize=(12, 6))

for class_name in class_names:
    probability_column = f"{class_name}_probability"

    plt.plot(
        results_df["frame_index"],
        results_df[probability_column],
        marker="o",
        label=class_name
    )

plt.title("Video kadrlarinin sinif ehtimallari")
plt.xlabel("Kadr indeksi")
plt.ylabel("Ehtimal faizi")
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    CHART_PATH,
    dpi=200
)

plt.show()

print("\nVideo analizi tamamlandi.")
print(f"CSV saxlanildi: {CSV_PATH}")
print(f"Qrafik saxlanildi: {CHART_PATH}")