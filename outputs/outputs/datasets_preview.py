import os
import random
import cv2
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
CLASSES = ["galaxy", "nebula", "star_cluster"]

fig, axes = plt.subplots(3, 4, figsize=(12, 10))

for row, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATASET_DIR, class_name)

    image_files = [
        file_name
        for file_name in os.listdir(class_dir)
        if file_name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    selected_files = random.sample(image_files, min(4, len(image_files)))

    for col in range(4):
        axes[row, col].axis("off")

        if col < len(selected_files):
            image_path = os.path.join(class_dir, selected_files[col])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            axes[row, col].imshow(image)
            axes[row, col].set_title(class_name)

plt.tight_layout()

output_path = os.path.join(PROJECT_DIR, "dataset_preview.png")
plt.savefig(output_path, dpi=200)
plt.show()

print(f"Yoxlama sekli saxlanildi: {output_path}")