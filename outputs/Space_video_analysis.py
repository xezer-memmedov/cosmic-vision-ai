import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(
    BASE_DIR,
    "Videos",
    "esawebb_dc2577dc6a8748e0846e2c8f5909a619.mp4"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "video_frames")
FRAME_INTERVAL = 60

os.makedirs(OUTPUT_DIR, exist_ok=True)

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise FileNotFoundError(f"Video acilmadi: {VIDEO_PATH}")

fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"Video FPS: {fps:.2f}")
print(f"Umumi kadr sayi: {total_frames}")
print(f"Video muddeti: {duration:.2f} saniye")

frame_number = 0
saved_frames = 0

while True:
    success, frame = video.read()

    if not success:
        break

    if frame_number % FRAME_INTERVAL == 0:
        frame_path = os.path.join(
            OUTPUT_DIR,
            f"space_frame_{saved_frames:03d}.jpg"
        )

        cv2.imwrite(frame_path, frame)
        saved_frames += 1

    frame_number += 1

video.release()

print(f"{saved_frames} kadr ugurla saxlanildi.")
print(f"Kadrlar burada: {OUTPUT_DIR}")

# Ilk kadrin Computer Vision analizi
SAMPLE_FRAME = os.path.join(OUTPUT_DIR, "space_frame_000.jpg")

image = cv2.imread(SAMPLE_FRAME)

if image is None:
    raise FileNotFoundError(f"Kadr tapilmadi: {SAMPLE_FRAME}")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
enhanced_image = cv2.equalizeHist(gray_image)
edge_image = cv2.Canny(enhanced_image, 30, 100)

analysis_path = os.path.join(BASE_DIR, "space_frame_analysis.png")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original telescope frame")
axes[0].axis("off")

axes[1].imshow(enhanced_image, cmap="gray")
axes[1].set_title("Contrast enhanced")
axes[1].axis("off")

axes[2].imshow(edge_image, cmap="gray")
axes[2].set_title("Canny Edge Detection")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(analysis_path, dpi=200)
plt.show()

print(f"Analysis image saved: {analysis_path}")

# Butun kadrlarin metrik analizi
frame_files = sorted(
    file_name
    for file_name in os.listdir(OUTPUT_DIR)
    if file_name.endswith(".jpg")
)

frame_results = []

for index, file_name in enumerate(frame_files):
    frame_path = os.path.join(OUTPUT_DIR, file_name)
    frame = cv2.imread(frame_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    mean_brightness = gray.mean()
    edge_pixels = cv2.countNonZero(edges)

    _, bright_mask = cv2.threshold(
        gray,
        200,
        255,
        cv2.THRESH_BINARY
    )

    _, _, stats, _ = cv2.connectedComponentsWithStats(
        bright_mask,
        connectivity=8
    )

    bright_regions = sum(
        1
        for area in stats[1:, cv2.CC_STAT_AREA]
        if 3 <= area <= 2000
    )

    frame_results.append(
        {
            "frame_index": index,
            "frame_file": file_name,
            "mean_brightness": round(float(mean_brightness), 2),
            "edge_pixels": int(edge_pixels),
            "bright_regions": bright_regions,
        }
    )

metrics_df = pd.DataFrame(frame_results)

metrics_path = os.path.join(BASE_DIR, "frame_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print("\nButun kadrlar analiz edildi:")
print(metrics_df)
print(f"\nNeticeler saxlanildi: {metrics_path}")

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

axes[0].plot(
    metrics_df["frame_index"],
    metrics_df["mean_brightness"],
    marker="o",
    color="orange"
)
axes[0].set_title("Kadrlar uzre orta parlaqliq")
axes[0].set_ylabel("Parlaqliq")
axes[0].grid(True)

axes[1].plot(
    metrics_df["frame_index"],
    metrics_df["edge_pixels"],
    marker="o",
    color="cyan"
)
axes[1].set_title("Kadrlar uzre kenar piksel sayi")
axes[1].set_xlabel("Kadr nomresi")
axes[1].set_ylabel("Kenar pikselleri")
axes[1].grid(True)

plt.tight_layout()

chart_path = os.path.join(BASE_DIR, "frame_metrics_chart.png")
plt.savefig(chart_path, dpi=200)
plt.show()

print(f"Qrafik saxlanildi: {chart_path}")
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
CLASS_NAMES = ["galaxy", "nebula", "star_cluster"]

for class_name in CLASS_NAMES:
    class_path = os.path.join(DATASET_DIR, class_name)
    os.makedirs(class_path, exist_ok=True)

print(f"Dataset qovluqlari hazirdir: {DATASET_DIR}")