# Cosmic Vision AI

Deep Learning and Computer Vision project for analyzing space telescope imagery and video frames.

The project uses real NASA image data to train a model that classifies astronomical images into three categories:

- Galaxy
- Nebula
- Star Cluster

## Project Features

- NASA image dataset downloader
- Dataset preview and validation
- OpenCV video frame extraction
- Image contrast enhancement
- Canny edge detection
- ResNet18 transfer learning model
- Galaxy / nebula / star cluster classification
- Single image prediction
- Full video-frame prediction analysis
- CSV reports and visualization charts

## Dataset

The dataset contains 60 real NASA telescope images:

| Class | Image count |
|---|---:|
| Galaxy | 20 |
| Nebula | 20 |
| Star cluster | 20 |

Image source metadata is saved in `dataset_sources.csv`.

Source: [NASA Image and Video Library API](https://images.nasa.gov/)

## Technologies

- Python
- PyTorch
- Torchvision
- OpenCV
- Pandas
- Matplotlib
- Scikit-learn
- NASA Images API

## Model

The classifier uses a pretrained ResNet18 model.

- Device: Apple Silicon MPS
- Training images: 48
- Validation images: 12
- Validation accuracy: 75%
- Epochs: 10

> The dataset is intentionally small for an educational prototype. Adding more labeled images will make the model more reliable.

## Project Structure

```text
cosmic-vision-ai/
│
├── dataset/
│   ├── galaxy/
│   ├── nebula/
│   └── star_cluster/
│
├── outputs/
│   ├── Videos/
│   ├── video_frames/
│   ├── space_classifier_resnet18.pth
│   ├── training_history.png
│   ├── confusion_matrix_space.png
│   ├── video_predictions.csv
│   ├── video_predictions_chart.png
│   ├── Space_video_analysis.py
│   ├── train_space_classifier.py
│   └── predict_space_image.py
│
├── dataset_sources.csv
├── pyproject.toml
└── README.md
