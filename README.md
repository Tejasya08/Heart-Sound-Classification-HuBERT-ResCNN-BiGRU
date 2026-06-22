# 🫀 HuBERT-ResCNN-BiGRU for Heart Sound Classification

## Overview

This project presents an AI-powered framework for automated heart sound classification using self-supervised audio representations and deep learning. The system analyzes phonocardiogram (PCG) recordings and classifies them into healthy and unhealthy cardiac conditions.

The proposed architecture combines **HuBERT Large**, **Residual Convolutional Neural Networks (ResCNN)**, and **Bidirectional Gated Recurrent Units (BiGRU)** to capture both local acoustic features and long-term temporal dependencies within heart sound signals.

---

## Key Features

* Self-supervised feature extraction using HuBERT Large
* Residual CNN-based feature refinement
* Bidirectional GRU for temporal sequence modeling
* Class imbalance handling using weighted sampling
* Data augmentation and SpecAugment
* Comprehensive evaluation using medical AI metrics
* High-performance binary heart sound classification

---

## Architecture

```text
Heart Sound Recording
          │
          ▼
Audio Preprocessing
(Resampling, Padding, Normalization)
          │
          ▼
HuBERT Large Feature Extraction
          │
          ▼
Feature Fusion
          │
          ▼
Residual CNN Blocks
          │
          ▼
BiGRU Layers
          │
          ▼
Fully Connected Layers
          │
          ▼
Healthy / Unhealthy Classification
```

---

## Dataset

The model is trained on phonocardiogram (PCG) recordings containing healthy and pathological heart sounds.

### Preprocessing Steps

* Resampling to 16 kHz
* Fixed-length padding/truncation
* Signal normalization
* Noise augmentation
* Time shifting
* SpecAugment

---

## Model Components

### 1. HuBERT Large

A self-supervised speech representation model used to extract robust acoustic embeddings from heart sound recordings.

### 2. Residual CNN

Residual convolutional blocks learn local cardiac acoustic patterns while reducing degradation issues in deep networks.

### 3. Bidirectional GRU

BiGRU layers capture temporal relationships and contextual information from sequential heart sound features.

### 4. Classification Head

Fully connected layers with dropout regularization perform the final binary classification.

---

## Training Configuration

| Parameter     | Value              |
| ------------- | ------------------ |
| Framework     | PyTorch            |
| Optimizer     | Adam               |
| Learning Rate | 1e-4               |
| Batch Size    | 32                 |
| Epochs        | 100                |
| Weight Decay  | 1e-5               |
| Scheduler     | Cosine Annealing   |
| Loss Function | Cross Entropy Loss |
| Hardware      | NVIDIA GPU         |

---

## Results

### Performance Metrics

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 96.34% |
| Precision | 98.39% |
| Recall    | 98.34% |
| F1 Score  | 98.34% |
| ROC-AUC   | 99.59% |

---

## Confusion Matrix

| Actual Class | Predicted Healthy | Predicted Unhealthy |
| ------------ | ----------------- | ------------------- |
| Healthy      | 145               | 5                   |
| Unhealthy    | 0                 | 151                 |

### Observations

* Only 5 misclassifications were observed.
* No unhealthy samples were incorrectly classified as healthy.
* The model achieved excellent discrimination capability with a ROC-AUC of 99.59%.

---

## Repository Structure

```text
HuBERT-ResCNN-BiGRU-for-Heart-Sound-Classification
│
├── README.md
├── requirements.txt
├── train.py
├── predict.py
├── model.py
│
├── checkpoints/
│   └── best_hybrid_model.pth
│
├── images/
│   ├── architecture.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_curve.png
│
├── results/
│   ├── metrics.txt
│   └── evaluation_plots
│
└── samples/
    ├── healthy.wav
    └── unhealthy.wav
```

---

## Installation

```bash
git clone https://github.com/yourusername/HuBERT-ResCNN-BiGRU-for-Heart-Sound-Classification.git

cd HuBERT-ResCNN-BiGRU-for-Heart-Sound-Classification

pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python train.py
```

### Inference

```bash
python predict.py
```

---

## Applications

* Computer-Aided Cardiac Diagnosis
* Healthcare AI
* Clinical Decision Support Systems
* Remote Patient Monitoring
* Smart Stethoscope Systems
* Biomedical Signal Analysis

---

## Future Work

* Attention-based architectures
* Explainable AI for cardiac diagnosis
* Multi-class heart disease classification
* Real-time deployment on edge devices
* ONNX optimization and model compression
* Integration with digital stethoscopes

---

## Technologies Used

* Python
* PyTorch
* Transformers
* HuBERT
* Torchaudio
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Librosa

---

## Author

**Tejasya Vashisht**

Bachelor of Engineering (Electrical and Computer Engineering)

Thapar Institute of Engineering and Technology, Patiala

Research Interests:

* Medical Artificial Intelligence
* Deep Learning
* Signal Processing
* Computer Vision
* Healthcare Analytics

---

⭐ If you find this project useful, consider giving the repository a star.
