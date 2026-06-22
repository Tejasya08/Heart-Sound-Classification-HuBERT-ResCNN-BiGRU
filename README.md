# 🫀 HuBERT-ResCNN-BiGRU for Heart Sound Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Medical AI](https://img.shields.io/badge/Application-Medical%20AI-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📖 Overview

This project presents an AI-powered framework for automated heart sound classification using self-supervised audio representations and deep learning. The proposed model combines **HuBERT Large embeddings**, **Residual Convolutional Neural Networks (ResCNN)**, and **Bidirectional Gated Recurrent Units (BiGRU)** to classify phonocardiogram (PCG) recordings into healthy and unhealthy cardiac conditions.

The framework leverages the representational power of self-supervised learning to capture rich acoustic patterns from heart sounds while utilizing deep neural architectures for robust classification.

---

## 🎯 Objectives

* Develop an automated heart sound classification system.
* Detect abnormal cardiac conditions from PCG recordings.
* Utilize self-supervised audio representations for improved feature extraction.
* Learn temporal dependencies in heart sound signals.
* Evaluate performance using standard medical AI metrics.

---

## 🏥 Clinical Motivation

Cardiovascular diseases remain one of the leading causes of mortality worldwide. Early diagnosis through cardiac auscultation can assist healthcare professionals in identifying abnormalities and initiating timely treatment.

Manual interpretation of heart sounds requires expertise and may be subjective. This project explores how deep learning and self-supervised audio models can support intelligent cardiac auscultation systems.

---

## 🧠 Why HuBERT?

Traditional heart sound analysis often relies on handcrafted acoustic features such as:

* MFCC
* Chroma Features
* Spectral Features

HuBERT (Hidden Unit BERT) learns contextual audio representations through self-supervised learning, enabling extraction of richer and more discriminative information directly from raw audio signals.

---


### Processing Pipeline

```text
Heart Sound Recording (.wav)
            │
            ▼
Audio Preprocessing
(Resampling, Normalization, Padding)
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
Bidirectional GRU
            │
            ▼
Fully Connected Layers
            │
            ▼
Healthy / Unhealthy Classification
```

---

## 📂 Dataset

### Data Type

Phonocardiogram (PCG) Recordings

### Classes

* Healthy
* Unhealthy

### Preprocessing

* Resampling to 16 kHz
* Fixed-length padding/truncation
* Signal normalization
* Noise augmentation
* Time shifting
* SpecAugment

---

## 🔬 Model Architecture

### Feature Extraction

* HuBERT Large (`facebook/hubert-large-ll60k`)
* Context-aware audio embeddings
* Feature fusion using the last hidden state and mean-pooled representations

### Deep Learning Backbone

* Residual CNN Blocks
* Batch Normalization
* Adaptive Average Pooling

### Sequence Modeling

* Bidirectional GRU
* Hidden Size: 256
* Number of Layers: 2
* Dropout Regularization

### Classification Head

* Fully Connected Layers
* ReLU Activation
* Softmax Output

---

## 🏋️ Training Configuration

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

---

## 📈 Results

### Performance Metrics

| Metric    | Score      |
| --------- | ---------- |
| Accuracy  | **96.34%** |
| Precision | **98.39%** |
| Recall    | **98.34%** |
| F1-Score  | **98.34%** |
| ROC-AUC   | **99.59%** |

---

## 📊 Confusion Matrix

<p align="center">
  <img src="images/Confusion Matrix.png" width="650">
</p>

### Analysis

| Actual Class | Predicted Healthy | Predicted Unhealthy |
| ------------ | ----------------- | ------------------- |
| Healthy      | 145               | 5                   |
| Unhealthy    | 0                 | 151                 |

### Key Observations

* Only **5 misclassifications** on the validation set.
* All unhealthy samples were correctly identified.
* No unhealthy cases were classified as healthy.
* Strong classification reliability for cardiac screening applications.

---

## 📉 ROC Curve

<p align="center">
  <img src="images/ROC curve.png" width="650">
</p>

The proposed model achieved a **ROC-AUC score of 99.59%**, indicating excellent class separability and diagnostic capability.

---

## 📈 Training Curves

<p align="center">
  <img src="images/Training and Validation Loss.png" width="650">
</p>

The training and validation curves demonstrate stable convergence and effective learning throughout training.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Transformers
* HuBERT
* Torchaudio
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Librosa

---

## 📁 Repository Structure

```text
HuBERT-ResCNN-BiGRU-for-Heart-Sound-Classification
│
├── README.md
├── requirements.txt
├── train.py
├── predict.py
│── results.txt
├── images/
│   ├── confusion_matrix.png
│   ├── ROC curve.png
│   ├── Training and validation Loss.png
│   └── Validation Accuracy.png
│

```

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/HuBERT-ResCNN-BiGRU-for-Heart-Sound-Classification.git

cd HuBERT-ResCNN-BiGRU-for-Heart-Sound-Classification

pip install -r requirements.txt
```

---

## ▶️ Training

```bash
python train.py
```

---

## 🔮 Inference

```bash
python predict.py
```

---

## 🌍 Applications

* Computer-Aided Cardiac Diagnosis
* Healthcare AI Systems
* Smart Stethoscopes
* Remote Patient Monitoring
* Biomedical Signal Processing
* Clinical Decision Support Systems

---

## 🔮 Future Work

* Attention-Based Architectures
* Explainable AI for Medical Decision Support
* Multi-Class Heart Disease Classification
* Edge AI Deployment
* ONNX Optimization
* Real-Time Cardiac Monitoring

---

## 📄 Results Summary

The proposed HuBERT-ResCNN-BiGRU framework demonstrates strong performance for automated heart sound classification, achieving:

✅ 96.34% Accuracy
✅ 98.39% Precision
✅ 98.34% Recall
✅ 98.34% F1-Score
✅ 99.59% ROC-AUC

These results highlight the effectiveness of combining self-supervised audio representations with deep neural architectures for intelligent cardiac auscultation.

---

## 👨‍💻 Author

**Tejasya Vashisht**

Bachelor of Engineering (Electrical and Computer Engineering)

Thapar Institute of Engineering and Technology, Patiala, India

Research Interests:

* Medical AI
* Deep Learning
* Computer Vision
* Signal Processing
* Healthcare Analytics

---

⭐ If you find this project useful, consider giving the repository a star.
