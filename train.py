"""
train.py
--------
Training script for HuBERT + ResCNN-BiGRU heart sound classification.

Usage:
    python train.py --data_path /path/to/dataset --epochs 100 --batch_size 32
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.preprocessing import label_binarize
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking

# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train HuBERT-ResCNN-BiGRU on heart sounds")
    parser.add_argument("--data_path",   type=str,   default="./dataset",  help="Path to dataset root")
    parser.add_argument("--epochs",      type=int,   default=100,          help="Number of training epochs")
    parser.add_argument("--batch_size",  type=int,   default=32,           help="Batch size")
    parser.add_argument("--lr",          type=float, default=1e-4,         help="Learning rate")
    parser.add_argument("--max_duration",type=int,   default=10,           help="Max audio duration in seconds")
    parser.add_argument("--save_dir",    type=str,   default=".",          help="Directory to save model and plots")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────────
def add_noise(waveform, noise_level=0.005):
    return waveform + torch.randn_like(waveform) * noise_level

def time_shift(waveform, shift_max=0.2):
    shift_amt = int(shift_max * waveform.shape[1])
    shift = np.random.randint(-shift_amt, shift_amt)
    return torch.roll(waveform, shifts=shift, dims=1)

def normalize(waveform):
    return waveform / (torch.max(torch.abs(waveform)) + 1e-6)

def spec_augment(specgram, freq_mask_param=15, time_mask_param=35):
    specgram = FrequencyMasking(freq_mask_param)(specgram)
    specgram = TimeMasking(time_mask_param)(specgram)
    return specgram


# ─────────────────────────────────────────────
# Audio Loading & Feature Preparation
# ─────────────────────────────────────────────
def load_and_resample(file_path, target_sr=16000, max_duration=10):
    waveform, sample_rate = torchaudio.load(file_path)
    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to target rate
    waveform = T.Resample(orig_freq=sample_rate, new_freq=target_sr)(waveform)

    # Trim or pad to fixed length
    target_samples = max_duration * target_sr
    if waveform.size(1) > target_samples:
        waveform = waveform[:, :target_samples]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.size(1)))

    # Clean waveform for HuBERT (normalize only)
    clean_waveform = normalize(waveform)

    # Augmented waveform for Mel Spectrogram
    aug_waveform = normalize(add_noise(time_shift(waveform)))
    mel_specgram = spec_augment(
        MelSpectrogram(sample_rate=target_sr, n_fft=1024, hop_length=512, n_mels=32)(aug_waveform)
    )

    return clean_waveform, mel_specgram, target_sr


# ─────────────────────────────────────────────
# HuBERT Feature Extraction
# ─────────────────────────────────────────────
def extract_features(file_list, labels_dict, label_mapping, feature_extractor,
                     hubert_model, max_duration=10):
    features, labels = [], []

    for file in tqdm(file_list, desc="Extracting features"):
        waveform, _, _ = load_and_resample(file, max_duration=max_duration)

        with torch.no_grad():
            inputs = feature_extractor(
                waveform.cpu().numpy().squeeze(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            # Build attention mask if missing
            if "attention_mask" not in inputs:
                attn_mask = torch.ones(inputs["input_values"].shape, dtype=torch.long)
                attn_mask = attn_mask.masked_fill(inputs["input_values"] == 0, 0)
                inputs["attention_mask"] = attn_mask

            inputs = {k: v.to(device) for k, v in inputs.items()}
            hidden = hubert_model(**inputs).last_hidden_state  # [1, T, 1024]

            # Combine last token + mean pool → [1, 2048]  (hubert-large: 1024-dim)
            last   = hidden[:, -1, :]       # [1, 1024]
            pooled = hidden.mean(dim=1)     # [1, 1024]

        combined = torch.cat((last, pooled), dim=-1).cpu().view(-1)  # [2048]
        features.append(combined)
        labels.append(label_mapping[labels_dict[os.path.basename(file)]])

    return {
        "features": torch.stack(features),
        "labels":   torch.tensor(labels, dtype=torch.long)
    }


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class HuBERTResCNNBiGRU(nn.Module):
    """
    Hybrid architecture:
      1. ResCNN backbone — extracts local temporal features from HuBERT embeddings
      2. Bidirectional GRU  — models sequential dependencies
      3. FC classifier       — final prediction
    """
    def __init__(self, input_dim=2048, hidden_dim=256, num_classes=2):
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(64,  128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )

        self.gru = nn.GRU(
            512, hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x   = x.unsqueeze(1)                       # [B, 1, input_dim]
        x   = self.initial_conv(x)                 # [B, 64, L]
        x   = self.res_blocks(x).squeeze(2)        # [B, 512]
        x, _ = self.gru(x.unsqueeze(1))            # [B, 1, hidden*2]
        return self.classifier(x.squeeze(1))       # [B, num_classes]


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate(model, loader, label_mapping):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            preds   = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted")
    rec  = recall_score(all_labels, all_preds, average="weighted")
    f1   = f1_score(all_labels, all_preds, average="weighted")
    auc  = roc_auc_score(all_labels, [p[1] for p in all_probs])
    return acc, prec, rec, f1, auc, all_labels, all_preds, all_probs


# ─────────────────────────────────────────────
# Plotting Utilities
# ─────────────────────────────────────────────
def plot_confusion_matrix(all_labels, all_preds, label_mapping, save_dir):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_mapping.keys(),
                yticklabels=label_mapping.keys())
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "images", "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved: images/confusion_matrix.png")


def plot_roc_curve(all_labels, all_probs, save_dir):
    fpr, tpr, _ = roc_curve(all_labels, [p[1] for p in all_probs])
    roc_auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "images", "roc_curve.png"), dpi=150)
    plt.close()
    print("Saved: images/roc_curve.png")


def plot_training_curves(train_losses, val_losses, val_accuracies, save_dir):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses,   label="Val Loss",   marker="o")
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Val Accuracy", marker="o", color="green")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "images", "training_curve.png"), dpi=150)
    plt.close()
    print("Saved: images/training_curve.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(os.path.join(args.save_dir, "images"),  exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "results"), exist_ok=True)

    # ── Load HuBERT ──────────────────────────
    print("Loading HuBERT model...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ll60k")
    hubert_model      = HubertModel.from_pretrained("facebook/hubert-large-ll60k").to(device)
    hubert_model.eval()

    # ── Dataset Paths ────────────────────────
    train_path = os.path.join(args.data_path, "train")
    val_path   = os.path.join(args.data_path, "val")
    train_files = glob.glob(os.path.join(train_path, "**", "*.wav"), recursive=True)
    val_files   = glob.glob(os.path.join(val_path,   "**", "*.wav"), recursive=True)

    labels_path = os.path.join(args.data_path, "labels.json")
    with open(labels_path, "r") as f:
        labels_dict = json.load(f)

    label_mapping = {label: idx for idx, label in enumerate(sorted(set(labels_dict.values())))}
    num_classes   = len(label_mapping)
    print(f"Classes: {label_mapping}")

    # ── Feature Extraction ───────────────────
    print("\nExtracting training features...")
    train_data = extract_features(train_files, labels_dict, label_mapping,
                                  feature_extractor, hubert_model, args.max_duration)
    print("Extracting validation features...")
    val_data   = extract_features(val_files, labels_dict, label_mapping,
                                  feature_extractor, hubert_model, args.max_duration)

    # ── Class Weights & Sampler ──────────────
    train_labels_np = train_data["labels"].cpu().numpy()
    class_weights   = compute_class_weight("balanced",
                                           classes=np.unique(train_labels_np),
                                           y=train_labels_np)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    sample_weights  = np.array([class_weights[l] for l in train_labels_np])
    sampler         = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double),
                                            num_samples=len(sample_weights),
                                            replacement=True)

    # ── DataLoaders ──────────────────────────
    train_loader = DataLoader(TensorDataset(train_data["features"], train_data["labels"]),
                              batch_size=args.batch_size, sampler=sampler)
    val_loader   = DataLoader(TensorDataset(val_data["features"], val_data["labels"]),
                              batch_size=args.batch_size, shuffle=False)

    # ── Model, Optimizer, Scheduler ─────────
    input_dim = train_data["features"].shape[1]
    model     = HuBERTResCNNBiGRU(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    # ── Training Loop ────────────────────────
    best_acc = 0.0
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            l1_reg  = 1e-6 * sum(torch.norm(p, 1) for p in model.parameters())
            loss    = criterion(outputs, labels) + l1_reg
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        correct, total, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                v_loss += criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total
        val_losses.append(v_loss / len(val_loader))
        val_accuracies.append(val_acc)
        scheduler.step(v_loss / len(val_loader))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, "best_model.pth"))

        print(f"Epoch [{epoch+1:3d}/{args.epochs}] | "
              f"Train Loss: {train_losses[-1]:.4f} | "
              f"Val Loss: {val_losses[-1]:.4f} | "
              f"Val Acc: {val_acc*100:.2f}%")

    # ── Final Evaluation ─────────────────────
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pth")))
    acc, prec, rec, f1, auc, all_labels, all_preds, all_probs = evaluate(
        model, val_loader, label_mapping
    )

    print(f"\n{'='*45}")
    print(f"  Final Accuracy : {acc  * 100:.2f}%")
    print(f"  Precision      : {prec * 100:.2f}%")
    print(f"  Recall         : {rec  * 100:.2f}%")
    print(f"  F1-Score       : {f1   * 100:.2f}%")
    print(f"  ROC-AUC        : {auc  * 100:.2f}%")
    print(f"{'='*45}\n")

    # ── Save Metrics ─────────────────────────
    metrics_path = os.path.join(args.save_dir, "results", "metrics.txt")
    with open(metrics_path, "w") as mf:
        mf.write("Heart Sound Classification — Model Evaluation Results\n")
        mf.write("="*52 + "\n\n")
        mf.write(f"Accuracy  : {acc  * 100:.2f}%\n")
        mf.write(f"Precision : {prec * 100:.2f}%\n")
        mf.write(f"Recall    : {rec  * 100:.2f}%\n")
        mf.write(f"F1-Score  : {f1   * 100:.2f}%\n")
        mf.write(f"ROC-AUC   : {auc  * 100:.2f}%\n\n")
        mf.write("Model     : HuBERT-ResCNN-BiGRU\n")
        mf.write(f"Input Dim : {input_dim}\n")
        mf.write(f"Classes   : {label_mapping}\n")
    print(f"Saved: {metrics_path}")

    # ── Save Plots ───────────────────────────
    plot_confusion_matrix(all_labels, all_preds, label_mapping, args.save_dir)
    plot_roc_curve(all_labels, all_probs, args.save_dir)
    plot_training_curves(train_losses, val_losses, val_accuracies, args.save_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
