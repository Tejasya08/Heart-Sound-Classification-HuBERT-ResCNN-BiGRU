"""
predict.py
----------
Run inference on a single heart sound audio file using a trained
HuBERT-ResCNN-BiGRU model.

Usage:
    python predict.py --audio /path/to/audio.wav --model best_model.pth
"""

import os
import json
import argparse
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Heart sound prediction")
    parser.add_argument("--audio",        type=str, required=True,
                        help="Path to input .wav file")
    parser.add_argument("--model",        type=str, default="best_model.pth",
                        help="Path to trained model weights (.pth)")
    parser.add_argument("--label_map",    type=str, default=None,
                        help="Path to label_mapping.json (optional)")
    parser.add_argument("--max_duration", type=int, default=10,
                        help="Max audio duration in seconds (must match training)")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# Model Architecture (must match train.py)
# ─────────────────────────────────────────────
import torch.nn as nn

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
        self.gru = nn.GRU(512, hidden_dim, batch_first=True,
                          bidirectional=True, num_layers=2, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x    = x.unsqueeze(1)
        x    = self.initial_conv(x)
        x    = self.res_blocks(x).squeeze(2)
        x, _ = self.gru(x.unsqueeze(1))
        return self.classifier(x.squeeze(1))


# ─────────────────────────────────────────────
# Audio Preprocessing
# ─────────────────────────────────────────────
def normalize(waveform):
    return waveform / (torch.max(torch.abs(waveform)) + 1e-6)

def load_audio(file_path, target_sr=16000, max_duration=10):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = T.Resample(orig_freq=sample_rate, new_freq=target_sr)(waveform)
    target_samples = max_duration * target_sr
    if waveform.size(1) > target_samples:
        waveform = waveform[:, :target_samples]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.size(1)))
    return normalize(waveform)


# ─────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────
def extract_hubert_features(waveform, feature_extractor, hubert_model):
    with torch.no_grad():
        inputs = feature_extractor(
            waveform.cpu().numpy().squeeze(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        if "attention_mask" not in inputs:
            attn_mask = torch.ones(inputs["input_values"].shape, dtype=torch.long)
            attn_mask = attn_mask.masked_fill(inputs["input_values"] == 0, 0)
            inputs["attention_mask"] = attn_mask

        inputs = {k: v.to(device) for k, v in inputs.items()}
        hidden = hubert_model(**inputs).last_hidden_state  # [1, T, 1024]

        last   = hidden[:, -1, :]       # [1, 1024]
        pooled = hidden.mean(dim=1)     # [1, 1024]

    return torch.cat((last, pooled), dim=-1).cpu().view(-1)  # [2048]


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict(audio_path, model, feature_extractor, hubert_model,
            label_mapping, max_duration=10):
    waveform = load_audio(audio_path, max_duration=max_duration)
    features = extract_hubert_features(waveform, feature_extractor, hubert_model)

    model.eval()
    with torch.no_grad():
        inputs  = features.unsqueeze(0).to(device)   # [1, 2048]
        outputs = model(inputs)
        probs   = torch.softmax(outputs, dim=1)
        pred_idx    = probs.argmax().item()
        confidence  = probs[0][pred_idx].item() * 100

    idx_to_label = {v: k for k, v in label_mapping.items()}
    return idx_to_label[pred_idx], confidence, probs[0].cpu().numpy()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # Label mapping (default: binary healthy/unhealthy)
    if args.label_map and os.path.exists(args.label_map):
        with open(args.label_map, "r") as f:
            label_mapping = json.load(f)
    else:
        label_mapping = {"healthy": 0, "unhealthy": 1}

    num_classes = len(label_mapping)

    # Load HuBERT
    print("Loading HuBERT...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ll60k")
    hubert_model      = HubertModel.from_pretrained("facebook/hubert-large-ll60k").to(device)
    hubert_model.eval()

    # Load classifier
    print(f"Loading model weights from: {args.model}")
    model = HuBERTResCNNBiGRU(input_dim=2048, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    # Predict
    print(f"\nAnalyzing: {args.audio}")
    label, confidence, probs = predict(
        args.audio, model, feature_extractor, hubert_model,
        label_mapping, args.max_duration
    )

    print(f"\n{'─'*35}")
    print(f"  Prediction : {label.upper()}")
    print(f"  Confidence : {confidence:.2f}%")
    print(f"{'─'*35}")
    for cls, idx in label_mapping.items():
        print(f"  {cls:<12}: {probs[idx]*100:.2f}%")
    print(f"{'─'*35}\n")


if __name__ == "__main__":
    main()
