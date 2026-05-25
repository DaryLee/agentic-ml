"""
Pre-built feature extractors, pipeline helpers, and neural architectures.
The agent can import these to avoid reinventing common operations.
This file is FIXED — the agent should not modify it.
"""

import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F


# === Feature extractors ===
# All take X of shape (N, window_size, channels) and return (N, n_features)

def extract_time_domain(X):
    """Standard time-domain stats per channel: mean, std, min, max, range, median,
    skewness, kurtosis, energy, SMA. Returns (N, channels*10)."""
    return np.hstack([
        X.mean(axis=1),
        X.std(axis=1),
        X.min(axis=1),
        X.max(axis=1),
        X.max(axis=1) - X.min(axis=1),
        np.median(X, axis=1),
        np.nan_to_num(skew(X, axis=1)),
        np.nan_to_num(kurtosis(X, axis=1)),
        np.sum(X**2, axis=1),
        np.sum(np.abs(X), axis=1),
    ])


def extract_frequency_domain(X):
    """FFT-based features per channel: mean, std, max magnitude,
    low-freq energy (1-5 bins), mid-freq energy (6-20 bins).
    Returns (N, channels*5)."""
    f = np.abs(np.fft.rfft(X, axis=1))
    return np.hstack([
        f.mean(axis=1),
        f.std(axis=1),
        f.max(axis=1),
        f[:, 1:6, :].mean(axis=1),
        f[:, 6:20, :].mean(axis=1),
    ])


def extract_jerk(X):
    """First derivative (jerk) features: mean, std, energy.
    Returns (N, channels*3)."""
    jerk = np.diff(X, axis=1)
    return np.hstack([
        jerk.mean(axis=1),
        jerk.std(axis=1),
        np.sum(jerk**2, axis=1),
    ])


def extract_magnitude(X):
    """Signal magnitude for 3-axis groups (acc, gyro, mag per IMU).
    Computes magnitude then stats. Returns (N, n_magnitude_features)."""
    features = []
    n_channels = X.shape[2]
    # Process each IMU (13 channels each: temp, acc16[3], acc6[3], gyro[3], mag[3])
    for imu_start in range(0, n_channels, 13):
        # acc16 magnitude
        acc16 = X[:, :, imu_start+1:imu_start+4]
        mag = np.sqrt(np.sum(acc16**2, axis=2))
        features.extend([mag.mean(axis=1, keepdims=True),
                        mag.std(axis=1, keepdims=True)])
        # gyro magnitude
        gyro = X[:, :, imu_start+7:imu_start+10]
        mag_g = np.sqrt(np.sum(gyro**2, axis=2))
        features.extend([mag_g.mean(axis=1, keepdims=True),
                        mag_g.std(axis=1, keepdims=True)])
    return np.hstack(features)


def extract_all(X):
    """Combined feature set: time + frequency + jerk + magnitude."""
    return np.hstack([
        extract_time_domain(X),
        extract_frequency_domain(X),
        extract_jerk(X),
        extract_magnitude(X),
    ])


# === Pipeline helpers ===

def make_pipeline(classifier, scaler="standard", pca_variance=None):
    """Build sklearn Pipeline with optional scaling and PCA.

    Args:
        classifier: sklearn-compatible estimator
        scaler: "standard", "robust", or None
        pca_variance: float (0-1) for PCA variance threshold, or None to skip
    """
    steps = []
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    if pca_variance is not None:
        steps.append(("pca", PCA(n_components=pca_variance, svd_solver="full")))
    steps.append(("clf", classifier))
    return Pipeline(steps)


def encode_labels(y_train, y_test):
    """Encode non-contiguous PAMAP2 labels to 0..N-1. Returns (y_train_enc, y_test_enc, encoder)."""
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    return le.transform(y_train), le.transform(y_test), le


# === Neural Network Architectures ===

class CNN1D(nn.Module):
    """Simple 1D CNN for time-series classification.

    Input: (batch, channels, time_steps)
    Output: (batch, num_classes)
    """
    def __init__(self, in_channels, num_classes, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.conv3 = nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim*2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (batch, channels, time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # (batch, hidden*2)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LSTMClassifier(nn.Module):
    """LSTM for time-series classification.

    Input: (batch, time_steps, channels)
    Output: (batch, num_classes)
    """
    def __init__(self, in_channels, num_classes, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden_dim, num_layers,
                           batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (batch, time, channels)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        x = h_n[-1]  # (batch, hidden_dim)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def train_torch_model(model, X_train, y_train, X_test, y_test,
                     epochs=50, batch_size=64, lr=0.001, device='cpu'):
    """Train a PyTorch model and return predictions.

    Args:
        model: nn.Module with forward(x) -> logits
        X_train: (N, time, channels) or (N, channels, time) numpy array
        y_train: (N,) labels (can be non-contiguous, will be auto-encoded)
        X_test: test data in same format as X_train
        y_test: test labels
        epochs: number of training epochs
        batch_size: batch size for training
        lr: learning rate
        device: 'cpu' or 'cuda'

    Returns:
        y_pred: (N_test,) predicted class indices in original label space
    """
    # Auto-detect if labels need encoding (non-contiguous or not starting from 0)
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    needs_encoding = (unique_labels.min() != 0 or
                     len(unique_labels) != unique_labels.max() + 1)

    if needs_encoding:
        # Encode labels to 0..N-1
        le = LabelEncoder()
        le.fit(unique_labels)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
    else:
        y_train_enc = y_train
        y_test_enc = y_test
        le = None

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train_enc).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    # Training loop
    model.train()
    for epoch in range(epochs):
        # Mini-batch training
        indices = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    # Prediction
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        y_pred_enc = logits.argmax(dim=1).cpu().numpy()

    # Inverse transform predictions back to original label space
    if le is not None:
        y_pred = le.inverse_transform(y_pred_enc)
    else:
        y_pred = y_pred_enc

    return y_pred

