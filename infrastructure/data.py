"""Fixed data loading infrastructure. The agent CANNOT modify this file."""

import zipfile
import numpy as np
import pandas as pd
import requests
from pathlib import Path

PAMAP2_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"

DATASET_DIR = Path(__file__).parent.parent / "data" / "PAMAP2_Dataset"

ACTIVITY_LABELS = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "nordic_walking",
    12: "ascending_stairs", 13: "descending_stairs",
    16: "vacuum_cleaning", 17: "ironing", 24: "rope_jumping",
}

SENSORS = ["hand", "chest", "ankle"]
SENSOR_CHANNELS = ["acc16_x", "acc16_y", "acc16_z", "acc6_x", "acc6_y", "acc6_z",
                   "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]

TRAIN_SUBJECTS = [101, 102, 103, 104, 105, 106, 107]
TEST_SUBJECTS = [108, 109]


def _build_column_names():
    imu_cols = ["temp"] + SENSOR_CHANNELS + ["orient_1", "orient_2", "orient_3", "orient_4"]
    cols = ["timestamp", "activityID", "heartrate"]
    for sensor in SENSORS:
        cols += [f"{sensor}_{c}" for c in imu_cols]
    return cols


COLUMN_NAMES = _build_column_names()
ORIENT_COLS = [c for c in COLUMN_NAMES if "orient" in c]
DROP_COLS = ["timestamp", "heartrate"] + ORIENT_COLS


def _download_dataset():
    if DATASET_DIR.exists() and (DATASET_DIR / "Protocol").exists():
        return DATASET_DIR

    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    zip_path = DATASET_DIR.parent / "pamap2.zip"

    if not zip_path.exists():
        print("Downloading PAMAP2 dataset...")
        try:
            response = requests.get(PAMAP2_URL, stream=True, verify=True)
            response.raise_for_status()
        except requests.exceptions.SSLError:
            response = requests.get(PAMAP2_URL, stream=True, verify=False)
            response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATASET_DIR.parent)
    zip_path.unlink()

    return DATASET_DIR


def load_subject(subject_id):
    filepath = DATASET_DIR / "Protocol" / f"subject{subject_id}.dat"
    df = pd.read_csv(filepath, sep=" ", header=None, names=COLUMN_NAMES)
    df = df[df["activityID"].isin(ACTIVITY_LABELS)].copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.interpolate(method="linear", limit=10).dropna()
    return df


def segment_windows(df, window_size, overlap):
    step = max(int(window_size * (1 - overlap)), 1)
    activity_col = df["activityID"].values
    data_cols = df.drop(columns=["activityID"]).values

    windows, labels = [], []
    for start in range(0, len(df) - window_size + 1, step):
        end = start + window_size
        window_labels = activity_col[start:end]
        if len(np.unique(window_labels)) == 1:
            windows.append(data_cols[start:end])
            labels.append(window_labels[0])

    if not windows:
        return np.array([]).reshape(0, window_size, data_cols.shape[1]), np.array([])
    return np.array(windows), np.array(labels, dtype=int)


def load_data(window_size=200, overlap=0.5):
    if not DATASET_DIR.exists():
        _download_dataset()

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for sid in TRAIN_SUBJECTS:
        try:
            df = load_subject(sid)
            X, y = segment_windows(df, window_size, overlap)
            if len(X) > 0:
                X_train_list.append(X)
                y_train_list.append(y)
        except Exception as e:
            print(f"  Warning: skipping subject {sid}: {e}")

    for sid in TEST_SUBJECTS:
        try:
            df = load_subject(sid)
            X, y = segment_windows(df, window_size, overlap)
            if len(X) > 0:
                X_test_list.append(X)
                y_test_list.append(y)
        except Exception as e:
            print(f"  Warning: skipping subject {sid}: {e}")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    print(f"Data loaded: train={X_train.shape}, test={X_test.shape}, "
          f"classes={len(np.unique(y_train))}")

    return X_train, y_train, X_test, y_test
