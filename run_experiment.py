"""
Simple baseline experiment
Receives: X_train, y_train, X_test, y_test as global numpy arrays.
Must print: RESULT: f1_macro=X.XXXX accuracy=X.XXXX
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# Simple feature extraction: mean and std per channel
def extract_features(X):
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    return np.hstack([mean, std])

X_train_feat = extract_features(X_train)
X_test_feat = extract_features(X_test)

# Basic Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_feat, y_train)
y_pred = model.predict(X_test_feat)

f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
acc = accuracy_score(y_test, y_pred)

print(f"RESULT: f1_macro={f1:.4f} accuracy={acc:.4f}")