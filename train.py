import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent

def encode_split_features(training, test):
    """Learn category columns from training data, then align held-out encodings."""
    encoded_training = pd.get_dummies(training, drop_first=True, dtype=int)
    encoded_test = pd.get_dummies(test, dtype=int).reindex(columns=encoded_training.columns, fill_value=0)
    return encoded_training, encoded_test


def main():
    parser = argparse.ArgumentParser(description="Train heart disease KNN")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "heart.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    required = {"Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
                "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope", "HeartDisease"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[list(sorted(required))].copy()
    df["HeartDisease"] = pd.to_numeric(df["HeartDisease"], errors="raise")
    if df["HeartDisease"].isna().any() or set(df["HeartDisease"].unique()) != {0, 1}:
        raise ValueError("HeartDisease requires both classes with labels 0 and 1")
    df[["Cholesterol", "RestingBP"]] = df[["Cholesterol", "RestingBP"]].replace(0, np.nan)
    X = df.drop(columns="HeartDisease")
    y = df["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test = encode_split_features(X_train, X_test)
    imputer = SimpleImputer(strategy="mean").set_output(transform="pandas")
    train_filled = imputer.fit_transform(X_train)
    test_filled = imputer.transform(X_test)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_filled)
    test_scaled = scaler.transform(test_filled)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_scaled, y_train)
    predictions = model.predict(test_scaled)
    metrics = {
        "model": "KNeighborsClassifier", "n_neighbors": 5,
        "random_state": 42, "test_size": 0.2, "stratified": True,
        "training_rows": len(X_train), "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib", "streamlit"]},
    }
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    for name, value in {"KNN_heart.pkl": model, "imputer.pkl": imputer, "scaler.pkl": scaler, "columns.pkl": X_train.columns.tolist()}.items():
        joblib.dump(value, model_dir / name)
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Accuracy: {metrics['accuracy']:.4f}; F1: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()
