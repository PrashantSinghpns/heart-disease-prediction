import argparse  # Accept the dataset path from the command line.
import hashlib  # Record which dataset produced the artifacts.
import importlib.metadata  # Record package versions used for training.
import json  # Save evaluation results in a readable format.
from pathlib import Path  # Handle portable file paths.
import joblib  # Save fitted models and preprocessing.
import numpy as np  # Represent missing readings.
import pandas as pd  # Load and encode the data.
from sklearn.impute import SimpleImputer  # Learn missing-value replacements from training data.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # Evaluate held-out predictions.
from sklearn.model_selection import train_test_split  # Keep a separate test set.
from sklearn.neighbors import KNeighborsClassifier  # Train the project's KNN model.
from sklearn.preprocessing import StandardScaler  # Learn feature scaling from training data.

ROOT = Path(__file__).resolve().parent  # Resolve all output paths from this script.

def main():  # Train and export a complete reproducible model bundle.
    parser = argparse.ArgumentParser(description="Train heart disease KNN")  # Describe the training command.
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "heart.csv")  # Allow an external dataset path.
    args = parser.parse_args()  # Read the supplied dataset path.
    df = pd.read_csv(args.data)  # Load the original unscaled data.
    df[["Cholesterol", "RestingBP"]] = df[["Cholesterol", "RestingBP"]].replace(0, np.nan)  # Mark invalid zero readings as missing.
    X = pd.get_dummies(df.drop(columns="HeartDisease"), drop_first=True, dtype=int)  # Encode categories while preserving numeric decimals.
    y = df["HeartDisease"]  # Keep the target out of the features.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Create a reproducible stratified holdout.
    imputer = SimpleImputer(strategy="mean").set_output(transform="pandas")  # Preserve feature names during imputation.
    train_filled = imputer.fit_transform(X_train)  # Learn means from the training subset only.
    test_filled = imputer.transform(X_test)  # Reuse those means for the test subset.
    scaler = StandardScaler()  # Create one shared scaler.
    train_scaled = scaler.fit_transform(train_filled)  # Fit scaling on training data only.
    test_scaled = scaler.transform(test_filled)  # Reuse training scaling for test data.
    model = KNeighborsClassifier(n_neighbors=5)  # Use the original model's neighbor count.
    model.fit(train_scaled, y_train)  # Train on the training subset only.
    predictions = model.predict(test_scaled)  # Predict unseen test rows.
    metrics = {  # Record observed evaluation and provenance.
        "model": "KNeighborsClassifier", "n_neighbors": 5,  # Describe the fitted model.
        "random_state": 42, "test_size": 0.2, "stratified": True,  # Document the holdout split.
        "training_rows": len(X_train), "test_rows": len(X_test),  # Record evaluation sizes.
        "accuracy": accuracy_score(y_test, predictions),  # Save unrounded accuracy.
        "f1_score": f1_score(y_test, predictions, zero_division=0),  # Save F1 for disease class 1.
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),  # Record errors in class order 0, 1.
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),  # Save precision and recall.
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),  # Identify the exact source data.
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib", "streamlit"]},  # Record artifact compatibility.
    }  # Finish metadata.
    model_dir = ROOT / "models"  # Keep all deployment artifacts together.
    model_dir.mkdir(exist_ok=True)  # Create the artifact folder if needed.
    for name, value in {"KNN_heart.pkl": model, "imputer.pkl": imputer, "scaler.pkl": scaler, "columns.pkl": X.columns.tolist()}.items():  # Save the matching bundle.
        joblib.dump(value, model_dir / name)  # Persist each fitted artifact.
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")  # Save readable results.
    print(f"Accuracy: {metrics['accuracy']:.4f}; F1: {metrics['f1_score']:.4f}")  # Display measured scores.

if __name__ == "__main__":  # Run training only when invoked directly.
    main()  # Start the training workflow.
