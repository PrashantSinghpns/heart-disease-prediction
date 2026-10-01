"""Inference contract for the repository's existing KNN artifact bundle."""
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"
NUMERIC = ("Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak")
CATEGORIES = {"Sex": {"M", "F"}, "ChestPainType": {"ATA", "NAP", "TA", "ASY"},
              "RestingECG": {"Normal", "ST", "LVH"}, "ExerciseAngina": {"N", "Y"},
              "ST_Slope": {"Up", "Flat", "Down"}}

def load_artifacts():
    """Load only the trusted repository bundle; joblib can execute serialized code."""
    return tuple(joblib.load(MODEL_DIR / name) for name in
                 ("KNN_heart.pkl", "imputer.pkl", "scaler.pkl", "columns.pkl"))

def prepare_input(raw_input, columns):
    """Apply the same zero-to-missing rule and categorical encoding as training."""
    required = set(NUMERIC) | set(CATEGORIES)
    missing = sorted(required - set(raw_input))
    if missing:
        raise ValueError(f"Missing input features: {missing}")
    for feature, valid in CATEGORIES.items():
        if raw_input[feature] not in valid:
            raise ValueError(f"Unsupported category for {feature}")
    for feature in NUMERIC:
        value = raw_input[feature]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{feature} must be a finite number")
    if raw_input["FastingBS"] not in {0, 1}:
        raise ValueError("FastingBS must be 0 or 1")
    frame = pd.DataFrame([{name: raw_input[name] for name in required}])
    frame[["Cholesterol", "RestingBP"]] = frame[["Cholesterol", "RestingBP"]].replace(0, np.nan)
    frame = pd.get_dummies(frame, dtype=int)
    return frame.reindex(columns=columns, fill_value=0)

def predict(raw_input, artifacts):
    model, imputer, scaler, columns = artifacts
    frame = prepare_input(raw_input, columns)
    return int(model.predict(scaler.transform(imputer.transform(frame)))[0])
