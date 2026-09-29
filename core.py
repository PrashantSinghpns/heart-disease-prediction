from pathlib import Path  # Resolve model paths relative to this project.
import joblib  # Read the trained artifacts.
import pandas as pd  # Build and encode prediction input.

MODEL_DIR = Path(__file__).resolve().parent / "models"  # Work from any launch folder.

def load_artifacts():  # Load matching preprocessing and model files together.
    return tuple(joblib.load(MODEL_DIR / name) for name in ("KNN_heart.pkl", "imputer.pkl", "scaler.pkl", "columns.pkl"))  # Preserve bundle order.

def prepare_input(raw_input, columns):  # Match the feature encoding used in training.
    frame = pd.DataFrame([raw_input])  # Create one row of raw values.
    frame = pd.get_dummies(frame, dtype=int)  # Encode M/F, chest pain, ECG, angina and slope.
    return frame.reindex(columns=columns, fill_value=0)  # Restore training column order and absent categories.

def predict(raw_input, artifacts):  # Apply the complete saved preprocessing sequence.
    model, imputer, scaler, columns = artifacts  # Unpack the matching bundle.
    frame = prepare_input(raw_input, columns)  # Encode the user's raw input.
    filled = imputer.transform(frame)  # Apply training means for missing readings.
    scaled = scaler.transform(filled)  # Apply training scaling exactly once.
    return int(model.predict(scaled)[0])  # Return class 0 or class 1.
