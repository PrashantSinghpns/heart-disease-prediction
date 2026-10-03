# Heart Disease Risk Prediction

[![App and input checks](https://github.com/PrashantSinghpns/heart-disease-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/PrashantSinghpns/heart-disease-prediction/actions/workflows/ci.yml)

[**Try the live Streamlit app →**](https://heart-disease-prediction-edwrnijvqc3zsdlf7otjcp.streamlit.app/)

An interactive machine learning project that predicts the **heart disease class** using a K-nearest neighbors (KNN) classifier and Streamlit.

This project demonstrates data exploration, preprocessing, model comparison, evaluation, and an interactive prediction interface.

> Educational project only. Predictions are not medical diagnoses, and a negative prediction does not rule out heart disease.

## Features

- Clean, wide Streamlit interface with a compact project sidebar.
- Three input sections: Patient Information, Cardiovascular Measurements, and Clinical Information.
- Human-readable category labels mapped to the trained model's exact values.
- Lower/Higher Predicted Risk result cards with the model's positive-class probability.
- Cached artifact loading and clear loading/prediction error messages.
- Comparison of Logistic Regression, KNN, Naive Bayes, Decision Tree, and SVM.
- Consistent category encoding between training and prediction.
- Missing-value replacement and scaling fitted on training data.
- Saved model and preprocessing files.
- Reproducible training script and Jupyter notebook.

## Technologies

Python · pandas · NumPy · scikit-learn · Streamlit · Joblib · Matplotlib · Seaborn

## Model Results

The included KNN classifier uses **5 neighbors**, an **80/20 stratified train/test split**, and `random_state=42`.

| Metric | Score |
|---|---:|
| Test accuracy | 89.13% |
| F1 score for heart disease class | 90.29% |
| Training rows | 734 |
| Test rows | 184 |

These results come from one held-out split. Detailed evaluation is available in `metrics.json`.

## Project Structure

| File or folder | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `core.py` | Input encoding and prediction |
| `train.py` | Training and model export |
| `models/` | Matching model, imputer, scaler, and feature columns |
| `notebooks/heart_disease.ipynb` | Exploration and model comparison |
| `requirements.txt` | App dependencies |
| `requirements-dev.txt` | Additional notebook dependencies |
| `metrics.json` | Evaluation results |
| `tests/` | Input encoding and app checks |

## Run Locally

The redesigned app was tested locally with **Python 3.12** and the pinned dependencies in `requirements.txt`.

```powershell
git clone https://github.com/PrashantSinghpns/heart-disease-prediction.git  # Download the project
cd heart-disease-prediction  # Enter the project folder
python -m venv .venv  # Create a project environment
.\.venv\Scripts\python -m pip install -r requirements.txt  # Install dependencies
.\.venv\Scripts\python -m streamlit run app.py  # Start the app
```

## Dataset

The project uses a 918-row dataset with 11 input features and the target `HeartDisease`.

- **0:** No heart disease label.
- **1:** Heart disease label.

The original dataset source and redistribution terms still need confirmation. The CSV is excluded from this repository; the app runs using the included trained artifacts.

## Retraining

Place an authorized copy of `heart.csv` in the `data/` folder.

```powershell
python train.py --data "data/heart.csv"  # Train and export matching model files
python -m unittest discover -s tests -v  # Run the project checks
```

## Limitations

The model predicts a dataset class. Its displayed KNN probability is the fraction of neighboring training samples in the heart disease class; it is not a calibrated estimate of personal medical risk. The model has not undergone external clinical validation and can make incorrect predictions.

## Author

Prashant Singh


## Inference contract

`app.py` collects 11 inputs using constrained controls and maps the displayed categories to their training values. `core.prepare_input` applies dummy encoding and restores the saved feature-column order. Prediction then applies the saved imputer, scaler, and KNN model. `predict_result` returns the predicted class and the probability for class 1. The redesigned interface reuses the existing artifact bundle without retraining or changing the recorded accuracy figures.

The current core helper does not independently validate arbitrary programmatic inputs or convert zero measurements to missing values. The UI restricts blood pressure and cholesterol to positive values.

The figures in `metrics.json` describe the included bundle's recorded 734/184-row split and dataset hash. The source CSV is not supplied, so its scores cannot be independently reproduced from this checkout. The revised training script learns dummy-column vocabulary from training rows and aligns held-out rows to those columns. Imputation and scaling are also fitted on training rows. The included historical artifact bundle is preserved; rerunning training is required to regenerate it under the revised code. Future evaluation should add CV model selection and external validation.

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q
```

CI checks the actual artifact bundle and Streamlit form on Python 3.14. Only load trusted joblib/pickle artifacts and preserve their training dependency versions.
