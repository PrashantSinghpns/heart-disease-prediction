# Heart Disease Prediction

An interactive machine learning project that predicts the **heart disease class** using a K-nearest neighbors (KNN) classifier and Streamlit.

This project demonstrates data exploration, preprocessing, model comparison, evaluation, and an interactive prediction interface.

> Educational project only. Predictions are not medical diagnoses, and a negative prediction does not rule out heart disease.

## Features

- Interactive sliders and category selectors.
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

Use **Python 3.14** with the pinned dependencies.

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

The model predicts a dataset class rather than a personal risk probability. It has not undergone external clinical validation and can make incorrect predictions.

## Author

Prashant Singh
