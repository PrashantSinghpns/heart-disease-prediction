import streamlit as st
from core import load_artifacts, predict_result  # Reuse the existing trained inference bundle.

st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="🩺", layout="wide")

# Use stable containers and our own cards; retain accessible native controls.
st.markdown("""
<style>
.stMainBlockContainer {max-width: 1180px; padding-top: 2.5rem; padding-bottom: 2rem;}
h1 {letter-spacing: -0.04em; font-weight: 750 !important;}
h3 {letter-spacing: -0.02em;}
[data-testid="stForm"] {border: 1px solid #dce5ec; border-radius: 18px; padding: 1.5rem; background: #fff;}
[data-testid="stSidebar"] {border-right: 1px solid #dce5ec;}
.eyebrow {color: #147d82; font-size: .75rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; margin-bottom: .6rem;}
.result-card {border: 1px solid #c6e1dd; border-left: 5px solid #167b72; border-radius: 16px; background: #f0faf7; padding: 1.6rem; margin-top: 1rem; color: #163331;}
.result-card.higher {border-color: #ead4b5; border-left-color: #ad702b; background: #fff8ee; color: #503716;}
.result-card h2 {font-size: 1.7rem; margin: .4rem 0; color: inherit;}
.result-card p {margin-bottom: .3rem;}
.result-label {font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;}
@media (max-width: 640px) {
 .stMainBlockContainer {padding: 1.25rem 1rem;}
 [data-testid="stForm"] {padding: 1rem;}
 .result-card {padding: 1.1rem;}
}
</style>
""", unsafe_allow_html=True)

# Readable labels map to the exact values used during training.
SEX = {"Male": "M", "Female": "F"}
CHEST_PAIN = {"Atypical angina": "ATA", "Non-anginal pain": "NAP", "Typical angina": "TA", "Asymptomatic": "ASY"}
ECG = {"Normal": "Normal", "ST–T wave abnormality": "ST", "Left ventricular hypertrophy": "LVH"}
ANGINA = {"No": "N", "Yes": "Y"}
FASTING_BS = {"No": 0, "Yes": 1}
SLOPE = {"Upsloping": "Up", "Flat": "Flat", "Downsloping": "Down"}


@st.cache_resource
def cached_artifacts():
    return load_artifacts()  # Read model, imputer, scaler and schema once per process.


try:
    artifacts = cached_artifacts()
except Exception:
    st.error("The trained model could not be loaded. Check the model files and installed package versions.")
    st.stop()  # Do not offer predictions without a working model.

with st.sidebar:
    st.markdown("### Project overview")
    st.caption("An educational machine learning application")
    st.divider()
    st.markdown("**Model**  \nK-nearest neighbors (KNN)")
    st.markdown("**Tech stack**  \nPython · scikit-learn · pandas · Streamlit")
    st.markdown("**Project type**  \nSupervised binary classification")
    st.divider()
    st.link_button("View GitHub project", "https://github.com/PrashantSinghpns/heart-disease-prediction", width="stretch")
    st.caption("Predictions reflect patterns in the training dataset and may be incorrect.")

st.markdown('<div class="eyebrow">Clinical data · Machine learning</div>', unsafe_allow_html=True)
st.title("Heart Disease Risk Prediction")
st.write("Explore a KNN model’s prediction using patient measurements and clinical observations.")
st.caption("Enter all 11 inputs below, then generate a model prediction.")

with st.form("patient_inputs"):
    st.subheader("01 · Patient Information")
    left, right = st.columns(2, gap="large")
    with left:
        age = st.slider("Age (years)", 18, 100, 40, key="age")
    with right:
        sex = st.selectbox("Sex", list(SEX), key="sex")

    st.divider()
    st.subheader("02 · Cardiovascular Measurements")
    left, right = st.columns(2, gap="large")
    with left:
        resting_bp = st.number_input("Resting blood pressure (mmHg)", 80, 200, 120, key="resting_bp")
        max_hr = st.slider("Maximum heart rate (bpm)", 60, 220, 150, key="max_hr", help="Maximum heart rate achieved during exercise testing.")
    with right:
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200, key="cholesterol")
        oldpeak = st.number_input("ST depression (Oldpeak)", -2.6, 6.2, 1.0, step=0.1, key="oldpeak", help="Exercise-related ST depression relative to rest. Use the recorded clinical value.")

    st.divider()
    st.subheader("03 · Clinical Information")
    left, right = st.columns(2, gap="large")
    with left:
        chest_pain = st.selectbox("Chest pain type", list(CHEST_PAIN), key="chest_pain")
        resting_ecg = st.selectbox("Resting ECG", list(ECG), key="resting_ecg")
        st_slope = st.selectbox("ST segment slope", list(SLOPE), key="st_slope", help="Slope of the peak exercise ST segment.")
    with right:
        fasting_bs = st.selectbox("Fasting blood sugar above 120 mg/dL", list(FASTING_BS), key="fasting_bs")
        exercise_angina = st.selectbox("Exercise-induced angina", list(ANGINA), key="exercise_angina", help="Whether angina was reported during exercise.")
    st.divider()
    submitted = st.form_submit_button("Predict Heart Disease Risk", type="primary", width="stretch")

if submitted:
    raw_input = {
        "Age": age, "Sex": SEX[sex],
        "ChestPainType": CHEST_PAIN[chest_pain], "RestingBP": resting_bp,
        "Cholesterol": cholesterol, "FastingBS": FASTING_BS[fasting_bs],
        "RestingECG": ECG[resting_ecg], "MaxHR": max_hr,
        "ExerciseAngina": ANGINA[exercise_angina], "Oldpeak": oldpeak,
        "ST_Slope": SLOPE[st_slope],
    }  # core.py restores the saved feature order before imputation and scaling.
    try:
        prediction, probability = predict_result(raw_input, artifacts)
    except Exception:
        st.error("Prediction could not be completed. Check that the model and preprocessing files belong to the same training run.")
    else:
        higher = prediction == 1
        title = "Higher Predicted Risk" if higher else "Lower Predicted Risk"
        detail = "The model assigned this input to the heart disease class." if higher else "The model assigned this input to the no heart disease class."
        probability_text = "" if probability is None else f"<p><strong>{probability:.0%}</strong> model probability for the heart disease class</p>"
        st.markdown(f'<div class="result-card {"higher" if higher else "lower"}" role="status"><div class="result-label">Model prediction</div><h2>{title}</h2><p>{detail}</p>{probability_text}</div>', unsafe_allow_html=True)
        if probability is not None:
            st.caption("KNN probability is the fraction of neighboring training samples in the positive class; it is not a calibrated estimate of personal medical risk.")
        st.caption("A lower prediction does not rule out heart disease. A higher prediction does not confirm it.")

st.caption("This ML prediction is for educational purposes and is not a medical diagnosis.")
