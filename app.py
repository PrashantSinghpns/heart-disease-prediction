import streamlit as st  # Build the browser interface.
from core import load_artifacts, predict  # Share the checked inference code.

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")  # Set the browser title.
st.title("Heart Disease Prediction")  # Use the target actually present in the dataset.
st.caption("A learning project using a K-nearest neighbors classifier.")  # Describe the model.
st.info("Educational demo. This prediction is not a diagnosis or a measure of your personal disease risk.")  # Explain what this output means.

@st.cache_resource  # Load the artifacts once per app process.
def cached_artifacts():  # Avoid reading model files on every slider change.
    return load_artifacts()  # Read the complete model bundle.

try:  # Show a clear message if artifacts are missing.
    artifacts = cached_artifacts()  # Load model, imputer, scaler and columns.
except FileNotFoundError:  # Handle an incomplete checkout.
    st.error("Model files are missing. Follow the training instructions in the README.")  # Explain the recovery step.
    st.stop()  # Stop before attempting prediction.

with st.form("patient_inputs"):  # Collect all values before predicting.
    age = st.slider("Age", 18, 100, 40)  # Select age in years.
    sex = st.selectbox("Sex", ["Male", "Female"])  # Collect the training category.
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])  # Avoid trailing spaces.
    resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)  # Select resting pressure.
    cholesterol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)  # Select cholesterol.
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])  # Match the dataset's binary feature.
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])  # Select ECG category.
    max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)  # Select maximum heart rate.
    exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])  # Preserve N/Y encoding.
    oldpeak = st.slider("Oldpeak (ST depression)", -2.6, 6.2, 1.0, step=0.1)  # Preserve decimals and dataset range.
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])  # Select slope category.
    submitted = st.form_submit_button("Predict")  # Run only when the form is submitted.

if submitted:  # Predict from the current form values.
    raw_input = {  # Keep raw categories identical to training.
        "Age": age, "Sex": "M" if sex == "Male" else "F",  # Convert the displayed sex label.
        "ChestPainType": chest_pain, "RestingBP": resting_bp,  # Preserve these inputs.
        "Cholesterol": cholesterol, "FastingBS": fasting_bs,  # Preserve numeric values.
        "RestingECG": resting_ecg, "MaxHR": max_hr,  # Preserve ECG and heart rate.
        "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak,  # Preserve angina category and decimal value.
        "ST_Slope": st_slope,  # Preserve the slope category.
    }  # Finish the input row.
    prediction = predict(raw_input, artifacts)  # Apply saved encoding, imputation and scaling.
    if prediction == 1:  # Display the model's positive class.
        st.warning("Model prediction: heart disease class (1).")  # Report a class, not a diagnosis.
    else:  # Display the model's negative class.
        st.success("Model prediction: no heart disease class (0).")  # Avoid promising absence of disease.
    st.caption("The model can be wrong. A negative prediction does not rule out heart disease.")  # State the limitation.
