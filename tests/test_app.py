import unittest  # Use the standard-library test runner.
from pathlib import Path  # Locate the app from the test folder.
from streamlit.testing.v1 import AppTest  # Exercise the actual Streamlit form.
from core import load_artifacts, prepare_input, predict  # Test the same code used by the app.

EXAMPLE = {  # Use the first recorded dataset example as input, without requiring a particular prediction.
    "Age": 40, "Sex": "M", "ChestPainType": "ATA",  # Keep the raw training categories.
    "RestingBP": 140, "Cholesterol": 289, "FastingBS": 0,  # Preserve numeric readings.
    "RestingECG": "Normal", "MaxHR": 172, "ExerciseAngina": "N",  # Preserve additional inputs.
    "Oldpeak": 0.0, "ST_Slope": "Up",  # Preserve the remaining features.
}  # Complete the example.

class InferenceTests(unittest.TestCase):  # Cover the original encoding and scaling failures.
    def test_category_features(self):  # Verify selections survive alignment to model columns.
        columns = load_artifacts()[3]  # Load the actual feature schema.
        encoded = prepare_input(EXAMPLE, columns).iloc[0]  # Encode one raw example.
        for name in ["Sex_M", "ChestPainType_ATA", "RestingECG_Normal", "ST_Slope_Up"]:  # Check selected dummy features.
            self.assertEqual(encoded[name], 1, name)  # Fail if a selected category was silently discarded.
        self.assertEqual(encoded["ExerciseAngina_Y"], 0)  # Verify the negative angina category.
        self.assertEqual(encoded["Age"], 40)  # Check that raw age is preserved until scaling.

    def test_ta_and_angina_features(self):  # Cover the original trailing-space and numeric-angina bugs.
        changed = dict(EXAMPLE, ChestPainType="TA", ExerciseAngina="Y", Sex="F")  # Exercise alternate categories.
        encoded = prepare_input(changed, load_artifacts()[3]).iloc[0]  # Encode with the saved schema.
        self.assertEqual(encoded["ChestPainType_TA"], 1)  # Verify TA is recognized.
        self.assertEqual(encoded["ExerciseAngina_Y"], 1)  # Verify Y is recognized.
        self.assertEqual(encoded["Sex_M"], 0)  # Verify F maps to the reference category.

    def test_artifact_bundle(self):  # Verify the model and preprocessing agree.
        model, imputer, scaler, columns = load_artifacts()  # Load the deployment bundle.
        self.assertEqual(list(scaler.feature_names_in_), columns)  # Verify feature order.
        self.assertEqual(model.n_features_in_, len(columns))  # Verify input dimensions.
        self.assertGreater(scaler.mean_[0], 20)  # Catch a scaler fitted to already-standardized ages.
        self.assertIn(predict(EXAMPLE, (model, imputer, scaler, columns)), [0, 1])  # Confirm inference succeeds.

    def test_streamlit_prediction(self):  # Exercise startup and a real form submission.
        app_path = Path(__file__).resolve().parents[1] / "app.py"  # Locate the app independently of working folder.
        app = AppTest.from_file(str(app_path), default_timeout=30).run()  # Render the app.
        self.assertEqual(len(app.exception), 0)  # Check startup for errors.
        app.button[0].click().run()  # Submit the default form inputs.
        self.assertEqual(len(app.exception), 0)  # Check prediction for errors.
        self.assertTrue(len(app.success) + len(app.warning) > 0)  # Confirm a result appears.

if __name__ == "__main__":  # Permit directly running this test file.
    unittest.main()  # Start the tests.
