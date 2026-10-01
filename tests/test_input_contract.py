import math
import pytest
from core import prepare_input

EXAMPLE = {"Age": 40, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 120,
           "Cholesterol": 200, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 150,
           "ExerciseAngina": "N", "Oldpeak": 1.5, "ST_Slope": "Up"}
COLUMNS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
           "Sex_M", "ChestPainType_ATA", "RestingECG_Normal", "ExerciseAngina_Y", "ST_Slope_Up"]

def test_zero_readings_match_training_imputation():
    row = prepare_input(dict(EXAMPLE, Cholesterol=0, RestingBP=0), COLUMNS).iloc[0]
    assert math.isnan(row.Cholesterol)
    assert math.isnan(row.RestingBP)
    assert row.Oldpeak == 1.5
    assert row.Sex_M == 1

def test_invalid_inputs_do_not_silently_become_reference_categories():
    for changed in [dict(EXAMPLE, Sex="invalid"), dict(EXAMPLE, Age=math.inf), dict(EXAMPLE, FastingBS=2)]:
        with pytest.raises(ValueError):
            prepare_input(changed, COLUMNS)
    with pytest.raises(ValueError):
        prepare_input({}, COLUMNS)
