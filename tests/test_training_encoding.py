import pandas as pd
from train import encode_split_features

def test_holdout_only_category_does_not_enter_training_vocabulary():
    training = pd.DataFrame({"Age": [40, 50], "Sex": ["F", "M"], "ChestPainType": ["ASY", "ATA"]})
    holdout = pd.DataFrame({"Age": [45], "Sex": ["M"], "ChestPainType": ["TA"]})
    encoded_training, encoded_test = encode_split_features(training, holdout)
    assert encoded_training.columns.equals(encoded_test.columns)
    assert "ChestPainType_TA" not in encoded_training.columns
    assert encoded_test.iloc[0]["Sex_M"] == 1
    assert encoded_test.iloc[0]["Age"] == 45
