# Dataset

The training file is named `heart.csv` and has 918 rows with the target column `HeartDisease`.

The original dataset URL and redistribution terms have not yet been supplied. The CSV is excluded from this repository until those details are confirmed. The included trained artifacts let you run the app without the CSV.

To retrain, place your authorized copy at `data/heart.csv`, or pass its location using `python train.py --data "path/to/heart.csv"`.

Features: `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, and `ST_Slope`.

Target: `0` = no heart disease label; `1` = heart disease label. This is a heart disease classification dataset, not a stroke prediction dataset.
