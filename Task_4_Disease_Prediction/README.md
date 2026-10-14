# Task 4: Disease prediction from medical data
Uses the **scikit-learn Wisconsin Diagnostic Breast Cancer** dataset (569 rows, 30 numeric features). Target is explicitly re-coded as **1 = malignant, 0 = benign**. Compares scaled Logistic Regression and Random Forest, reports cross-validated training F1 and untouched holdout precision/recall/F1/ROC-AUC.

Run `python Task_4_Disease_Prediction/train.py`. The features are digitized cell image measurements, **not symptoms or blood tests**. This notebook is strictly an educational classification experiment, not for clinical or patient decisions.
