# Heart Disease Severity Prediction (UCI)

This repository trains an XGBoost model on the UCI Heart Disease dataset (processed variants) to classify disease severity levels (0-4).

Features:

- Merges Cleveland, Hungarian, Switzerland, and VA datasets (processed versions)
- Data augmentation for underrepresented classes
- XGBoost multiclass model with cross-validated evaluation
- Normalized confusion matrix plot (`confusion_matrix_normalized.png`)
- Feature importance plot (`feature_importance.png`)

Usage:

1. `python main.py`
2. View output metrics in stdout and plots in repository root.

Dataset attributes:

- `age`: age in years
- `sex`: 1 = male, 0 = female
- `cp`: chest pain type (0..3)
- `trestbps`: resting blood pressure (mm Hg)
- `chol`: serum cholesterol (mg/dl)
- `fbs`: fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- `restecg`: resting electrocardiographic results (0..2)
- `thalach`: maximum heart rate achieved
- `exang`: exercise-induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: slope of the peak exercise ST segment
- `ca`: number of major vessels colored by fluoroscopy (0..3)
- `thal`: thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)
- `num`: diagnosis status (0 = normal, 1-4 = disease severity)

License: MIT
