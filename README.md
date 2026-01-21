# Diabetes Classification

A reproducible Jupyter notebook project that performs end-to-end binary classification for diabetes (Pima Indians style dataset). This repository contains a single notebook (`diabetes-classification.ipynb`) and the dataset (`diabetes.csv`). The notebook follows a clear data-science workflow: EDA → cleaning → feature engineering → model training and evaluation.

Repository contents
- `diabetes-classification.ipynb` — Main analysis and experiments (notebook).
  - Sections found in the notebook: Problem statement, Libraries, Data collection, Data exploration (describe/info), Data cleaning, Feature engineering (identify and fix invalid zero values), EDA/visualization, Model training & evaluation.
  - Models used (explicit in notebook): Logistic Regression and Linear Discriminant Analysis (LDA). GridSearchCV is imported for hyperparameter tuning; typical metrics include accuracy, precision, recall, F1-score, ROC AUC.
- `diabetes.csv` — Dataset used by the notebook (expected to be the Pima Indians / Kaggle-style file with columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome)

Quick summary of what the notebook does
- Loads `diabetes.csv` and inspects shape and statistics (dataset: 768 rows × 9 columns).
- Identifies physiologically-invalid zero values in features: Glucose, BloodPressure, SkinThickness, Insulin, BMI.
- Replaces invalid zero values with the median of the respective column (median chosen for robustness to outliers).
- Conducts EDA, distribution checks, and summary statistics.
- Prepares data for modeling (scaling via StandardScaler is imported).
- Trains and compares Logistic Regression and LDA models and evaluates them using:
  - Accuracy, Precision, Recall, F1-score
  - ROC AUC and ROC curves
  - Confusion matrix
- Uses sklearn utilities such as train_test_split, Stratified folds are recommended for robust evaluation, and GridSearchCV for hyperparameter tuning.

## Results

Replace the numbers below with the actual model evaluation results found in `diabetes-classification.ipynb`. If you want, I can extract these values from the notebook and update this file automatically — say "Extract metrics and update README".

| Model | Test Accuracy | Precision | Recall | F1-score | ROC AUC |
|-------|--------------:|----------:|-------:|---------:|--------:|
| Logistic Regression | 0.76 | 0.74 | 0.65 | 0.69 | 0.81 |
| Linear Discriminant Analysis (LDA) | 0.75 | 0.73 | 0.64 | 0.68 | 0.80 |

Notes:
- The table above shows example values. Please replace them with the exact numbers from your experiment.
- Include confusion matrix and ROC curve images below (or link to notebook cells that display them).

How to reproduce locally
1. Clone the repo:
   git clone https://github.com/anant1123/Diabetes-classification.git
2. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows
3. Install dependencies (see suggested list below) and start Jupyter:
   pip install -r requirements.txt  # create this file if not present
   jupyter lab   # or jupyter notebook
4. Open `diabetes-classification.ipynb` and run cells in order.

Suggested requirements.txt (create or ask me to add)
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost        # optional
jupyterlab
shap           # optional (for explainability)
```

Notes, recommendations and next steps
- The notebook handles zero values by replacing them with the median — which is appropriate here, but consider:
  - Verifying that zeros are truly missing / invalid (based on domain knowledge).
  - Optionally imputing using KNNImputer or modeling-based imputation for improved fidelity.
- Consider adding:
  - A results section with final test metrics and plots (ROC, confusion matrix).  <-- Added above
  - A small `src/` folder with reusable functions (data loading, preprocessing, training) so results can be run programmatically (good for CI).
  - A `requirements.txt` and small `run.sh` or README section for Colab (Colab badge: https://colab.research.google.com/github/anant1123/Diabetes-classification/blob/main/diabetes-classification.ipynb).
- If you want model-versioning or reproducible experiments, consider adding:
  - seed setting for numpy/sklearn and any frameworks used
  - a small `experiments/` folder or a `results/` folder with exported model artifacts and plots

License
- Add a LICENSE file (e.g., MIT) to clarify reuse permissions.

Contact
- Maintainer: @anant1123
