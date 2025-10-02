# Diabetes Prediction (ML) — Starter Project

A clean, beginner-friendly machine learning project that predicts the likelihood of diabetes from clinical features.
This repo is **ATS/Recruiter friendly** with clear structure, documentation, and reproducible scripts.

## ✨ Features
- Clean project structure (`src/`, `data/`, `models/`).
- Trains and evaluates **Logistic Regression** and **Random Forest**.
- Saves the best model to `models/best_model.joblib` and a `metrics.json` report.
- Includes CLI scripts:
  - `python src/train.py --data data/diabetes_sample.csv`
  - `python src/predict.py --model models/best_model.joblib --input "6,148,72,35,0,33.6,0.627,50"`

## 📂 Project Structure
```
diabetes-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── diabetes_sample.csv
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── models/
│   └── .gitkeep
└── notebooks/
    └── .gitkeep
```

## 🧪 Dataset
This repo includes a small **sample** dataset (`data/diabetes_sample.csv`) derived from the well-known Pima Indians Diabetes dataset schema.
Columns:
`Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome`

> For real experiments, replace the sample with the full dataset (same columns).

## 🚀 Quickstart
1. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Train models**
   ```bash
   python src/train.py --data data/diabetes_sample.csv
   ```
   Outputs:
   - `models/best_model.joblib`
   - `models/metrics.json`
4. **Run a single prediction**
   ```bash
   python src/predict.py --model models/best_model.joblib --input "6,148,72,35,0,33.6,0.627,50"
   ```

## 📝 Tech Stack
- Python, pandas, numpy, scikit-learn
- matplotlib (for saved plots)
- joblib (model persistence)

## 🧹 Replace the Sample Data (Optional)
Use a full dataset with identical columns to improve accuracy:
- Save as `data/diabetes.csv` and run:
  ```bash
  python src/train.py --data data/diabetes.csv
  ```

## 📈 Metrics
`models/metrics.json` contains accuracy, precision, recall, f1 for the selected best model.
Confusion matrix and ROC curve are also saved as images (optional).

## 📄 License
For personal/educational use. You may reuse/adapt this template with attribution.
