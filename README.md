# SustainFlow AI – Food Loss, Demand, Spoilage & Recommendation Suite

## Project Overview

SustainFlow AI is a 4-module food-waste intelligence dashboard built with Streamlit. It predicts food loss, forecasts demand, detects spoilage risk, and generates AI-style prevention recommendations for the same set of food items across all modules.

| Module | Title | Model | Output |
|---|---|---|---|
| M1 | Food Loss Prediction | Random Forest Regressor | Expected loss % per commodity / country / stage / year |
| M2 | Demand Forecast (Units) | LightGBM | Monthly demand in units |
| M3 | Spoilage Detection | Gradient Boosting | Spoilage risk % + CRITICAL/HIGH/MEDIUM/LOW level |
| M4 | AI Recommendation Engine | Rule-based advisor over M1–M3 | Diagnosis + ranked decision matrix + action plan |

## Folder Structure

```
SustainFlow AI/
│
├── Codes/
│   ├── app.py                      # Streamlit dashboard (run this)
│   ├── train_fao_model.py          # Trains M1 Random Forest model
│   ├── train_demand_model.py       # Trains M2 LightGBM model
│   ├── train_spoilage_model.py     # Trains M3 Gradient Boosting model
│   ├── build_module3_dataset.py    # Builds M3 spoilage dataset
│   ├── predict_loss.py             # Standalone M1 CLI predictor (+ charts/PDF)
│   ├── predict_demand.py           # Standalone M2 CLI predictor (+ charts/PDF)
│   ├── predict_spoilage.py         # Standalone M3 CLI predictor (+ charts/PDF)
│   ├── risk_classifier.py          # Helper: loss % → risk level
│   ├── xai_attribution.py          # SHAP / leave-one-out explanations
│
├── Datasets/
│   ├── Fao_merged_clean.csv        # M1 food loss data
│   ├── module2_matched_demand.csv  # M2 demand data
│   └── module3_fao_spoilage_v2.csv # M3 spoilage data (leakage-free)
│
├── Results/
│   ├── fao_model.joblib            # M1 trained model
│   ├── demand_model.joblib         # M2 trained model
│   ├── spoilage_model.joblib       # M3 trained model
│   ├── *_metrics.txt               # evaluation reports
│   ├── *.png                       # model comparison / SHAP charts
│   └── shap_report.pdf
│
├── requirements.txt
└── README.md
```

## Steps to Run

Install requirements:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run Codes/app.py
```

Retrain the models (optional, only if data changed):

```bash
python Codes/train_fao_model.py
python Codes/train_demand_model.py
python Codes/train_spoilage_model.py
```

Note: `build_module3_dataset.py` generates the M3 spoilage dataset file. The models only need the CSV files in `Datasets/` to exist.

## What Each Program Does

- **app.py** — the whole dashboard: sidebar navigation, M1 loss predictor, M2 demand forecast, M3 spoilage detector (with SHAP explanations), M4 decision matrix + AI advisor + markdown simulator, light/dark mode, multi-currency display.
- **train_fao_model.py** — trains and evaluates the M1 Random Forest regressor (log-transformed target), saves the model + comparison charts + metrics.
- **train_demand_model.py** — trains and evaluates the M2 LightGBM regressor against a linear baseline, saves the model + metrics.
- **train_spoilage_model.py** — trains the M3 Gradient Boosting classifier on leakage-free spoilage risk (balanced class weights), saves the model + metrics.
- **predict_loss.py / predict_demand.py / predict_spoilage.py** — command-line predictors that load the saved models, compute a prediction for chosen inputs, and export charts / SHAP reports / PDF.
- **xai_attribution.py** — shared explainability helpers (SHAP where available, leave-one-out fallback).
- **risk_classifier.py** — small utility that maps a loss % to Low / Medium / High.

## Evaluation Summary

- M1 Food Loss: Random Forest (R², RMSE, MAE vs a Ridge baseline in `Results/metrics.txt`)
- M2 Demand: LightGBM over Linear Regression (see `Results/demand_metrics.txt`)
- M3 Spoilage: Gradient Boosting, 73.2% accuracy, balanced accuracy 71.4%, ROC-AUC 0.76 (see `Results/spoilage_metrics.txt`)

## How the Module 3 Spoilage Risk Score is Built (important)

The M3 spoilage label is **built exclusively from an external shelf-life reference table** – it is
**not** computed from the features the model is trained on, so there is no target leakage:

1. **Label** – each commodity's typical shelf life (unrefrigerated, literature-based) is mapped to a
   "fragility" score; a threshold plus light noise and a 5% label flip mark `high_spoilage_risk`.
   Only commodity identity and that external table go into this label.
2. **Features** – the model is trained only on operational signals that did **not** produce the label:
   FAO loss %, logged average demand, demand volatility, peak month, supply-stage code and year.
   `commodity`, `country` and `perishable` are deliberately withheld (all three are directly linked to
   the shelf-life table, so including them would re-introduce leakage / allow per-commodity memorization).
3. **Result** – because the model cannot lean on commodity identity, its ~73% accuracy (ROC-AUC 0.76)
   reflects the real, weak association between loss/demand/season signals and spoilage fragility. The
   accuracy is **honest and generalizable**, not formula-inversion. Balanced class weights keep the
   imbalanced label (13.7% positive) from inflating the numbers.

`build_module3_dataset.py` produces `module3_fao_spoilage_v2.csv`; `train_spoilage_model.py` trains on it.