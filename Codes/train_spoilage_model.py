import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "Datasets", "module3_fao_spoilage_v2.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
MODEL_PATH = os.path.join(RESULTS_DIR, "spoilage_model.joblib")
METRICS_PATH = os.path.join(RESULTS_DIR, "spoilage_metrics.txt")

# LEAKAGE-FREE feature set: none of these build the label.
# The label comes only from the external shelf-life reference table
# (see build_module3_dataset.py). commodity/country/perishable are withheld.
FAO_FEATURES = ["loss_percentage", "avg_demand_log", "volatility_ratio",
                "peak_month", "stage_code", "year"]

print("MODULE 3 v2: LEAKAGE-FREE SPOILAGE RISK MODEL TRAINING")
print("Gradient Boosting on the same food items as Module 1 & 2")
print("Label = external shelf-life fragility; features exclude label ingredients.\n")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("module3_fao_spoilage_v2.csv not found in Datasets/ — run build_module3_dataset.py first!")

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} records across {df['commodity'].nunique()} commodities "
      f"and {df['country'].nunique()} countries.")
print(f"Positive (high risk) share: {df['high_spoilage_risk'].mean() * 100:.1f}%")

X = df[FAO_FEATURES]
y = df["high_spoilage_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balanced sample weights so that the accuracy is not trivially inflated
# by the imbalanced label (13-14% positive class).
class_counts = np.bincount(y_train)
sample_weight = y_train.map(
    lambda c: len(y_train) / (2.0 * class_counts[c])
).to_numpy()

model = GradientBoostingClassifier(
    n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42
)
model.fit(X_train, y_train, sample_weight=sample_weight)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\nMODULE 3 EVALUATION METRICS (test set)")
print(f"  Accuracy:          {acc * 100:.2f}%")
print(f"  Balanced accuracy: {bal_acc * 100:.2f}%")
print(f"  Precision:         {prec:.4f}")
print(f"  Recall:            {rec:.4f}")
print(f"  F1-Score:          {f1:.4f}")
print(f"  ROC-AUC:           {auc:.4f}")

joblib.dump(model, MODEL_PATH)

metrics_report = f"""======================================================================
MODULE 3 - SPOILAGE RISK MODEL METRICS (LEAKAGE-FREE V2 DATASET)
======================================================================

Training rows: {len(X_train):,}
Testing rows:  {len(X_test):,}

Features: {", ".join(FAO_FEATURES)}
Model: Gradient Boosting (150 stages, depth 4, lr 0.08)
Label: external shelf-life fragility score (>= 55 -> high risk)

----------------------------------------------------------------------
GRADIENT BOOSTING CLASSIFIER (MAIN MODEL)
----------------------------------------------------------------------
Accuracy:          {acc * 100:.2f}%
Balanced accuracy: {bal_acc * 100:.2f}%
Precision:         {prec:.4f}
Recall:            {rec:.4f}
F1-Score:          {f1:.4f}
ROC-AUC:           {auc:.4f}

Confusion Matrix (test):
  True negative:  {tn:,}
  False positive: {fp:,}
  False negative: {fn:,}
  True positive:  {tp:,}

Classification Report:
{classification_report(y_test, y_pred)}
======================================================================
"""

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    f.write(metrics_report)

print(f"\nSaved Module 3 Spoilage Model to: {MODEL_PATH}")
print(f"Saved Module 3 Evaluation Metrics to: {METRICS_PATH}")
print("DONE")