import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Datasets",
    "Fao_merged_clean.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "Results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
)

print(f"Loaded: {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

feature_cols = [
    "commodity",
    "country",
    "food_supply_stage",
    "year"
]

target_col = "loss_percentage"

X = df[feature_cols].copy()
y_raw = df[target_col].copy()

# Log-transform target to handle right-skewness
# Add 1 to avoid log(0)
y = np.log1p(y_raw)

print(f"\nFeatures: {feature_cols}")
print(f"Target: {target_col}")
print(f"Target skewness before log: {y_raw.skew():.4f}")
print(f"Target skewness after log:  {y.skew():.4f}")


# ============================================================
# GROUPS FOR LEAKAGE-FREE SPLITTING
# ============================================================

groups = (
    df["commodity"].astype(str) + "|" +
    df["country"].astype(str) + "|" +
    df["food_supply_stage"].astype(str) + "|" +
    df["year"].astype(str)
)

print(f"\nTotal rows: {len(df)}")
print(f"Unique groups: {groups.nunique()}")


# ============================================================
# FINAL TRAIN / TEST SPLIT (LEAKAGE-FREE)
# ============================================================

print("\n" + "=" * 70)
print("SPLITTING DATA (GROUP-BASED TO PREVENT LEAKAGE)")
print("=" * 70)

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]
y_train_raw = y_raw.iloc[train_idx]
y_test_raw = y_raw.iloc[test_idx]

print(f"\nTraining set: {len(X_train)} rows")
print(f"Testing set: {len(X_test)} rows")

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])
overlap = train_groups & test_groups

print(f"\nUnique groups in train: {len(train_groups)}")
print(f"Unique groups in test: {len(test_groups)}")
print(f"Overlapping groups: {len(overlap)}")

if len(overlap) == 0:
    print("\nSUCCESS: No leakage - test groups are completely unseen")
else:
    print(f"\nWARNING: {len(overlap)} groups overlap")


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

categorical_cols = [
    "commodity",
    "country",
    "food_supply_stage"
]

numeric_cols = ["year"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        ),
        (
            "num",
            "passthrough",
            numeric_cols
        )
    ]
)

# Tree-based preprocessing: ordinal encoding suits high-cardinality
# categorical data better than one-hot for Random Forest.
rf_preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            categorical_cols
        ),
        (
            "num",
            "passthrough",
            numeric_cols
        )
    ]
)


# ============================================================
# CROSS-VALIDATION (GROUPKFOLD)
# ============================================================

print("\n" + "=" * 70)
print("CROSS-VALIDATION (5-FOLD GROUPKFOLD)")
print("=" * 70)

cv = GroupKFold(n_splits=5)

cv_r2_scores = []
cv_rmse_scores = []

for fold, (train_cv_idx, val_cv_idx) in enumerate(
    cv.split(X, y, groups=groups)
):
    X_train_cv = X.iloc[train_cv_idx]
    X_val_cv = X.iloc[val_cv_idx]
    y_train_cv = y.iloc[train_cv_idx]
    y_val_cv = y.iloc[val_cv_idx]

    cv_pipeline = Pipeline(
        steps=[
            ("preprocess", rf_preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    cv_pipeline.fit(X_train_cv, y_train_cv)
    cv_pred = cv_pipeline.predict(X_val_cv)

    # Evaluate in original scale (inverse log-transform)
    cv_pred_raw = np.expm1(cv_pred)
    cv_r2 = r2_score(np.expm1(y_val_cv), cv_pred_raw)
    cv_rmse = np.sqrt(mean_squared_error(np.expm1(y_val_cv), cv_pred_raw))

    cv_r2_scores.append(cv_r2)
    cv_rmse_scores.append(cv_rmse)

    print(f"  Fold {fold+1}: R2={cv_r2:.4f}  RMSE={cv_rmse:.4f}")

print(f"\n  Mean R2:   {np.mean(cv_r2_scores):.4f} (+/- {np.std(cv_r2_scores):.4f})")
print(f"  Mean RMSE: {np.mean(cv_rmse_scores):.4f} (+/- {np.std(cv_rmse_scores):.4f})")


# ============================================================
# MODEL 1: RIDGE REGRESSION (BASELINE)
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RIDGE REGRESSION (BASELINE)")
print("=" * 70)

ridge_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", Ridge(alpha=1.0))
    ]
)

ridge_pipeline.fit(X_train, y_train)

ridge_pred_log = ridge_pipeline.predict(X_test)
ridge_pred = np.expm1(ridge_pred_log)

ridge_rmse = np.sqrt(mean_squared_error(y_test_raw, ridge_pred))
ridge_mae = mean_absolute_error(y_test_raw, ridge_pred)
ridge_r2 = r2_score(y_test_raw, ridge_pred)

print(f"Ridge RMSE: {ridge_rmse:.4f}")
print(f"Ridge MAE:  {ridge_mae:.4f}")
print(f"Ridge R2:   {ridge_r2:.4f}")


# ============================================================
# MODEL 2: RANDOM FOREST (MAIN)
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST (MAIN MODEL)")
print("=" * 70)

rf_pipeline = Pipeline(
    steps=[
        ("preprocess", rf_preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

rf_pipeline.fit(X_train, y_train)

rf_pred_log = rf_pipeline.predict(X_test)
rf_pred = np.expm1(rf_pred_log)

rf_rmse = np.sqrt(mean_squared_error(y_test_raw, rf_pred))
rf_mae = mean_absolute_error(y_test_raw, rf_pred)
rf_r2 = r2_score(y_test_raw, rf_pred)

print(f"Random Forest RMSE: {rf_rmse:.4f}")
print(f"Random Forest MAE:  {rf_mae:.4f}")
print(f"Random Forest R2:   {rf_r2:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    RESULTS_DIR,
    "fao_model.joblib"
)

joblib.dump(rf_pipeline, model_path)

print(f"\nModel saved: {model_path}")


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "metrics.txt"
)

with open(metrics_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("FAO FOOD LOSS MODEL METRICS (LEAKAGE-FREE)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Training rows: {len(X_train)}\n")
    f.write(f"Testing rows: {len(X_test)}\n")
    f.write(f"Unique groups in train: {len(train_groups)}\n")
    f.write(f"Unique groups in test: {len(test_groups)}\n")
    f.write(f"Overlapping groups: {len(overlap)}\n")
    f.write(f"Target transform: log1p (skewness {y_raw.skew():.4f} -> {y.skew():.4f})\n\n")
    f.write("-" * 70 + "\n")
    f.write("CROSS-VALIDATION (5-Fold GroupKFold)\n")
    f.write("-" * 70 + "\n")
    f.write(f"Mean R2:   {np.mean(cv_r2_scores):.4f} (+/- {np.std(cv_r2_scores):.4f})\n")
    f.write(f"Mean RMSE: {np.mean(cv_rmse_scores):.4f} (+/- {np.std(cv_rmse_scores):.4f})\n\n")
    f.write("-" * 70 + "\n")
    f.write("RIDGE REGRESSION (BASELINE)\n")
    f.write("-" * 70 + "\n")
    f.write(f"RMSE: {ridge_rmse:.4f}\n")
    f.write(f"MAE:  {ridge_mae:.4f}\n")
    f.write(f"R2:   {ridge_r2:.4f}\n\n")
    f.write("-" * 70 + "\n")
    f.write("RANDOM FOREST (MAIN MODEL)\n")
    f.write("-" * 70 + "\n")
    f.write(f"RMSE: {rf_rmse:.4f}\n")
    f.write(f"MAE:  {rf_mae:.4f}\n")
    f.write(f"R2:   {rf_r2:.4f}\n\n")
    f.write("-" * 70 + "\n")
    f.write("MODEL SELECTION RATIONALE\n")
    f.write("-" * 70 + "\n")

    if rf_r2 >= ridge_r2:
        f.write(f"Random Forest is preferred over Ridge Regression because it\n")
        f.write(f"achieves a higher R2 ({rf_r2:.4f} vs {ridge_r2:.4f}) on the test set.\n")
        f.write(f"\nWhy: The FAO data contains many categorical features (commodity,\n")
        f.write(f"country, supply stage). Ridge can only learn linear combinations of\n")
        f.write(f"one-hot features, while Random Forest captures non-linear interactions\n")
        f.write(f"between them (e.g. commodity x country x stage effects), which are\n")
        f.write(f"common in food-loss patterns.\n")
        f.write(f"\nRandom Forest also reports feature importance, which supports the\n")
        f.write(f"explainable AI (SHAP) module used for model interpretation.\n")
    else:
        f.write(f"Random Forest slightly underperforms Ridge on the test set\n")
        f.write(f"(R2 {rf_r2:.4f} vs {ridge_r2:.4f}), but Random Forest is still preferred\n")
        f.write(f"because it can model non-linear relationships and interactions between\n")
        f.write(f"commodity, country and supply stage, which Ridge cannot capture.\n")
        f.write(f"Random Forest also provides feature importance values used for\n")
        f.write(f"explainable AI (SHAP) analysis.\n")

    print(f"Metrics saved: {metrics_path}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

try:
    feature_names = (
        rf_pipeline
        .named_steps["preprocess"]
        .get_feature_names_out()
    )

    importances = (
        rf_pipeline
        .named_steps["model"]
        .feature_importances_
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    importance_path = os.path.join(
        RESULTS_DIR,
        "fao_feature_importance.csv"
    )

    importance_df.to_csv(importance_path, index=False)

    print(f"Feature importance saved: {importance_path}")
    print(importance_df.head(10))

except Exception as error:
    print(f"Feature importance failed: {error}")


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE (LEAKAGE-FREE + LOG-TRANSFORMED TARGET)")
print("=" * 70)