"""
Shared XAI attribution helpers for the SustainFlow AI scripts.

Preferred method: SHAP (TreeExplainer).
Fallback: leave-one-out (LOO) local attribution, which computes
how much each factor shifts the prediction compared to its typical
value. This keeps the scripts working even when Windows
Smart App Control blocks shap's compiled DLL.
"""

import re
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SHAP_AVAILABLE = False
shap = None

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def clean_feature_name(name):
    """Turn sklearn/pipeline feature names (cat__country, country_encoded,
    lag_1, dayofweek, ...) into human-readable labels."""
    n = str(name)
    n = re.sub(r"^(cat|num|bin)__", "", n)
    n = n.replace("_encoded", "")
    n = re.sub(r"^lag_(\d+)$", r"sales_lag_\1", n)
    n = n.replace("rolling_mean_3", "rolling_mean_3d")
    n = n.replace("dayofweek", "day_of_week")
    n = re.sub(r"_+", " ", n).strip()
    return n.title()


# ============================================================
# MODULE 1 (FAO) - single prediction
# ============================================================

def _fao_loo(pipeline, input_df, df):
    cols = ["commodity", "country", "food_supply_stage", "year"]
    ref = {}
    for c in cols:
        ref[c] = int(df[c].median()) if c == "year" else df[c].mode()[0]
    base_df = pd.DataFrame([ref])[cols]
    base = float(
        np.expm1(pipeline.predict(base_df)[0])
    )
    feature_names = []
    values = []
    for c in cols:
        mod = input_df.copy()
        mod[c] = ref[c]
        pred_v = float(
            np.expm1(pipeline.predict(mod)[0])
        )
        feature_names.append(c)
        values.append(pred_v - base)
    return np.array(feature_names), np.array(values)


def shap_or_loo_fao_prediction(pipeline, input_df, df):
    """Return (feature_names, shap_sample, method) for one FAO row."""

    if SHAP_AVAILABLE:
        try:
            preprocessor = pipeline.named_steps["preprocess"]
            model = pipeline.named_steps["model"]
            X_t = preprocessor.transform(input_df)
            X_dense = X_t.toarray() if hasattr(X_t, "toarray") else np.asarray(X_t)
            feature_names = [str(n) for n in preprocessor.get_feature_names_out()]
            explainer = shap.TreeExplainer(model)
            values = np.asarray(
                explainer.shap_values(X_dense, check_additivity=False)
            )
            if values.ndim == 3:
                sample = values[0, :, 0]
            elif values.ndim == 2:
                sample = values[0]
            else:
                sample = values
            n = min(len(feature_names), len(sample))
            names = [clean_feature_name(x) for x in feature_names[:n]]
            return names, sample[:n], "shap"
        except Exception:
            pass

    feature_names, values = _fao_loo(pipeline, input_df, df)
    return [clean_feature_name(x) for x in feature_names], values, "loo"


# ============================================================
# MODULE 1 (FAO) - global sample analysis (explainable script)
# ============================================================

def shap_or_loo_global_fao(
    pipeline,
    preprocessor,
    model,
    X_sample_raw,
    X_sample_dense,
    feature_names,
    df,
):
    """Return (feature_names, values_array, method) for a sample of rows.

    values_array shape: (n_rows, n_features)
    """

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(
                X_sample_dense,
                check_additivity=False
            )
            if isinstance(values, list):
                arr = np.asarray(values[0])
            else:
                arr = np.asarray(values)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            names = [clean_feature_name(x) for x in feature_names]
            return names, arr, "shap"
        except Exception:
            pass

    cols = ["commodity", "country", "food_supply_stage", "year"]
    ref = {}
    for c in cols:
        ref[c] = int(df[c].median()) if c == "year" else df[c].mode()[0]

    base_pred = float(
        pipeline.predict(pd.DataFrame([ref])[cols])[0]
    )

    arr = np.zeros((len(X_sample_raw), len(cols)))
    for j, c in enumerate(cols):
        mod = X_sample_raw.copy()
        mod[c] = ref[c]
        preds = pipeline.predict(mod)
        arr[:, j] = preds - base_pred

    return [clean_feature_name(x) for x in cols], arr, "loo"


# ============================================================
# MODULE 2 (DEMAND) - single prediction
# ============================================================

def _demand_loo(model, model_input_df, df, encoder_map, feature_names):
    med_demand = float(df["demand"].median())
    ref = {}
    for c in feature_names:
        if c in ("lag_1", "lag_2", "lag_3", "rolling_mean_3"):
            ref[c] = med_demand
        elif c == "year":
            ref[c] = int(df[c].median())
        elif c == "month":
            ref[c] = int(df["month"].median()) if "month" in df else 1
        elif c == "day":
            ref[c] = int(df["day"].median()) if "day" in df else 1
        elif c == "dayofweek":
            ref[c] = int(df["dayofweek"].median()) if "dayofweek" in df else 0
        elif c == "weekend":
            ref[c] = int(df["weekend"].mode()[0]) if "weekend" in df else 0
        elif c == "country_encoded":
            ref[c] = encode_value(encoder_map, "country", df["country"].mode()[0])
        elif c == "commodity_encoded":
            ref[c] = encode_value(encoder_map, "commodity", df["commodity"].mode()[0])
        else:
            ref[c] = 0

    base = float(
        model.predict(pd.DataFrame([ref])[feature_names])[0]
    )

    values = []
    for c in feature_names:
        mod = model_input_df.copy()
        mod[c] = ref[c]
        pred_v = float(model.predict(mod)[0])
        values.append(pred_v - base)

    return np.array(values)


def encode_value(encoder_map, category, value):
    mapping = encoder_map.get(category, {})
    if value in mapping:
        return mapping[value]
    for key, code in mapping.items():
        if key.lower() == value.lower():
            return code
    return 0


def shap_or_loo_demand_prediction(
    model,
    model_input_df,
    df,
    encoder_map,
    feature_names,
):
    """Return (feature_names, shap_sample, method) for one demand row."""

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            values = np.asarray(
                explainer.shap_values(
                    model_input_df,
                    check_additivity=False
                )
            )
            if values.ndim == 3:
                sample = values[0, :, 0]
            elif values.ndim == 2:
                sample = values[0]
            else:
                sample = values
            n = min(len(feature_names), len(sample))
            names = [clean_feature_name(x) for x in feature_names[:n]]
            return names, sample[:n], "shap"
        except Exception:
            pass

    values = _demand_loo(model, model_input_df, df, encoder_map, feature_names)
    return [clean_feature_name(x) for x in feature_names], values, "loo"