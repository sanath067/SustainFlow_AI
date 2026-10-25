import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# OPTIONAL LIBRARIES
# ============================================================

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "Datasets", "module3_fao_spoilage_v2.csv")
MODEL_PATH = os.path.join(BASE_DIR, "Results", "spoilage_model.joblib")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")

SPOIL_FEATURES = ["loss_percentage", "avg_demand_log", "volatility_ratio",
                  "peak_month", "stage_code", "year"]
SPOIL_MODE_COLS = {"peak_month", "stage_code", "year"}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_dataset():
    print("\n" + "=" * 70)
    print("LOADING MODULE 3 SPOILAGE DATASET")
    print("=" * 70)
    print(f"\nDataset path:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip().str.lower()
    print(f"\nDataset loaded successfully.")
    print(f"Total rows: {len(df)}")

    required = ["commodity", "country", "food_supply_stage", "year"] + SPOIL_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def load_model():
    print("\n" + "=" * 70)
    print("LOADING SPOILAGE RISK MODEL (Gradient Boosting)")
    print("=" * 70)
    print(f"\nModel path:\n{MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found — run Codes/train_spoilage_model.py first.")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
    return model


def select_option(label, options):
    print(f"\n{label} options available:")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    while True:
        choice = input(f"\nEnter {label} (number, exact name or partial text): ").strip()
        if not choice:
            print("Input cannot be empty.")
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        exact = [o for o in options if clean_text(o).lower() == choice.lower()]
        if exact:
            return exact[0]
        partial = [o for o in options if choice.lower() in clean_text(o).lower()]
        if partial:
            if len(partial) == 1:
                return partial[0]
            print("Multiple matches found:")
            for i, p in enumerate(partial[:10], start=1):
                print(f"  {i}. {p}")
            print("Please type the exact complete name to choose.")
            continue
        print("No match found. Please try again.")


def select_year(years):
    print("\nYear options available:")
    for y in years:
        print(f"  {y}")
    while True:
        choice = input(f"\nEnter Year (e.g., {years[0]} to {years[-1]}): ").strip()
        if not choice.isdigit():
            print("Please enter a numeric year.")
            continue
        y = int(choice)
        if y in years:
            return y
        print(f"Year {y} not in dataset. Try one of: {years}")


def spoilage_risk_level(prob):
    if prob >= 70:
        return "CRITICAL"
    if prob >= 50:
        return "HIGH"
    if prob >= 30:
        return "MEDIUM"
    return "LOW"


def shap_or_loo_spoilage(model, input_df, df):
    """SHAP values for the spoilage prediction; LOO fallback if unavailable."""
    feature_names = list(input_df.columns)
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            raw = explainer.shap_values(input_df, check_additivity=False)
            if isinstance(raw, list):
                sample = raw[1][0]
            elif raw.ndim == 3:
                sample = raw[0, :, 1]
            elif raw.ndim == 2:
                sample = raw[0]
            else:
                sample = raw
            n = min(len(feature_names), len(sample))
            out = pd.DataFrame({
                "feature": feature_names[:n],
                "shap_value": sample[:n],
                "absolute_shap_value": np.abs(sample[:n]),
            }).sort_values("absolute_shap_value", ascending=False).reset_index(drop=True)
            return out, "shap"
        except Exception:
            pass

    ref = {}
    for c in feature_names:
        ref[c] = int(df[c].mode()[0]) if c in SPOIL_MODE_COLS else float(df[c].median())
    base = float(model.predict_proba(pd.DataFrame([ref])[feature_names])[:, 1][0])
    rows = []
    for c in feature_names:
        mod = input_df.copy()
        mod[c] = ref[c]
        pred_v = float(model.predict_proba(mod)[:, 1][0])
        rows.append({"feature": c, "shap_value": round(pred_v - base, 4),
                     "absolute_shap_value": round(abs(pred_v - base), 4)})
    return (pd.DataFrame(rows)
            .sort_values("absolute_shap_value", ascending=False)
            .reset_index(drop=True)), "loo"


def save_prediction(input_df, prob, level):
    row = input_df.copy()
    row["predicted_spoilage_risk_pct"] = prob
    row["spoilage_risk_level"] = level
    out_path = os.path.join(RESULTS_DIR, "latest_spoilage_prediction.csv")
    row.to_csv(out_path, index=False)
    print(f"\nSaved prediction to: {out_path}")


def generate_report(model, input_df, df, prob, level):
    """Save a PDF report with the spoilage score + factor chart."""
    factors, method = shap_or_loo_spoilage(model, input_df, df)
    top = factors.head(10).sort_values("shap_value")

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#1d6b4d" if v >= 0 else "#b71c1c" for v in top["shap_value"]]
    ax.barh(top["feature"], top["shap_value"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1.2)
    ax.set_title(f"Spoilage risk {prob:.1f}% ({level}) — factor attribution ({method})")
    ax.set_xlabel("Effect on spoilage probability")
    ax.set_ylabel("Factor")
    fig.tight_layout()

    report_path = os.path.join(RESULTS_DIR, "spoilage_shap_report.pdf")
    with PdfPages(report_path) as pdf:
        pdf.savefig(fig)
        plt.close(fig)
    print(f"Saved SHAP/attribution report to: {report_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("SUSTAINFLOW AI")
    print("MODULE 3 - SPOILAGE RISK PREDICTION")
    print("=" * 70)

    df = load_dataset()
    model = load_model()

    commodities = sorted(df["commodity"].dropna().astype(str).unique().tolist(), key=str.lower)
    countries = sorted(df["country"].dropna().astype(str).unique().tolist(), key=str.lower)
    stages = sorted(df["food_supply_stage"].dropna().astype(str).unique().tolist(), key=str.lower)
    years = sorted(df["year"].dropna().astype(int).unique().tolist())

    print("\nYou can select using: 1) number from the list 2) exact name 3) partial name.")
    commodity = select_option("Commodity", commodities)
    country = select_option("Country", countries)
    stage = select_option("Food Supply Stage", stages)
    year = select_year(years)

    combo = df[(df["commodity"] == commodity) &
               (df["country"] == country) &
               (df["food_supply_stage"] == stage) &
               (df["year"] == year)]
    if len(combo) == 0:
        print(f"\nNo records for {commodity} / {country} / {stage.replace('_', ' ')} / {year}.")
        return
    row = combo.iloc[0]
    input_df = pd.DataFrame([{f: float(row[f]) for f in SPOIL_FEATURES}])

    print("\n" + "=" * 70)
    print("MAKING PREDICTION")
    print("=" * 70)

    prob = float(model.predict_proba(input_df)[:, 1][0]) * 100.0
    level = spoilage_risk_level(prob)

    print("\n" + "=" * 70)
    print("SPOILAGE RISK PREDICTION RESULT")
    print("=" * 70)
    print(f"\nCommodity:          {commodity}")
    print(f"Country:            {country}")
    print(f"Food Supply Stage:  {stage}")
    print(f"Year:               {year}")
    print(f"Perishable:         {'yes' if int(row['perishable']) else 'no'}")
    print(f"FAO loss:           {float(row['loss_percentage']):,.2f}%")
    print(f"Avg demand:         {float(row['avg_demand']):,.0f} units/mo")
    print(f"Volatility:         {float(row['volatility_ratio']):.2f}")
    print(f"Peak season:        month {int(row['peak_month'])}")
    print("\n" + "-" * 70)
    print(f"PREDICTED SPOILAGE RISK:  {prob:.1f}%  ({level})")
    print("-" * 70)

    save_prediction(input_df, round(prob, 2), level)
    generate_report(model, input_df, df, prob, level)

    print("\n" + "=" * 70)
    print("PREDICTION PROCESS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()