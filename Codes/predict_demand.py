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

try:
    import shap

    SHAP_AVAILABLE = True

except ImportError:

    SHAP_AVAILABLE = False

from xai_attribution import (
    encode_value as xai_encode_value,
    shap_or_loo_demand_prediction,
)


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
    "module2_matched_demand.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Results",
    "demand_model.joblib"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "Results",
    "demand_encoders.joblib"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "Results"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 70)
    print("LOADING MODULE 2 DATASET")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print(f"\nDataset loaded successfully.")
    print(f"Total rows: {len(df)}")

    for column in ["country", "commodity"]:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")

    if "month" not in df.columns:
        df["month"] = 1

    if "day" not in df.columns:
        df["day"] = 1

    if "dayofweek" not in df.columns:
        df["dayofweek"] = 0

    if "weekend" not in df.columns:
        df["weekend"] = 0

    df = df.dropna(subset=["country", "commodity", "year", "demand"])

    print(f"Usable rows: {len(df)}")

    return df


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 70)
    print("LOADING TRAINED DEMAND MODEL")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"\nModel not found:\n"
            f"{MODEL_PATH}\n\n"
            f"Run train_demand_model.py first."
        )

    model = joblib.load(MODEL_PATH)

    print("\nModel loaded successfully.")

    return model


# ============================================================
# LOAD ENCODER MAPPINGS
# ============================================================

def load_encoders():

    print("\n" + "=" * 70)
    print("LOADING ENCODER MAPPINGS")
    print("=" * 70)

    if not os.path.exists(ENCODER_PATH):

        raise FileNotFoundError(
            f"\nEncoder mappings not found:\n"
            f"{ENCODER_PATH}\n\n"
            f"Run train_demand_model.py first."
        )

    encoder_map = joblib.load(ENCODER_PATH)

    print("\nEncoder mappings loaded successfully.")

    return encoder_map


# ============================================================
# ENCODE USING SAVED MAPPINGS
# ============================================================

def encode_value(encoder_map, category, value):
    """Encode a value using the saved encoder mapping."""

    mapping = encoder_map.get(category, {})

    if value in mapping:
        return mapping[value]

    # Fallback: try case-insensitive match
    value_lower = value.lower()
    for key, code in mapping.items():
        if key.lower() == value_lower:
            return code

    # Unknown value: return 0 (safe fallback)
    print(f"  WARNING: Unknown {category} '{value}', using code 0")
    return 0


# ============================================================
# DISPLAY OPTIONS
# ============================================================

def display_all_options(title, values):

    print("\n" + "=" * 70)
    print(title.upper())
    print("=" * 70)
    print(f"\nTotal available options: {len(values)}\n")

    for index, value in enumerate(values, start=1):
        print(f"{index}. {value}")

    print("\n" + "=" * 70)


# ============================================================
# SELECT OPTION
# ============================================================

def select_option(title, values):

    display_all_options(title, values)

    while True:

        user_input = input(
            f"\nEnter {title} (number or name): "
        ).strip()

        if user_input.isdigit():

            selected_index = int(user_input) - 1

            if 0 <= selected_index < len(values):

                selected_value = values[selected_index]
                print(f"\nSelected {title}: {selected_value}")
                return selected_value

            else:
                print("\nInvalid number.")
                continue

        for value in values:

            if value.lower() == user_input.lower():

                print(f"\nSelected {title}: {value}")
                return value

        suggestions = [
            value
            for value in values
            if user_input.lower() in value.lower()
        ]

        if suggestions:

            print("\nNo exact match found.")
            print("\nSuggestions:")

            for index, value in enumerate(suggestions, start=1):
                print(f"{index}. {value}")

            suggestion_input = input(
                "\nEnter suggestion number or type again: "
            ).strip()

            if suggestion_input.isdigit():

                suggestion_index = int(suggestion_input) - 1

                if 0 <= suggestion_index < len(suggestions):

                    selected_value = suggestions[suggestion_index]
                    print(f"\nSelected {title}: {selected_value}")
                    return selected_value

            print("\nPlease try again.")

        else:

            print("\nNo matching option found.")


# ============================================================
# SELECT YEAR
# ============================================================

def select_year(years):

    years = sorted(years)

    display_all_options("Available Years", years)

    while True:

        user_input = input(
            "\nEnter Year (number or year): "
        ).strip()

        if user_input.isdigit():

            possible_index = int(user_input) - 1

            if 0 <= possible_index < len(years):

                selected_year = years[possible_index]
                print(f"\nSelected Year: {selected_year}")
                return selected_year

            entered_year = int(user_input)

            if entered_year in years:

                print(f"\nSelected Year: {entered_year}")
                return entered_year

        print("\nInvalid year.")


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(input_df, prediction):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    result_df = input_df.copy()
    result_df["predicted_demand"] = prediction

    latest_prediction_path = os.path.join(
        RESULTS_DIR,
        "latest_demand_prediction.csv"
    )

    result_df.to_csv(latest_prediction_path, index=False)

    print(f"\nPrediction saved to:\n{latest_prediction_path}")

    return latest_prediction_path


# ============================================================
# GET DEMAND LEVEL
# ============================================================

def get_demand_level(prediction):

    if prediction < 500:
        return "LOW"

    elif prediction < 1000:
        return "MODERATE"

    elif prediction < 2000:
        return "HIGH"

    else:
        return "VERY HIGH"


# ============================================================
# SHAP REPORT
# ============================================================

def generate_shap_report(
    model,
    input_df,
    prediction,
    feature_names,
    country_code,
    commodity_code,
    actual_month,
    actual_day,
    actual_dayofweek,
    actual_weekend,
    lag_1,
    lag_2,
    lag_3,
    rolling_mean_3,
    df=None,
    encoder_map=None
):

    print("\n" + "=" * 70)
    print("GENERATING SHAP EXPLANATION")
    print("=" * 70)

    if not SHAP_AVAILABLE:

        print("\nWARNING: SHAP is not installed or is blocked.")
        print("Falling back to leave-one-out attribution.")

    try:

        model_input = pd.DataFrame(
            [{
                "country_encoded": country_code,
                "commodity_encoded": commodity_code,
                "year": int(input_df.iloc[0]["year"]),
                "month": actual_month,
                "day": actual_day,
                "dayofweek": actual_dayofweek,
                "weekend": actual_weekend,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "rolling_mean_3": rolling_mean_3
            }]
        )

        model_input = model_input[feature_names]

        feature_names_cut, shap_sample, method = shap_or_loo_demand_prediction(
            model,
            model_input,
            df,
            encoder_map,
            feature_names
        )

        if method == "loo":
            print("\nWARNING: shap unavailable (blocked by Windows policy).")
            print("Using leave-one-out local attribution instead.")
        else:
            print("\nSHAP values calculated.")

        shap_df = pd.DataFrame(
            {
                "feature": feature_names_cut,
                "shap_value": shap_sample,
                "absolute_shap_value": np.abs(shap_sample)
            }
        )

        shap_df = (
            shap_df
            .sort_values("absolute_shap_value", ascending=False)
            .reset_index(drop=True)
        )

        shap_csv_path = os.path.join(
            RESULTS_DIR,
            "demand_shap_importance.csv"
        )

        shap_df.to_csv(shap_csv_path, index=False)

        print(f"\nSHAP importance saved to:\n{shap_csv_path}")

        top_shap = shap_df.head(15).copy()

        plt.figure(figsize=(12, 8))

        ordered_features = top_shap["feature"].iloc[::-1]
        ordered_values = top_shap["shap_value"].iloc[::-1]

        bars = plt.barh(ordered_features, ordered_values)

        plt.xlabel("SHAP Contribution")
        plt.ylabel("Feature")
        plt.title("SHAP Feature Contribution for Demand Prediction")
        plt.grid(axis="x", alpha=0.3)

        for bar, value in zip(bars, ordered_values):

            plt.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f" {value:.4f}",
                va="center",
                fontsize=8
            )

        plt.tight_layout()

        shap_graph_path = os.path.join(
            RESULTS_DIR,
            "demand_shap_explanation.png"
        )

        plt.savefig(shap_graph_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"\nSHAP graph saved to:\n{shap_graph_path}")

        shap_pdf_path = os.path.join(
            RESULTS_DIR,
            "demand_shap_report.pdf"
        )

        demand_level = get_demand_level(prediction)

        with PdfPages(shap_pdf_path) as pdf:

            # Page 1
            fig1, ax1 = plt.subplots(figsize=(11.7, 8.3))
            ax1.axis("off")

            ax1.text(0.5, 0.94, "SUSTAINFLOW AI", fontsize=22,
                     fontweight="bold", ha="center")
            ax1.text(0.5, 0.88, "SHAP Demand Prediction Report",
                     fontsize=15, ha="center")

            ax1.text(0.08, 0.78, "PREDICTION INPUT", fontsize=14,
                     fontweight="bold")

            input_y = 0.71

            input_labels = [
                ("Country", input_df.iloc[0]["country"]),
                ("Commodity", input_df.iloc[0]["commodity"]),
                ("Year", input_df.iloc[0]["year"]),
                ("Month", actual_month),
                ("lag_1", lag_1),
                ("lag_2", lag_2),
                ("lag_3", lag_3)
            ]

            for label, value in input_labels:

                ax1.text(0.10, input_y, f"{label}: {value}", fontsize=11)
                input_y -= 0.055

            ax1.text(0.08, 0.35, "MODEL PREDICTION", fontsize=14,
                     fontweight="bold")

            ax1.text(0.10, 0.28,
                     f"Predicted Demand: {prediction:.2f}",
                     fontsize=18, fontweight="bold")

            ax1.text(0.10, 0.22,
                     f"Demand Level: {demand_level}",
                     fontsize=14)

            ax1.text(0.08, 0.15, "TOP FACTORS", fontsize=14,
                     fontweight="bold")

            factor_y = 0.09

            for index, (_, row) in enumerate(
                top_shap.head(5).iterrows(), start=1
            ):

                direction = (
                    "increases" if row["shap_value"] >= 0
                    else "decreases"
                )

                feature_name = str(row["feature"])

                if len(feature_name) > 55:
                    feature_name = feature_name[:52] + "..."

                ax1.text(0.10, factor_y,
                         f"{index}. {feature_name} ({direction} prediction)",
                         fontsize=10)

                factor_y -= 0.040

            ax1.text(0.5, 0.03, "Generated by SustainFlow AI",
                     fontsize=9, ha="center", style="italic")

            plt.tight_layout()
            pdf.savefig(fig1, bbox_inches="tight")
            plt.close()

            # Page 2
            fig2, ax2 = plt.subplots(figsize=(11.7, 8.3))

            chart_features = top_shap["feature"].iloc[::-1]
            chart_values = top_shap["shap_value"].iloc[::-1]

            bars = ax2.barh(chart_features, chart_values)

            ax2.set_xlabel("SHAP Contribution")
            ax2.set_ylabel("Feature")
            ax2.set_title("TOP FEATURE CONTRIBUTIONS",
                          fontsize=15, fontweight="bold")
            ax2.grid(axis="x", alpha=0.3)

            for bar, value in zip(bars, chart_values):

                ax2.text(value,
                         bar.get_y() + bar.get_height() / 2,
                         f" {value:.4f}",
                         va="center", fontsize=8)

            plt.tight_layout()
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close()

            # Page 3
            fig3, ax3 = plt.subplots(figsize=(11.7, 8.3))
            ax3.axis("off")

            ax3.text(0.5, 0.95, "SHAP FEATURE CONTRIBUTION TABLE",
                     fontsize=16, fontweight="bold", ha="center")

            table_rows = [["Rank", "Feature", "SHAP Value", "Impact"]]

            for index, (_, row) in enumerate(
                top_shap.head(15).iterrows(), start=1
            ):

                feature_name = str(row["feature"])

                if len(feature_name) > 45:
                    feature_name = feature_name[:42] + "..."

                impact = (
                    "Increases prediction"
                    if row["shap_value"] >= 0
                    else "Decreases prediction"
                )

                table_rows.append([
                    str(index),
                    feature_name,
                    f"{row['shap_value']:.5f}",
                    impact
                ])

            table = ax3.table(
                cellText=table_rows,
                loc="center",
                cellLoc="left",
                colWidths=[0.08, 0.50, 0.18, 0.24]
            )

            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.45)

            for column in range(4):
                table[(0, column)].set_text_props(weight="bold")

            plt.tight_layout()
            pdf.savefig(fig3, bbox_inches="tight")
            plt.close()

        print("\n" + "=" * 70)
        print("SHAP EXPLANATION GENERATED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nSHAP PDF report:\n{shap_pdf_path}")

    except Exception as error:

        print("\nWARNING: SHAP report could not be generated.")
        print(f"Reason: {str(error)}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("SUSTAINFLOW AI")
    print("MODULE 2 - DEMAND PREDICTION")
    print("=" * 70)

    df = load_dataset()
    model = load_model()
    encoder_map = load_encoders()

    countries = sorted(
        df["country"].dropna().astype(str).unique().tolist(),
        key=str.lower
    )

    commodities = sorted(
        df["commodity"].dropna().astype(str).unique().tolist(),
        key=str.lower
    )

    years = sorted(
        df["year"].dropna().astype(int).unique().tolist()
    )

    print("\nYou can select using:")
    print("1. Number from the list")
    print("2. Exact name")
    print("3. Partial name to get suggestions")

    country = select_option("Country", countries)
    commodity = select_option("Commodity", commodities)
    year = select_year(years)

    input_df = pd.DataFrame(
        {
            "country": [country],
            "commodity": [commodity],
            "year": [year]
        }
    )

    # ========================================================
    # GET LATEST DATA FOR THIS COMBINATION
    # ========================================================

    print("\n" + "=" * 70)
    print("LOOKING UP LATEST DATA FOR THIS COMBINATION")
    print("=" * 70)

    combo_data = df[
        (df["country"] == country) &
        (df["commodity"] == commodity) &
        (df["year"] == year)
    ].sort_values("month")

    if len(combo_data) > 0:

        print(f"Found {len(combo_data)} rows for this combination")
        print(combo_data[["country", "commodity", "year", "month", "demand"]].to_string())

        latest = combo_data.iloc[-1]

        if len(combo_data) >= 3:
            lag_1 = combo_data.iloc[-1]["demand"]
            lag_2 = combo_data.iloc[-2]["demand"]
            lag_3 = combo_data.iloc[-3]["demand"]
            rolling_mean_3 = (lag_1 + lag_2 + lag_3) / 3
        elif len(combo_data) == 2:
            lag_1 = combo_data.iloc[-1]["demand"]
            lag_2 = combo_data.iloc[-2]["demand"]
            lag_3 = 0
            rolling_mean_3 = (lag_1 + lag_2) / 2
        else:
            lag_1 = combo_data.iloc[-1]["demand"]
            lag_2 = 0
            lag_3 = 0
            rolling_mean_3 = lag_1

        actual_month = int(latest["month"])
        actual_day = int(latest.get("day", 1))
        actual_dayofweek = int(latest.get("dayofweek", 0))
        actual_weekend = int(latest.get("weekend", 0))

        print(f"\nUsing latest month data:")
        print(f"  Month: {actual_month}")
        print(f"  lag_1: {lag_1}")
        print(f"  lag_2: {lag_2}")
        print(f"  lag_3: {lag_3}")
        print(f"  rolling_mean_3: {rolling_mean_3:.2f}")

    else:

        print(f"No data found. Using defaults.")

        actual_month = 1
        actual_day = 1
        actual_dayofweek = 0
        actual_weekend = 0
        lag_1 = 0
        lag_2 = 0
        lag_3 = 0
        rolling_mean_3 = 0

    # ========================================================
    # ENCODE USING SAVED MAPPINGS
    # ========================================================

    country_code = encode_value(encoder_map, 'country', country)
    commodity_code = encode_value(encoder_map, 'commodity', commodity)

    feature_names = [
        "country_encoded",
        "commodity_encoded",
        "year",
        "month",
        "day",
        "dayofweek",
        "weekend",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3"
    ]

    model_input = pd.DataFrame(
        [{
            "country_encoded": country_code,
            "commodity_encoded": commodity_code,
            "year": int(year),
            "month": actual_month,
            "day": actual_day,
            "dayofweek": actual_dayofweek,
            "weekend": actual_weekend,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "rolling_mean_3": rolling_mean_3
        }]
    )

    model_input = model_input[feature_names]

    # ========================================================
    # PREDICTION
    # ========================================================

    print("\n" + "=" * 70)
    print("MAKING PREDICTION")
    print("=" * 70)

    prediction = float(model.predict(model_input)[0])
    demand_level = get_demand_level(prediction)

    print("\n" + "=" * 70)
    print("DEMAND PREDICTION RESULT")
    print("=" * 70)

    print(f"\nCountry: {country}")
    print(f"Commodity: {commodity}")
    print(f"Year: {year}")

    print("\n" + "-" * 70)

    print(f"\nPREDICTED DEMAND: {prediction:.2f}")
    print(f"\nDEMAND LEVEL: {demand_level}")

    print("\n" + "=" * 70)

    if len(combo_data) > 0:

        actual_values = combo_data["demand"].values
        actual_mean = actual_values.mean()

        print("\n" + "=" * 70)
        print("ACTUAL VS PREDICTED COMPARISON")
        print("=" * 70)

        print(f"\nActual values: {actual_values}")
        print(f"Actual average: {actual_mean:.2f}")
        print(f"Predicted: {prediction:.2f}")
        print(f"Difference: {abs(prediction - actual_mean):.2f}")

    save_prediction(input_df, prediction)

    generate_shap_report(
        model, input_df, prediction, feature_names,
        country_code, commodity_code,
        actual_month, actual_day, actual_dayofweek, actual_weekend,
        lag_1, lag_2, lag_3, rolling_mean_3,
        df, encoder_map
    )

    print("\n" + "=" * 70)
    print("PREDICTION PROCESS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()