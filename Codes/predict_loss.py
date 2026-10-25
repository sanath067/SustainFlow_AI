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

from xai_attribution import shap_or_loo_fao_prediction


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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Results",
    "fao_model.joblib"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "Results"
)


# ============================================================
# HELPER FUNCTION: CLEAN TEXT
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 70)
    print("LOADING FAO DATASET")
    print("=" * 70)

    print(
        f"\nDataset path:\n{DATA_PATH}"
    )

    df = pd.read_csv(
        DATA_PATH
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print(
        f"\nDataset loaded successfully."
    )

    print(
        f"Total rows: {len(df)}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    required_columns = [
        "commodity",
        "country",
        "food_supply_stage",
        "year",
        "loss_percentage"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print(
            "\nERROR: Required columns are missing."
        )

        print(
            f"Missing columns: {missing_columns}"
        )

        print(
            f"Available columns: "
            f"{df.columns.tolist()}"
        )

        raise ValueError(
            "Dataset does not contain the "
            "required columns."
        )

    # Clean categorical columns
    for column in [
        "commodity",
        "country",
        "food_supply_stage"
    ]:

        df[column] = (
            df[column]
            .apply(clean_text)
        )

    # Convert numeric columns
    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce"
    )

    df["loss_percentage"] = pd.to_numeric(
        df["loss_percentage"],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "commodity",
            "country",
            "food_supply_stage",
            "year",
            "loss_percentage"
        ]
    )

    # Remove empty categorical values
    df = df[
        (
            df["commodity"] != ""
        )
        &
        (
            df["country"] != ""
        )
        &
        (
            df["food_supply_stage"] != ""
        )
    ]

    print(
        f"\nUsable rows: {len(df)}"
    )

    return df


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 70)
    print("LOADING TRAINED FAO MODEL")
    print("=" * 70)

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\nModel not found:\n"
            f"{MODEL_PATH}\n\n"
            f"Run train_fao_model.py first."
        )

    pipeline = joblib.load(
        MODEL_PATH
    )

    print(
        "\nModel loaded successfully."
    )

    return pipeline


# ============================================================
# DISPLAY ALL OPTIONS
# ============================================================

def display_all_options(
    title,
    values
):

    print("\n" + "=" * 70)

    print(
        title.upper()
    )

    print("=" * 70)

    print(
        f"\nTotal available options: "
        f"{len(values)}\n"
    )

    # Display ALL items
    for index, value in enumerate(
        values,
        start=1
    ):

        print(
            f"{index}. {value}"
        )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# SELECT OPTION
# ============================================================

def select_option(
    title,
    values
):

    display_all_options(
        title,
        values
    )

    while True:

        user_input = input(
            f"\nEnter {title} "
            f"(number or name): "
        ).strip()

        # Number selection
        if user_input.isdigit():

            selected_index = (
                int(user_input) - 1
            )

            if (
                0 <= selected_index
                < len(values)
            ):

                selected_value = (
                    values[
                        selected_index
                    ]
                )

                print(
                    f"\nSelected {title}: "
                    f"{selected_value}"
                )

                return selected_value

            else:

                print(
                    "\nInvalid number."
                )

                print(
                    "Please enter a number "
                    "from the displayed list."
                )

                continue

        # Exact text selection
        for value in values:

            if (
                value.lower()
                ==
                user_input.lower()
            ):

                print(
                    f"\nSelected {title}: "
                    f"{value}"
                )

                return value

        # Suggestions
        suggestions = [
            value
            for value in values
            if (
                user_input.lower()
                in
                value.lower()
            )
        ]

        if suggestions:

            print(
                "\nNo exact match found."
            )

            print(
                "\nSuggestions:"
            )

            for index, value in enumerate(
                suggestions,
                start=1
            ):

                print(
                    f"{index}. {value}"
                )

            suggestion_input = input(
                "\nEnter suggestion number "
                "or type again: "
            ).strip()

            if suggestion_input.isdigit():

                suggestion_index = (
                    int(
                        suggestion_input
                    ) - 1
                )

                if (
                    0 <= suggestion_index
                    < len(suggestions)
                ):

                    selected_value = (
                        suggestions[
                            suggestion_index
                        ]
                    )

                    print(
                        f"\nSelected {title}: "
                        f"{selected_value}"
                    )

                    return selected_value

            print(
                "\nPlease try again."
            )

        else:

            print(
                "\nNo matching option found."
            )

            print(
                "Please select from "
                "the available list."
            )


# ============================================================
# SELECT YEAR
# ============================================================

def select_year(
    years
):

    years = sorted(
        years
    )

    display_all_options(
        "Available Years",
        years
    )

    while True:

        user_input = input(
            "\nEnter Year "
            "(number or year): "
        ).strip()

        # Number from displayed list
        if user_input.isdigit():

            possible_index = (
                int(user_input) - 1
            )

            if (
                0 <= possible_index
                < len(years)
            ):

                selected_year = years[
                    possible_index
                ]

                print(
                    f"\nSelected Year: "
                    f"{selected_year}"
                )

                return selected_year

            # Direct year input
            entered_year = int(
                user_input
            )

            if entered_year in years:

                print(
                    f"\nSelected Year: "
                    f"{entered_year}"
                )

                return entered_year

        print(
            "\nInvalid year."
        )

        print(
            "Please select a year "
            "from the displayed list."
        )


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    input_df,
    prediction
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    result_df = input_df.copy()

    result_df[
        "predicted_loss_percentage"
    ] = prediction

    latest_prediction_path = os.path.join(
        RESULTS_DIR,
        "latest_prediction.csv"
    )

    result_df.to_csv(
        latest_prediction_path,
        index=False
    )

    print(
        f"\nPrediction saved to:\n"
        f"{latest_prediction_path}"
    )

    return latest_prediction_path


# ============================================================
# GET RISK LEVEL
# ============================================================

def get_risk_level(
    prediction
):

    if prediction < 5:

        return "LOW"

    elif prediction < 15:

        return "MEDIUM"

    else:

        return "HIGH"


# ============================================================
# GENERATE SHAP EXPLANATION
# ============================================================

def generate_shap_report(
    pipeline,
    input_df,
    prediction,
    df=None
):

    print("\n" + "=" * 70)
    print("GENERATING SHAP EXPLANATION")
    print("=" * 70)

    if not SHAP_AVAILABLE:

        print(
            "\nWARNING: SHAP is not installed or is blocked."
        )

        print(
            "Falling back to leave-one-out attribution."
        )

    try:

        # ====================================================
        # GET PIPELINE COMPONENTS
        # ====================================================

        preprocessor = (
            pipeline
            .named_steps[
                "preprocess"
            ]
        )

        model = (
            pipeline
            .named_steps[
                "model"
            ]
        )

        # ====================================================
        # TRANSFORM INPUT USING SAME PREPROCESSOR
        # ====================================================

        X_transformed = (
            preprocessor.transform(
                input_df
            )
        )

        # Convert sparse matrix if necessary
        if hasattr(
            X_transformed,
            "toarray"
        ):

            X_dense = (
                X_transformed
                .toarray()
            )

        else:

            X_dense = np.asarray(
                X_transformed
            )

        print(
            f"\nOriginal input shape: "
            f"{input_df.shape}"
        )

        print(
            f"Transformed input shape: "
            f"{X_dense.shape}"
        )

        # ====================================================
        # FEATURE NAMES
        # ====================================================

        try:

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

        except Exception:

            feature_names = [
                f"Feature_{index + 1}"
                for index in range(
                    X_dense.shape[1]
                )
            ]

        feature_names = [
            str(feature)
            for feature in feature_names
        ]

        # ====================================================
        # CREATE SHAP EXPLAINER (with LOO fallback)
        # ====================================================

        print(
            "\nCalculating feature contributions..."
        )

        feature_names, shap_sample, method = shap_or_loo_fao_prediction(
            pipeline,
            input_df,
            df
        )

        feature_names = [
            str(feature)
            for feature in feature_names
        ]

        if method == "loo":
            print(
                "\nWARNING: shap unavailable (blocked by Windows policy)."
            )
            print(
                "Using leave-one-out local attribution instead."
            )
        else:
            print(
                "\nSHAP values calculated."
            )

        # ====================================================
        # CREATE SHAP DATAFRAME
        # ====================================================

        shap_df = pd.DataFrame(
            {
                "feature":
                    feature_names,

                "shap_value":
                    shap_sample,

                "absolute_shap_value":
                    np.abs(
                        shap_sample
                    )
            }
        )

        shap_df = (
            shap_df
            .sort_values(
                "absolute_shap_value",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        # ====================================================
        # SAVE SHAP CSV
        # ====================================================

        shap_csv_path = os.path.join(
            RESULTS_DIR,
            "shap_importance.csv"
        )

        shap_df.to_csv(
            shap_csv_path,
            index=False
        )

        print(
            f"\nSHAP importance saved to:\n"
            f"{shap_csv_path}"
        )

        # ====================================================
        # TOP FEATURES
        # ====================================================

        top_shap = (
            shap_df
            .head(15)
            .copy()
        )

        # ====================================================
        # GENERATE SHAP GRAPH
        # ====================================================

        plt.figure(
            figsize=(12, 8)
        )

        ordered_features = (
            top_shap[
                "feature"
            ].iloc[::-1]
        )

        ordered_values = (
            top_shap[
                "shap_value"
            ].iloc[::-1]
        )

        bars = plt.barh(
            ordered_features,
            ordered_values
        )

        plt.xlabel(
            "SHAP Contribution"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            "SHAP Feature Contribution "
            "for Food Loss Prediction"
        )

        plt.grid(
            axis="x",
            alpha=0.3
        )

        for bar, value in zip(
            bars,
            ordered_values
        ):

            plt.text(
                value,
                bar.get_y()
                +
                bar.get_height() / 2,
                f" {value:.4f}",
                va="center",
                fontsize=8
            )

        plt.tight_layout()

        shap_graph_path = os.path.join(
            RESULTS_DIR,
            "shap_explanation.png"
        )

        plt.savefig(
            shap_graph_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"\nSHAP graph saved to:\n"
            f"{shap_graph_path}"
        )

        # ====================================================
        # GENERATE SHAP PDF REPORT
        # ====================================================

        shap_pdf_path = os.path.join(
            RESULTS_DIR,
            "shap_report.pdf"
        )

        risk_level = get_risk_level(
            prediction
        )

        with PdfPages(
            shap_pdf_path
        ) as pdf:

            # ================================================
            # PAGE 1 - PREDICTION SUMMARY
            # ================================================

            fig1, ax1 = plt.subplots(
                figsize=(11.7, 8.3)
            )

            ax1.axis(
                "off"
            )

            ax1.text(
                0.5,
                0.94,
                "SUSTAINFLOW AI",
                fontsize=22,
                fontweight="bold",
                ha="center"
            )

            ax1.text(
                0.5,
                0.88,
                "SHAP Prediction Explanation Report",
                fontsize=15,
                ha="center"
            )

            ax1.text(
                0.08,
                0.78,
                "PREDICTION INPUT",
                fontsize=14,
                fontweight="bold"
            )

            input_y = 0.71

            input_labels = [
                (
                    "Commodity",
                    input_df.iloc[
                        0
                    ]["commodity"]
                ),
                (
                    "Country",
                    input_df.iloc[
                        0
                    ]["country"]
                ),
                (
                    "Food Supply Stage",
                    input_df.iloc[
                        0
                    ][
                        "food_supply_stage"
                    ]
                ),
                (
                    "Year",
                    input_df.iloc[
                        0
                    ]["year"]
                )
            ]

            for label, value in input_labels:

                ax1.text(
                    0.10,
                    input_y,
                    f"{label}: {value}",
                    fontsize=11
                )

                input_y -= 0.055

            ax1.text(
                0.08,
                0.42,
                "MODEL PREDICTION",
                fontsize=14,
                fontweight="bold"
            )

            ax1.text(
                0.10,
                0.34,
                f"Predicted Food Loss: "
                f"{prediction:.2f}%",
                fontsize=18,
                fontweight="bold"
            )

            ax1.text(
                0.10,
                0.28,
                f"Risk Level: "
                f"{risk_level}",
                fontsize=14,
                fontweight="bold"
            )

            ax1.text(
                0.08,
                0.20,
                "TOP FACTORS "
                "INFLUENCING THE PREDICTION",
                fontsize=14,
                fontweight="bold"
            )

            factor_y = 0.14

            for index, (
                _,
                row
            ) in enumerate(
                top_shap
                .head(5)
                .iterrows(),
                start=1
            ):

                if (
                    row[
                        "shap_value"
                    ]
                    >= 0
                ):

                    direction = (
                        "increases"
                    )

                else:

                    direction = (
                        "decreases"
                    )

                feature_name = str(
                    row[
                        "feature"
                    ]
                )

                if len(
                    feature_name
                ) > 55:

                    feature_name = (
                        feature_name[:52]
                        +
                        "..."
                    )

                ax1.text(
                    0.10,
                    factor_y,
                    f"{index}. "
                    f"{feature_name} "
                    f"({direction} "
                    f"prediction)",
                    fontsize=10
                )

                factor_y -= 0.040

            ax1.text(
                0.5,
                0.03,
                "Generated by SustainFlow AI",
                fontsize=9,
                ha="center",
                style="italic"
            )

            plt.tight_layout()

            pdf.savefig(
                fig1,
                bbox_inches="tight"
            )

            plt.close()

            # ================================================
            # PAGE 2 - SHAP BAR CHART
            # ================================================

            fig2, ax2 = plt.subplots(
                figsize=(11.7, 8.3)
            )

            chart_features = (
                top_shap[
                    "feature"
                ].iloc[::-1]
            )

            chart_values = (
                top_shap[
                    "shap_value"
                ].iloc[::-1]
            )

            bars = ax2.barh(
                chart_features,
                chart_values
            )

            ax2.set_xlabel(
                "SHAP Contribution"
            )

            ax2.set_ylabel(
                "Feature"
            )

            ax2.set_title(
                "TOP FEATURE CONTRIBUTIONS "
                "TO THIS PREDICTION",
                fontsize=15,
                fontweight="bold"
            )

            ax2.grid(
                axis="x",
                alpha=0.3
            )

            for bar, value in zip(
                bars,
                chart_values
            ):

                ax2.text(
                    value,
                    bar.get_y()
                    +
                    bar.get_height()
                    /
                    2,
                    f" {value:.4f}",
                    va="center",
                    fontsize=8
                )

            plt.tight_layout()

            pdf.savefig(
                fig2,
                bbox_inches="tight"
            )

            plt.close()

            # ================================================
            # PAGE 3 - SHAP TABLE
            # ================================================

            fig3, ax3 = plt.subplots(
                figsize=(11.7, 8.3)
            )

            ax3.axis(
                "off"
            )

            ax3.text(
                0.5,
                0.95,
                "SHAP FEATURE CONTRIBUTION TABLE",
                fontsize=16,
                fontweight="bold",
                ha="center"
            )

            table_rows = []

            table_rows.append(
                [
                    "Rank",
                    "Feature",
                    "SHAP Value",
                    "Impact"
                ]
            )

            for index, (
                _,
                row
            ) in enumerate(
                top_shap
                .head(15)
                .iterrows(),
                start=1
            ):

                feature_name = str(
                    row[
                        "feature"
                    ]
                )

                if len(
                    feature_name
                ) > 45:

                    feature_name = (
                        feature_name[:42]
                        +
                        "..."
                    )

                impact = (
                    "Increases prediction"
                    if row[
                        "shap_value"
                    ]
                    >= 0
                    else
                    "Decreases prediction"
                )

                table_rows.append(
                    [
                        str(
                            index
                        ),
                        feature_name,
                        f"{row['shap_value']:.5f}",
                        impact
                    ]
                )

            table = ax3.table(
                cellText=table_rows,
                loc="center",
                cellLoc="left",
                colWidths=[
                    0.08,
                    0.50,
                    0.18,
                    0.24
                ]
            )

            table.auto_set_font_size(
                False
            )

            table.set_fontsize(
                8
            )

            table.scale(
                1,
                1.45
            )

            for column in range(
                4
            ):

                table[
                    (
                        0,
                        column
                    )
                ].set_text_props(
                    weight="bold"
                )

            plt.tight_layout()

            pdf.savefig(
                fig3,
                bbox_inches="tight"
            )

            plt.close()

        print(
            "\n" + "=" * 70
        )

        print(
            "SHAP EXPLANATION GENERATED SUCCESSFULLY"
        )

        print(
            "=" * 70
        )

        print(
            f"\nSHAP PDF report:\n"
            f"{shap_pdf_path}"
        )

    except Exception as error:

        print(
            "\nWARNING: SHAP report could not "
            "be generated."
        )

        print(
            f"Reason: {str(error)}"
        )


# ============================================================
# MAIN PREDICTION PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "SUSTAINFLOW AI"
    )

    print(
        "FAO FOOD LOSS PREDICTION"
    )

    print(
        "=" * 70
    )

    # Load dataset
    df = load_dataset()

    # Load model
    pipeline = load_model()

    # ========================================================
    # GET AVAILABLE OPTIONS
    # ========================================================

    commodities = sorted(
        df[
            "commodity"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist(),
        key=str.lower
    )

    countries = sorted(
        df[
            "country"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist(),
        key=str.lower
    )

    stages = sorted(
        df[
            "food_supply_stage"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist(),
        key=str.lower
    )

    years = sorted(
        df[
            "year"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    # ========================================================
    # USER SELECTION
    # ========================================================

    print(
        "\nYou can select using:"
    )

    print(
        "1. Number from the list"
    )

    print(
        "2. Exact name"
    )

    print(
        "3. Partial name to get suggestions"
    )

    print(
        "\nAll available dataset items "
        "will be displayed."
    )

    # Commodity
    commodity = select_option(
        "Commodity",
        commodities
    )

    # Country
    country = select_option(
        "Country",
        countries
    )

    # Food Supply Stage
    stage = select_option(
        "Food Supply Stage",
        stages
    )

    # Year
    year = select_year(
        years
    )

    # ========================================================
    # CREATE MODEL INPUT
    # ========================================================

    input_df = pd.DataFrame(
        {
            "commodity":
                [commodity],

            "country":
                [country],

            "food_supply_stage":
                [stage],

            "year":
                [year]
        }
    )

    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "MAKING PREDICTION"
    )

    print(
        "=" * 70
    )

    prediction_array = (
        pipeline.predict(
            input_df
        )
    )

    # Inverse log-transform: model outputs log1p(loss_percentage)
    prediction = float(
        np.expm1(prediction_array[0])
    )

    risk_level = get_risk_level(
        prediction
    )

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "FOOD LOSS PREDICTION RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nCommodity: "
        f"{commodity}"
    )

    print(
        f"Country: "
        f"{country}"
    )

    print(
        f"Food Supply Stage: "
        f"{stage}"
    )

    print(
        f"Year: "
        f"{year}"
    )

    print(
        "\n" + "-" * 70
    )

    print(
        f"\nPREDICTED FOOD LOSS: "
        f"{prediction:.2f}%"
    )

    print(
        f"\nRISK LEVEL: "
        f"{risk_level}"
    )

    print(
        "\n" + "=" * 70
    )

    # ========================================================
    # SAVE PREDICTION
    # ========================================================

    save_prediction(
        input_df,
        prediction
    )

    # ========================================================
    # GENERATE SHAP REPORT
    # ========================================================

    generate_shap_report(
        pipeline,
        input_df,
        prediction,
        df
    )

    print("\n" + "=" * 70)

    print(
        "PREDICTION PROCESS COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()