import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAO_PATH = os.path.join(BASE_DIR, "Datasets", "Fao_merged_clean.csv")
DEMAND_PATH = os.path.join(BASE_DIR, "Datasets", "module2_matched_demand.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "Datasets", "module3_fao_spoilage_v2.csv")

print("MODULE 3 v2: LEAKAGE-FREE SPOILAGE DATASET BUILDER")
print("Label comes from an external shelf-life reference table only.")
print("Model features come from Module 1 + Module 2 data only.\n")

# ============================================================
# EXTERNAL REFERENCE: TYPICAL SHELF LIFE (DAYS), BY COMMODITY FAMILY
# Approximate literature values for standard commercial storage.
# This table is independent of any column in this project's data -
# it is not calculated from loss_percentage, volatility, or stage.
# ============================================================

SHELF_LIFE_DAYS = [
    ("fish", 3), ("seafood", 3), ("shrimp", 3),
    ("lettuce", 5), ("spinach", 5), ("leafy", 5), ("herbs", 5),
    ("raspberr", 5), ("mushroom", 6), ("berry", 6), ("berries", 6),
    ("strawberr", 6),
    ("cauliflower", 7), ("broccoli", 7),
    ("milk", 7), ("cream", 7), ("peach", 7), ("plum", 7), ("apricot", 7),
    ("papaya", 7), ("okra", 7),
    ("tomato", 10), ("cucumber", 10), ("pepper", 10), ("eggplant", 10),
    ("cherry", 10), ("cherries", 10), ("fig", 5), ("avocado", 10),
    ("mango", 10), ("banana", 10), ("sausage", 10),
    ("meat of", 5), ("poultry", 5), ("chicken", 5), ("beef", 5),
    ("pork", 5), ("mutton", 5), ("cattle", 5), ("pig", 5),
    ("sheep", 5), ("goat", 5),
    ("yoghurt", 14), ("yogurt", 14), ("melon", 14), ("watermelon", 14),
    ("grape", 14), ("radish", 14), ("celery", 14),
    ("carrot", 21), ("cabbage", 21), ("turnip", 21), ("beet", 21),
    ("egg", 21), ("kiwi", 21), ("juice", 21), ("pineapple", 14),
    ("apple", 30), ("pear", 30), ("orange", 30), ("citrus", 30),
    ("lemon", 30), ("lime", 30),
    ("sweet potato", 45), ("cheese", 45),
    ("potato", 60), ("onion", 60), ("yam", 60), ("butter", 60),
    ("cassava", 14),
    ("garlic", 90),
    ("sweet corn", 14),
    ("dry", 270), ("flour", 180), ("oil", 180), ("starch", 270),
    ("nut", 180), ("almond", 180), ("cashew", 180), ("walnut", 180),
    ("bean", 180), ("pea", 180), ("lentil", 365), ("chick pea", 365),
    ("rice", 365), ("wheat", 365), ("maize", 365), ("barley", 365),
    ("oat", 365), ("sorghum", 365), ("millet", 365), ("cereal", 270),
    ("sugar", 365), ("cocoa", 365), ("coffee", 365), ("tea", 365),
    ("spice", 365),
]

DEFAULT_SHELF_LIFE_DAYS = 90  # generic packaged/processed goods fallback


def shelf_life_for(commodity: str) -> int:
    c = str(commodity).lower()
    for keyword, days in SHELF_LIFE_DAYS:
        if keyword in c:
            return days
    return DEFAULT_SHELF_LIFE_DAYS


# ============================================================
# LOAD MODULE 1 (FAO LOSS) DATA
# ============================================================

fao = pd.read_csv(FAO_PATH)
fao.columns = fao.columns.astype(str).str.strip().str.lower()
fao = fao.rename(columns={"losspercentage": "loss_percentage", "loss%": "loss_percentage"})
if "loss_percentage" not in fao.columns:
    raise KeyError("FAO dataset needs a loss_percentage column")
fao["loss_percentage"] = pd.to_numeric(fao["loss_percentage"], errors="coerce").fillna(0.0)
fao["commodity"] = fao["commodity"].astype(str).str.strip()
fao["country"] = fao["country"].astype(str).str.strip()
fao["food_supply_stage"] = fao["food_supply_stage"].astype(str).str.strip().str.lower()

# ============================================================
# LOAD MODULE 2 (DEMAND) DATA -> AGGREGATE FEATURES
# ============================================================

dem = pd.read_csv(DEMAND_PATH)
dem.columns = dem.columns.astype(str).str.strip().str.lower()
dem["month"] = dem["month"].astype(int)
dem["commodity"] = dem["commodity"].astype(str).str.strip()
dem["country"] = dem["country"].astype(str).str.strip()

key = ["country", "commodity", "year"]
agg = dem.groupby(key).agg(
    avg_demand=("demand", "mean"),
    std_demand=("demand", "std"),
).reset_index()
agg["volatility_ratio"] = (agg["std_demand"] / agg["avg_demand"].clip(lower=0.01)).clip(0, 5).fillna(0.5)
peak = (dem.loc[dem.groupby(key)["demand"].idxmax(), key + ["month"]]
        .rename(columns={"month": "peak_month"}))
agg = agg.merge(peak, on=key, how="left")
agg["peak_month"] = agg["peak_month"].fillna(6).astype(int)

sub = dem.groupby(["country", "commodity"]).agg(
    avg_demand=("demand", "mean"), std_demand=("demand", "std")).reset_index()
sub["volatility_ratio"] = (sub["std_demand"] / sub["avg_demand"].clip(lower=0.01)).clip(0, 5).fillna(0.5)
sub_peak = (dem.loc[dem.groupby(["country", "commodity"])["demand"].idxmax(),
                     ["country", "commodity", "month"]]
            .rename(columns={"month": "peak_month"}))
sub = sub.merge(sub_peak, on=["country", "commodity"], how="left")
sub["peak_month"] = sub["peak_month"].fillna(6).astype(int)

fao = fao.merge(agg, on=["country", "commodity", "year"], how="left")
fao = fao.merge(sub, on=["country", "commodity"], how="left", suffixes=("", "_sub"))
fao["avg_demand"] = fao["avg_demand"].fillna(fao["avg_demand_sub"])
fao["volatility_ratio"] = fao["volatility_ratio"].fillna(fao["volatility_ratio_sub"])
fao["peak_month"] = fao["peak_month"].fillna(fao["peak_month_sub"])
fao["avg_demand"] = fao["avg_demand"].fillna(200.0)
fao["volatility_ratio"] = fao["volatility_ratio"].fillna(0.6)
fao["peak_month"] = fao["peak_month"].fillna(6).astype(int)
fao["avg_demand_log"] = np.log1p(fao["avg_demand"])

# ============================================================
# STAGE ENCODING (categorical feature - used as a PREDICTOR only,
# not used to build the label)
# ============================================================

STAGES_SORTED = sorted(fao["food_supply_stage"].unique().tolist())
fao["stage_code"] = fao["food_supply_stage"].apply(lambda s: STAGES_SORTED.index(s))

# ============================================================
# LABEL: built ONLY from the external shelf-life table + light noise.
# Independent of loss_percentage, volatility_ratio, and stage.
# ============================================================

fao["shelf_life_days"] = fao["commodity"].apply(shelf_life_for)

rng = np.random.default_rng(42)
# Convert shelf life to a 0-100 "fragility" score (shorter life = higher score)
# then add a modest amount of noise so this isn't a perfectly clean cutoff.
fragility = 100 * (1 - np.log1p(fao["shelf_life_days"]) / np.log1p(365))
fragility = fragility.clip(0, 100)
noisy_fragility = (fragility + rng.normal(0, 12.0, len(fao))).clip(0, 100)

fao["spoilage_risk_score"] = noisy_fragility.round(2)
fao["high_spoilage_risk"] = (fao["spoilage_risk_score"] >= 55.0).astype(int)

# small measurement-noise flip, to avoid an unrealistically perfect boundary
flip = rng.random(len(fao)) < 0.05
fao.loc[flip, "high_spoilage_risk"] = 1 - fao.loc[flip, "high_spoilage_risk"]

# perishable flag kept for reference/app display only (NOT a model feature,
# since it's derived from the same commodity identity as the label)
PERISHABLE_KEYS = [k for k, d in SHELF_LIFE_DAYS if d <= 14]
fao["perishable"] = fao["commodity"].str.lower().apply(
    lambda c: 1 if any(k in c for k in PERISHABLE_KEYS) else 0
)

# ============================================================
# SAVE
# ============================================================

keep = ["commodity", "country", "food_supply_stage", "stage_code", "year",
        "loss_percentage", "avg_demand", "avg_demand_log", "volatility_ratio",
        "peak_month", "shelf_life_days", "perishable",
        "spoilage_risk_score", "high_spoilage_risk"]
out = fao[keep].copy()
out.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(out):,} rows -> {OUTPUT_PATH}")
print(f"Perishable share (reference only): {out['perishable'].mean()*100:.1f}%")
print(f"Positive (high risk) share: {out['high_spoilage_risk'].mean()*100:.1f}%")
print("\nMODEL-TRAINING FEATURES should be:")
print("  loss_percentage, avg_demand_log, volatility_ratio, peak_month, stage_code, year")
print("NOT: shelf_life_days, perishable, commodity, country")
print("  (those either leak into the label or would make the model just")
print("   memorize a per-commodity lookup instead of learning a pattern)")