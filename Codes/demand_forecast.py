import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("MODULE 2: DEMAND FORECASTING WITH SHAP")
print("="*60)

os.makedirs('../Results', exist_ok=True)
print("Results folder created")

print("\nLoading datasets...")

try:
    stores = pd.read_csv('../Datasets/stores.csv')
    print(f"stores.csv loaded: {len(stores)} rows")
except FileNotFoundError:
    print("ERROR: stores.csv not found in Datasets folder")
    exit()

try:
    items = pd.read_csv('../Datasets/items.csv')
    print(f"items.csv loaded: {len(items)} rows")
except FileNotFoundError:
    print("ERROR: items.csv not found in Datasets folder")
    exit()

try:
    print("\nLoading train.csv (500,000 rows)...")
    train = pd.read_csv('../Datasets/train.csv', nrows=500000, parse_dates=['date'])
    print(f"train.csv loaded: {len(train):,} rows")
    print(f"Date range: {train['date'].min()} to {train['date'].max()}")
except MemoryError:
    print("Memory Error: Try reducing nrows further")
    exit()
except FileNotFoundError:
    print("ERROR: train.csv not found in Datasets folder")
    exit()

print("\n" + "="*60)
print("DATA EXPLORATION & QUALITY CHECK")
print("="*60)

print(f"Date range: {train['date'].min()} to {train['date'].max()}")
print(f"Unique dates: {train['date'].nunique()} days")

# Fix onpromotion - fill NaN with 0
train['onpromotion'] = train['onpromotion'].fillna(0).astype(int)
print(f"\nOnpromotion values:\n{train['onpromotion'].value_counts()}")

# Fix negative sales
negative_sales = (train['unit_sales'] < 0).sum()
if negative_sales > 0:
    print(f"Found {negative_sales:,} rows with negative sales - converting to 0")
    train.loc[train['unit_sales'] < 0, 'unit_sales'] = 0

# Remove duplicates
duplicates = train.duplicated(['date', 'store_nbr', 'item_nbr']).sum()
if duplicates > 0:
    print(f"Found {duplicates:,} duplicate entries - removing")
    train = train.drop_duplicates(['date', 'store_nbr', 'item_nbr'])

print(f"\nAfter cleaning: {len(train):,} rows")

print("\n" + "="*60)
print("MERGING DATASETS")
print("="*60)

df = train.merge(stores, on='store_nbr', how='left')
df = df.merge(items, on='item_nbr', how='left')
print(f"Merged dataset: {len(df):,} rows")

print("\n" + "="*60)
print("CREATING TIME FEATURES")
print("="*60)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['weekend'] = (df['dayofweek'] >= 5).astype(int)
df['dayofyear'] = df['date'].dt.dayofyear

# Seasonal encodings
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

print("Created time features (including seasonal encodings)")

print("\n" + "="*60)
print("CREATING LAG FEATURES")
print("="*60)

df = df.sort_values(['store_nbr', 'item_nbr', 'date'])

for lag in [1, 7, 14]:
    df[f'lag_{lag}'] = df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(lag).fillna(0)
    print(f"Created lag_{lag}")

for lag in [1, 7]:
    df[f'onpromotion_lag_{lag}'] = df.groupby(['store_nbr', 'item_nbr'])['onpromotion'].shift(lag).fillna(0)
    print(f"Created onpromotion_lag_{lag}")

print("\n" + "="*60)
print("CREATING ROLLING FEATURES")
print("="*60)

for window in [7, 14]:
    df[f'rolling_mean_{window}'] = df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].rolling(window).mean().reset_index(level=[0,1], drop=True).fillna(0)
    print(f"Created rolling_mean_{window}")
    
    df[f'rolling_std_{window}'] = df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].rolling(window).std().reset_index(level=[0,1], drop=True).fillna(0)
    print(f"Created rolling_std_{window}")

print("\n" + "="*60)
print("CREATING INTERACTION FEATURES")
print("="*60)

df['store_item_avg'] = df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform('mean').fillna(0)
print("Created store_item_avg")

df['item_popularity'] = df.groupby('item_nbr')['unit_sales'].transform('mean').fillna(0)
print("Created item_popularity")

print("\n" + "="*60)
print("PREPARING FEATURES FOR MODEL")
print("="*60)

categorical_cols = ['store_nbr', 'item_nbr', 'family', 'class', 'type', 'cluster']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

feature_cols = [
    'store_nbr', 'item_nbr',
    'year', 'month', 'day', 'dayofweek', 'quarter', 'weekend', 'dayofyear',
    'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos',
    'onpromotion',
    'lag_1', 'lag_7', 'lag_14',
    'onpromotion_lag_1', 'onpromotion_lag_7',
    'rolling_mean_7', 'rolling_mean_14',
    'rolling_std_7', 'rolling_std_14',
    'store_item_avg', 'item_popularity'
]

available_features = [col for col in feature_cols if col in df.columns]
print(f"Using {len(available_features)} features")

X = df[available_features]
y = df['unit_sales']

print("\n" + "="*60)
print("TRAIN TEST SPLIT (TEMPORAL)")
print("="*60)

split_date = df['date'].quantile(0.8)
train_mask = df['date'] < split_date
test_mask = df['date'] >= split_date

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f"Training set: {len(X_train):,} rows")
print(f"Testing set: {len(X_test):,} rows")
print(f"Split date: {split_date}")

print("\n" + "="*60)
print("TRAINING LIGHTGBM MODEL")
print("="*60)

from lightgbm import LGBMRegressor

model = LGBMRegressor(
    n_estimators=50,
    learning_rate=0.1,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(X_train, y_train)
print("Model training completed")

print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2: {r2:.4f}")

# ============ SAVE METRICS.TXT (Matching Module 1) ============
with open('../Results/demand_metrics.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("MODULE 2: DEMAND FORECASTING RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"R2: {r2:.4f}\n")
    f.write("\n" + "="*60 + "\n")
    f.write("Model: LightGBM\n")
    f.write(f"Training samples: {len(X_train):,}\n")
    f.write(f"Testing samples: {len(X_test):,}\n")
    f.write(f"Features used: {len(available_features)}\n")
print("Metrics saved to Results/demand_metrics.txt")

print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

importance = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=False)

# ============ SAVE FEATURE IMPORTANCE CSV (Matching Module 1) ============
importance_df.to_csv('../Results/demand_feature_importance.csv', index=False)
print("Feature importance saved to Results/demand_feature_importance.csv")

non_zero = importance_df[importance_df['Importance'] > 0]
print(f"\nFeatures with non-zero importance: {len(non_zero)} out of {len(importance_df)}")
print("\nTop 10 Features:")
print(non_zero.head(10))

print("\n" + "="*60)
print("ADDING SHAP EXPLANATIONS (Matching Module 1)")
print("="*60)

try:
    import shap
    import matplotlib.pyplot as plt
    
    print("\nCreating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    # Use smaller sample for SHAP to save memory
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    
    print(f"Calculating SHAP values for {sample_size} samples...")
    shap_values = explainer.shap_values(X_sample)
    
    # ============ SHAP FEATURE IMPORTANCE CSV ============
    shap_importance = pd.DataFrame({
        'Feature': X_sample.columns,
        'SHAP_Mean_Abs': np.abs(shap_values).mean(axis=0)
    }).sort_values('SHAP_Mean_Abs', ascending=False)
    shap_importance.to_csv('../Results/demand_shap_importance.csv', index=False)
    print("SHAP feature importance CSV saved to Results/demand_shap_importance.csv")
    
    # ============ SHAP GLOBAL IMPORTANCE PLOT (Matching Module 1) ============
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig('../Results/demand_shap_global_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP global importance plot saved to Results/demand_shap_global_importance.png")
    
    # ============ ACTUAL VS PREDICTED PLOT (Matching Module 1) ============
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=1)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Unit Sales')
    plt.ylabel('Predicted Unit Sales')
    plt.title('Actual vs Predicted Unit Sales')
    plt.savefig('../Results/demand_actual_vs_predicted.png', dpi=300)
    plt.close()
    print("Actual vs Predicted plot saved to Results/demand_actual_vs_predicted.png")
    
    # ============ FEATURE IMPORTANCE PLOT (Matching Module 1) ============
    plt.figure(figsize=(12, 8))
    top_features = non_zero.head(15)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance')
    plt.savefig('../Results/demand_feature_importance_plot.png', dpi=300)
    plt.close()
    print("Feature importance plot saved to Results/demand_feature_importance_plot.png")

except ImportError:
    print("SHAP not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'shap'])
    print("Please run the script again")
    exit()
except Exception as e:
    print(f"SHAP error: {e}")
    print("Continuing without SHAP...")

print("\n" + "="*60)
print("SAVING PREDICTIONS")
print("="*60)

predictions_df = pd.DataFrame({
    'date': df[test_mask]['date'].values,
    'store_nbr': df[test_mask]['store_nbr'].values,
    'item_nbr': df[test_mask]['item_nbr'].values,
    'actual_unit_sales': y_test.values,
    'predicted_unit_sales': y_pred
})
predictions_df.to_csv('../Results/demand_predictions.csv', index=False)
print("Predictions saved to Results/demand_predictions.csv")

import joblib
joblib.dump(model, '../Results/demand_model.joblib')
print("Model saved to Results/demand_model.joblib")

# ============ SAVE FINAL RESULTS EXCEL (Matching Module 1) ============
with pd.ExcelWriter('../Results/demand_final_results.xlsx') as writer:
    # Sheet 1: Metrics
    metrics_data = {
        'Metric': ['RMSE', 'MAE', 'R2'],
        'Value': [rmse, mae, r2]
    }
    pd.DataFrame(metrics_data).to_excel(writer, sheet_name='Metrics', index=False)
    
    # Sheet 2: Feature Importance
    importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)
    
    # Sheet 3: SHAP Importance
    shap_importance.to_excel(writer, sheet_name='SHAP Importance', index=False)
    
    # Sheet 4: Sample Predictions
    sample_predictions = predictions_df.head(1000)
    sample_predictions.to_excel(writer, sheet_name='Sample Predictions', index=False)
    
    print("Final results Excel saved to Results/demand_final_results.xlsx")

print("\n" + "="*60)
print("MODULE 2 COMPLETED SUCCESSFULLY")
print("="*60)

print("\n" + "="*60)
print("SUMMARY FOR DEMO")
print("="*60)
print(f"\nKey Results:")
print(f"  - RMSE: {rmse:.4f}")
print(f"  - MAE: {mae:.4f}")
print(f"  - R2: {r2:.4f}")
print(f"\nTop Features: {non_zero.head(3)['Feature'].tolist()}")
print(f"Active Features: {len(non_zero)} out of {len(importance_df)}")
print(f"SHAP Explanations: Added (matching Module 1)")

print("\nOutput files saved to Results/ folder:")
print("  - demand_metrics.txt")
print("  - demand_feature_importance.csv")
print("  - demand_feature_importance_plot.png")
print("  - demand_actual_vs_predicted.png")
print("  - demand_predictions.csv")
print("  - demand_model.joblib")
print("  - demand_shap_importance.csv")
print("  - demand_shap_global_importance.png")
print("  - demand_final_results.xlsx")