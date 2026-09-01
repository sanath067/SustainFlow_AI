import pandas as pd
import numpy as np
import joblib
import shap
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*60)
print("SUSTAINFLOW AI - DEMAND PREDICTION")
print("="*60)

# Load model and data
print("\nLoading model and data...")
model = joblib.load('../Results/demand_model.joblib')

# Get exact features from model
trained_features = model.feature_name_
print(f"Model expects {len(trained_features)} features")

# Get categorical features from the model
categorical_features = []
if hasattr(model, 'categorical_feature_'):
    categorical_features = model.categorical_feature_
    print(f"Categorical features from model: {categorical_features}")
elif hasattr(model, 'booster_'):
    try:
        categorical_features = model.booster_.categorical_feature
        print(f"Categorical features from booster: {categorical_features}")
    except:
        print("No categorical features found in booster")

stores = pd.read_csv('../Datasets/stores.csv')
items = pd.read_csv('../Datasets/items.csv')
train = pd.read_csv('../Datasets/train.csv', nrows=200000, parse_dates=['date'])

# Prepare data
train['onpromotion'] = train['onpromotion'].fillna(0).astype(int)
df = train.merge(stores, on='store_nbr', how='left')
df = df.merge(items, on='item_nbr', how='left')

# Create ALL features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['weekend'] = (df['dayofweek'] >= 5).astype(int)
df['dayofyear'] = df['date'].dt.dayofyear

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

df = df.sort_values(['store_nbr', 'item_nbr', 'date'])

# ALL lag features
for lag in [1, 7, 14, 28]:
    df[f'lag_{lag}'] = df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(lag).fillna(0)

for lag in [1, 7, 14]:
    df[f'onpromotion_lag_{lag}'] = df.groupby(['store_nbr', 'item_nbr'])['onpromotion'].shift(lag).fillna(0)

# ALL rolling features
for window in [7, 14, 30]:
    df[f'rolling_mean_{window}'] = df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].rolling(window).mean().reset_index(level=[0,1], drop=True).fillna(0)
    df[f'rolling_std_{window}'] = df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].rolling(window).std().reset_index(level=[0,1], drop=True).fillna(0)

df['store_item_avg'] = df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform('mean').fillna(0)
df['item_popularity'] = df.groupby('item_nbr')['unit_sales'].transform('mean').fillna(0)

# Display stores
print("\n" + "="*60)
print("SELECT STORE")
print("="*60)
print(f"{'No.':<6} {'Store ID':<10} {'City':<20} {'State':<20}")
print("-"*60)
store_list = []
for idx, (_, row) in enumerate(stores.iterrows(), 1):
    print(f"{idx:<6} {row['store_nbr']:<10} {row['city']:<20} {row['state']:<20}")
    store_list.append(row['store_nbr'])

while True:
    try:
        store_choice = int(input("\nSelect Store (1-54): "))
        if 1 <= store_choice <= len(store_list):
            store_nbr = store_list[store_choice - 1]
            store_info = stores[stores['store_nbr'] == store_nbr].iloc[0]
            print(f"\nSelected: Store {store_nbr} - {store_info['city']}, {store_info['state']}")
            break
        else:
            print(f"Enter number between 1 and {len(store_list)}")
    except ValueError:
        print("Enter a valid number.")

# Display items
print("\n" + "="*60)
print("SELECT ITEM")
print("="*60)
print(f"{'No.':<6} {'Item ID':<10} {'Family':<25} {'Perishable':<10}")
print("-"*60)
item_list = []
for idx, (_, row) in enumerate(items.head(50).iterrows(), 1):
    perishable = "Yes" if row['perishable'] == 1 else "No"
    print(f"{idx:<6} {row['item_nbr']:<10} {row['family']:<25} {perishable:<10}")
    item_list.append(row['item_nbr'])

print(f"\n... and {len(items) - 50} more items")

while True:
    try:
        item_choice = int(input("\nSelect Item (1-50): "))
        if 1 <= item_choice <= len(item_list):
            item_nbr = item_list[item_choice - 1]
            item_info = items[items['item_nbr'] == item_nbr].iloc[0]
            perishable_text = "Yes (spoils quickly)" if item_info['perishable'] == 1 else "No (long shelf life)"
            print(f"\nSelected: Item {item_nbr} - {item_info['family']}")
            print(f"          Perishable: {perishable_text}")
            break
        else:
            print(f"Enter number between 1 and {len(item_list)}")
    except ValueError:
        print("Enter a valid number.")

# Get date with validation
print("\n" + "="*60)
print("ENTER DATE")
print("="*60)

while True:
    try:
        year = int(input("Year (2013-2017): "))
        if 2013 <= year <= 2017:
            break
        else:
            print("Enter year between 2013-2017.")
    except ValueError:
        print("Enter a valid year.")

while True:
    try:
        month = int(input("Month (1-12): "))
        if 1 <= month <= 12:
            break
        else:
            print("Enter month between 1-12.")
    except ValueError:
        print("Enter a valid month.")

while True:
    try:
        day = int(input("Day (1-31): "))
        if 1 <= day <= 31:
            break
        else:
            print("Enter day between 1-31.")
    except ValueError:
        print("Enter a valid day.")

# Validate date
try:
    pd.Timestamp(f'{year}-{month:02d}-{day:02d}')
except ValueError:
    print(f"\nInvalid date: {year}-{month:02d}-{day:02d}")
    print("Please enter a valid date.")
    exit()

# Build features
historical = df[(df['store_nbr'] == store_nbr) & (df['item_nbr'] == item_nbr)].copy()
historical = historical.sort_values('date')

if len(historical) > 0:
    recent = historical.iloc[-1]
    input_dict = {
        'store_nbr': store_nbr,
        'item_nbr': item_nbr,
        'year': year,
        'month': month,
        'day': day,
        'dayofweek': pd.Timestamp(f'{year}-{month}-{day}').dayofweek,
        'quarter': pd.Timestamp(f'{year}-{month}-{day}').quarter,
        'weekend': 1 if pd.Timestamp(f'{year}-{month}-{day}').dayofweek >= 5 else 0,
        'dayofyear': pd.Timestamp(f'{year}-{month}-{day}').dayofyear,
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'dayofweek_sin': np.sin(2 * np.pi * pd.Timestamp(f'{year}-{month}-{day}').dayofweek / 7),
        'dayofweek_cos': np.cos(2 * np.pi * pd.Timestamp(f'{year}-{month}-{day}').dayofweek / 7),
        'onpromotion': 0
    }
    
    for lag in [1, 7, 14, 28]:
        input_dict[f'lag_{lag}'] = recent.get(f'lag_{lag}', 0)
    
    for lag in [1, 7, 14]:
        input_dict[f'onpromotion_lag_{lag}'] = recent.get(f'onpromotion_lag_{lag}', 0)
    
    for window in [7, 14, 30]:
        input_dict[f'rolling_mean_{window}'] = recent.get(f'rolling_mean_{window}', historical['unit_sales'].mean())
        input_dict[f'rolling_std_{window}'] = recent.get(f'rolling_std_{window}', 0)
    
    input_dict['store_item_avg'] = recent.get('store_item_avg', historical['unit_sales'].mean())
    input_dict['item_popularity'] = recent.get('item_popularity', historical['unit_sales'].mean())
    
    input_data = pd.DataFrame([input_dict])
else:
    store_avg = df[df['store_nbr'] == store_nbr]['unit_sales'].mean() if len(df[df['store_nbr'] == store_nbr]) > 0 else 10
    item_avg = df[df['item_nbr'] == item_nbr]['unit_sales'].mean() if len(df[df['item_nbr'] == item_nbr]) > 0 else 10
    
    input_dict = {
        'store_nbr': store_nbr,
        'item_nbr': item_nbr,
        'year': year,
        'month': month,
        'day': day,
        'dayofweek': pd.Timestamp(f'{year}-{month}-{day}').dayofweek,
        'quarter': pd.Timestamp(f'{year}-{month}-{day}').quarter,
        'weekend': 1 if pd.Timestamp(f'{year}-{month}-{day}').dayofweek >= 5 else 0,
        'dayofyear': pd.Timestamp(f'{year}-{month}-{day}').dayofyear,
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'dayofweek_sin': np.sin(2 * np.pi * pd.Timestamp(f'{year}-{month}-{day}').dayofweek / 7),
        'dayofweek_cos': np.cos(2 * np.pi * pd.Timestamp(f'{year}-{month}-{day}').dayofweek / 7),
        'onpromotion': 0,
        'lag_1': 0,
        'lag_7': 0,
        'lag_14': 0,
        'lag_28': 0,
        'onpromotion_lag_1': 0,
        'onpromotion_lag_7': 0,
        'onpromotion_lag_14': 0,
        'rolling_mean_7': item_avg,
        'rolling_mean_14': item_avg,
        'rolling_mean_30': item_avg,
        'rolling_std_7': item_avg * 0.3,
        'rolling_std_14': item_avg * 0.3,
        'rolling_std_30': item_avg * 0.3,
        'store_item_avg': (store_avg + item_avg) / 2,
        'item_popularity': item_avg
    }
    input_data = pd.DataFrame([input_dict])

# IMPORTANT: Convert categorical features to match training
# Determine which features should be categorical
cat_features = []
if categorical_features:
    cat_features = categorical_features
else:
    # Fallback: treat store_nbr and item_nbr as categorical
    cat_features = ['store_nbr', 'item_nbr']

# Convert categorical features to pandas categorical dtype
for col in cat_features:
    if col in input_data.columns:
        input_data[col] = input_data[col].astype('category')
        print(f"Converted {col} to categorical")

# Align features with trained model
input_data = input_data[trained_features]

print(f"\nInput data has {len(input_data.columns)} features matching model")

# Convert input data to proper format for LightGBM
try:
    # Convert to numpy array
    X_pred = input_data.values
    
    # Check if any categorical columns need special handling
    if cat_features:
        # Get indices of categorical features
        cat_indices = [list(input_data.columns).index(col) for col in cat_features if col in input_data.columns]
        print(f"Categorical feature indices: {cat_indices}")
    
except Exception as e:
    print(f"Data conversion error: {e}")

# Prediction
try:
    # Try using the original data
    prediction = model.predict(input_data)[0]
    predicted_units = max(0, round(prediction, 2))
except Exception as e:
    print(f"Prediction failed: {e}")
    predicted_units = 0

# SHAP Explanation
try:
    import shap
    import matplotlib.pyplot as plt
    
    print("\nCreating SHAP explanation...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    shap_importance = pd.DataFrame({
        'Feature': input_data.columns,
        'SHAP_Value': shap_values[0]
    }).sort_values('SHAP_Value', key=abs, ascending=False)
    
    top_features = shap_importance.head(4)
    
    plt.figure(figsize=(14, 10))
    
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[0]
    
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=input_data.iloc[0].values,
            feature_names=input_data.columns.tolist()
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig('../Results/demand_prediction_explanation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nExplanation saved to Results/demand_prediction_explanation.png")
    
except Exception as e:
    print(f"SHAP explanation failed: {e}")
    top_features = pd.DataFrame({'Feature': [], 'SHAP_Value': []})

# Determine demand level
if predicted_units < 1:
    demand_level = "Very Low"
elif predicted_units < 5:
    demand_level = "Low"
elif predicted_units < 20:
    demand_level = "Moderate"
elif predicted_units < 50:
    demand_level = "High"
else:
    demand_level = "Very High"

# OUTPUT
print("\n" + "="*60)
print("PREDICTION RESULT")
print("="*60)
print(f"\nPredicted Unit Sales: {predicted_units:.2f}")
print(f"Demand Level: {demand_level}")

print(f"\nStore: {store_nbr} - {store_info['city']}")
print(f"Item: {item_nbr} - {item_info['family']}")
print(f"Date: {year}-{month:02d}-{day:02d}")

print("\n" + "="*60)
print("SHAP EXPLANATION - TOP 4 FACTORS")
print("="*60)
print("\nFactors influencing this prediction:")
print("-"*60)

if len(top_features) > 0:
    for _, row in top_features.iterrows():
        impact = "Increases" if row['SHAP_Value'] > 0 else "Decreases"
        print(f"{row['Feature']:<20} : {row['SHAP_Value']:>8.4f}  {impact} demand")
else:
    print("No SHAP values available")

print("-"*60)

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

if predicted_units > 0:
    perishable = item_info['perishable']
    if perishable == 1:
        rec = f"Order {int(predicted_units * 0.8)} units (80% of prediction)"
        print(f"\nPerishable item - {rec}")
        print("Reason: Order slightly less to avoid spoilage waste")
    else:
        rec = f"Order {int(predicted_units)} units"
        print(f"\nNon-perishable item - {rec}")
        print("Reason: Can order safely up to predicted value")
else:
    rec = "No recommendation"

# SAVE TO EXCEL
print("\n" + "="*60)
print("SAVING TO EXCEL")
print("="*60)

try:
    excel_file = '../Results/demand_predictions_log.xlsx'
    
    prediction_record = pd.DataFrame([{
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Store': store_nbr,
        'Store_City': store_info['city'],
        'Store_State': store_info['state'],
        'Item': item_nbr,
        'Item_Family': item_info['family'],
        'Perishable': 'Yes' if item_info['perishable'] == 1 else 'No',
        'Prediction_Date': f'{year}-{month:02d}-{day:02d}',
        'Predicted_Units': predicted_units,
        'Demand_Level': demand_level,
        'SHAP_Top1': top_features.iloc[0]['Feature'] if len(top_features) > 0 else '',
        'SHAP_Top1_Value': top_features.iloc[0]['SHAP_Value'] if len(top_features) > 0 else 0,
        'SHAP_Top2': top_features.iloc[1]['Feature'] if len(top_features) > 1 else '',
        'SHAP_Top2_Value': top_features.iloc[1]['SHAP_Value'] if len(top_features) > 1 else 0,
        'SHAP_Top3': top_features.iloc[2]['Feature'] if len(top_features) > 2 else '',
        'SHAP_Top3_Value': top_features.iloc[2]['SHAP_Value'] if len(top_features) > 2 else 0,
        'SHAP_Top4': top_features.iloc[3]['Feature'] if len(top_features) > 3 else '',
        'SHAP_Top4_Value': top_features.iloc[3]['SHAP_Value'] if len(top_features) > 3 else 0,
        'Recommendation': rec
    }])
    
    if os.path.exists(excel_file):
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            existing = pd.read_excel(excel_file, sheet_name='Predictions')
            combined = pd.concat([existing, prediction_record], ignore_index=True)
            combined.to_excel(writer, sheet_name='Predictions', index=False)
    else:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            prediction_record.to_excel(writer, sheet_name='Predictions', index=False)
    
    print(f"\nPrediction saved to: {excel_file}")
    print(f"Sheet: Predictions")
    
except Exception as e:
    print(f"\nCould not save to Excel: {e}")

print("\n" + "="*60)
print("PREDICTION COMPLETE")
print("="*60)