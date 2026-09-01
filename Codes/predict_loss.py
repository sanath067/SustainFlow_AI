import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SUSTAINFLOW AI - FOOD LOSS PREDICTION WITH SHAP REPORT")
print("="*60)

# Load model and data
print("\nLoading model and data...")
model = joblib.load('Results/fao_model.joblib')
df = pd.read_csv('Datasets/Fao_merged_clean.csv')

# ============================================
# SELECT COMMODITY
# ============================================
print("\n" + "="*60)
print("SELECT COMMODITY")
print("="*60)

commodities = sorted(df['commodity'].unique())
for i, c in enumerate(commodities, 1):
    print(f"{i}. {c}")

while True:
    try:
        choice = int(input("\nEnter your choice (1-{}): ".format(len(commodities))))
        if 1 <= choice <= len(commodities):
            commodity = commodities[choice - 1]
            print(f"\nSelected: {commodity}")
            break
        else:
            print(f"Please enter a number between 1 and {len(commodities)}")
    except ValueError:
        print("Please enter a valid number.")

# ============================================
# SELECT COUNTRY
# ============================================
print("\n" + "="*60)
print("SELECT COUNTRY")
print("="*60)

countries = sorted(df['country'].unique())
for i, c in enumerate(countries, 1):
    print(f"{i}. {c}")

while True:
    try:
        choice = int(input("\nEnter your choice (1-{}): ".format(len(countries))))
        if 1 <= choice <= len(countries):
            country = countries[choice - 1]
            print(f"\nSelected: {country}")
            break
        else:
            print(f"Please enter a number between 1 and {len(countries)}")
    except ValueError:
        print("Please enter a valid number.")

# ============================================
# SELECT FOOD SUPPLY STAGE
# ============================================
print("\n" + "="*60)
print("SELECT FOOD SUPPLY STAGE")
print("="*60)

stages = sorted(df['food_supply_stage'].unique())
for i, s in enumerate(stages, 1):
    print(f"{i}. {s}")

while True:
    try:
        choice = int(input("\nEnter your choice (1-{}): ".format(len(stages))))
        if 1 <= choice <= len(stages):
            stage = stages[choice - 1]
            print(f"\nSelected: {stage}")
            break
        else:
            print(f"Please enter a number between 1 and {len(stages)}")
    except ValueError:
        print("Please enter a valid number.")

# ============================================
# SELECT YEAR
# ============================================
print("\n" + "="*60)
print("SELECT YEAR")
print("="*60)

years = sorted(df['year'].unique())
for i, y in enumerate(years, 1):
    print(f"{i}. {y}")

while True:
    try:
        choice = int(input("\nEnter your choice (1-{}): ".format(len(years))))
        if 1 <= choice <= len(years):
            year = years[choice - 1]
            print(f"\nSelected: {year}")
            break
        else:
            print(f"Please enter a number between 1 and {len(years)}")
    except ValueError:
        print("Please enter a valid number.")

# ============================================
# PREDICTION
# ============================================
print("\n" + "="*60)
print("PREDICTION RESULT")
print("="*60)

input_data = pd.DataFrame([{
    'commodity': commodity,
    'country': country,
    'food_supply_stage': stage,
    'year': year
}])

try:
    prediction = model.predict(input_data)[0]
    print(f"\nFood item          : {commodity}")
    print(f"Country            : {country}")
    print(f"Food supply stage  : {stage}")
    print(f"Year               : {year}")
    print(f"\nPredicted food loss: {prediction:.2f}%")
except Exception as e:
    print(f"Prediction failed: {e}")
    exit()

exists = len(df[(df['commodity'] == commodity) & 
                (df['country'] == country) & 
                (df['food_supply_stage'] == stage) & 
                (df['year'] == year)]) > 0

if exists:
    actual = df[(df['commodity'] == commodity) & 
                (df['country'] == country) & 
                (df['food_supply_stage'] == stage) & 
                (df['year'] == year)]['loss_percentage'].values[0]
    print(f"\nData status        : Found in dataset")
    print(f"Actual food loss   : {actual:.2f}%")
    print(f"Prediction error   : {abs(prediction - actual):.2f}%")
else:
    print(f"\nData status        : Combination not directly found in FAO dataset")
    print(f"Actual food loss   : Not available")
    print(f"Prediction error   : Not available")
    print(f"\nThis exact combination is not present in the dataset.")
    print(f"The displayed loss percentage is a model-based estimate.")

# ============================================
# SHAP EXPLANATION
# ============================================
print("\n" + "="*60)
print("EXPLAINABLE AI ANALYSIS")
print("="*60)

print(f"\nAnalyzing why the model produced a prediction of {prediction:.2f}%...")

try:
    # Get preprocessed data
    preprocessed_data = model.named_steps['preprocess'].transform(input_data)
    
    if hasattr(preprocessed_data, 'toarray'):
        preprocessed_data = preprocessed_data.toarray()
    
    feature_names = model.named_steps['preprocess'].get_feature_names_out()
    
    explainer = shap.TreeExplainer(model.named_steps['model'])
    shap_values = explainer.shap_values(preprocessed_data, check_additivity=False)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, -1)
    
    shap_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Value': shap_values[0]
    }).sort_values('SHAP_Value', key=abs, ascending=False)
    
    abs_shap = np.abs(shap_values[0])
    total_impact = abs_shap.sum()
    
    if total_impact > 0:
        shap_importance['Contribution_%'] = (abs_shap / total_impact * 100).round(2)
        shap_success = True
    else:
        shap_success = False
        importances = model.named_steps['model'].feature_importances_
        shap_importance['Contribution_%'] = (importances / importances.sum() * 100).round(2)
        shap_success = True
    
    if shap_success:
        print("\nThe following factors influenced this prediction:\n")
        
        for i, (idx, row) in enumerate(shap_importance.head(4).iterrows(), 1):
            effect = "Increased" if row['SHAP_Value'] > 0 else "Decreased"
            contrib = row['Contribution_%']
            
            if contrib >= 30:
                level = "HIGH"
            elif contrib >= 15:
                level = "MEDIUM"
            else:
                level = "LOW"
            
            feature_name = row['Feature']
            if '_' in feature_name:
                parts = feature_name.split('_')
                display = parts[0].capitalize()
                value = '_'.join(parts[1:])
            else:
                display = feature_name
                value = ''
            
            print(f"{i}. {display}")
            print(f"   Selected value : {value if value else 'N/A'}")
            print(f"   Effect         : {effect} predicted food loss")
            print(f"   Influence      : {contrib:.2f}%")
            print(f"   Influence level: {level}")
            print()
        
        top_factor = shap_importance.iloc[0]
        top_factor_name = top_factor['Feature']
        top_factor_value = top_factor['SHAP_Value']
        top_factor_contribution = top_factor['Contribution_%']
        top_factor_effect = "Increased" if top_factor_value > 0 else "Decreased"
        
        print("-"*60)
        print("EXPLANATION SUMMARY")
        print("-"*60)
        
        print(f"\nPredicted food loss: {prediction:.2f}%")
        print(f"\nThe strongest influencing factor was:\n")
        print(f"{top_factor_name}")
        print(f"Effect         : {top_factor_effect} predicted food loss")
        print(f"Influence      : {top_factor_contribution:.2f}%")
        print(f"Influence level: {'HIGH' if top_factor_contribution >= 30 else 'MEDIUM' if top_factor_contribution >= 15 else 'LOW'}")
        
        # ============================================
        # GENERATE SHAP REPORT (PDF)
        # ============================================
        
        print("\n" + "="*60)
        print("GENERATING SHAP REPORT")
        print("="*60)
        
        report_path = 'Results/shap_report.pdf'
        
        try:
            with PdfPages(report_path) as pdf:
                
                # Page 1: Summary + Prediction + Top Factor
                fig1, ax1 = plt.subplots(figsize=(11, 8.5))
                ax1.axis('off')
                
                ax1.text(0.5, 0.95, 'SHAP EXPLANATION REPORT', fontsize=20, ha='center', weight='bold')
                ax1.text(0.5, 0.90, 'Food Loss Prediction - Explainable AI Report', fontsize=14, ha='center', style='italic')
                
                ax1.text(0.1, 0.80, 'PREDICTION RESULT', fontsize=14, weight='bold')
                ax1.text(0.1, 0.74, f'Predicted Food Loss: {prediction:.2f}%', fontsize=16, weight='bold', color='blue')
                
                ax1.text(0.1, 0.65, 'INPUT DETAILS', fontsize=14, weight='bold')
                ax1.text(0.1, 0.59, f'Commodity : {commodity}', fontsize=12)
                ax1.text(0.1, 0.55, f'Country   : {country}', fontsize=12)
                ax1.text(0.1, 0.51, f'Stage     : {stage}', fontsize=12)
                ax1.text(0.1, 0.47, f'Year      : {year}', fontsize=12)
                
                ax1.text(0.55, 0.80, 'TOP INFLUENCING FACTOR', fontsize=14, weight='bold')
                ax1.text(0.55, 0.74, f'Feature    : {top_factor_name}', fontsize=14, weight='bold')
                ax1.text(0.55, 0.68, f'Effect     : {top_factor_effect} prediction', fontsize=14)
                ax1.text(0.55, 0.62, f'Influence  : {top_factor_contribution:.1f}%', fontsize=14, color='red')
                ax1.text(0.55, 0.56, f'SHAP Value : {top_factor_value:.4f}', fontsize=12)
                
                if top_factor_value > 0:
                    ax1.text(0.55, 0.48, 'This feature INCREASED the prediction', fontsize=12, color='green')
                else:
                    ax1.text(0.55, 0.48, 'This feature DECREASED the prediction', fontsize=12, color='red')
                
                ax1.text(0.1, 0.35, 'SUMMARY', fontsize=14, weight='bold')
                ax1.text(0.1, 0.29, f'The model predicted {prediction:.2f}% food loss for {commodity} in {country}.', fontsize=11)
                ax1.text(0.1, 0.25, f'The strongest influencing factor was "{top_factor_name}" which {top_factor_effect.lower()} the prediction by {top_factor_contribution:.1f}%.', fontsize=11)
                
                ax1.text(0.5, 0.05, 'SustainFlow AI - Generated Report', fontsize=10, ha='center', style='italic')
                
                plt.tight_layout()
                pdf.savefig(fig1, dpi=300)
                plt.close()
                
                # Page 2: Bar Chart
                fig2, ax2 = plt.subplots(figsize=(11, 8.5))
                
                top_10 = shap_importance.head(10)
                colors = ['#FF6B6B' if x < 0 else '#51CF66' for x in top_10['SHAP_Value']]
                bars = ax2.barh(top_10['Feature'], top_10['SHAP_Value'], color=colors)
                
                ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax2.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
                ax2.set_ylabel('Features', fontsize=12)
                ax2.set_title('Top 10 Features Affecting the Prediction', fontsize=14, weight='bold')
                
                for bar, value in zip(bars, top_10['SHAP_Value']):
                    ax2.text(value + (0.1 if value > 0 else -0.1), 
                             bar.get_y() + bar.get_height()/2, 
                             f'{value:.2f}', 
                             ha='left' if value > 0 else 'right', 
                             va='center', 
                             fontsize=10)
                
                for i, (idx, row) in enumerate(top_10.iterrows()):
                    ax2.text(0.02, i, f"{row['Contribution_%']:.1f}%", 
                             va='center', fontsize=9, color='black', weight='bold')
                
                ax2.text(0.02, -0.8, 'Note: Green = Increases prediction, Red = Decreases prediction', 
                         fontsize=10, style='italic', color='gray')
                
                plt.tight_layout()
                pdf.savefig(fig2, dpi=300)
                plt.close()
                
                # Page 3: Pie Chart
                fig3, ax3 = plt.subplots(figsize=(11, 8.5))
                
                top5 = shap_importance.head(5)
                others_value = total_impact - top5['SHAP_Value'].abs().sum()
                
                pie_labels = list(top5['Feature'])
                pie_values = list(top5['SHAP_Value'].abs())
                if others_value > 0:
                    pie_labels.append('Other Features')
                    pie_values.append(others_value)
                
                colors_pie = plt.cm.Set3(np.linspace(0, 1, len(pie_labels)))
                explode = [0.05] * len(pie_labels)
                
                wedges, texts, autotexts = ax3.pie(pie_values, 
                                                    labels=pie_labels, 
                                                    autopct='%1.1f%%', 
                                                    startangle=90,
                                                    colors=colors_pie,
                                                    explode=explode)
                
                for autotext in autotexts:
                    autotext.set_fontsize(11)
                    autotext.set_weight('bold')
                
                ax3.set_title('Top 5 Features Contribution to Prediction', fontsize=14, weight='bold')
                ax3.axis('equal')
                ax3.text(0.5, -0.08, 'Based on absolute SHAP values', fontsize=10, ha='center', style='italic', color='gray')
                
                plt.tight_layout()
                pdf.savefig(fig3, dpi=300)
                plt.close()
                
                # Page 4: Full Feature Table
                fig4, ax4 = plt.subplots(figsize=(11, 8.5))
                ax4.axis('off')
                
                ax4.text(0.5, 0.95, 'COMPLETE FEATURE IMPACT TABLE', fontsize=16, ha='center', weight='bold')
                ax4.text(0.5, 0.90, 'All features with their SHAP values and contribution percentages', fontsize=12, ha='center', style='italic')
                
                table_data = []
                table_data.append(['Rank', 'Feature', 'SHAP Value', 'Contribution %', 'Effect'])
                
                for i, (idx, row) in enumerate(shap_importance.iterrows(), 1):
                    effect = 'Increases' if row['SHAP_Value'] > 0 else 'Decreases'
                    table_data.append([
                        str(i),
                        row['Feature'],
                        f"{row['SHAP_Value']:.4f}",
                        f"{row['Contribution_%']:.1f}%",
                        effect
                    ])
                
                table = ax4.table(cellText=table_data, 
                                   loc='center', 
                                   cellLoc='center',
                                   colWidths=[0.08, 0.45, 0.15, 0.17, 0.15])
                
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                
                for i in range(5):
                    table[(0, i)].set_facecolor('#4472C4')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                for i in range(1, len(table_data)):
                    if 'Decreases' in table_data[i][4]:
                        for j in range(5):
                            table[(i, j)].set_facecolor('#FFE6E6')
                    else:
                        for j in range(5):
                            table[(i, j)].set_facecolor('#E6FFE6')
                
                plt.tight_layout()
                pdf.savefig(fig4, dpi=300)
                plt.close()
                
                # Page 5: Waterfall Plot - SKIPPED to avoid errors
                print("   - Page 5: Waterfall Plot (skipped to avoid error)")
            
            print(f"\nSHAP Report generated: {report_path}")
            print("   Report contains:")
            print("   - Page 1: Prediction Summary + Top Factor")
            print("   - Page 2: Bar Chart (Top 10 Features)")
            print("   - Page 3: Pie Chart (Top 5 Contribution)")
            print("   - Page 4: Complete Feature Impact Table")
            
        except Exception as e:
            print(f"PDF generation failed: {e}")
    else:
        print("Could not generate SHAP explanation.")
    
except Exception as e:
    print(f"SHAP analysis failed: {e}")
    print("Prediction completed successfully, but SHAP explanation could not be generated.")

# ============================================
# SAVE TO EXCEL
# ============================================

print("\n" + "="*60)
print("SAVING TO EXCEL")
print("="*60)

excel_path = 'Results/final_results.xlsx'

try:
    from openpyxl import load_workbook
    import os
    
    if os.path.exists(excel_path):
        wb = load_workbook(excel_path)
        if 'Predictions' not in wb.sheetnames:
            wb.create_sheet('Predictions')
        ws = wb['Predictions']
        row_num = ws.max_row + 1
        if row_num == 2:
            headers = ['Commodity', 'Country', 'Food Supply Stage', 'Year', 'Predicted Loss %']
            for col, header in enumerate(headers, 1):
                ws.cell(row=1, column=col, value=header)
            row_num = 2
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = 'Predictions'
        headers = ['Commodity', 'Country', 'Food Supply Stage', 'Year', 'Predicted Loss %']
        for col, header in enumerate(headers, 1):
            ws.cell(row=1, column=col, value=header)
        row_num = 2
    
    ws.cell(row=row_num, column=1, value=commodity)
    ws.cell(row=row_num, column=2, value=country)
    ws.cell(row=row_num, column=3, value=stage)
    ws.cell(row=row_num, column=4, value=year)
    ws.cell(row=row_num, column=5, value=round(prediction, 2))
    
    wb.save(excel_path)
    print(f"Results saved to: {excel_path}")
    print("Previous predictions are preserved.")
    print("The new prediction has been added below the previous results.")
    
except Exception as e:
    print(f"Could not save to Excel: {e}")

print("\n" + "="*60)
print("PREDICTION AND EXPLAINABLE AI ANALYSIS COMPLETED SUCCESSFULLY")
print("="*60)