"""
Demonstrates SHAP vs. Built-in Feature Importance for a mock
American Airlines Customer Satisfaction Analysis scenario.

This script implements a simplified version of the concepts discussed in
feature_importance_shap_vs_builtin.md, showcasing:
- Training an XGBoost model on synthetic customer satisfaction data.
- Extracting built-in feature importances.
- Calculating and visualizing SHAP values for global and local explanations, and feature interactions.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. Generate Synthetic Customer Satisfaction Data ---
def generate_synthetic_data(num_samples=1000):
    """Generates synthetic data resembling airline customer satisfaction features."""
    np.random.seed(42) # for reproducibility
    data                           = pd.DataFrame()
    data['FlightDelayMinutes']     = np.random.lognormal(mean=3, sigma=0.8, size=num_samples).astype(int)
    data['FlightDelayMinutes']     = np.clip(data['FlightDelayMinutes'], 0, 300) # Cap delays
    data['IsEliteStatus']          = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])
    data['SeatComfort']            = np.random.randint(1, 6, size=num_samples) # 1-5 scale
    data['InflightEntertainment']  = np.random.randint(1, 6, size=num_samples) # 1-5 scale
    data['StaffAttitude']          = np.random.randint(1, 6, size=num_samples) # 1-5 scale
    data['ProactiveCommunication'] = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])

    # Create a synthetic target: CustomerSatisfaction (1=Satisfied, 0=Dissatisfied)
    # Satisfaction decreases with delays, increases with other positive factors
    satisfaction_score = (
        -0.01 * data['FlightDelayMinutes'] +
        1.0 * data['IsEliteStatus'] +
        0.5 * data['SeatComfort'] +
        0.4 * data['InflightEntertainment'] +
        0.6 * data['StaffAttitude'] +
        0.8 * data['ProactiveCommunication'] +
        np.random.normal(0, 1.5, num_samples) # Add some noise
    )
    data['CustomerSatisfaction'] = (satisfaction_score > np.median(satisfaction_score)).astype(int)
    return data

print("Generating synthetic customer satisfaction data...")
df = generate_synthetic_data()
print("Sample of generated data:")
print(df.head())

# --- 2. Data Preprocessing and Model Training ---
X = df.drop('CustomerSatisfaction', axis=1)
y = df['CustomerSatisfaction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining XGBoost model...")
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# --- 3. Built-in GBDT Feature Importance ---
print("\n--- Built-in GBDT Feature Importance ---")
print("Methodology: Typically measures how often a feature is used in trees (e.g., 'weight') or its average gain ('gain').")
print("Output: Global importance scores only.")

# XGBoost offers different importance types: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
# 'gain' is often a good default as it reflects the improvement in accuracy brought by a feature.
built_in_importance = model.get_booster().get_score(importance_type='gain')
if not built_in_importance:
    # Fallback if 'gain' is not available or model is simple (e.g., single tree)
    built_in_importance_values = model.feature_importances_
    built_in_importance = {X_train.columns[i]: built_in_importance_values[i] for i in range(len(X_train.columns))}

# Sort features by importance
sorted_importance = sorted(built_in_importance.items(), key=lambda x: x[1], reverse=True)

print("\nTop built-in feature importances (by gain):")
for feature, score in sorted_importance:
    print(f"- {feature}: {score:.4f}")

plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='gain', max_num_features=10, title='Built-in Feature Importance (Gain)')
plt.tight_layout()
plt.savefig('builtin_feature_importance.png')
print("Saved built-in feature importance plot to 'builtin_feature_importance.png'")
plt.show()

print("\nAdvantages for AA (Built-in):")
print("- Computational efficiency: Faster for initial exploration, especially with large datasets.")
print("- Implementation simplicity: Readily available in GBDT packages.")

# --- 4. SHAP (SHapley Additive exPlanations) Values ---
print("\n--- SHAP Values ---")
print("Methodology: Based on cooperative game theory, fairly distributes prediction credit.")
print("Output: Local (per-prediction) and global (aggregated) importance.")

# SHAP works well with tree-based models like XGBoost
explainer = shap.Explainer(model, X_train) # Using X_train as background data for TreeExplainer
shap_values = explainer(X_test)

# Correctly access SHAP values for a binary classification model
# For XGBClassifier, explainer(X_test) returns an Explanation object.
# shap_values.values will be an array where for binary classification,
# if it's (num_samples, num_features, num_classes), we usually use the values for the positive class.
# If it's (num_samples, num_features), it's often already for the positive class.
# Let's inspect the shape

# For binary classification with XGBoost, shap_values often has shape (n_samples, n_features)
# representing SHAP values for the positive class. If it has 2 classes, use shap_values_for_class_1

if len(shap_values.values.shape) == 3: # (samples, features, classes)
    shap_values_for_class_1 = shap_values.values[:, :, 1] # Assuming class 1 is 'Satisfied'
    expected_value_class_1 = explainer.expected_value[1]
else: # (samples, features)
    shap_values_for_class_1 = shap_values.values
    expected_value_class_1 = explainer.expected_value

# 4a. SHAP Global Feature Importance (Summary Bar Plot)
print("\nGlobal SHAP Importance (mean absolute SHAP value):")
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values_for_class_1, X_test, plot_type="bar", show=False)
plt.title('SHAP Global Feature Importance')
plt.tight_layout()
plt.savefig('shap_global_importance_bar.png')
print("Saved SHAP global importance bar plot to 'shap_global_importance_bar.png'")
plt.show()

# 4b. SHAP Summary Plot (Beeswarm - shows distribution and magnitude)
print("\nSHAP Summary Plot (Beeswarm - shows direction and distribution):")
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values_for_class_1, X_test, show=False)
plt.title('SHAP Summary Plot (Beeswarm)')
plt.tight_layout()
plt.savefig('shap_summary_beeswarm.png')
print("Saved SHAP summary beeswarm plot to 'shap_summary_beeswarm.png'")
plt.show()

# 4c. SHAP Local Explanations (Waterfall plot for a single prediction)
print("\nLocal SHAP Explanation (Waterfall plot for one passenger's prediction):")
passenger_index = 0 # Explain the first passenger in the test set

plt.figure(figsize=(12,8))
# Create an Explanation object for the single instance for waterfall plot
individual_shap_values = shap.Explanation(
    values=shap_values_for_class_1[passenger_index,:],
    base_values=expected_value_class_1,
    data=X_test.iloc[passenger_index,:],
    feature_names=X_test.columns
)
shap.plots.waterfall(individual_shap_values, max_display=10, show=False)
plt.title(f'SHAP Waterfall Plot for Passenger {passenger_index} (Prediction: {model.predict(X_test.iloc[[passenger_index]])[0]})')
plt.tight_layout()
plt.savefig('shap_local_waterfall.png')
print("Saved SHAP local waterfall plot to 'shap_local_waterfall.png'")
plt.show()

# 4d. SHAP Dependence Plot (Interaction Effects)
# Example: How 'FlightDelayMinutes' interacts with 'IsEliteStatus'
feature_to_plot = 'FlightDelayMinutes'
interaction_feature = 'IsEliteStatus'
print(f"\nSHAP Dependence Plot (Interaction of '{feature_to_plot}' with '{interaction_feature}'):")

plt.figure()
shap.dependence_plot(
    feature_to_plot,
    shap_values_for_class_1,
    X_test,
    interaction_index=interaction_feature,
    show=False
)
plt.title(f'SHAP Dependence: {feature_to_plot} (Interaction with {interaction_feature})')
plt.tight_layout()
plt.savefig(f'shap_dependence_{feature_to_plot}_vs_{interaction_feature}.png')
print(f"Saved SHAP dependence plot to 'shap_dependence_{feature_to_plot}_vs_{interaction_feature}.png'")
plt.show()

print("\nAdvantages of SHAP for American Airlines' Customer Satisfaction Analysis:")
print("- Consistency across models: Provides consistent interpretation.")
print("- Direction of impact: Shows if a feature positively or negatively affects satisfaction (e.g., beeswarm plot).")
print("- Interaction detection: Reveals how features interact (e.g., dependence plots).")
print("- Local explanations: Provides passenger-specific insights (e.g., waterfall plots) for personalized service.")
print("- Regulatory alignment & Trust building: More defensible and intuitive.")

# --- 5. Practical Application at American Airlines (Hybrid Approach) ---
print("\n--- Practical Application at American Airlines (Hybrid Approach) ---")
print("1. Use built-in importance for initial exploration and model iteration:")
print("   - Quick feedback during model development.")
print("   - Faster processing for high-level trend analysis.")
print("2. Use SHAP for deeper operational insights and action planning:")
print("   - Detailed analysis of how service disruptions affect different customer segments.")
print("   - Generating personalized explanations for customer service agents.")
print("   - Root cause analysis for satisfaction outliers.")

print("\nSpecific AA use cases for SHAP (as demonstrated by the plots):")
print("- Understanding how elite status moderates negative impact of delays (via dependence plots).")
print("- Quantifying the effect of proactive communication during IRROPs (via summary/beeswarm plots).")
print("- Analyzing how different aircraft configurations (mocked by 'SeatComfort') affect ratings.")

print("\nThis script provides a basic framework. For a real-world AA scenario, you would:")
print("- Use actual customer survey data and operational metrics.")
print("- Perform more extensive feature engineering.")
print("- Tune models rigorously.")
print("- Integrate these insights into dashboards and operational workflows.")

print("\nEnd of demonstration.")

