import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("user_skill_assessment_dataset_v2.csv")

# 1. Data Preparation
# Add small noise to prevent perfect scores
np.random.seed(42)
df['calculated_skill_score'] = df['calculated_skill_score'].apply(
    lambda x: np.clip(x + np.random.normal(0, 0.1), 0, 5)
)

# Select only question score features (Q1_score to Q5_score)
question_columns = [f'Q{i}_score' for i in range(1, 6)]
X = df[question_columns]
y = df['calculated_skill_score']

# 2. Model Development
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define models to compare
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'Linear Regression': LinearRegression()
}

# Evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'model': model
    }
    
    print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

# 3. Hyperparameter Tuning (skip for Linear Regression)
print("\nTuning best performing model...")
best_model_name = max(results, key=lambda k: results[k]['R2'])
print(f"Selected model for tuning: {best_model_name}")

if best_model_name == 'Linear Regression':
    print("No hyperparameters to tune for Linear Regression. Using default model.")
    best_model = results[best_model_name]['model']
else:
    # Define parameter grids only for models that need tuning
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
    elif best_model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }

    grid_search = GridSearchCV(
        estimator=results[best_model_name]['model'],
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

# Evaluate best model (tuned or default)
y_pred_tuned = best_model.predict(X_test)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"\nFinal {best_model_name} - MAE: {mae_tuned:.2f}, RMSE: {rmse_tuned:.2f}, R2: {r2_tuned:.2f}")
if best_model_name != 'Linear Regression':
    print("Best parameters:", grid_search.best_params_)

# 4. Feature Importance Analysis
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_names = X.columns
    feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp = feature_imp.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp)
    plt.title(f'{best_model_name} Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

# 5. Save the best model
joblib.dump(best_model, 'skill_assessment_model.pkl')

# 6. Prediction Function
def predict_skill_score(answers, model=best_model):
    """
    Predict skill score based on questionnaire answers only
    
    Args:
        answers: Dictionary of question scores (e.g., {'Q1_score': 1, 'Q2_score': 0.5, ...})
        model: Trained model (defaults to best model)
    
    Returns:
        Predicted skill score (0-5)
    """
    # Convert answers to dataframe
    input_data = pd.DataFrame([answers])
    
    # Ensure all question scores are present
    for q in [f'Q{i}_score' for i in range(1, 6)]:
        if q not in input_data:
            input_data[q] = 0  # Default to 0 if missing
    
    # Reorder columns to match training data
    input_data = input_data[X.columns]
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Clip to 0-5 range
    return np.clip(prediction[0], 0, 5)

# Example usage:
sample_answers = {
    'Q1_score': 1,
    'Q2_score': 2,
    'Q3_score': 1,
    'Q4_score': 0.5,
    'Q5_score': 1
}

print(f"\nSample prediction: {predict_skill_score(sample_answers):.1f}/5")

# Model comparison bar plot
model_names = list(results.keys())
r2_scores = [results[name]['R2'] for name in model_names]

plt.figure(figsize=(10, 5))
bars = plt.bar(model_names, r2_scores, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
plt.title('Model Performance Comparison (R² Score)')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')

plt.savefig('model_performance.png', bbox_inches='tight')
plt.show()

# Actual vs Predicted values scatter plot
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred_tuned, alpha=0.6, color='#4C72B0')
plt.plot([0, 5], [0, 5], '--', color='red')  # Perfect prediction line
plt.title('Actual vs Predicted Skill Scores')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.grid(True, linestyle='--', alpha=0.5)

# Add R² and MAE annotations
plt.text(0.5, 4.7, f'R² = {r2_tuned:.2f}', fontsize=12)
plt.text(0.5, 4.5, f'MAE = {mae_tuned:.2f}', fontsize=12)

plt.savefig('actual_vs_predicted.png', bbox_inches='tight')
plt.show()

# Error distribution plot
errors = y_test - y_pred_tuned
plt.figure(figsize=(10, 5))
sns.histplot(errors, kde=True, color='#DD8452', bins=20)
plt.title('Prediction Error Distribution')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.axvline(x=0, color='red', linestyle='--')
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('error_distribution.png', bbox_inches='tight')
plt.show()