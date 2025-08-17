# =============================================================================
# train_model.py
# This script trains the final machine learning model based on the best
# parameters found during experimentation in the Jupyter Notebook.
# It saves the final, trained pipeline for use in the web application.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("--- [START] Model Training Script ---")

# --- 1. Load Data ---
# Build a robust path to the data file
try:
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_dataset.csv')
    df = pd.read_csv(data_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file was not found at {data_path}")
    print("Please make sure 'raw_dataset.csv' is in the 'data' folder.")
    exit()


# --- 2. Clean Data ---
# This cleaning logic is the same as the one finalized in the notebook
print("Cleaning data...")
df.drop(['Timestamp', 'Country', 'state', 'comments'], axis=1, inplace=True, errors='ignore')

# Standardize Gender column
df['Gender'] = df['Gender'].str.lower()
male_synonyms = ['m', 'male', 'male-ish', 'maile', 'mal', 'male (cis)', 'make', 'cis male', 'man', 'msle', 'mail']
female_synonyms = ['f', 'female', 'woman', 'cis female', 'femake', 'female (cis)', 'femail', 'cis-female/femme', 'female (trans)']

def standardize_gender(g):
    if g in male_synonyms:
        return 'Male'
    elif g in female_synonyms:
        return 'Female'
    else:
        return 'Other'

df['Gender'] = df['Gender'].apply(standardize_gender)
df = df[df['Gender'].isin(['Male', 'Female'])]

# Handle missing values
df['self_employed'].fillna(df['self_employed'].mode()[0], inplace=True)
df['work_interfere'].fillna(df['work_interfere'].mode()[0], inplace=True)
df.dropna(inplace=True)

# --- NEW CODE START ---
# Save the cleaned DataFrame to a new file for record-keeping and future use
processed_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_dataset.csv')
df.to_csv(processed_data_path, index=False)
# --- NEW CODE END ---

print("Data cleaning complete and processed_dataset.csv saved.")


# --- 3. Define Features and Target ---
X = df.drop('treatment', axis=1)
y = df['treatment']


# --- 4. Set up The Final Preprocessing & Model Pipeline ---
# This pipeline uses the best model (RandomForest) and best parameters from our notebook
print("Setting up the final model pipeline...")

numerical_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(include='object').columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Using the best parameters found from GridSearchCV in the notebook
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=1
    ))
])


# --- 5. Train the Final Model ---
# For the production model, we train on the ENTIRE dataset so it learns as much as possible
print("Training the final model on the entire dataset...")
final_model.fit(X, y)
print("Model training complete.")


# --- 6. Save the Model ---
# Save the entire pipeline to the 'ml_model' folder
model_dir = os.path.dirname(__file__)
model_save_path = os.path.join(model_dir, 'mental_health_model.pkl')

joblib.dump(final_model, model_save_path)
print(f"Final model saved successfully to: {model_save_path}")
print("--- [END] Model Training Script ---")