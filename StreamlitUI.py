import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --- Page Setup ---
st.set_page_config(page_title="Insurance Claim Prediction", layout="wide")
st.title(" Vehicle Insurance Claim Prediction")

# --- Preprocessing Functions ---

def extract_vals(val):
    """Regex to extract torque/power and RPM values from strings"""
    res = re.findall(r'(\d+\.?\d*)', str(val))
    if len(res) >= 2: 
        return float(res[0]), float(res[1])
    return 0.0, 0.0

@st.cache_data
def prepare_model():
    """Trains the KNN model using the top 88 features by correlation"""
    # 1. Load Dataset
    df = pd.read_csv("dataset.csv") 
    df.drop_duplicates(inplace=True)
    
    # 2. Clean Population Density (Notebook step)
    df['population_density'] = df['population_density'].astype(str).str.replace(r'\D', '', regex=True)
    df['population_density'] = pd.to_numeric(df['population_density'], errors='coerce')
    
    # 3. Feature Engineering (Derived Columns)
    df['torque'], df['torque_rpm'] = zip(*df['max_torque'].apply(extract_vals))
    df['power'], df['power_rpm'] = zip(*df['max_power'].apply(extract_vals))
    df['torque_to_rpm_ratio'] = df['torque'] / df['torque_rpm'].replace(0, np.nan)
    df['power_to_rpm_ratio'] = df['power'] / df['power_rpm'].replace(0, np.nan)
    df.fillna(0, inplace=True)
    
    # Store df for UI (before encoding)
    raw_df = df.copy()
    
    # 4. Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'policy_id' in categorical_cols:
        categorical_cols.remove('policy_id')
    
    df_one_hot = pd.get_dummies(df.drop(columns=['policy_id']), columns=categorical_cols)
    
    # 5. TOP 88 FEATURES LOGIC (Your Request)
    correlations = df_one_hot.corr()['is_claim'].abs().sort_values(ascending=False)
    top_88_names = correlations.head(88).index.tolist()
    
    # Select the columns and split X/y
    df_final = df_one_hot[top_88_names]
    X = df_final.drop(columns=['is_claim']) # 87 input features
    y = df_final['is_claim']
    
    # 6. Scaling & KNN Training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_scaled, y)
    
    return model, scaler, raw_df, categorical_cols, X.columns.tolist()

# Initialize Model & Data
try:
    model, scaler, raw_df, cat_cols, final_features = prepare_model()
    st.sidebar.success(" KNN Model ")
except Exception as e:
    st.error(f"Error loading model: {e}. Check if 'dataset.csv' is present.")
    st.stop()

# --- UI Input Form ---

st.subheader("Enter Policy & Vehicle Information")
user_input = {}
cols = st.columns(3)
idx = 0

# Skip target, IDs, and columns that we calculate manually
skip_cols = ["is_claim", "policy_id", "torque", "torque_rpm", "power", "power_rpm", "torque_to_rpm_ratio", "power_to_rpm_ratio"]

for column in raw_df.columns:
    if column in skip_cols:
        continue

    with cols[idx % 3]:
        # Dropdowns for categories
        if column in cat_cols:
            user_input[column] = st.selectbox(
                f"Select {column.replace('_', ' ').title()}",
                options=sorted(raw_df[column].dropna().unique().astype(str))
            )
        # Text inputs for Torque/Power (to match Notebook extraction)
        elif column in ["max_torque", "max_power"]:
            user_input[column] = st.text_input(
                f"Enter {column.replace('_', ' ').title()}",
                value=str(raw_df[column].mode()[0])
            )
        # Number inputs for standard numerics
        else:
            user_input[column] = st.number_input(
                f"Enter {column.replace('_', ' ').title()}",
                value=float(raw_df[column].mean())
            )
    idx += 1

# --- Prediction Logic ---

if st.button("Predict Insurance Claim", use_container_width=True):
    # 1. Convert input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # 2. Re-apply Notebook Preprocessing
    input_df['torque'], input_df['torque_rpm'] = zip(*input_df['max_torque'].apply(extract_vals))
    input_df['power'], input_df['power_rpm'] = zip(*input_df['max_power'].apply(extract_vals))
    input_df['torque_to_rpm_ratio'] = input_df['torque'] / input_df['torque_rpm'].replace(0, np.nan)
    input_df['power_to_rpm_ratio'] = input_df['power'] / input_df['power_rpm'].replace(0, np.nan)
    input_df.fillna(0, inplace=True)
    
    # 3. Encoding & Feature Alignment (88 Column Check)
    input_encoded = pd.get_dummies(input_df, columns=cat_cols)
    input_encoded = input_encoded.reindex(columns=final_features, fill_value=0)
    
    # 4. KNN Prediction
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    
    # 5. Result Display
    st.divider()
    if prediction == 1:
        st.error(f"### Predicted: CLAIM LIKELY\n**Confidence Level:** {prob:.2%}")
    else:
        st.success(f"### Predicted: NO CLAIM\n**Confidence Level:** {(1-prob):.2%}")