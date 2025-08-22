# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import numpy as np

# --- 1. Caching the Model Training ---
# This function runs only once, and its result is stored (cached) for performance.
@st.cache_data
def train_model():
    df = pd.read_csv('COVID-19-Hospitals-Treatment-Plan.csv')

    # --- Data Cleaning and Preprocessing ---
    df = df.drop(columns=['case_id', 'patientid'], errors='ignore')
    df['Bed_Grade'] = df['Bed_Grade'].fillna(df['Bed_Grade'].mode()[0])
    df['City_Code_Patient'] = df['City_Code_Patient'].fillna(df['City_Code_Patient'].mode()[0])

    target = 'Stay_Days'
    ordinal_features = ['Age', 'Illness_Severity']
    nominal_features = ['Hospital', 'Hospital_type', 'Hospital_city', 'Hospital_region', 
                        'Department', 'Ward_Type', 'Ward_Facility', 'Type_of_Admission', 
                        'City_Code_Patient']

    age_order = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    severity_order = ['Minor', 'Moderate', 'Extreme']

    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(categories=[age_order, severity_order]), ordinal_features),
            ('nominal', OneHotEncoder(handle_unknown='ignore'), nominal_features)
        ],
        remainder='passthrough'
    )

    X = df.drop(target, axis=1)
    y = df[target]

    # We only need to train the model once, so we use the full dataset here for the final app
    X_train_processed = preprocessor.fit_transform(X)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y)

    # Train the final, constrained model
    final_model = XGBClassifier(
        objective='multi:softprob', n_estimators=100, learning_rate=0.1, max_depth=5,
        use_label_encoder=False, eval_metric='mlogloss', random_state=42
    )
    final_model.fit(X_train_processed, y_train_encoded)

    return final_model, preprocessor, le, df # Return the trained objects

# --- Load the trained model and other objects ---
model, preprocessor, le, df = train_model()

# --- Dictionaries and Helper Functions for Cost/Delay ---
cost_schedule = {
    ('gynecology', 'Emergency'): 8000, ('gynecology', 'Trauma'): 7000, ('gynecology', 'Urgent'): 6500,
    ('radiotherapy', 'Emergency'): 15000, ('radiotherapy', 'Trauma'): 12000, ('radiotherapy', 'Urgent'): 10000,
    ('anesthesia', 'Emergency'): 11000, ('anesthesia', 'Trauma'): 9500, ('anesthesia', 'Urgent'): 8500,
    ('TB & Chest disease', 'Emergency'): 9000, ('TB & Chest disease', 'Trauma'): 7500, ('TB & Chest disease', 'Urgent'): 7000,
    ('surgery', 'Emergency'): 13000, ('surgery', 'Trauma'): 11000, ('surgery', 'Urgent'): 9000
}
stay_to_days_mapping = {
    '0-10': 5, '11-20': 15.5, '21-30': 25.5, '31-40': 35.5, '41-50': 45.5,
    '51-60': 55.5, '61-70': 65.5, '71-80': 75.5, '81-90': 85.5, '91-100': 95.5,
    'More than 100 Days': 110
}
administrative_day_rate = 2500

def calculate_administrative_delay_score(medical_cost, severity, admission_type, age):
    delay_score = 0
    if medical_cost > 200000: delay_score += 5
    elif medical_cost > 125000: delay_score += 3
    elif medical_cost > 75000: delay_score += 1
    if severity == 'Extreme': delay_score += 3
    elif severity == 'Moderate': delay_score += 1
    if admission_type == 'Emergency': delay_score += 2
    elif admission_type == 'Trauma': delay_score += 1
    if age in ['61-70', '71-80', '81-90', '91-100']: delay_score += 1
    return delay_score

# --- Building the User Interface ---
st.set_page_config(page_title="Hospital Stay & Cost Predictor", layout="wide")
st.title("🏥 Hospital Stay & Cost Predictor")

st.sidebar.header("Enter Patient Information")

departments = df['Department'].unique()
admission_types = df['Type_of_Admission'].unique()
severity_levels = ['Minor', 'Moderate', 'Extreme']
age_brackets = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']

department = st.sidebar.selectbox("Select Department", departments)
admission = st.sidebar.selectbox("Select Admission Type", admission_types)
severity = st.sidebar.selectbox("Select Illness Severity", severity_levels)
age = st.sidebar.selectbox("Select Age Bracket", age_brackets)
visitors = st.sidebar.number_input("Enter Number of Visitors", min_value=0, max_value=50, value=5)
admission_deposit = st.sidebar.number_input("Enter Admission Deposit (INR)", min_value=1000, max_value=15000, value=5000)

if st.sidebar.button("Predict Stay and Cost"):
    new_patient_data = {
        'Hospital': [8], 'Hospital_type': [2], 'Hospital_city': [3], 'Hospital_region': [2],
        'Available_Extra_Rooms_in_Hospital': [3], 'Department': [department], 'Ward_Type': ['R'],
        'Ward_Facility': ['F'], 'Bed_Grade': [2.0], 'City_Code_Patient': [7.0],
        'Type_of_Admission': [admission], 'Illness_Severity': [severity],
        'Patient_Visitors': [visitors], 'Age': [age], 'Admission_Deposit': [admission_deposit]
    }
    new_patient_df = pd.DataFrame(new_patient_data)

    new_patient_processed = preprocessor.transform(new_patient_df)
    prediction_encoded = model.predict(new_patient_processed)
    predicted_stay_category = le.inverse_transform(prediction_encoded)[0]
    predicted_days = stay_to_days_mapping[predicted_stay_category]

    daily_rate = cost_schedule.get((department, admission), 6000)
    medical_cost = predicted_days * daily_rate
    admin_delay_days = calculate_administrative_delay_score(medical_cost, severity, admission, age)
    administrative_cost = admin_delay_days * administrative_day_rate
    total_stay_days = predicted_days + admin_delay_days
    total_final_cost = medical_cost + administrative_cost
    
    st.header("Final Patient Stay and Cost Report")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Medical Stay", f"{predicted_days} days", f"({predicted_stay_category})")
    col2.metric("Simulated Admin Delay", f"{admin_delay_days} days")
    col3.metric("Total Estimated Stay", f"{total_stay_days} days")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated Medical Cost", f"₹{medical_cost:,.0f}")
    col2.metric("Administrative Cost", f"₹{administrative_cost:,.0f}")
    col3.metric("TOTAL ESTIMATED COST", f"₹{total_final_cost:,.0f}")