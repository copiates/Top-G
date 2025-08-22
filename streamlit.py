# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import numpy as np

# --- 1. Model Training Function with Caching ---
# This function loads data, preprocesses it, and trains the final model.
# Streamlit's @st.cache_data decorator ensures this complex process runs only once.
@st.cache_data
def train_model():
    """
    Loads data, preprocesses it, and trains the final XGBoost model.
    Returns the trained model, preprocessor, label encoder, and dataframe.
    """
    # Load the dataset from the local file
    df = pd.read_csv('COVID-19-Hospitals-Treatment-Plan.csv')

    # --- Data Cleaning and Preprocessing ---
    df = df.drop(columns=['case_id', 'patientid'], errors='ignore')
    df['Bed_Grade'] = df['Bed_Grade'].fillna(df['Bed_Grade'].mode()[0])
    df['City_Code_Patient'] = df['City_Code_Patient'].fillna(df['City_Code_Patient'].mode()[0])

    # Define feature categories for the preprocessor
    target = 'Stay_Days'
    ordinal_features = ['Age', 'Illness_Severity']
    nominal_features = ['Hospital', 'Hospital_type', 'Hospital_city', 'Hospital_region', 
                        'Department', 'Ward_Type', 'Ward_Facility', 'Type_of_Admission', 
                        'City_Code_Patient']

    # Define the explicit order for ordinal features to ensure consistency
    age_order = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    severity_order = ['Minor', 'Moderate', 'Extreme']

    # Create the column transformer to handle different feature types
    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(categories=[age_order, severity_order]), ordinal_features),
            ('nominal', OneHotEncoder(handle_unknown='ignore'), nominal_features)
        ],
        remainder='passthrough' # Keep numerical columns as they are
    )

    X = df.drop(target, axis=1)
    y = df[target]

    # Fit the preprocessor and label encoder on the entire dataset
    X_processed = preprocessor.fit_transform(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train the final, constrained model to prevent overfitting
    final_model = XGBClassifier(
        objective='multi:softprob', 
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, # This is the key to preventing extreme, biased predictions
        use_label_encoder=False, 
        eval_metric='mlogloss', 
        random_state=42
    )
    final_model.fit(X_processed, y_encoded)

    return final_model, preprocessor, le, df

# --- Load the trained model and other necessary objects ---
model, preprocessor, le, df = train_model()

# --- 2. Dictionaries & Helper Functions for Calculations ---
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
    """Calculates a nuanced administrative delay based on a scoring system."""
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

# --- 3. Building the Streamlit User Interface ---
st.set_page_config(page_title="Hospital Stay & Cost Predictor", layout="wide")
st.title("🏥 Hospital Stay & Cost Predictor")
st.markdown("Enter patient details in the sidebar to get an estimated length of stay and the associated costs.")

# Create a sidebar for all user inputs
st.sidebar.header("Enter Patient Information")

# Get unique values for dropdowns directly from the dataframe
departments = df['Department'].unique()
admission_types = df['Type_of_Admission'].unique()
severity_levels = ['Minor', 'Moderate', 'Extreme']
age_brackets = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
ward_types = df['Ward_Type'].unique()
ward_facilities = df['Ward_Facility'].unique()

# Create interactive input widgets in the sidebar
department = st.sidebar.selectbox("Select Department", departments)
admission = st.sidebar.selectbox("Select Admission Type", admission_types)
severity = st.sidebar.selectbox("Select Illness Severity", severity_levels)
age = st.sidebar.selectbox("Select Age Bracket", age_brackets)
ward_type_selection = st.sidebar.selectbox("Select Ward Type", ward_types)
ward_facility_selection = st.sidebar.selectbox("Select Ward Facility", ward_facilities)
visitors = st.sidebar.number_input("Number of Visitors", min_value=0, max_value=50, value=5)
admission_deposit = st.sidebar.number_input("Admission Deposit (INR)", min_value=1000, max_value=20000, value=5000)

# Create a button to trigger the prediction
if st.sidebar.button("Predict Stay and Cost"):
    # --- 4. Prediction and Calculation Logic ---
    # Create a DataFrame from the user's inputs
    new_patient_data = {
        # Using default values for features not included in the UI for simplicity
        'Hospital': [8], 'Hospital_type': [2], 'Hospital_city': [3], 'Hospital_region': [2],
        'Available_Extra_Rooms_in_Hospital': [3], 'Bed_Grade': [2.0], 'City_Code_Patient': [7.0],
        
        # Using the dynamic inputs from the UI
        'Department': [department],
        'Ward_Type': [ward_type_selection],
        'Ward_Facility': [ward_facility_selection],
        'Type_of_Admission': [admission],
        'Illness_Severity': [severity],
        'Patient_Visitors': [visitors],
        'Age': [age],
        'Admission_Deposit': [admission_deposit]
    }
    new_patient_df = pd.DataFrame(new_patient_data)

    # Preprocess the input and make a prediction
    new_patient_processed = preprocessor.transform(new_patient_df)
    prediction_encoded = model.predict(new_patient_processed)
    predicted_stay_category = le.inverse_transform(prediction_encoded)[0]
    predicted_days = stay_to_days_mapping[predicted_stay_category]

    # Calculate costs and delays based on the prediction
    daily_rate = cost_schedule.get((department, admission), 6000)
    medical_cost = predicted_days * daily_rate
    admin_delay_days = calculate_administrative_delay_score(medical_cost, severity, admission, age)
    administrative_cost = admin_delay_days * administrative_day_rate
    total_stay_days = predicted_days + admin_delay_days
    total_final_cost = medical_cost + administrative_cost
    
    # --- 5. Display the Final Report ---
    st.header("Final Patient Stay and Cost Report")
    st.markdown("---")
    
    # Display key metrics in columns for a clean layout
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Medical Stay", f"{predicted_days} days", f"({predicted_stay_category})")
    col2.metric("Simulated Admin Delay", f"{admin_delay_days} days")
    col3.metric("Total Estimated Stay", f"{total_stay_days} days")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated Medical Cost", f"₹{medical_cost:,.0f}")
    col2.metric("Administrative Cost", f"₹{administrative_cost:,.0f}")
    col3.metric("TOTAL ESTIMATED COST", f"₹{total_final_cost:,.0f}")
