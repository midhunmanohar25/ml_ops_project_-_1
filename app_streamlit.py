import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data_path = "models/dataset.joblib"
df = joblib.load(data_path)

st.set_page_config(
    page_title="Used Car Price Recommendation System",
    layout="wide"
)

st.title("Used Car Price Recommendation System")

with st.container():
    st.subheader("Enter Car Details")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_company = st.selectbox('Company',df['company'].unique().tolist())

    
    with col2:
        if selected_company == 'Bmw':
    
            names = df['name'].str.startswith('BMW')
            fetched_names = df[names]['name']
            selected_name = st.selectbox('Car Name',fetched_names.unique().tolist())
    
        elif selected_company in df['company'].values:
    
            names = df['name'].str.startswith(selected_company)
            fetched_names = df[names]['name']
            selected_name = st.selectbox('Car Name',fetched_names.unique().tolist())

    with col3:
        selected_year = st.selectbox('Select Year',sorted(df['year'].unique().tolist(),reverse=True))
        input_kilometers = float(st.number_input('Enter Kilometers',min_value=0,max_value=400000,step=1))
        

    with col4:
        selected_fueltype = st.selectbox('Select Fuel Type',['Petrol','Diesel'])
    
       
luxury_brand = ['bmw', 'audi', 'mercedes', 'jaguar', 'mini']
is_luxury = int(selected_company.lower() in luxury_brand)
current_year = 2020
car_age = current_year - selected_year
kms_per_year = input_kilometers / (car_age + 1)

# MODEL LOADING PERFORMANCE
@st.cache_resource
def load_pipeline():
    pipeline_path = "models/pipeline.joblib"
    pipeline = joblib.load(pipeline_path)
    return pipeline

pipeline = load_pipeline()

# Predicting the value
if st.button("Predict"):
    data = [[selected_name,selected_company,selected_year,input_kilometers,selected_fueltype, car_age, is_luxury, kms_per_year]]
    
    columns=['name','company','year','kms_driven','fuel_type', 'car_age', 'is_luxury', 'kms_per_year']
    
    one_df = pd.DataFrame(data, columns=columns)
    
    # Predict
    pred_score = pipeline.predict(one_df)
    
    base_price = np.expm1(pred_score)[0]
    # display
    st.success(f"Predicted Price: ₹ {round(base_price, 2)}")