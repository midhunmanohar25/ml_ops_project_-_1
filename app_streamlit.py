import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime

data_path = "models/dataset.joblib"
df = joblib.load(data_path)

st.set_page_config(
    page_title="Used Car Price Recommendation System",
    layout="wide"
)

# st.markdown(
#     """
#     <style>
#     div[data-testid="stImage"] {
#         display: flex;
#         justify-content: center;
#     }

#     div[data-testid="stImage"] img {
#         max-height: 700px !important;
#         width: 200 !important;
#         object-fit: cover;
#         object-position: center;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.image("Images/car_banner.png", use_container_width=True)

st.title("Used Car Price Recommendation System")

with st.container():
    st.subheader("Enter Car Details")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_company = st.selectbox('Company',df['company'].unique().tolist())
        luxury_brand = ['bmw', 'audi', 'mercedes', 'jaguar', 'mini']
        is_luxury = df[selected_company].str.lower().isin(luxury_brand).astype(int)
    
    with col2:
        selected_name = st.selectbox('Car Name',df['name'].unique().tolist())

    with col3:
        selected_year = st.selectbox('Select Year',sorted(df['year'].unique().tolist(),reverse=True))
        input_kilometers = float(st.number_input('Enter Kilometers',min_value=0,max_value=400000,step=1))
        current_year = datetime.now().year
        car_age = current_year - selected_year
        kms_per_year = input_kilometers / (car_age + 1)

    with col4:
        selected_fueltype = st.selectbox('Select Fuel Type',['Petrol','Diesel'])
    
    
    
        

# MODEL LOADING PERFORMANCE
@st.cache_resource
def load_model():
    model_path = "models/model.joblib"
    model = joblib.load(model_path)
    return model

model = load_model()

# Predicting the value
if st.button("Predict"):
    data = [[selected_name,selected_company,selected_year,input_kilometers,selected_fueltype, car_age, is_luxury, kms_per_year]]
    
    columns=['name','company','year','kms_driven','fuel_type', 'car_age', 'is_luxury', 'kms_per_year']
    
    one_df = pd.DataFrame(data, columns=columns)
    
    numerical_col = df.select_dtypes(['int','float']).columns.to_list()
    categorical_col = df.select_dtypes(['object']).columns.to_list()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_col),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_col)
        ],
        remainder='passthrough'
    )
    
    one_df = preprocessor.fit_transform(one_df)
    # Predict
    pred_score = model.predict(one_df)
    
    base_price = np.expm1(pred_score)[0]
    # display
    st.success(f"Predicted Price: ₹ {round(base_price, 2)}")