# from fastapi import FastAPI
# import joblib
# import pandas as pd
# import numpy as np
# from pydantic import BaseModel, Field, computed_field
# from typing import Annotated, Literal
# import uvicorn

# # 1. Initialize FastAPI
# app = FastAPI(title="Used Car Price API")

# # 2. Define the Input Schema using Pydantic
# class CarInput(BaseModel):
#     name: Annotated[str, Field(..., description='Name of the Car', examples=['Maruti Suzuki Alto'])]
#     company: Annotated[str, Field(..., description='Name of the Company that the Car belongs to', examples=['Maruti'])]
#     year: Annotated[int, Field(..., description='Car Purchased Year')]
#     kms_driven: Annotated[float, Field(...,gt=0, description='Kilometer Driven in Float')]
#     fuel_type: Annotated[Literal['Diesel', 'Petrol'], Field(..., description='Fuel Type of the Car')]
    
#     # @computed_field
#     # @property
#     # def luxury(self) -> str:
#     #     luxury_brands = ['bmw', 'audi', 'mercedes', 'jaguar', 'mini', 'land rover']
#     #     is_luxury = 1 if self.company.lower() in luxury_brands else 0
#     #     return is_luxury

# # 3. Load the pre-trained Pipeline
# # Using a Pipeline is better because it handles preprocessing automatically
# model_path = "pipeline.joblib"
# pipeline = joblib.load(model_path)


# @app.get("/")
# def home():
#     return {"status": "API is online", "message": "Send a POST request to /predict"}

# @app.get("/welcome")
# def welcome():
#     return {"message": "Hi welcome to the Mlops project app created by Midhun Manohar"}

# @app.post("/predict")
# def predict(data: CarInput):
#     # --- Feature Engineering (Must match your training logic) ---
#     luxury_brands = ['bmw', 'audi', 'mercedes', 'jaguar', 'mini', 'land rover']
#     is_luxury = 1 if data.company.lower() in luxury_brands else 0
#     current_year = 2020
#     car_age = current_year - data.year
#     kms_per_year = data.kms_driven / (car_age + 1)

#     # 4. Create DataFrame for Pipeline
#     # Pipelines expect a DataFrame with matching column names
#     input_df = pd.DataFrame([{
#         'name': data.name,
#         'company': data.company,
#         'year': data.year,
#         'kms_driven': data.kms_driven,
#         'fuel_type': data.fuel_type,
#         'car_age': car_age,
#         'is_luxury': is_luxury,
#         'kms_per_year': kms_per_year
#     }])

#     # 5. Make Prediction
#     pred_log_score = pipeline.predict(input_df)
    
#     # Reverse log transform if necessary (np.expm1)
#     # If you didn't log-transform during training, just use pred_log_score[0]
#     final_price = np.expm1(pred_log_score[0])

#     return {
#         "input_received": data.dict(),
#         "predicted_price": round(float(final_price), 2)
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
#     # 127.0.0.1


import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.title("Car Price Prediction App")
st.markdown("Enter your car details below:")

# Input fields
name = st.text_input("Name of the Car")
company = st.text_input("Enter the Company the Car belongs to")
year = st.number_input("Car Purchase Year", min_value=2007, max_value=2020)
kms_driven = st.number_input("Kilometer Driven", min_value=100)
fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel'])

if st.button("Predict Price"):
    input_data = {
        "name" : name,
        "company" : company,
        "year" : year,
        "kms_driven" : kms_driven,
        "fuel_type" : fuel_type
    }
    
    try:
        response = requests.post(API_URL, json=input_data)
        result = response.json()
        
        if response.status_code == 200:
            # Handle whether your FastAPI returns the Pydantic model at the root 
            # or nested inside a "response" key. This line handles both safely:
            prediction = result.get("response", result)

            # 1. Main Price Prediction
            st.success(f"Predicted Resale Price: **₹{prediction['predicted_price']:,}**")
            
            # 2. Confidence Metric
            st.metric(label="Model Confidence Score", value=f"{prediction['model_confidence_percent']}%")
            
            # 3. Confidence Interval Range
            ci = prediction["confidence_interval"]
            st.info(f"💡 **Estimated Price Range:** ${ci['lower_bound']:,} to ${ci['upper_bound']:,}")
            
            # 4. Prediction Summary Features (Displayed in columns)
            st.write("---")
            st.write("📊 **Vehicle Analysis Summary:**")
            summary = prediction["prediction_summary"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Car Age:**\n{summary['car_age']} years")
            with col2:
                st.markdown(f"**Luxury Status:**\n{'✨ Luxury' if summary['is_luxury'] else 'Standard'}")
            with col3:
                st.markdown(f"**Avg KMs/Year:**\n{summary['kms_per_year']:,} km")

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(result)

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server. Make sure it's running.")
            
