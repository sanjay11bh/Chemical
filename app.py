import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load the trained model
model_pipeline = joblib.load("model_pipeline.pkl")  

# Define a cost model (basic calculation based on yield and costs)
def calculate_cost(yield_kg, electricity_cost, H2_cost, capex):
    # Calculate the variable cost based on yield, electricity cost, and H2 cost
    variable_cost = yield_kg * (electricity_cost * 0.5 + H2_cost * 0.3)
    fixed_cost = capex / yield_kg  # Capital cost per kg
    total_cost = variable_cost + fixed_cost
    
    # Debugging: Output the components of the cost calculation
    st.write(f"**Debugging Cost Calculation:**")
    st.write(f"SAF Yield (kg): {yield_kg:.2f}")
    st.write(f"Variable Cost: {variable_cost:.2f}")
    st.write(f"Fixed Cost (CAPEX component): {fixed_cost:.2f}")
    st.write(f"Total Production Cost: {total_cost:.2f}")
    
    return total_cost

# Define feasibility check based on cost threshold
def is_feasible(cost, threshold=10.0):
    return cost < threshold

# Streamlit UI
st.title("SAF Production Prediction and Feasibility App")

# User inputs for SAF prediction
electrolysis_type = st.selectbox("Electrolysis Type", ["LTE", "HTE"])
voltage = st.slider("Voltage (V)", min_value=1.0, max_value=4.0, step=0.1)
current_density = st.slider("Current Density (mA/cm²)", min_value=150, max_value=1000, step=10)
temperature = st.slider("Temperature (°C)", min_value=30, max_value=800, step=10)
CO2_conversion_rate = st.slider("CO2 Conversion Rate", min_value=0.4, max_value=0.55, step=0.01)
CO2_source = st.selectbox("CO2 Source", ["bioethanol", "cement", "NGCC", "DAC"])
electricity_cost = st.slider("Electricity Cost ($/kWh)", min_value=0.01, max_value=0.07, step=0.01)
H2_required = st.slider("H2 Required (kg)", min_value=3.5, max_value=6.5, step=0.1)
capex = st.slider("CAPEX ($)", min_value=5000, max_value=25000, step=500)

# Feasibility threshold input for dynamic testing
threshold = st.slider("Set Feasibility Cost Threshold ($/kg)", min_value=50, max_value=200.0, value=10.0, step=0.5)

# Prepare the input for prediction
input_data = pd.DataFrame({
    'Electrolysis_Type': [electrolysis_type],
    'Voltage_V': [voltage],
    'Current_Density_mA_cm2': [current_density],
    'Temperature_C': [temperature],
    'CO2_Conversion_Rate': [CO2_conversion_rate],
    'CO2_Source': [CO2_source],
    'Electricity_Cost_per_kWh': [electricity_cost],
    'H2_Required_kg': [H2_required],
    'CAPEX_$': [capex],
    'Carbon_Intensity_gCO2_MJ': [0.0]  # Placeholder value
})

try:
    # Predict SAF Yield
    saf_yield = model_pipeline.predict(input_data)[0]
    
    # Calculate cost
    cost_per_kg = calculate_cost(saf_yield, electricity_cost, H2_required, capex)
    
    # Check feasibility based on adjustable threshold
    feasibility = is_feasible(cost_per_kg, threshold)
    
    # Display results
    st.subheader("Results")
    st.write(f"**Predicted SAF Yield (kg):** {saf_yield:.2f}")
    st.write(f"**Estimated Production Cost ($ per kg):** {cost_per_kg:.2f}")
    st.write(f"**Production Feasibility (Threshold ${threshold:.2f}/kg):** {'Feasible' if feasibility else 'Not Feasible'}")
except Exception as e:
    st.write("An error occurred during prediction:", e)

# Additional Information
st.write("---")
st.write("**Explanation:**")
st.write("This app predicts the sustainable aviation fuel (SAF) yield based on input parameters, "
         "calculates the production cost per kg, and checks if the production is feasible "
         "based on a threshold cost value. Adjust the inputs to see how different settings impact feasibility.")
