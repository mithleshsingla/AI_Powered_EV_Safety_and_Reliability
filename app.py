import streamlit as st
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import xgboost as xgb

# Page configuration
st.set_page_config(
    page_title="Battery Health Monitor",
    page_icon="🔋",
    layout="wide"
)

# Title
st.title("🔋 Battery Health Monitoring System")

# Load models
@st.cache_resource
def load_models():
    """Load XGBoost and Q-Learning models"""
    try:
        # Load XGBoost model
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model('xgboost_anomaly_model.json')
        st.sidebar.success("✅ XGBoost model loaded")
        
        # Load Q-Learning model
        with open('q_learning_battery_model.pkl', 'rb') as f:
            ql_data = pickle.load(f)
            q_table = defaultdict(lambda: np.zeros(4), ql_data['q_table'])
        st.sidebar.success("✅ Q-Learning model loaded")
        
        return xgb_model, q_table, ql_data
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info(f"📁 Looking in: {os.getcwd()}")
        st.code("Required files:\n- xgboost_anomaly_model.pkl\n- q_learning_battery_model.pkl")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

# Helper functions
def calculate_bhi(voltage, temperature):
    """Calculate Battery Health Index (0-100)"""
    voltage_normalized = np.clip((voltage - 5) / (8.6 - 5), 0, 1)
    voltage_health = 100 * (1 - np.abs(voltage_normalized - 0.5) * 2)
    
    temp_normalized = np.clip((temperature + 20) / (70 + 20), 0, 1)
    temp_health = 100 * (1 - np.abs(temp_normalized - 0.4) * 2.5)
    temp_health = np.clip(temp_health, 0, 100)
    
    bhi = 0.5 * voltage_health + 0.5 * temp_health
    return np.clip(bhi, 0, 100)

def estimate_anomaly_probability(voltage, temperature, bhi):
    """Estimate anomaly probability"""
    prob = 0.0
    
    if temperature > 60:
        prob += 0.4
    elif temperature > 50:
        prob += 0.2
    elif temperature < 0:
        prob += 0.3
    
    if voltage < 5 or voltage > 8.6:
        prob += 0.5
    elif voltage < 5.5 or voltage > 8.3:
        prob += 0.2
    
    if bhi < 40:
        prob += 0.3
    elif bhi < 60:
        prob += 0.1
    
    return min(prob, 1.0)

def predict_anomaly(model, mode, voltage, temperature):
    """Predict anomaly using XGBoost"""
    try:
        # Create simplified feature vector (adjust based on your model's expected features)
        if mode == -1:
            features = [mode, voltage, temperature, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            features = [mode, voltage, temperature, np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan, np.nan, np.nan, np.nan]
        
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]
        return prediction, probability
    except Exception as e:
        st.warning(f"Prediction error: {e}. Using fallback method.")
        # Fallback
        prob = estimate_anomaly_probability(voltage, temperature, calculate_bhi(voltage, temperature))
        return (1 if prob > 0.5 else 0), prob

def get_rl_recommendation(q_table, voltage, temperature, bhi, anomaly_prob):
    """Get RL recommendation"""
    voltage_bin = np.digitize(voltage, bins=[5, 6, 7, 8, 8.6])
    temp_bin = np.digitize(temperature, bins=[-20, 0, 20, 40, 60, 70])
    bhi_bin = np.digitize(bhi, bins=[0, 40, 60, 80, 100])
    anomaly_bin = np.digitize(anomaly_prob, bins=[0, 0.3, 0.7, 1.0])
    
    discrete_state = (voltage_bin, temp_bin, bhi_bin, anomaly_bin)
    q_values = q_table[discrete_state]
    action = np.argmax(q_values)
    
    actions = {
        0: '⚡ Fast Charge',
        1: '🔋 Normal Charge',
        2: '🐌 Trickle Charge',
        3: '⏸️ Pause Charging'
    }
    
    return action, actions[action], q_values

# Main app
st.sidebar.header("📊 Input Parameters")

# Load models
with st.spinner("Loading models..."):
    xgb_model, q_table, ql_data = load_models()

if xgb_model is None or q_table is None:
    st.stop()

# Input controls
mode = st.sidebar.selectbox(
    "Operating Mode",
    options=[1, 0, -1],
    format_func=lambda x: {1: "🔌 Charging", 0: "⏸️ Idle", -1: "🔋 Discharging"}[x]
)

voltage = st.sidebar.slider("Voltage (V)", 4.0, 9.5, 7.0, 0.1)
temperature = st.sidebar.slider("Temperature (°C)", -25.0, 80.0, 25.0, 0.5)

# Calculate metrics
bhi = calculate_bhi(voltage, temperature)
anomaly_prob = estimate_anomaly_probability(voltage, temperature, bhi)

# Get predictions
anomaly_pred, anomaly_pred_prob = predict_anomaly(xgb_model, mode, voltage, temperature)

# Risk classification
if temperature > 70 or temperature < -20 or voltage < 5 or voltage > 8.6:
    risk_level = "🔴 HIGH"
    risk_color = "red"
elif anomaly_prob >= 0.7:
    risk_level = "🔴 HIGH"
    risk_color = "red"
elif anomaly_prob >= 0.3:
    risk_level = "🟡 MEDIUM"
    risk_color = "orange"
else:
    risk_level = "🟢 LOW"
    risk_color = "green"

# Display metrics
st.header("📊 Current Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Battery Health Index", f"{bhi:.1f}%", 
              delta="Good" if bhi > 80 else "Fair" if bhi > 60 else "Poor")

with col2:
    st.metric("Anomaly Risk", f"{anomaly_prob*100:.1f}%",
              delta="High" if anomaly_prob > 0.7 else "Medium" if anomaly_prob > 0.3 else "Low")

with col3:
    st.metric("Risk Level", risk_level)

with col4:
    status = "⚠️ ANOMALY" if anomaly_pred == 1 else "✅ NORMAL"
    st.metric("Detection", status, delta=f"{anomaly_pred_prob*100:.0f}% conf.")

# Detailed status
st.header("🔍 Detailed Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Battery Parameters")
    
    params_df = pd.DataFrame({
        'Parameter': ['Mode', 'Voltage', 'Temperature', 'BHI'],
        'Value': [
            {1: "Charging", 0: "Idle", -1: "Discharging"}[mode],
            f"{voltage:.2f} V",
            f"{temperature:.2f} °C",
            f"{bhi:.1f}%"
        ],
        'Status': [
            '🔌',
            '✅' if 5 <= voltage <= 8.6 else '⚠️',
            '✅' if 0 <= temperature <= 60 else '⚠️',
            '✅' if bhi > 80 else '⚠️' if bhi > 60 else '❌'
        ]
    })
    st.dataframe(params_df, hide_index=True, use_container_width=True)
    
    # Safety warnings
    st.subheader("⚠️ Safety Alerts")
    warnings = []
    
    if temperature > 60:
        warnings.append("🔥 High temperature!")
    if temperature < 0:
        warnings.append("❄️ Very low temperature!")
    if voltage < 5:
        warnings.append("⚡ Low voltage!")
    if voltage > 8.6:
        warnings.append("⚡ High voltage!")
    if anomaly_prob > 0.7:
        warnings.append("⚠️ High anomaly risk!")
    
    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.success("✅ All parameters normal")

with col2:
    st.subheader("🤖 RL Recommendation")
    
    if mode == 1:
        action, action_name, q_values = get_rl_recommendation(q_table, voltage, temperature, bhi, anomaly_prob)
        
        st.info(f"**Recommended Action:** {action_name}")
        
        st.write("**Q-Values for all actions:**")
        q_df = pd.DataFrame({
            'Action': ['⚡ Fast Charge', '🔋 Normal Charge', '🐌 Trickle Charge', '⏸️ Pause Charging'],
            'Q-Value': [f"{v:.2f}" for v in q_values],
            'Selected': ['✅' if i == action else '' for i in range(4)]
        })
        st.dataframe(q_df, hide_index=True, use_container_width=True)
        
        # Explanation
        explanations = {
            0: "Fast charging recommended - conditions are optimal",
            1: "Normal charging provides good balance",
            2: "Trickle charging is safer under current conditions",
            3: "Pause recommended - allow battery to stabilize"
        }
        st.info(f"**Why?** {explanations[action]}")
        
    else:
        st.info("ℹ️ RL recommendations only available in Charging Mode")
    
    # Progress bars
    st.subheader("📊 Visual Indicators")
    
    st.write("**Voltage**")
    voltage_pct = ((voltage - 4) / (9.5 - 4)) * 100
    st.progress(min(max(int(voltage_pct), 0), 100))
    
    st.write("**Temperature**")
    temp_pct = ((temperature + 25) / (80 + 25)) * 100
    st.progress(min(max(int(temp_pct), 0), 100))
    
    st.write("**Battery Health Index**")
    st.progress(int(bhi))

# Safety thresholds
st.header("📋 Safety Thresholds")

threshold_df = pd.DataFrame({
    'Parameter': ['Voltage', 'Temperature'],
    'Min Safe': ['5.0 V', '0°C'],
    'Optimal': ['6.0 - 8.0 V', '20 - 40°C'],
    'Max Safe': ['8.6 V', '60°C'],
    'Current': [f"{voltage:.2f} V", f"{temperature:.2f}°C"],
    'Status': [
        '✅' if 5 <= voltage <= 8.6 else '❌',
        '✅' if 0 <= temperature <= 60 else '❌'
    ]
})
st.dataframe(threshold_df, hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Battery Health Monitoring System v1.0 | XGBoost + Q-Learning RL")