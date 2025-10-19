"""
Streamlit App for Battery Charging Inference Results
With diagnostic checks for missing data
"""

import streamlit as st
import pandas as pd
import json
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path("include")
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_CSV = DATA_DIR / "battery20.csv"
INFERENCE_OUTPUT = RESULTS_DIR / "inference_results.json"
TRANSFORMED_DATA = RESULTS_DIR / "transformed_data.csv"
ANOMALY_DATA = RESULTS_DIR / "anomaly_detected_data.csv"

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Battery Charging RL Inference Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .title-section {
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .error-box {
        background-color: #ffcccc;
        border: 2px solid #ff0000;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #ccffcc;
        border: 2px solid #00cc00;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def check_files():
    """Check if all required files exist."""
    status = {
        'input_csv': INPUT_CSV.exists(),
        'transformed_data': TRANSFORMED_DATA.exists(),
        'anomaly_data': ANOMALY_DATA.exists(),
        'inference_output': INFERENCE_OUTPUT.exists(),
    }
    return status

def get_file_info():
    """Get info about existing files."""
    info = {}
    
    for name, path in [
        ('input_csv', INPUT_CSV),
        ('transformed_data', TRANSFORMED_DATA),
        ('anomaly_data', ANOMALY_DATA),
        ('inference_output', INFERENCE_OUTPUT),
    ]:
        if path.exists():
            info[name] = {
                'exists': True,
                'size_mb': path.stat().st_size / (1024*1024),
                'modified': datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            info[name] = {'exists': False}
    
    return info

@st.cache_data
def load_inference_results():
    """Load inference results from JSON."""
    try:
        if not INFERENCE_OUTPUT.exists():
            return None
        
        file_size = INFERENCE_OUTPUT.stat().st_size
        if file_size == 0:
            return None
        
        with open(INFERENCE_OUTPUT, 'r') as f:
            data = json.load(f)
            if not data or len(data) == 0:
                return None
            return data
            
    except json.JSONDecodeError as e:
        st.warning(f"JSON decode error: {str(e)}")
        return None
    except Exception as e:
        st.warning(f"Error loading JSON: {str(e)}")
    return None

@st.cache_data
def load_anomaly_data():
    """Load anomaly detection data."""
    try:
        if ANOMALY_DATA.exists():
            return pd.read_csv(ANOMALY_DATA)
    except Exception as e:
        st.error(f"Error loading anomaly data: {str(e)}")
    return None

@st.cache_data
def load_battery_data():
    """Load original battery data."""
    try:
        if INPUT_CSV.exists():
            return pd.read_csv(INPUT_CSV)
    except Exception as e:
        st.error(f"Error loading battery data: {str(e)}")
    return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_action_distribution_chart(action_dist):
    """Create action distribution pie chart."""
    if not action_dist:
        return None
    
    fig = go.Figure(data=[go.Pie(
        labels=list(action_dist.keys()),
        values=list(action_dist.values()),
        marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    )])
    fig.update_layout(title="RL Action Distribution", height=400)
    return fig

def create_anomaly_chart(normal, anomaly):
    """Create anomaly distribution chart."""
    fig = go.Figure(data=[
        go.Bar(x=['Normal', 'Anomaly'], y=[normal, anomaly],
               marker_color=['#2ECC71', '#E74C3C'])
    ])
    fig.update_layout(title="Anomaly Detection Results", height=400, showlegend=False)
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="title-section">
        <h1>🔋 Battery Charging RL Inference Dashboard</h1>
        <p>Real-time inference results from RL agent and anomaly detection model</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Diagnostics
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        file_status = check_files()
        file_info = get_file_info()
        
        # File status indicator
        all_files_exist = all(file_status.values())
        
        if all_files_exist:
            st.markdown('<div class="success-box">✅ All pipeline files detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">❌ Missing pipeline files</div>', unsafe_allow_html=True)
        
        st.write("**File Status:**")
        for name, exists in file_status.items():
            status_icon = "✅" if exists else "❌"
            st.write(f"{status_icon} {name}")
            if name in file_info and file_info[name].get('exists'):
                size = file_info[name]['size_mb']
                modified = file_info[name]['modified']
                st.caption(f"Size: {size:.2f}MB | Modified: {modified}")
        
        st.markdown("---")
        st.markdown("### 📊 Dashboard Navigation")
        page = st.radio("Select View:", 
            ["📈 Overview", "🤖 RL Inference", "🚨 Anomaly Detection", "📋 Data Analysis", "🔍 Diagnostics"]
        )
    
    # Load data
    inference_results = load_inference_results()
    anomaly_data = load_anomaly_data()
    battery_data = load_battery_data()
    
    # ====================================================================
    # DIAGNOSTICS PAGE (Always Available)
    # ====================================================================
    
    if page == "🔍 Diagnostics":
        st.header("System Diagnostics")
        
        st.subheader("1. File Status")
        file_info = get_file_info()
        
        for name, info in file_info.items():
            if info['exists']:
                st.success(f"✅ {name}: {info['size_mb']:.2f}MB (Modified: {info['modified']})")
            else:
                st.error(f"❌ {name}: NOT FOUND")
        
        st.subheader("2. What to do if files are missing:")
        st.markdown("""
        **Option A: Run the Airflow DAG**
        1. Go to Airflow UI: http://localhost:8080
        2. Find "battery_inference_pipeline" DAG
        3. Click the Play button to trigger
        4. Wait for completion (should take 1-2 minutes)
        5. Refresh this Streamlit app
        
        **Option B: Generate sample data**
        """)
        
        if st.button("Generate Sample Results (for testing)"):
            try:
                # Create sample data
                sample_results = {
                    'timestamp': datetime.now().isoformat(),
                    'total_records': 100,
                    'features_created': 100,
                    'anomalies_detected': 15,
                    'pipeline_stages': {
                        '1_feature_engineering': True,
                        '2_xgboost_anomaly_detection': True,
                        '3_rl_agent_prediction': True
                    },
                    'results': {
                        'anomaly_rate': 15.0,
                        'avg_bhi': 75.5,
                        'risk_distribution': {'LOW': 60, 'MEDIUM': 25, 'HIGH': 15},
                        'rl_actions': {'STOP': 10, 'TRICKLE': 25, 'NORMAL': 50, 'FAST': 15},
                        'sample_data': [
                            {
                                'voltage_charger': 6.5 + i*0.05,
                                'temperature_battery': 25 + i*2,
                                'BHI': 80 - i*0.5,
                                'is_anomaly': 1 if i % 7 == 0 else 0,
                                'risk_level': 'HIGH' if i % 7 == 0 else ('MEDIUM' if i % 3 == 0 else 'LOW'),
                                'action_name': ['STOP', 'TRICKLE', 'NORMAL', 'FAST'][i % 4]
                            }
                            for i in range(20)
                        ]
                    }
                }
                
                # Save sample results
                RESULTS_DIR.mkdir(exist_ok=True)
                with open(INFERENCE_OUTPUT, 'w') as f:
                    json.dump(sample_results, f, indent=2)
                
                st.success("✅ Sample results generated! Refresh the page to see data.")
                
            except Exception as e:
                st.error(f"Error generating sample: {str(e)}")
        
        st.subheader("3. Check Airflow Logs")
        st.markdown("""
        Run this command to see Airflow logs:
        ```bash
        astro dev logs
        ```
        """)
        
        st.subheader("4. Check Data Files")
        
        if battery_data is not None:
            st.write(f"✅ Battery data loaded: {battery_data.shape}")
            with st.expander("View sample battery data"):
                st.dataframe(battery_data.head(10))
        else:
            st.warning("❌ Battery data not found")
        
        if anomaly_data is not None:
            st.write(f"✅ Anomaly data loaded: {anomaly_data.shape}")
            with st.expander("View sample anomaly data"):
                st.dataframe(anomaly_data.head(10))
        else:
            st.warning("❌ Anomaly data not found")
        
        return
    
    # ====================================================================
    # Check if inference results exist
    # ====================================================================
    
    if inference_results is None:
        st.markdown("""
        <div class="error-box">
        <h3>❌ No Inference Results Found</h3>
        <p>The pipeline hasn't run yet. Please do one of the following:</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.error("""
        **Option 1: Run the Airflow DAG**
        1. Go to Airflow UI: http://localhost:8080
        2. Find "battery_inference_pipeline"
        3. Click Play button
        4. Wait 1-2 minutes for completion
        5. Refresh this page
        
        **Option 2: Generate Sample Data**
        Go to 🔍 Diagnostics tab → Click "Generate Sample Results"
        """)
        return
    
    # ====================================================================
    # PAGE: OVERVIEW
    # ====================================================================
    
    if page == "📈 Overview":
        st.header("Inference Pipeline Overview")
        
        summary = inference_results.get('summary', {})
        results = inference_results.get('results', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = summary.get('total_records', 0)
            st.metric("📦 Total Records", f"{total:,}")
        
        with col2:
            anomalies = summary.get('anomalies_detected', 0)
            st.metric("🚨 Anomalies", f"{anomalies:,}")
        
        with col3:
            normal = total - anomalies if total > 0 else 0
            st.metric("✅ Normal", f"{normal:,}")
        
        with col4:
            anomaly_pct = results.get('anomaly_rate', 0)
            st.metric("📊 Anomaly %", f"{anomaly_pct:.2f}%")
        
        # Timestamp
        if 'timestamp' in summary:
            st.info(f"⏱️ Last inference: {summary['timestamp']}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            rl_actions = results.get('rl_actions', {})
            if rl_actions:
                chart = create_action_distribution_chart(rl_actions)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
        
        with col2:
            normal_count = total - anomalies if total > 0 else 0
            chart = create_anomaly_chart(normal_count, anomalies)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
        
        # Risk distribution
        st.subheader("Risk Level Distribution")
        risk_dist = results.get('risk_distribution', {})
        if risk_dist:
            risk_df = pd.DataFrame(list(risk_dist.items()), columns=['Risk Level', 'Count'])
            st.bar_chart(risk_df.set_index('Risk Level'))
        
        # Sample results table
        st.subheader("Sample Results (First 10)")
        sample_data = results.get('sample_data', [])
        if sample_data:
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True)
    
    # ====================================================================
    # PAGE: RL INFERENCE
    # ====================================================================
    
    elif page == "🤖 RL Inference":
        st.header("RL Agent Action Recommendations")
        
        results = inference_results.get('results', {})
        rl_actions = results.get('rl_actions', {})
        
        col1, col2, col3, col4 = st.columns(4)
        action_names = {'STOP': '⏹️', 'TRICKLE': '🐢', 'NORMAL': '⚡', 'FAST': '🚀'}
        
        for action, col in zip(['STOP', 'TRICKLE', 'NORMAL', 'FAST'], [col1, col2, col3, col4]):
            count = rl_actions.get(action, 0)
            total = sum(rl_actions.values())
            pct = (count / total * 100) if total > 0 else 0
            
            with col:
                st.metric(f"{action_names.get(action, '')} {action}", f"{count:,}", f"{pct:.1f}%")
        
        st.markdown("**Action Meanings:**")
        st.markdown("""
        - **⏹️ STOP**: No charging - Battery fully charged or risky conditions
        - **🐢 TRICKLE**: Slow charging (~5A) - Safe maintenance
        - **⚡ NORMAL**: Standard charging (~15A) - Normal operation  
        - **🚀 FAST**: Rapid charging (~25A) - When safe
        """)
    
    # ====================================================================
    # PAGE: ANOMALY DETECTION
    # ====================================================================
    
    elif page == "🚨 Anomaly Detection":
        st.header("Anomaly Detection Results")
        
        summary = inference_results.get('summary', {})
        results = inference_results.get('results', {})
        
        total = summary.get('total_records', 0)
        anomalies = summary.get('anomalies_detected', 0)
        normal = total - anomalies
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🚨 Anomalies", f"{anomalies:,}")
        with col2:
            st.metric("✅ Normal", f"{normal:,}")
        with col3:
            st.metric("📊 Rate", f"{results.get('anomaly_rate', 0):.2f}%")
        
        # Chart
        chart = create_anomaly_chart(normal, anomalies)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        
        st.markdown("""
        **What is an anomaly?**
        - Unusual battery behavior detected by XGBoost model
        - Indicates: overheating, rapid degradation, charging faults
        
        **Average BHI:** """ + f"{results.get('avg_bhi', 0):.2f}")
    
    # ====================================================================
    # PAGE: DATA ANALYSIS
    # ====================================================================
    
    elif page == "📋 Data Analysis":
        st.header("Battery Data Analysis")
        
        if anomaly_data is not None:
            st.subheader("Data Summary")
            st.dataframe(anomaly_data.describe(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'temperature_battery' in anomaly_data.columns:
                    fig = px.histogram(anomaly_data, x='temperature_battery', nbins=30,
                                      title='Temperature Distribution')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'BHI' in anomaly_data.columns:
                    fig = px.histogram(anomaly_data, x='BHI', nbins=30,
                                      title='BHI Distribution')
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()