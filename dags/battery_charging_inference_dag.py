"""
Airflow DAG: Battery Charging Inference Pipeline
Flow: CSV → Features → XGBoost Anomaly Detection → RL Agent Prediction → Results
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
#from airflow.utils.dates import days_ago
import pendulum


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path("/usr/local/airflow/include")
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

INPUT_CSV = DATA_DIR / "battery20.csv"
Q_MODEL_PATH = MODEL_DIR / "q_learning_battery_model.pkl"
XGBOOST_MODEL_PATH = MODEL_DIR / "xgboost_anomaly_model.json"
INFERENCE_OUTPUT = RESULTS_DIR / "inference_results.json"
TRANSFORMED_DATA = RESULTS_DIR / "transformed_data.csv"
ANOMALY_DATA = RESULTS_DIR / "anomaly_detected_data.csv"

# ============================================================================
# TASK 1: LOAD AND CREATE FEATURES
# ============================================================================

def create_featured_dataset(**context):
    """
    Create featured dataset with selected columns and their derivatives
    """
    print("="*70)
    print("TASK 1: FEATURE ENGINEERING")
    print("="*70)
    
    try:
        # Load data
        df = pd.read_csv(INPUT_CSV)
        print(f"\n✓ Loaded data shape: {df.shape}")
        
        # Select required columns
        selected_cols = ['mode', 'voltage_charger', 'temperature_battery', 
                         'voltage_load', 'current_load', 'temperature_mosfet', 
                         'temperature_resistor']
        
        featured_df = df[selected_cols].copy()
        
        # Convert to numeric
        print("\n[1/3] Converting columns to numeric...")
        for col in selected_cols:
            if col != 'mode':
                featured_df[col] = pd.to_numeric(featured_df[col], errors='coerce')
        
        featured_df['mode'] = pd.to_numeric(featured_df['mode'], errors='coerce').astype('Int64')
        print("✓ All columns converted to numeric")
        
        # Calculate derivatives
        print("\n[2/3] Calculating derivatives...")
        derivative_cols = ['voltage_charger', 'temperature_battery', 
                          'voltage_load', 'current_load', 
                          'temperature_mosfet', 'temperature_resistor']
        
        for col in derivative_cols:
            derivative_col_name = f'{col}_derivative'
            featured_df[derivative_col_name] = featured_df[col].diff()
            
            # Handle mode-specific nulls
            if col in ['voltage_load', 'current_load', 'temperature_mosfet', 'temperature_resistor']:
                mask = (featured_df['mode'] == 0) | (featured_df['mode'] == 1)
                featured_df.loc[mask, derivative_col_name] = None
        
        print("✓ Derivatives calculated")
        
        # Drop rows with null derivatives
        print("\n[3/3] Cleaning data...")
        initial_rows = len(featured_df)
        
        rows_to_drop = []
        for col in derivative_cols:
            derivative_col_name = f'{col}_derivative'
            
            if col in ['voltage_load', 'current_load', 'temperature_mosfet', 'temperature_resistor']:
                mask = (featured_df['mode'] == -1) & \
                       (featured_df[col].notna()) & \
                       (featured_df[derivative_col_name].isna())
            else:
                mask = (featured_df[col].notna()) & \
                       (featured_df[derivative_col_name].isna())
            
            rows_to_drop.extend(featured_df[mask].index.tolist())
        
        featured_df = featured_df.drop(list(set(rows_to_drop)))
        rows_dropped = initial_rows - len(featured_df)
        
        print(f"✓ Dropped {rows_dropped} rows with null derivatives")
        
        # Handle remaining nulls with forward fill
        featured_df = featured_df.fillna(method='ffill').fillna(0)
        
        # Save transformed data
        featured_df.to_csv(TRANSFORMED_DATA, index=False)
        print(f"✓ Saved to {TRANSFORMED_DATA}")
        
        # Store info in XCom
        context['task_instance'].xcom_push(
            key='featured_rows',
            value=len(featured_df)
        )
        context['task_instance'].xcom_push(
            key='feature_columns',
            value=featured_df.columns.tolist()
        )
        
        print(f"\nFeature Engineering Summary:")
        print(f"  Input shape: {df.shape}")
        print(f"  Output shape: {featured_df.shape}")
        print(f"  Features: {len(featured_df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in feature engineering: {str(e)}")
        raise

# ============================================================================
# TASK 2: XGBOOST ANOMALY DETECTION
# ============================================================================

def run_xgboost_anomaly_detection(**context):
    """
    Run XGBoost model to detect anomalies
    """
    print("\n" + "="*70)
    print("TASK 2: XGBOOST ANOMALY DETECTION")
    print("="*70)
    
    try:
        # Load featured data
        featured_df = pd.read_csv(TRANSFORMED_DATA)
        print(f"\n✓ Loaded featured data: {featured_df.shape}")
        
        # Define feature columns for model
        feature_cols = [
            'mode', 'voltage_charger', 'temperature_battery', 
            'voltage_load', 'current_load', 'temperature_mosfet', 'temperature_resistor',
            'voltage_charger_derivative', 'temperature_battery_derivative',
            'voltage_load_derivative', 'current_load_derivative',
            'temperature_mosfet_derivative', 'temperature_resistor_derivative'
        ]
        
        X = featured_df[feature_cols].copy()
        
        # Load XGBoost model
        if XGBOOST_MODEL_PATH.exists():
            import xgboost as xgb
            
            model = xgb.Booster()
            model.load_model(str(XGBOOST_MODEL_PATH))
            print(f"✓ Loaded XGBoost model from {XGBOOST_MODEL_PATH}")
            
            # Prepare data
            dmatrix = xgb.DMatrix(X)
            
            # Get predictions
            anomaly_scores = model.predict(dmatrix)
            anomaly_predictions = (anomaly_scores > 0.5).astype(int)
            
            print(f"✓ Generated anomaly predictions")
            
        else:
            print(f"⚠ XGBoost model not found, using default predictions")
            anomaly_scores = np.random.rand(len(featured_df))
            anomaly_predictions = (anomaly_scores > 0.5).astype(int)
        
        # Add predictions to dataframe
        result_df = featured_df.copy()
        result_df['anomaly_score'] = anomaly_scores
        result_df['is_anomaly'] = anomaly_predictions
        
        # Detect specific anomaly types
        print("\nDetecting specific anomaly types...")
        
        result_df['sudden_voltage_drop'] = (
            result_df['voltage_charger_derivative'] < -0.003
        ).astype(int)
        
        result_df['rapid_temp_rise'] = (
            result_df['temperature_battery_derivative'] > 0.066
        ).astype(int)
        
        result_df['abnormal_discharge'] = (
            (result_df['mode'] == -1) & 
            (result_df['current_load'] > result_df['current_load'].quantile(0.9)) &
            (result_df['voltage_load_derivative'] < -0.01)
        ).fillna(False).astype(int)
        
        result_df['overheating_risk'] = (
            result_df['temperature_battery'] > 60
        ).astype(int)
        
        result_df['hardware_anomaly'] = (
            (result_df['temperature_battery'] > 70) |
            (result_df['temperature_battery'] < -20) |
            (result_df['voltage_charger'] < 5) |
            (result_df['voltage_charger'] > 8.6)
        ).astype(int)
        
        # Calculate Battery Health Index (BHI)
        print("Calculating Battery Health Index (BHI)...")
        
        # Voltage health (30% weight)
        voltage_normalized = np.clip(
            (result_df['voltage_charger'] - 5) / (8.6 - 5), 0, 1
        )
        voltage_health = 100 * (1 - np.abs(voltage_normalized - 0.5) * 2)
        
        # Temperature health (30% weight)
        temp_normalized = np.clip(
            (result_df['temperature_battery'] + 20) / (70 + 20), 0, 1
        )
        temp_health = 100 * (1 - np.abs(temp_normalized - 0.4) * 2.5)
        temp_health = np.clip(temp_health, 0, 100)
        
        # Stability health (20% weight)
        voltage_stability = 100 * (
            1 - np.clip(np.abs(result_df['voltage_charger_derivative']) / 0.1, 0, 1)
        )
        temp_stability = 100 * (
            1 - np.clip(np.abs(result_df['temperature_battery_derivative']) / 0.2, 0, 1)
        )
        stability_health = (voltage_stability + temp_stability) / 2
        
        # Anomaly-free operation (20% weight)
        anomaly_health = 100 * (1 - result_df['anomaly_score'])
        
        # Combined BHI
        result_df['BHI'] = (
            0.35 * voltage_health +
            0.25 * temp_health +
            0.20 * stability_health +
            0.20 * anomaly_health
        ).round(2)
        
        result_df['BHI'] = np.clip(result_df['BHI'], 0, 100)
        
        # Risk Classification
        def classify_risk(row):
            if row['hardware_anomaly'] == 1:
                return 'HIGH'
            if row['is_anomaly'] == 0:
                return 'LOW'
            
            if (row['sudden_voltage_drop'] == 1 or 
                row['rapid_temp_rise'] == 1 or 
                row['overheating_risk'] == 1):
                return 'HIGH'
            
            if row['anomaly_score'] >= 0.9:
                return 'HIGH'
            else:
                return 'MEDIUM'
        
        result_df['risk_level'] = result_df.apply(classify_risk, axis=1)
        
        # Save anomaly detected data
        result_df.to_csv(ANOMALY_DATA, index=False)
        print(f"✓ Saved anomaly detected data to {ANOMALY_DATA}")
        
        # Summary
        print("\n" + "-"*70)
        print("ANOMALY DETECTION SUMMARY")
        print("-"*70)
        print(f"Total records: {len(result_df)}")
        print(f"Anomalies detected: {result_df['is_anomaly'].sum()} ({result_df['is_anomaly'].mean()*100:.2f}%)")
        print(f"Normal records: {(1-result_df['is_anomaly']).sum()}")
        print(f"\nRisk Distribution:")
        print(result_df['risk_level'].value_counts())
        print(f"\nBHI Statistics:")
        print(f"  Mean: {result_df['BHI'].mean():.2f}")
        print(f"  Min: {result_df['BHI'].min():.2f}")
        print(f"  Max: {result_df['BHI'].max():.2f}")
        
        # Store in XCom
        context['task_instance'].xcom_push(
            key='anomaly_rows',
            value=int(result_df['is_anomaly'].sum())
        )
        context['task_instance'].xcom_push(
            key='bhi_values',
            value=result_df['BHI'].tolist()
        )
        
        return True
        
    except Exception as e:
        print(f"ERROR in anomaly detection: {str(e)}")
        raise

# ============================================================================
# TASK 3: RL AGENT PREDICTION
# ============================================================================

def run_rl_agent_prediction(**context):
    """
    Run RL agent to recommend charging actions
    """
    print("\n" + "="*70)
    print("TASK 3: RL AGENT ACTION RECOMMENDATION")
    print("="*70)
    
    try:
        # Load anomaly detected data
        df_anomaly = pd.read_csv(ANOMALY_DATA)
        print(f"\n✓ Loaded anomaly data: {df_anomaly.shape}")
        
        # Create state representation for RL
        print("\nPreparing RL state features...")
        
        # Normalize features for RL input
        state_features = pd.DataFrame()
        
        # State components
        state_features['soc'] = (df_anomaly['voltage_charger'] - 5) / (8.6 - 5)
        state_features['soc'] = np.clip(state_features['soc'], 0, 1)
        
        state_features['temperature'] = (df_anomaly['temperature_battery'] + 20) / (70 + 20)
        state_features['temperature'] = np.clip(state_features['temperature'], 0, 1)
        
        state_features['bhi'] = df_anomaly['BHI'] / 100
        
        state_features['anomaly_prob'] = df_anomaly['anomaly_score']
        
        state_features['voltage_deriv'] = df_anomaly['voltage_charger_derivative']
        state_features['temp_deriv'] = df_anomaly['temperature_battery_derivative']
        
        # Load RL model
        if Q_MODEL_PATH.exists():
            rl_model = joblib.load(Q_MODEL_PATH)
            print(f"✓ Loaded RL model from {Q_MODEL_PATH}")
            
            # Discretize states and get actions
            actions = []
            action_names_map = {0: 'STOP', 1: 'TRICKLE', 2: 'NORMAL', 3: 'FAST'}
            
            print("\nGenerating RL action recommendations...")
            
            for idx, row in state_features.iterrows():
                # Discretize state
                voltage_bin = np.digitize(row['soc'], bins=[0.2, 0.5, 0.8])
                temp_bin = np.digitize(row['temperature'], bins=[0.3, 0.6])
                bhi_bin = np.digitize(row['bhi'], bins=[0.4, 0.6, 0.8])
                anomaly_bin = np.digitize(row['anomaly_prob'], bins=[0.3, 0.7])
                
                discrete_state = (voltage_bin, temp_bin, bhi_bin, anomaly_bin)
                
                # Get action from Q-table
                if hasattr(rl_model, 'q_table'):
                    if discrete_state in rl_model.q_table:
                        action = np.argmax(rl_model.q_table[discrete_state])
                    else:
                        # Default action if state not in Q-table
                        action = 2  # NORMAL charging
                else:
                    action = 2
                
                actions.append(action)
            
            df_anomaly['rl_action'] = actions
            
        else:
            print(f"⚠ RL model not found, using default actions")
            # Default strategy based on state
            def default_action(row):
                if row['is_anomaly'] == 1 or row['BHI'] < 60:
                    return 1  # TRICKLE
                elif row['BHI'] > 90:
                    return 0  # STOP
                elif row['temperature_battery'] > 50:
                    return 2  # NORMAL
                else:
                    return 3  # FAST
            
            df_anomaly['rl_action'] = df_anomaly.apply(default_action, axis=1)
        
        # Map action indices to names
        action_names = {0: 'STOP', 1: 'TRICKLE', 2: 'NORMAL', 3: 'FAST'}
        df_anomaly['action_name'] = df_anomaly['rl_action'].map(action_names)
        
        print("✓ RL actions generated")
        
        # Summary
        print("\n" + "-"*70)
        print("RL ACTION DISTRIBUTION")
        print("-"*70)
        for action in range(4):
            count = (df_anomaly['rl_action'] == action).sum()
            pct = count / len(df_anomaly) * 100
            print(f"{action_names[action]:8s}: {count:8,} ({pct:5.1f}%)")
        
        # Store in XCom
        action_dist = {
            action_names[i]: int((df_anomaly['rl_action'] == i).sum())
            for i in range(4)
        }
        context['task_instance'].xcom_push(
            key='action_distribution',
            value=action_dist
        )
        df_anomaly.to_csv(ANOMALY_DATA, index=False)
        return True
        
    except Exception as e:
        print(f"ERROR in RL prediction: {str(e)}")
        raise

# ============================================================================
# TASK 4: COMBINE AND SAVE RESULTS
# ============================================================================

# 
def combine_and_save_results(**context):
    """
    Combine all results and save for Streamlit
    """
    print("\n" + "="*70)
    print("TASK 4: COMBINING RESULTS")
    print("="*70)
    
    try:
        # Load final data with predictions
        if not ANOMALY_DATA.exists():
            print(f"ERROR: {ANOMALY_DATA} not found!")
            raise FileNotFoundError(f"Anomaly data file not found: {ANOMALY_DATA}")
        
        df_results = pd.read_csv(ANOMALY_DATA)
        print(f"✓ Loaded results: {df_results.shape}")
        print(f"  Columns: {df_results.columns.tolist()}")
        
        # Get XCom data with debugging
        featured_rows = context['task_instance'].xcom_pull(
            task_ids='feature_engineering',
            key='featured_rows'
        )
        print(f"✓ featured_rows from XCom: {featured_rows}")
        
        anomaly_rows = context['task_instance'].xcom_pull(
            task_ids='anomaly_detection',
            key='anomaly_rows'
        )
        print(f"✓ anomaly_rows from XCom: {anomaly_rows}")
        
        action_dist = context['task_instance'].xcom_pull(
            task_ids='rl_prediction',
            key='action_distribution'
        )
        print(f"✓ action_dist from XCom: {action_dist}")
        
        # Validate data exists
        if featured_rows is None:
            featured_rows = len(df_results)
            print(f"⚠ featured_rows was None, using: {featured_rows}")
        
        if anomaly_rows is None:
            anomaly_rows = int(df_results['is_anomaly'].sum())
            print(f"⚠ anomaly_rows was None, using: {anomaly_rows}")
        
        if action_dist is None:
            action_dist = {}
            print(f"⚠ action_dist was None, using empty dict")
        
        # Calculate metrics
        total_records = len(df_results)
        anomaly_rate = float(df_results['is_anomaly'].mean() * 100) if len(df_results) > 0 else 0.0
        avg_bhi = float(df_results['BHI'].mean()) if 'BHI' in df_results.columns and len(df_results) > 0 else 0.0
        
        # Risk distribution
        if 'risk_level' in df_results.columns:
            risk_dist = df_results['risk_level'].value_counts().to_dict()
        else:
            risk_dist = {}
        
        # Sample data
        sample_cols = ['voltage_charger', 'temperature_battery', 'BHI', 'is_anomaly', 'risk_level', 'action_name']
        available_cols = [col for col in sample_cols if col in df_results.columns]
        sample_data = df_results[available_cols].head(20).to_dict(orient='records')
        
        # Create summary with explicit structure
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_records': int(total_records),
            'features_created': int(featured_rows),
            'anomalies_detected': int(anomaly_rows),
            'pipeline_stages': {
                '1_feature_engineering': True,
                '2_xgboost_anomaly_detection': True,
                '3_rl_agent_prediction': True
            }
        }
        
        results = {
            'anomaly_rate': anomaly_rate,
            'avg_bhi': avg_bhi,
            'risk_distribution': risk_dist,
            'rl_actions': action_dist if isinstance(action_dist, dict) else {},
            'sample_data': sample_data
        }
        
        summary['results'] = results
        
        print(f"\n[DEBUG] Summary structure:")
        print(f"  timestamp: {summary['timestamp']}")
        print(f"  total_records: {summary['total_records']}")
        print(f"  features_created: {summary['features_created']}")
        print(f"  anomalies_detected: {summary['anomalies_detected']}")
        print(f"  anomaly_rate: {results['anomaly_rate']:.2f}%")
        print(f"  avg_bhi: {results['avg_bhi']:.2f}")
        
        # Ensure directory exists
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save complete results
        print(f"\nSaving to {INFERENCE_OUTPUT}...")
        with open(INFERENCE_OUTPUT, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Verify file was written
        file_size = INFERENCE_OUTPUT.stat().st_size
        print(f"✓ File saved successfully")
        print(f"  File size: {file_size} bytes")
        
        # Verify by reading back
        with open(INFERENCE_OUTPUT, 'r') as f:
            verify_data = json.load(f)
        print(f"✓ Verification successful: {len(verify_data)} keys in JSON")
        
        # Print pipeline summary
        print("\n" + "="*70)
        print("INFERENCE PIPELINE COMPLETE")
        print("="*70)
        print(f"\nPipeline Summary:")
        print(f"  Stage 1 (Features): {featured_rows:,} records")
        print(f"  Stage 2 (Anomalies): {anomaly_rows:,} detected")
        print(f"  Stage 3 (RL Actions): Recommended for {total_records:,} records")
        print(f"\nKey Metrics:")
        print(f"  Anomaly Rate: {results['anomaly_rate']:.2f}%")
        print(f"  Avg BHI: {results['avg_bhi']:.2f}")
        print(f"  Risk Levels: {risk_dist}")
        print(f"  RL Actions: {action_dist}")
        print(f"\nOutput file: {INFERENCE_OUTPUT}")
        print(f"Output size: {file_size} bytes")
        
        return True
        
    except Exception as e:
        print(f"ERROR combining results: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# ============================================================================
# DAG DEFINITION
# ============================================================================

default_args = {
    'owner': 'battery-team',
    'start_date': pendulum.now().subtract(days=1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='battery_inference_pipeline',
    default_args=default_args,
    description='Features → XGBoost → RL Agent Inference Pipeline',
    schedule='@daily',
    catchup=False,
    tags=['battery', 'inference', 'xgboost', 'rl-agent']
)

# ============================================================================
# TASKS
# ============================================================================

task_features = PythonOperator(
    task_id='feature_engineering',
    python_callable=create_featured_dataset,
    dag=dag
)

task_anomaly = PythonOperator(
    task_id='anomaly_detection',
    python_callable=run_xgboost_anomaly_detection,
    dag=dag
)

task_rl = PythonOperator(
    task_id='rl_prediction',
    python_callable=run_rl_agent_prediction,
    dag=dag
)

task_combine = PythonOperator(
    task_id='combine_results',
    python_callable=combine_and_save_results,
    dag=dag
)

# ============================================================================
# DEPENDENCIES: Features → Anomaly → RL → Results
# ============================================================================

task_features >> task_anomaly >> task_rl >> task_combine