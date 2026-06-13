import os
import yaml
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from kafka import KafkaConsumer
import json

# Ensure configuration values are locked down
ST_CONFIG_PATH = "config/config.yaml"
SENSOR_MODEL_PATH = "data/virtual_sensor.pt"
AE_MODEL_PATH = "data/isolation_engine.pt"

@st.cache_resource
def bootstrap_neural_engines():
    """Loads and caches both PyTorch model state checkpoints for real-time inference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Safely load configurations and models
    with open(ST_CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
        
    # Reconstruct networks using our saved checkpoint architectures
    sensor_payload = torch.load(SENSOR_MODEL_PATH, map_location=device, weights_only=False) # 👈 ADD weights_only=False HERE
    from src.models.virtual_sensor import HybridVirtualSensor
    virtual_sensor = HybridVirtualSensor(
        input_dim=sensor_payload['input_dim'],
        hidden_dim=sensor_payload['hidden_dim'],
        sequence_length=sensor_payload['sequence_length']
    )
    virtual_sensor.load_state_dict(sensor_payload['state_dict'])
    virtual_sensor.to(device).eval()
    
    ae_payload = torch.load(AE_MODEL_PATH, map_location=device, weights_only=False) # 👈 ADD weights_only=False HERE

    from src.models.autoencoder import AnomalyAutoencoder
    autoencoder = AnomalyAutoencoder(input_dim=1)
    autoencoder.load_state_dict(ae_payload['state_dict'])
    autoencoder.to(device).eval()
    
    return virtual_sensor, autoencoder, ae_payload['alert_threshold'], sensor_payload['scalar_metadata']['scaler'], config

# 1. Setup Page Configurations
st.set_page_config(page_title="F1 Pit Wall AI Analytics Engine", layout="wide", page_icon="🏎️")
st.markdown("<h2 style='text-align: center; color: #FF1801;'>🏎️ F1 VIRTUAL THERMAL SENSOR & ANOMALY ISOLATION SUITE</h2>", unsafe_allow_html=True)

# Verify models exist before launching consumer stream
if not (os.path.exists(SENSOR_MODEL_PATH) and os.path.exists(AE_MODEL_PATH)):
    st.error("❌ Deep Learning weights missing in data/ directory. Please run 'python train.py' to generate checkpoints first!")
    st.stop()

# Load models and configurations into memory
virtual_sensor, autoencoder, alert_threshold, scaler, system_config = bootstrap_neural_engines()
feature_cols = system_config["features"]["raw_channels"] + system_config["features"]["physics_engineered"]
seq_len = system_config["model_hyperparameters"]["sequence_length"]

# 2. Configure Kafka Live Stream Consumer Connection
kafka_broker = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
@st.cache_resource
def get_kafka_consumer():
    return KafkaConsumer(
        "f1-telemetry-bus",
        bootstrap_servers=kafka_broker,
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        consumer_timeout_ms=1000 # Break loop if stream pauses
    )

consumer = get_kafka_consumer()

# Initialize localized runtime memory buffers using Streamlit state values
if "stream_buffer" not in st.session_state:
    st.session_state.stream_buffer = []
if "ui_history" not in st.session_state:
    st.session_state.ui_history = pd.DataFrame(columns=["TimeSec", "Speed", "Brake", "Predicted_Temp", "Actual_Temp", "Anomaly_Score"])

# 3. Create Pit Wall UI Container Grid Layout
metric_row = st.columns(4)
chart_row = st.columns([2, 1])

# Continuous stream consumption and screen-rendering loop
for message in consumer:
    frame = message.value
    st.session_state.stream_buffer.append(frame)
    
    # Bound sliding buffer window to prevent memory leaks
    if len(st.session_state.stream_buffer) > seq_len * 3:
        st.session_state.stream_buffer.pop(0)
        
    # Trigger AI calculations once enough sequential telemetry frames are buffered
    if len(st.session_state.stream_buffer) >= seq_len:
        window_df = pd.DataFrame(st.session_state.stream_buffer[-seq_len:])
        
        # Format current sequence slice into a 3D input tensor [Batch=1, Seq, Features]
        input_tensor = torch.tensor(window_df[feature_cols].values, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            # Estimate core brake temperature via the Virtual Sensor
            pred_temp_scalar = virtual_sensor(input_tensor).item()
            
            # Extract real target stream value and evaluate current residual error
            actual_temp_scalar = float(window_df.iloc[-1]["Brake_Temp_Target"])
            residual_error = np.abs(actual_temp_scalar - pred_temp_scalar)
            
            # Pass residual delta through Autoencoder to evaluate structural health
            residual_tensor = torch.tensor([[residual_error]], dtype=torch.float32)
            anomaly_loss, _ = autoencoder.calculate_reconstruction_loss(residual_tensor)
            anomaly_score = anomaly_loss.item()
            
        # Isolate baseline telemetry fields for display metrics
        current_time = window_df.iloc[-1]["TimeSec"]
        current_speed = window_df.iloc[-1]["Speed"] * 360 # Invert scaled values for display
        current_brake = window_df.iloc[-1]["Brake"] * 100
        
        # Check system health status against calculated autoencoder alert thresholds
        is_anomaly = anomaly_score > alert_threshold
        
        # Append current evaluations directly into layout tracking histories
        new_row = pd.DataFrame([{
            "TimeSec": current_time, "Speed": current_speed, "Brake": current_brake,
            "Predicted_Temp": pred_temp_scalar, "Actual_Temp": actual_temp_scalar, "Anomaly_Score": anomaly_score
        }])
        st.session_state.ui_history = pd.concat([st.session_state.ui_history, new_row]).tail(100) # Track last 100 points
        
        # 4. Render Active Live UI Control Room Elements
        with metric_row[0]:
            st.metric(label="🏎️ Vehicle Velocity", value=f"{int(current_speed)} km/h")
        with metric_row[1]:
            st.metric(label="🛑 Brake Pressure Application", value=f"{int(current_brake)} %")
        with metric_row[2]:
            st.metric(label="🌡️ Virtual Core Brake Temperature", value=f"{pred_temp_scalar:.1f} °C")
        with metric_row[3]:
            status_color = "🔴 FAILURE CRITICAL" if is_anomaly else "🟢 NOMINAL"
            st.metric(label="🚨 Telemetry Health Status", value=status_color, delta=f"Score: {anomaly_score:.4f}")
            
        # Add visual flashing structural danger warning alert banner when anomaly triggers
        if is_anomaly:
            st.error(f"⚠️ PIT WALL CRITICAL ALERT: Brake structural anomaly identified! Reconstruction Outlier Score ({anomaly_score:.4f}) exceeds threshold limits ({alert_threshold:.4f}). Check cooling ducts immediately!")
            
        # Render dynamic real-time performance line graphs
        with chart_row[0]:
            fig_thermal = px.line(
                st.session_state.ui_history, x="TimeSec", y=["Actual_Temp", "Predicted_Temp"],
                title="Live Thermal Profile: Physical Telemetry Stream vs. AI Virtual Predictions",
                labels={"value": "Celsius (°C)", "TimeSec": "Timeline Offset (s)"},
                color_discrete_map={"Actual_Temp": "#FF1801", "Predicted_Temp": "#00FF66"}
            )
            fig_thermal.update_layout(template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_thermal, use_container_width=True)
            
        with chart_row[1]:
            fig_anomaly = px.line(
                st.session_state.ui_history, x="TimeSec", y="Anomaly_Score",
                title="Isolation Engine Reconstruction Loss Profile",
                labels={"Anomaly_Score": "Loss Vector", "TimeSec": "Timeline Offset (s)"},
                color_discrete_sequence=["#FFA500"]
            )
            fig_anomaly.add_hline(y=alert_threshold, line_dash="dash", line_color="red", annotation_text="Alert Threshold Limit")
            fig_anomaly.update_layout(template="plotly_dark")
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
        # Add a microscopic rerun delay loop to synchronize UI drawing frames
        st.rerun()