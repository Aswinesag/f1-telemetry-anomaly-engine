import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from kafka import KafkaConsumer
import json
import time

# FORCE PYTHON TO RECOGNISE THE PROJECT ROOT DIRECTORY
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.virtual_sensor import HybridVirtualSensor
from src.models.autoencoder import AnomalyAutoencoder

# Define file paths centrally
ST_CONFIG_PATH = "config/config.yaml"
SENSOR_MODEL_PATH = "data/virtual_sensor.pt"
AE_MODEL_PATH = "data/isolation_engine.pt"

@st.cache_resource
def bootstrap_neural_engines():
    """Loads and caches both PyTorch model state checkpoints for real-time inference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(ST_CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
        
    sensor_payload = torch.load(SENSOR_MODEL_PATH, map_location=device, weights_only=False)
    virtual_sensor = HybridVirtualSensor(
        input_dim=sensor_payload['input_dim'],
        hidden_dim=sensor_payload['hidden_dim'],
        sequence_length=sensor_payload['sequence_length']
    )
    virtual_sensor.load_state_dict(sensor_payload['state_dict'])
    virtual_sensor.to(device).eval()
    
    ae_payload = torch.load(AE_MODEL_PATH, map_location=device, weights_only=False)
    autoencoder = AnomalyAutoencoder(input_dim=1)
    autoencoder.load_state_dict(ae_payload['state_dict'])
    autoencoder.to(device).eval()
    
    return virtual_sensor, autoencoder, ae_payload['alert_threshold'], sensor_payload['scalar_metadata']['scaler'], config

# 1. Page Configuration
st.set_page_config(page_title="F1 Pit Wall AI Analytics Engine", layout="wide", page_icon="🏎️")
st.markdown("<h2 style='text-align: center; color: #FF1801;'>🏎️ F1 VIRTUAL THERMAL SENSOR & ANOMALY ISOLATION SUITE</h2>", unsafe_allow_html=True)

if not (os.path.exists(SENSOR_MODEL_PATH) and os.path.exists(AE_MODEL_PATH)):
    st.error("❌ Deep Learning weights missing in data/. Please run 'python train.py' to generate checkpoints first!")
    st.stop()

# Initialize models
virtual_sensor, autoencoder, alert_threshold, scaler, system_config = bootstrap_neural_engines()
feature_cols = system_config["features"]["raw_channels"] + system_config["features"]["physics_engineered"]
seq_len = system_config["model_hyperparameters"]["sequence_length"]

# 2. Kafka Connection
kafka_broker = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
@st.cache_resource
def get_kafka_consumer():
    return KafkaConsumer(
        "f1-telemetry-bus",
        bootstrap_servers=kafka_broker,
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        consumer_timeout_ms=100  # Low timeout prevents the dashboard from freezing
    )

consumer = get_kafka_consumer()

# Initialize localized state persistence buffers
if "stream_buffer" not in st.session_state:
    st.session_state.stream_buffer = []
if "ui_history" not in st.session_state:
    st.session_state.ui_history = pd.DataFrame(columns=["TimeSec", "Speed", "Brake", "Predicted_Temp", "Actual_Temp", "Anomaly_Score"])

# 3. Create a single empty container wrapper to overwrite values dynamically
placeholder = st.empty()

# 4. Asynchronous Live Engine Consumption Loop
while True:
    # Poll for any new packets waiting on the Kafka topic broker
    raw_messages = consumer.poll(timeout_ms=50)
    
    new_data_received = False
    for topic_partition, messages in raw_messages.items():
        for message in messages:
            frame = message.value
            st.session_state.stream_buffer.append(frame)
            new_data_received = True

    # Limit baseline boundary queue allocation sizes to prevent memory creeping
    if len(st.session_state.stream_buffer) > seq_len * 3:
        st.session_state.stream_buffer = st.session_state.stream_buffer[-seq_len*2:]

    # Run inference only when new metrics are buffered and sequence length matches requirements
    if new_data_received and len(st.session_state.stream_buffer) >= seq_len:
        window_df = pd.DataFrame(st.session_state.stream_buffer[-seq_len:])
        
        # Format current matrix sequences into 3D structural inputs
        input_tensor = torch.tensor(window_df[feature_cols].values, dtype=torch.float32).unsqueeze(0)
        device = next(virtual_sensor.parameters()).device
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            pred_temp_scalar = virtual_sensor(input_tensor).item()
            actual_temp_scalar = float(window_df.iloc[-1]["Brake_Temp_Target"])
            residual_error = np.abs(actual_temp_scalar - pred_temp_scalar)
            
            residual_tensor = torch.tensor([[residual_error]], dtype=torch.float32).to(device)
            anomaly_loss, _ = autoencoder.calculate_reconstruction_loss(residual_tensor)
            anomaly_score = anomaly_loss.item()
            
        current_time = window_df.iloc[-1]["TimeSec"]
        current_speed = window_df.iloc[-1]["Speed"] * 360  # Invert scaling configuration mapping
        current_brake = window_df.iloc[-1]["Brake"] * 100
        is_anomaly = anomaly_score > alert_threshold
        
        new_row = pd.DataFrame([{
            "TimeSec": current_time, "Speed": current_speed, "Brake": current_brake,
            "Predicted_Temp": pred_temp_scalar, "Actual_Temp": actual_temp_scalar, "Anomaly_Score": anomaly_score
        }])
        st.session_state.ui_history = pd.concat([st.session_state.ui_history, new_row]).tail(100)

        # 5. Non-Blocking Virtual Render Block Replacement
        with placeholder.container():
            metric_row = st.columns(4)
            metric_row[0].metric(label="🏎️ Vehicle Velocity", value=f"{int(current_speed)} km/h")
            metric_row[1].metric(label="🛑 Brake Application", value=f"{int(current_brake)} %")
            metric_row[2].metric(label="🌡️ Virtual Thermal State", value=f"{pred_temp_scalar:.1f} °C")
            status_text = "🔴 CRITICAL FAILURE" if is_anomaly else "🟢 NOMINAL"
            metric_row[3].metric(label="🚨 Telemetry Health", value=status_text, delta=f"Score: {anomaly_score:.4f}", delta_color="inverse" if is_anomaly else "normal")
            
            if is_anomaly:
                st.error(f"⚠️ CRITICAL SYSTEM ALARM: Anomaly score ({anomaly_score:.4f}) breaches background threshold safety limits ({alert_threshold:.4f})! Inspect physical brake ducts.")
                
            fig_thermal = px.line(
                st.session_state.ui_history, x="TimeSec", y=["Actual_Temp", "Predicted_Temp"],
                title="Live Performance Tracks: Real-time Sensors vs. AI Predictions",
                labels={"value": "Celsius (°C)", "TimeSec": "Stint Timeline Offset (s)"},
                color_discrete_map={"Actual_Temp": "#FF1801", "Predicted_Temp": "#00FF66"}
            )
            fig_thermal.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_thermal, use_container_width=True)
            
            fig_anomaly = px.line(
                st.session_state.ui_history, x="TimeSec", y="Anomaly_Score",
                title="Isolation Engine Loss Signature Profile",
                labels={"Anomaly_Score": "Loss Intensity", "TimeSec": "Stint Timeline Offset (s)"},
                color_discrete_sequence=["#FFA500"]
            )
            fig_anomaly.add_hline(y=alert_threshold, line_dash="dash", line_color="red", annotation_text="Safety Threshold Boundary")
            fig_anomaly.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_anomaly, use_container_width=True)

    # Microscopic sleep window keeps the processor cool and ensures smooth frame refreshes
    time.sleep(0.01)