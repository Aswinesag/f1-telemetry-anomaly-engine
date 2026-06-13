import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import json
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from kafka import KafkaConsumer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.virtual_sensor import HybridVirtualSensor
from src.models.autoencoder import AnomalyAutoencoder

app = FastAPI(title="F1 Inference Engine")

# CORS for local Next.js development
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Load Configurations & Models (Same logic from your Streamlit app)
ST_CONFIG_PATH = "config/config.yaml"
with open(ST_CONFIG_PATH, "r") as file:
    system_config = yaml.safe_load(file)

device = "cuda" if torch.cuda.is_available() else "cpu"
sensor_payload = torch.load("data/virtual_sensor.pt", map_location=device, weights_only=False)
virtual_sensor = HybridVirtualSensor(input_dim=sensor_payload['input_dim'], hidden_dim=sensor_payload['hidden_dim'], sequence_length=sensor_payload['sequence_length'])
virtual_sensor.load_state_dict(sensor_payload['state_dict'])
virtual_sensor.to(device).eval()

ae_payload = torch.load("data/isolation_engine.pt", map_location=device, weights_only=False)
autoencoder = AnomalyAutoencoder(input_dim=1)
autoencoder.load_state_dict(ae_payload['state_dict'])
autoencoder.to(device).eval()

scaler = sensor_payload['scalar_metadata']['scaler']
alert_threshold = ae_payload['alert_threshold']
seq_len = system_config["model_hyperparameters"]["sequence_length"]
feature_cols = system_config["features"]["raw_channels"] + system_config["features"]["physics_engineered"]

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    consumer = KafkaConsumer(
        "f1-telemetry-bus",
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    stream_buffer = []

    try:
        while True:
            # Non-blocking poll
            raw_messages = consumer.poll(timeout_ms=100)
            
            for topic_partition, messages in raw_messages.items():
                for message in messages:
                    stream_buffer.append(message.value)
            
            if len(stream_buffer) > seq_len * 2:
                stream_buffer = stream_buffer[-seq_len:]
                
            if len(stream_buffer) == seq_len:
                raw_window_df = pd.DataFrame(stream_buffer)
                
                # Dynamic Feature Engineering
                raw_window_df["Speed_ms"] = raw_window_df["Speed"] / 3.6
                raw_window_df["Delta_KE"] = raw_window_df["Speed_ms"].pow(2).diff().fillna(0.0)
                raw_window_df["Brake_Work_EMA"] = (raw_window_df["Brake"] * raw_window_df["Speed_ms"]).ewm(alpha=0.05, adjust=False).mean()
                
                if "Brake_Temp_Target" not in raw_window_df.columns:
                     raw_window_df["Brake_Temp_Target"] = 180.0 + (raw_window_df["Brake_Work_EMA"] * 1.8) - (raw_window_df["Delta_KE"] * 0.4)
                
                scaled_df = raw_window_df.copy()
                scaling_features = system_config["features"]["raw_channels"] + system_config["features"]["physics_engineered"]
                scaled_df[scaling_features] = scaler.transform(raw_window_df[scaling_features])
                
                input_tensor = torch.tensor(scaled_df[feature_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred_temp = virtual_sensor(input_tensor).item()
                    actual_temp = float(raw_window_df.iloc[-1]["Brake_Temp_Target"])
                    anomaly_score, _ = autoencoder.calculate_reconstruction_loss(torch.tensor([[np.abs(actual_temp - pred_temp)]], dtype=torch.float32).to(device))
                    anomaly_score = anomaly_score.item()
                
                # Push the cleanly formatted payload to the Next.js UI
                payload = {
                    "TimeSec": raw_window_df.iloc[-1]["TimeSec"],
                    "Speed": raw_window_df.iloc[-1]["Speed"],
                    "Brake": raw_window_df.iloc[-1]["Brake"],
                    "Predicted_Temp": pred_temp,
                    "Actual_Temp": actual_temp,
                    "Anomaly_Score": anomaly_score,
                    "Is_Anomaly": anomaly_score > alert_threshold
                }
                
                await websocket.send_json(payload)
                
            await asyncio.sleep(0.05) # Yield control back to the event loop

    except Exception as e:
        print(f"WebSocket closed: {e}")
    finally:
        consumer.close()