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
        auto_offset_reset="latest",
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    )

    # --- Persistent state for streaming feature engineering (prevents EMA/DIFF reset “amnesia”) ---
    prev_speed_ms: float | None = None
    brake_work_ema: float | None = None

    STREAM_WINDOW = seq_len
    stream_buffer: list[dict] = []

    # Match the backend’s EMA smoothing factor used previously
    EMA_ALPHA = 0.05

    try:
        while True:
            raw_messages = consumer.poll(timeout_ms=100)

            for _, messages in raw_messages.items():
                for message in messages:
                    pkt = message.value

                    # Compute streaming derived features for THIS packet only
                    speed_ms = float(pkt["Speed"]) / 3.6

                    # Delta_KE using persistent previous speed (instead of per-window .diff())
                    # Delta KE = (v^2 - v_prev^2)
                    if prev_speed_ms is None:
                        delta_ke = 0.0
                    else:
                        delta_ke = (speed_ms**2) - (prev_speed_ms**2)

                    prev_speed_ms = speed_ms

                    # Brake_Work_EMA using persistent EMA state (instead of per-window .ewm())
                    brake_val = float(pkt["Brake"])
                    brake_work = brake_val * speed_ms
                    if brake_work_ema is None:
                        brake_work_ema = brake_work
                    else:
                        brake_work_ema = (EMA_ALPHA * brake_work) + (1.0 - EMA_ALPHA) * brake_work_ema

                    # Attach derived features so the scaler gets exactly what the model expects
                    pkt = dict(pkt)  # avoid mutating Kafka payload
                    pkt["Speed_ms"] = speed_ms
                    pkt["Delta_KE"] = delta_ke
                    pkt["Brake_Work_EMA"] = brake_work_ema

                    # If the target is missing from the incoming packet, reconstruct it consistently.
                    # (Use the same formula you previously used, but based on persistent derived features.)
                    if "Brake_Temp_Target" not in pkt:
                        pkt["Brake_Temp_Target"] = 180.0 + (pkt["Brake_Work_EMA"] * 1.8) - (pkt["Delta_KE"] * 0.4)

                    stream_buffer.append(pkt)

                    # Strictly cap buffer to the last seq_len packets
                    if len(stream_buffer) > STREAM_WINDOW:
                        stream_buffer = stream_buffer[-STREAM_WINDOW:]

                    # Run inference whenever we have a full window
                    if len(stream_buffer) >= STREAM_WINDOW:
                        raw_window_df = pd.DataFrame(stream_buffer[-STREAM_WINDOW:])

                        # Scale only the features that were used in training
                        scaling_features = system_config["features"]["raw_channels"] + system_config["features"]["physics_engineered"]
                        scaled_df = raw_window_df.copy()
                        scaled_df[scaling_features] = scaler.transform(raw_window_df[scaling_features])

                        input_tensor = torch.tensor(
                            scaled_df[feature_cols].values,
                            dtype=torch.float32,
                        ).unsqueeze(0).to(device)

                        with torch.no_grad():
                            pred_temp = virtual_sensor(input_tensor).item()
                            actual_temp = float(raw_window_df.iloc[-1]["Brake_Temp_Target"])
                            anomaly_score, _ = autoencoder.calculate_reconstruction_loss(
                                torch.tensor([[np.abs(actual_temp - pred_temp)]], dtype=torch.float32).to(device)
                            )
                            anomaly_score = anomaly_score.item()

                        payload = {
                            "TimeSec": raw_window_df.iloc[-1]["TimeSec"],
                            "Speed": raw_window_df.iloc[-1]["Speed"],
                            "Brake": raw_window_df.iloc[-1]["Brake"],
                            "Predicted_Temp": pred_temp,
                            "Actual_Temp": actual_temp,
                            "Anomaly_Score": anomaly_score,
                            "Is_Anomaly": anomaly_score > alert_threshold,
                        }

                        await websocket.send_json(payload)

            await asyncio.sleep(0.05)  # Yield control back to the event loop

    except Exception as e:
        print(f"WebSocket closed: {e}")
    finally:
        consumer.close()

