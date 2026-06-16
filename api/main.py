import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import json
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from kafka import KafkaConsumer
from kafka.admin import KafkaAdminClient
from kafka.errors import KafkaError, NoBrokersAvailable

# FORCE PYTHON TO RECOGNISE THE PROJECT ROOT DIRECTORY
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.virtual_sensor import HybridVirtualSensor
from src.models.autoencoder import AnomalyAutoencoder

app = FastAPI(title="F1 Inference Engine")

# CORS for local Next.js development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Configurations & Models
ST_CONFIG_PATH = "config/config.yaml"
with open(ST_CONFIG_PATH, "r") as file:
    system_config = yaml.safe_load(file)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SYSTEM] FastAPI Inference Engine initializing on: {device.upper()}")

# Bootstrap Virtual Sensor
sensor_payload = torch.load("data/virtual_sensor.pt", map_location=device, weights_only=False)
virtual_sensor = HybridVirtualSensor(
    input_dim=sensor_payload['input_dim'], 
    hidden_dim=sensor_payload['hidden_dim'], 
    sequence_length=sensor_payload['sequence_length']
)
virtual_sensor.load_state_dict(sensor_payload['state_dict'])
virtual_sensor.to(device).eval()

# Bootstrap Isolation Engine
ae_payload = torch.load("data/isolation_engine.pt", map_location=device, weights_only=False)
autoencoder = AnomalyAutoencoder(input_dim=1)
autoencoder.load_state_dict(ae_payload['state_dict'])
autoencoder.to(device).eval()

# Extract metadata
scaler = sensor_payload['scalar_metadata']['scaler']
alert_threshold = ae_payload['alert_threshold']
seq_len = system_config["model_hyperparameters"]["sequence_length"]
feature_cols = system_config["features"]["raw_channels"] + system_config["features"]["physics_engineered"]
latest_telemetry_payload = None

def get_kafka_bootstrap_servers() -> str:
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

def check_kafka_ready() -> tuple[bool, str | None]:
    bootstrap_servers = get_kafka_bootstrap_servers()
    admin_client = None

    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            request_timeout_ms=3000,
            api_version_auto_timeout_ms=3000,
            client_id="inference-api-readiness",
        )
        admin_client.list_topics()
        return True, None
    except (NoBrokersAvailable, KafkaError, OSError) as exc:
        return False, str(exc)
    finally:
        if admin_client is not None:
            admin_client.close()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "inference-api",
        "device": device,
    }

@app.get("/ready")
def ready():
    kafka_ready, kafka_error = check_kafka_ready()
    models_ready = virtual_sensor is not None and autoencoder is not None and scaler is not None

    payload = {
        "status": "ready" if kafka_ready and models_ready else "not_ready",
        "checks": {
            "models": models_ready,
            "kafka": kafka_ready,
        },
        "kafka_bootstrap_servers": get_kafka_bootstrap_servers(),
    }

    if kafka_error:
        payload["kafka_error"] = kafka_error

    if not kafka_ready or not models_ready:
        return JSONResponse(status_code=503, content=payload)

    return payload

@app.get("/telemetry/latest")
def latest_telemetry():
    if latest_telemetry_payload is None:
        return JSONResponse(
            status_code=404,
            content={
                "status": "empty",
                "message": "No telemetry has been processed yet.",
            },
        )

    return latest_telemetry_payload

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    global latest_telemetry_payload
    await websocket.accept()
    print("[WEBSOCKET] Client connected to live telemetry stream.")
    
    consumer = KafkaConsumer(
        "f1-telemetry-bus",
        bootstrap_servers=get_kafka_bootstrap_servers(),
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    stream_buffer = []

    try:
        while True:
            # Non-blocking poll batching
            raw_messages = consumer.poll(timeout_ms=100)
            
            for topic_partition, messages in raw_messages.items():
                for message in messages:
                    stream_buffer.append(message.value)
            
            # Memory boundary constraint
            if len(stream_buffer) > seq_len * 2:
                stream_buffer = stream_buffer[-seq_len:]
                
            # Trigger inference when buffer threshold is met
            if len(stream_buffer) >= seq_len:
                raw_window_df = pd.DataFrame(stream_buffer[-seq_len:])
                
                # --- ADVANCED PHYSICS ENGINE ---
                # 1. Base Kinematics
                raw_window_df["Speed_ms"] = raw_window_df["Speed"] / 3.6
                raw_window_df["Delta_KE"] = raw_window_df["Speed_ms"].pow(2).diff().fillna(0.0)
                
                # dt is 0.02 for a 50Hz telemetry stream
                raw_window_df["Acceleration"] = raw_window_df["Speed_ms"].diff().fillna(0.0) / 0.02
                raw_window_df["Longitudinal_G"] = raw_window_df["Acceleration"] / 9.81
                
                # 2. Aerodynamic Profiling
                raw_window_df["Aero_Drag_N"] = 0.5 * 1.225 * (raw_window_df["Speed_ms"] ** 2) * 1.15
                raw_window_df["Aero_Downforce_N"] = 0.5 * 1.225 * (raw_window_df["Speed_ms"] ** 2) * 3.5
                
                # 3. Dynamic Vehicle Weight
                raw_window_df["Effective_Weight_N"] = (798 * 9.81) + raw_window_df["Aero_Downforce_N"]
                
                # 4. Advanced Thermodynamics
                instantaneous_brake_work = raw_window_df["Brake"] * raw_window_df["Speed_ms"] * (raw_window_df["Effective_Weight_N"] / 10000)
                raw_window_df["Brake_Work_EMA"] = instantaneous_brake_work.ewm(alpha=0.05, adjust=False).mean()
                raw_window_df["Convective_Cooling_Factor"] = (raw_window_df["Speed_ms"] ** 0.8) * 0.05
                
                # 5. Synthesize Target
                if "Brake_Temp_Target" not in raw_window_df.columns:
                    base_temp = 180.0
                    heat_added = raw_window_df["Brake_Work_EMA"] * 2.2
                    heat_extracted = raw_window_df["Convective_Cooling_Factor"] * 1.5
                    raw_window_df["Brake_Temp_Target"] = base_temp + heat_added - heat_extracted
                # --- END ADVANCED PHYSICS ENGINE ---

                # Dynamic scaling
                scaled_df = raw_window_df.copy()
                scaling_features = system_config["features"]["raw_channels"] + system_config["features"]["physics_engineered"]
                scaled_df[scaling_features] = scaler.transform(raw_window_df[scaling_features])
                
                # Prepare PyTorch Tensors
                input_tensor = torch.tensor(scaled_df[feature_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred_temp = virtual_sensor(input_tensor).item()
                    actual_temp = float(raw_window_df.iloc[-1]["Brake_Temp_Target"])
                    
                    residual_error = np.abs(actual_temp - pred_temp)
                    residual_tensor = torch.tensor([[residual_error]], dtype=torch.float32).to(device)
                    
                    anomaly_score, _ = autoencoder.calculate_reconstruction_loss(residual_tensor)
                    anomaly_score = anomaly_score.item()
                
                # Broadcast payload to Next.js Client
                payload = {
                    "TimeSec": raw_window_df.iloc[-1]["TimeSec"],
                    "Speed": raw_window_df.iloc[-1]["Speed"],
                    "Brake": raw_window_df.iloc[-1]["Brake"],
                    "Predicted_Temp": pred_temp,
                    "Actual_Temp": actual_temp,
                    "Anomaly_Score": anomaly_score,
                    "Alert_Threshold": alert_threshold,
                    "Is_Anomaly": anomaly_score > alert_threshold
                }
                latest_telemetry_payload = payload
                
                await websocket.send_json(payload)
                
            # Yield control back to the asynchronous event loop to maintain WebSocket heartbeat
            await asyncio.sleep(0.02) 

    except Exception as e:
        print(f"[WEBSOCKET] Connection closed: {e}")
    finally:
        consumer.close()
