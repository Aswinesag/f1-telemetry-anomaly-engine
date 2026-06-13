import os
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import developed repository modules
from src.pipeline.etl_processor import F1TelemetryProcessor
from src.models.virtual_sensor import F1TelemetryDataset, HybridVirtualSensor
from src.models.autoencoder import AnomalyAutoencoder

def load_system_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def execute_training_pipeline():
    print("="*70)
    print("🚀 INITIALISING FORMULA 1 VIRTUAL SENSOR & ISOLATION TRAINING ENGINE")
    print("="*70)
    
    # 1. Load System Settings
    config = load_system_config()
    hyperparams = config["model_hyperparameters"]
    feature_cols = config["features"]["raw_channels"] + config["features"]["physics_engineered"]
    target_col = config["features"]["target_channel"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SYSTEM] Hardware Ingestion Target: {device.upper()}")
    
    # 2. Extract and Preprocess High-Frequency Telemetry
    print("\n[PHASE 1] Extracting data from FastF1 and running Cubic Spline alignment...")
    processor = F1TelemetryProcessor()
    # Profiled on Monza (high-braking circuit) using Verstappen's baseline telemetry stint
    processed_df = processor.process_session_telemetry(
        year=2023, 
        location="Monza", 
        session_type="R", 
        driver="VER"
    )
    print(f"[DATA SUCCESS] Aligned Matrix Shape: {processed_df.shape}")
    
    # Save a local cache copy to act as the seed file for the Kafka live replayer
    os.makedirs("data/raw_samples", exist_ok=True)
    processed_df.to_csv("data/raw_samples/monza_ver_cleaned.csv", index=False)
    
    # 3. Partition Sequences & Build Data Loaders
    seq_len = hyperparams["sequence_length"]
    dataset = F1TelemetryDataset(processed_df, feature_cols, target_col, sequence_length=seq_len)
    
    # Maintain chronological order for sequential validation (shuffle=False is mandatory)
    train_loader = DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=False)
    
    # 4. Instantiate and Train the Hybrid Virtual Sensor Network
    print("\n[PHASE 2] Initialising Hybrid 1D-CNN + Bidirectional LSTM Model Architecture...")
    virtual_sensor = HybridVirtualSensor(
        input_dim=len(feature_cols),
        hidden_dim=hyperparams["hidden_dimension"],
        sequence_length=seq_len,
        dropout_rate=hyperparams["dropout_rate"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(virtual_sensor.parameters(), lr=hyperparams["learning_rate"], weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    print(f"Beginning training across {hyperparams['epochs']} epochs...")
    virtual_sensor.train()
    for epoch in range(hyperparams["epochs"]):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = virtual_sensor(batch_x).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"  • Epoch {epoch+1:02d}/{hyperparams['epochs']} | Mean Trajectory MSE Loss: {epoch_loss/len(train_loader):.6f}")
        
    # Export Virtual Sensor weights along with scaling parameters
    sensor_model_path = "data/virtual_sensor.pt"
    virtual_sensor.save_checkpoint(sensor_model_path, scalar_metadata={"scaler": processor.scaler})
    
    # 5. Extract Residual footprint profile to seed the Isolation Engine
    print("\n[PHASE 3] Generating residuals footprint matrix to train Isolation Engine...")
    virtual_sensor.eval()
    all_residuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            preds = virtual_sensor(batch_x).squeeze()
            residual_deltas = torch.abs(batch_y.to(device) - preds)
            all_residuals.append(residual_deltas.cpu().numpy())
            
    # Standardise residual matrix representation dimensions
    residual_array = np.concatenate(all_residuals).reshape(-1, 1)
    residual_tensor = torch.tensor(residual_array, dtype=torch.float32).to(device)
    
    # 6. Instantiate and Train the Autoencoder Engine
    autoencoder = AnomalyAutoencoder(input_dim=1).to(device)
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
    ae_criterion = nn.MSELoss()
    
    autoencoder.train()
    print("Training Anomaly Isolation Network on nominal residual distributions...")
    for ae_epoch in range(15):
        ae_optimizer.zero_grad()
        reconstructed = autoencoder(residual_tensor)
        ae_loss = ae_criterion(reconstructed, residual_tensor)
        ae_loss.backward()
        ae_optimizer.step()
        if (ae_epoch + 1) % 5 == 0:
            print(f"  • Autoencoder Epoch {ae_epoch+1:02d}/15 | Extraction Loss: {ae_loss.item():.6f}")
            
    # Calculate adaptive safety threshold bounds based on maximum clean deviations
    sample_losses, _ = autoencoder.calculate_reconstruction_loss(residual_tensor)
    calculated_threshold = float(torch.mean(sample_losses).item() + (3.0 * torch.std(sample_losses).item()))
    print(f"[THRESHOLD DETECTED] Dynamic Mahalanobis Alert Trigger Limit set at: {calculated_threshold:.6f}")
    
    # Export Isolation Engine configurations
    ae_model_path = "data/isolation_engine.pt"
    autoencoder.save_model(ae_model_path, threshold_metadata=calculated_threshold)
    
    print("\n" + "="*70)
    print("🎉 PIPELINE RUN COMPLETED: SYSTEM READY FOR LIVE PIPELINE DEPLOYMENT")
    print("="*70)

if __name__ == "__main__":
    execute_training_pipeline()