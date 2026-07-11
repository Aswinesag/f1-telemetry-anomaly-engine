# F1 Telemetry Anomaly Engine

A production-grade, asynchronous engine designed to process real-time F1 telemetry, synthesize virtual sensor data, and detect mechanical anomalies using deep learning.

## Architecture
This system decouples data ingestion, physics-based feature engineering, and ML inference to ensure low-latency performance during live telemetry streaming.

- **FastAPI Backend:** Asynchronous API handling WebSocket streams.
- **Physics Engine:** A vectorized, stateless module for real-time thermodynamic modeling (Drag, EMA brake work).
- **Async Inference:** Background worker pattern for ML model execution to prevent I/O blocking.
- **SQLAlchemy 2.0:** Async-first persistence for telemetry snapshots and anomaly records.

## Quick Start
1. **Environment:** Ensure Python 3.13+ and a running PostgreSQL instance.
2. **Setup:**
   ```bash
   pip install -r requirements.txt
   alembic upgrade head