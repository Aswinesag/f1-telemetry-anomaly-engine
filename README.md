# F1 Telemetry Anomaly Engine

Real-time F1 telemetry intelligence system for streaming race data, generating physics-informed features, running virtual thermal sensing, detecting anomalies, explaining model decisions, tracking tire degradation, and comparing telemetry across cars.

The project is implemented as a production-style asynchronous stack:

- **Kafka** transports high-frequency telemetry frames.
- **FastAPI** serves ingestion, health/readiness, WebSocket telemetry, and comparison APIs.
- **InferenceWorker** runs model inference outside request handlers.
- **PhysicsEngine** generates shared training/runtime features.
- **Redis** stores the latest inference payload for low-latency dashboard reads.
- **PostgreSQL** stores historical telemetry snapshots and model outputs.
- **Next.js** renders the pitwall dashboard on port `8502`.

# Demo


# If you want to see a higher resolution video demo

https://vimeo.com/1209867702?share=copy&fl=sv&fe=ci

---
## System Architecture

```text
FastF1 / replay CSV
      |
      v
Telemetry replay producer
      |
      v
Kafka topic: f1-telemetry-bus
      |
      v
FastAPI inference-api
      |
      +--> async Kafka consumer
      +--> bounded worker queue
      +--> PhysicsEngine feature generation
      +--> PyTorch virtual thermal sensor
      +--> PyTorch anomaly autoencoder
      +--> Captum XAI attribution
      +--> tire degradation trend classification
      +--> Redis latest-result cache
      +--> PostgreSQL telemetry_snapshots table
      |
      v
HTTP / WebSocket telemetry
      |
      v
Next.js pitwall dashboard
```

---

## Implemented Features

- Async FastAPI service with `/health`, `/ready`, `/telemetry`, `/telemetry/latest`, `/ws/telemetry`, and multi-car comparison endpoints.
- Kafka replay pipeline for historical telemetry playback.
- Stateless vectorized physics engine shared by training and inference.
- Tire dynamics features:
  - `Wheel_Speed_ms`
  - `Slip_Ratio`
  - `Normal_Load_N`
  - `Tire_Load_Factor`
  - `Tire_Stress_Index`
  - `Thermal_Decay_Indicator`
  - `Calculated_Degradation_Index`
  - `Sustained_Tire_Decay_Flag`
- PyTorch virtual thermal sensor using 1D CNN + bidirectional LSTM.
- PyTorch residual autoencoder for anomaly scoring.
- Captum Integrated Gradients XAI attribution module.
- Fault classification:
  - `Nominal`
  - `Mechanical Fault`
  - `Tire Degradation`
- SQLAlchemy 2.0 async persistence for telemetry snapshots.
- Alembic migrations for schema evolution.
- Redis-backed latest telemetry cache.
- Multi-car comparison API using server-side `pandas.merge_asof`.
- Next.js pitwall dashboard with:
  - live telemetry cards
  - anomaly/loss chart
  - thermal waveform chart
  - XAI explanation panel
  - tire degradation panel
  - reusable comparison chart component
- Docker Compose orchestration for Kafka, ZooKeeper, Redis, PostgreSQL, API, dashboard, and replay.

---

## Operating Ports

| Component | Host URL / Port | Container Port |
|---|---:|---:|
| Dashboard | `http://localhost:8502` | `8502` |
| Inference API | `http://localhost:18080` | `8000` |
| Kafka external listener | `localhost:9092` | `9092` |
| PostgreSQL | `localhost:5432` | `5432` |
| Redis | `localhost:6379` | `6379` |
| ZooKeeper | `localhost:2181` | `2181` |

The API is exposed on host port `18080` because Windows may reserve port `8000`.

---

## Repository Layout

```text
.
├── alembic/
│   └── versions/
│       ├── 20260714_0001_add_tire_degradation_fields.py
│       └── 20260714_0002_add_multi_car_identity.py
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── worker.py
│   └── database/
│       └── models.py
├── config/
│   └── config.yaml
├── data/
│   ├── virtual_sensor.pt
│   ├── isolation_engine.pt
│   └── raw_samples/
├── f1-dashboard/
│   ├── app/
│   │   ├── Dashboard.tsx
│   │   └── page.tsx
│   └── components/
│       ├── ComparisonChart.tsx
│       ├── ExplanationPanel.tsx
│       └── TireDegradationPanel.tsx
├── src/
│   ├── engine/
│   │   └── physics.py
│   ├── ml/
│   │   └── xai_engine.py
│   ├── models/
│   │   ├── autoencoder.py
│   │   └── virtual_sensor.py
│   └── pipeline/
│       ├── etl_processor.py
│       ├── ingestion.py
│       └── replay_simulation.py
├── tests/
│   └── test_physics.py
├── alembic.ini
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.api
├── Dockerfile.replay
├── requirements.txt
├── requirements-api.txt
├── requirements-replay.txt
└── train.py
```

---

## Technical Stack

### Backend

- **FastAPI** for async HTTP and WebSocket APIs.
- **Uvicorn** for ASGI runtime.
- **aiokafka** for async Kafka consumption.
- **kafka-python** for replay producer support.
- **Redis asyncio client** for latest-result caching.
- **SQLAlchemy 2.0 Async** with **asyncpg** for PostgreSQL persistence.
- **Alembic** for schema migrations.
- **Pydantic** for API schemas.

### Machine Learning

- **PyTorch** for the virtual sensor and anomaly autoencoder.
- **Captum** for Integrated Gradients attribution.
- **scikit-learn** for `MinMaxScaler` feature scaling.
- **NumPy / Pandas** for vectorized physics math and comparison alignment.
- **FastF1** for historical telemetry extraction.
- **SciPy** for training-time resampling/interpolation.

### Frontend

- **Next.js 16**
- **React**
- **Tailwind CSS 4**
- **Recharts**
- **lucide-react**

### Infrastructure

- **Docker Compose**
- **Confluent Kafka + ZooKeeper**
- **PostgreSQL 16 Alpine**
- **Redis 7 Alpine**
- **CPU-only PyTorch wheel** in the API image

---

## Physics Engine

Main file:

```text
src/engine/physics.py
```

The `PhysicsEngine` is stateless and used by both training and live inference to avoid training-serving skew.

Core generated features:

```text
Speed_ms
Delta_KE
Acceleration
Longitudinal_G
Aero_Drag_N
Aero_Downforce_N
Effective_Weight_N
Brake_Work_EMA
Convective_Cooling_Factor
Brake_Temp_Target
Wheel_Speed_ms
Slip_Ratio
Normal_Load_N
Tire_Load_Factor
Tire_Stress_Index
Thermal_Decay_Indicator
Calculated_Degradation_Index
Sustained_Tire_Decay_Flag
```

Required columns:

```text
Speed, Brake
```

Optional tire inputs:

```text
WheelSpeed
Wheel_Speed
WheelSpeedFL
WheelSpeedFR
WheelSpeedRL
WheelSpeedRR
tire_compound
stint_lap_number
```

If wheel speed is not present, the engine falls back to vehicle speed, yielding zero slip ratio.

---

## ML Pipeline

### Virtual Thermal Sensor

Implemented in:

```text
src/models/virtual_sensor.py
```

Architecture:

```text
Input feature sequence
  -> Conv1D
  -> BatchNorm + ReLU + Dropout
  -> 2-layer bidirectional LSTM
  -> dense regressor head
  -> predicted thermal state
```

Target:

```text
Brake_Temp_Target
```

### Anomaly Isolation Engine

Implemented in:

```text
src/models/autoencoder.py
```

The autoencoder operates on residuals:

```text
abs(actual_temperature - predicted_temperature)
```

Inference:

```text
anomaly_score = reconstruction_loss(residual)
is_anomaly = anomaly_score > alert_threshold
```

Threshold:

```text
mean(validation_losses) + initial_threshold * std(validation_losses)
```

---

## Explainable AI

Main file:

```text
src/ml/xai_engine.py
```

`XAIEngine` wraps the autoencoder with Captum Integrated Gradients and returns normalized feature contribution scores.

Explanation payload:

```json
{
  "top_factor": "Thermal_Residual",
  "importance_score": 1.0,
  "fault_type": "Mechanical Fault",
  "recommendation": "Inspect brake, cooling, and mechanical telemetry for a sudden fault signature.",
  "feature_importance": {
    "Thermal_Residual": 1.0
  }
}
```

XAI is skipped for low-score nominal frames by default. Override with:

```text
XAI_TRIGGER_THRESHOLD
XAI_IG_STEPS
```

---

## Tire Degradation

Tire degradation is treated as a continuous trend, not a binary anomaly.

Worker output fields:

```text
tire_compound
stint_lap_number
degradation_index
degradation_trend
fault_type
```

Classification rules use:

- current `Calculated_Degradation_Index`
- rolling degradation slope
- sustained slip/load stress

Config:

```yaml
anomaly_detection:
  degradation_slope_threshold: 0.0025
  degradation_index_threshold: 0.15
```

The default replay data is single-car and does not currently provide wheel-speed or compound columns, so tire degradation validates as `0.0` / `Nominal` unless richer telemetry is ingested.

---

## Multi-Car Comparison

Endpoint:

```text
GET /telemetry/compare/{car_a}/{car_b}?session_id={id}
```

The API performs temporal alignment server-side using `pandas.merge_asof` with nearest-sample matching.

Returned deltas include:

```text
speed
predicted_temperature
actual_temperature
anomaly_score
degradation_index
```

The comparison query is backed by the composite index:

```text
ix_telemetry_snapshots_session_car_time(session_id, car_id, time_sec)
```

The frontend component is:

```text
f1-dashboard/components/ComparisonChart.tsx
```

It is a reusable component. It will only show meaningful data after at least two distinct `car_id` values exist in the same `session_id`.

---

## API Endpoints

Base URL:

```text
http://localhost:18080
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | API liveness. |
| `GET` | `/ready` | Redis, DB, Kafka task, and worker readiness. |
| `POST` | `/telemetry` | Direct telemetry ingestion into worker queue. |
| `GET` | `/telemetry/latest` | Latest processed inference payload from Redis. |
| `GET` | `/telemetry/compare/{car_a}/{car_b}?session_id={id}` | Server-aligned multi-car comparison. |
| `WS` | `/ws/telemetry` | Live dashboard telemetry stream. |

Telemetry input:

```json
{
  "TimeSec": 85.2,
  "Speed": 280.0,
  "Throttle": 40.0,
  "Brake": 92.0,
  "RPM": 11200.0,
  "nGear": 6,
  "session_id": "monza-2023-race",
  "car_id": "VER"
}
```

Inference output:

```json
{
  "CapturedAt": "2026-07-14T12:17:37.814753+00:00",
  "TimeSec": 85.22,
  "Speed": 311.91,
  "Brake": 0.0,
  "Predicted_Temp": 202.9,
  "Actual_Temp": 177.3,
  "Anomaly_Score": 607.62,
  "Alert_Threshold": 40767.36,
  "Is_Anomaly": false,
  "anomaly_score": 607.62,
  "tire_compound": null,
  "stint_lap_number": null,
  "degradation_index": 0.0,
  "degradation_trend": 0.0,
  "fault_type": "Nominal",
  "explanation": null
}
```

Comparison output:

```json
{
  "session_id": "monza-2023-race",
  "car_a": "VER",
  "car_b": "HAM",
  "alignment": {
    "method": "merge_asof",
    "direction": "nearest",
    "tolerance_seconds": 0.25
  },
  "samples": [
    {
      "time_sec": 85.2,
      "delta": {
        "speed": 3.6,
        "actual_temperature": 2.6,
        "anomaly_score": -1.0,
        "degradation_index": -0.1
      }
    }
  ]
}
```

---

## Database

Main model:

```text
api/database/models.py
```

Table:

```text
telemetry_snapshots
```

Important identity fields:

```text
session_id
car_id
time_sec
```

Migration command:

```powershell
docker compose run --rm inference-api alembic upgrade head
```

Current migrations:

| Revision | Purpose |
|---|---|
| `20260714_0001` | Add tire degradation persistence fields. |
| `20260714_0002` | Add multi-car identity fields and comparison index. |

Check migration state:

```powershell
docker compose exec postgres psql -U f1_telemetry -d f1_telemetry -c "SELECT version_num FROM alembic_version;"
```

---

## Quick Start

### 1. Build and start services

```powershell
docker compose up --build
```

### 2. Apply migrations

```powershell
docker compose run --rm inference-api alembic upgrade head
```

### 3. Check API

```powershell
curl http://localhost:18080/health
curl http://localhost:18080/ready
```

### 4. Open dashboard

```text
http://localhost:8502
```

### 5. Replay telemetry

```powershell
docker compose --profile replay run --rm telemetry-replay
```

---

## Local Development

### Backend

```powershell
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 18080
```

### Dashboard

```powershell
cd f1-dashboard
npm install
npm run dev -- --hostname 127.0.0.1 -p 8502
```

Dashboard WebSocket:

```text
NEXT_PUBLIC_WS_URL=ws://localhost:18080/ws/telemetry
```

### Training

```powershell
python train.py
```

Training saves:

```text
data/virtual_sensor.pt
data/isolation_engine.pt
data/raw_samples/monza_ver_cleaned.csv
```

---

## Configuration

Main config:

```text
config/config.yaml
```

Important keys:

| Key | Purpose |
|---|---|
| `system.target_frequency_hz` | Replay/model frequency. |
| `features.raw_channels` | Raw model inputs. |
| `features.physics_engineered` | Physics-generated model inputs. |
| `features.target_channel` | Training target. |
| `model_hyperparameters.sequence_length` | Rolling model window size. |
| `anomaly_detection.initial_threshold` | Autoencoder alert multiplier. |
| `anomaly_detection.degradation_slope_threshold` | Tire degradation trend threshold. |
| `anomaly_detection.degradation_index_threshold` | Tire degradation index threshold. |

---

## Environment Variables

| Variable | Default | Used By |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | API, replay |
| `KAFKA_TELEMETRY_TOPIC` | `f1-telemetry-bus` | API |
| `KAFKA_CONSUMER_GROUP` | `inference-api` | API |
| `DATABASE_URL` | `postgresql+asyncpg://f1_telemetry:f1_telemetry@localhost:5432/f1_telemetry` | API, Alembic |
| `REDIS_URL` | `redis://localhost:6379/0` | API |
| `WEBSOCKET_POLL_SECONDS` | `0.05` | API |
| `INFERENCE_QUEUE_MAX_SIZE` | `2000` | API |
| `MODEL_VERSION` | `unknown` | DB persistence |
| `XAI_TRIGGER_THRESHOLD` | model alert threshold | XAI |
| `XAI_IG_STEPS` | `32` | XAI |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:18080/ws/telemetry` | Dashboard |

---

## Validation

Backend syntax:

```powershell
python -m py_compile api/main.py api/worker.py api/database/models.py api/schemas.py src/engine/physics.py src/ml/xai_engine.py
```

Physics tests:

```powershell
python -m pytest tests/test_physics.py -q
```

Dashboard:

```powershell
cd f1-dashboard
npm run lint
npm run build
```

Compose:

```powershell
docker compose config --quiet
```

Database migration:

```powershell
docker compose run --rm inference-api alembic upgrade head
```

End-to-end smoke path:

```text
replay -> Kafka -> inference-api -> Redis -> WebSocket -> dashboard
```

---

## Operational Notes

### Dashboard data behavior

The dashboard needs processed telemetry. Start services, wait for readiness, then run replay.

### Multi-car comparison behavior

The default replay CSV is single-car. `ComparisonChart` requires two distinct `car_id` values under the same `session_id`. Until multi-car telemetry is ingested, the comparison endpoint may return no aligned samples.

### Windows port reservations

Check excluded ranges:

```powershell
netsh interface ipv4 show excludedportrange protocol=tcp
```

### Docker image downloads

The API uses CPU-only PyTorch in `requirements-api.txt`. If builds stall, check access to:

```text
download.pytorch.org
```

### Git hygiene

Generated folders and local caches are ignored:

- `__pycache__/`
- `.pytest_cache/`
- `node_modules/`
- `.next/`
- `data/f1_cache/`
- SQLite cache files

Model checkpoints and replay CSVs are currently retained because they make the prototype runnable.

---

## License

MIT License. See `LICENSE.md`.
