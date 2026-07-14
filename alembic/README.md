# Alembic Migration Path

Apply schema changes against an existing PostgreSQL database before running telemetry replay:

```powershell
docker compose run --rm inference-api alembic upgrade head
```

For local development:

```powershell
$env:DATABASE_URL="postgresql+asyncpg://f1_telemetry:f1_telemetry@localhost:5432/f1_telemetry"
alembic upgrade head
```

The tire degradation migration adds nullable columns, so existing telemetry rows remain valid.
