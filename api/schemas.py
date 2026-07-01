from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TelemetryInput(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    time_sec: float = Field(alias="TimeSec")
    speed: float = Field(alias="Speed")
    throttle: float = Field(alias="Throttle")
    brake: float = Field(alias="Brake")
    rpm: float = Field(alias="RPM")
    gear: int = Field(alias="nGear")

    def as_worker_payload(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json")


class TelemetryAccepted(BaseModel):
    status: str
    queue_depth: int


class ServiceHealth(BaseModel):
    status: str
    service: str
    device: str
