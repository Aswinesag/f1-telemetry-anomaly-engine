"use client";

import {
  Activity,
  CheckCircle2,
  Gauge,
  RadioTower,
  RotateCw,
  ShieldAlert,
  Thermometer,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import type { ReactNode } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface TelemetryPacket {
  CapturedAt?: string;
  TimeSec: number;
  Speed: number;
  Brake: number;
  RPM?: number;
  nGear?: number;
  Predicted_Temp: number;
  Actual_Temp: number;
  Anomaly_Score: number;
  Alert_Threshold: number;
  Is_Anomaly: boolean;
}

interface MetricCardProps {
  label: string;
  value: string;
  unit?: string;
  tone: "nominal" | "warning" | "critical" | "neutral";
  icon: ReactNode;
  sublabel: string;
}

const MAX_HISTORY_LENGTH = 180;
const NOMINAL = "#00FF66";
const CRITICAL = "#FF0033";
const WARNING = "#FFB000";

const subscribeToHydration = () => () => undefined;
const getClientSnapshot = () => true;
const getServerSnapshot = () => false;

function getTelemetrySnapshotUrl(wsUrl: string) {
  const parsedUrl = new URL(wsUrl);
  parsedUrl.protocol = parsedUrl.protocol === "wss:" ? "https:" : "http:";
  parsedUrl.pathname = "/telemetry/latest";
  parsedUrl.search = "";
  parsedUrl.hash = "";
  return parsedUrl.toString();
}

function formatNumber(value: number | undefined, digits = 0) {
  if (value === undefined || Number.isNaN(value)) return "--";
  return value.toLocaleString("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function getConnectionAge(packet: TelemetryPacket | null) {
  if (!packet?.CapturedAt) return "NO SYNC";
  const capturedAt = new Date(packet.CapturedAt).getTime();
  if (Number.isNaN(capturedAt)) return "SYNCED";
  const deltaSeconds = Math.max(0, Math.round((Date.now() - capturedAt) / 1000));
  return `${deltaSeconds}s AGO`;
}

export default function Dashboard() {
  const [history, setHistory] = useState<TelemetryPacket[]>([]);
  const [currentStatus, setCurrentStatus] = useState<TelemetryPacket | null>(null);
  const [connectionState, setConnectionState] = useState<"connecting" | "live" | "offline">("connecting");
  const isMounted = useSyncExternalStore(subscribeToHydration, getClientSnapshot, getServerSnapshot);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:18080/ws/telemetry";

    fetch(getTelemetrySnapshotUrl(wsUrl))
      .then((response) => (response.ok ? response.json() : null))
      .then((packet: TelemetryPacket | null) => {
        if (!packet) return;
        setCurrentStatus(packet);
        setHistory([packet]);
      })
      .catch(() => undefined);

    const socket = new WebSocket(wsUrl);
    wsRef.current = socket;

    socket.onopen = () => setConnectionState("live");
    socket.onclose = () => setConnectionState("offline");
    socket.onerror = () => setConnectionState("offline");
    socket.onmessage = (event) => {
      const packet: TelemetryPacket = JSON.parse(event.data);
      setCurrentStatus(packet);
      setHistory((previous) => {
        const nextHistory = [...previous, packet];
        return nextHistory.length > MAX_HISTORY_LENGTH
          ? nextHistory.slice(nextHistory.length - MAX_HISTORY_LENGTH)
          : nextHistory;
      });
    };

    return () => socket.close();
  }, []);

  const latestThreshold = currentStatus?.Alert_Threshold ?? 0;
  const sensorStreams = useMemo(
    () => [
      { name: "TEMP", value: currentStatus?.Actual_Temp ?? 0, max: 450, active: Boolean(currentStatus) },
      { name: "VIRT", value: currentStatus?.Predicted_Temp ?? 0, max: 450, active: Boolean(currentStatus) },
      { name: "LOSS", value: currentStatus?.Anomaly_Score ?? 0, max: Math.max(latestThreshold * 1.35, 1), active: Boolean(currentStatus) },
      { name: "BRAKE", value: currentStatus?.Brake ?? 0, max: 100, active: Boolean(currentStatus) },
      { name: "RPM", value: currentStatus?.RPM ?? 0, max: 13000, active: currentStatus?.RPM !== undefined },
    ],
    [currentStatus, latestThreshold],
  );

  const anomalyDelta = currentStatus
    ? currentStatus.Alert_Threshold - currentStatus.Anomaly_Score
    : undefined;

  return (
    <main className="min-h-screen overflow-hidden bg-neutral-950 p-4 text-neutral-100 md:p-6">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_top_left,rgba(0,255,102,0.10),transparent_32%),radial-gradient(circle_at_bottom_right,rgba(255,0,51,0.10),transparent_30%)]" />
      <section className="relative mx-auto flex min-h-[calc(100vh-3rem)] max-w-[1800px] flex-col gap-4">
        <header className="grid grid-cols-1 gap-4 lg:grid-cols-[1.4fr_0.8fr_0.8fr]">
          <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5 shadow-2xl shadow-black/30 backdrop-blur">
            <div className="flex items-center gap-3">
              <div className="flex size-11 items-center justify-center rounded-xl border border-red-500/40 bg-red-500/10 text-red-500">
                <Activity className="size-6" />
              </div>
              <div>
                <p className="font-mono text-xs uppercase tracking-[0.45em] text-neutral-500">F1 Telemetry Anomaly Engine</p>
                <h1 className="text-2xl font-black uppercase tracking-tight text-white md:text-4xl">
                  Pitwall Isolation Console
                </h1>
              </div>
            </div>
          </div>
          <StatusPill
            label="Stream Link"
            value={connectionState === "live" ? "LIVE" : connectionState.toUpperCase()}
            tone={connectionState === "live" ? "nominal" : connectionState === "connecting" ? "warning" : "critical"}
            icon={<RadioTower className="size-5" />}
          />
          <StatusPill
            label="System State"
            value={currentStatus?.Is_Anomaly ? "ANOMALY" : "NOMINAL"}
            tone={currentStatus?.Is_Anomaly ? "critical" : "nominal"}
            icon={currentStatus?.Is_Anomaly ? <ShieldAlert className="size-5" /> : <CheckCircle2 className="size-5" />}
          />
        </header>

        <div className="grid flex-1 grid-cols-1 gap-4 xl:grid-cols-4 xl:grid-rows-3">
          <HeroChartCard
            className="xl:col-span-2 xl:row-span-2"
            title="Thermal Sensor Waveform"
            subtitle="Actual brake thermal model vs hybrid virtual sensor"
            data={history}
            yDomain={["auto", "auto"]}
            isMounted={isMounted}
          >
            <defs>
              <linearGradient id="actualTemperatureFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={CRITICAL} stopOpacity={0.48} />
                <stop offset="95%" stopColor={CRITICAL} stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="predictedTemperatureFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={NOMINAL} stopOpacity={0.42} />
                <stop offset="95%" stopColor={NOMINAL} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <Area type="monotone" dataKey="Actual_Temp" stroke={CRITICAL} fill="url(#actualTemperatureFill)" strokeWidth={2} dot={false} isAnimationActive={false} name="Actual °C" />
            <Area type="monotone" dataKey="Predicted_Temp" stroke={NOMINAL} fill="url(#predictedTemperatureFill)" strokeWidth={2} dot={false} isAnimationActive={false} name="Virtual °C" />
          </HeroChartCard>

          <HeroChartCard
            className="xl:col-span-2 xl:row-span-2"
            title="Isolation Engine Loss Signature"
            subtitle="Autoencoder residual trend against backend model threshold"
            data={history}
            yDomain={[0, "auto"]}
            isMounted={isMounted}
          >
            <defs>
              <linearGradient id="anomalyFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={WARNING} stopOpacity={0.55} />
                <stop offset="95%" stopColor={WARNING} stopOpacity={0.03} />
              </linearGradient>
            </defs>
            {currentStatus?.Alert_Threshold !== undefined && (
              <ReferenceLine
                y={currentStatus.Alert_Threshold}
                stroke={CRITICAL}
                strokeDasharray="4 4"
                label={{ value: `LIMIT ${currentStatus.Alert_Threshold.toFixed(2)}`, fill: CRITICAL, fontSize: 11, fontFamily: "monospace" }}
              />
            )}
            <Area type="monotone" dataKey="Anomaly_Score" stroke={WARNING} fill="url(#anomalyFill)" strokeWidth={2} dot={false} isAnimationActive={false} name="Loss" />
          </HeroChartCard>

          <MetricCard
            label="RPM"
            value={formatNumber(currentStatus?.RPM)}
            unit="rev/min"
            tone="neutral"
            icon={<RotateCw className="size-5" />}
            sublabel={`GEAR ${currentStatus?.nGear ?? "--"}`}
          />
          <MetricCard
            label="Speed"
            value={formatNumber(currentStatus?.Speed)}
            unit="km/h"
            tone="nominal"
            icon={<Gauge className="size-5" />}
            sublabel="VELOCITY BUS"
          />
          <MetricCard
            label="Brake Pressure"
            value={formatNumber(currentStatus?.Brake)}
            unit="%"
            tone={(currentStatus?.Brake ?? 0) > 80 ? "warning" : "neutral"}
            icon={<Zap className="size-5" />}
            sublabel="PEDAL LOAD"
          />
          <MetricCard
            label="Tire Temp"
            value={formatNumber(currentStatus?.Actual_Temp, 1)}
            unit="°C"
            tone={currentStatus?.Is_Anomaly ? "critical" : "nominal"}
            icon={<Thermometer className="size-5" />}
            sublabel={`AI ${formatNumber(currentStatus?.Predicted_Temp, 1)}°C`}
          />

          <StatusTicker
            streams={sensorStreams}
            isAnomaly={Boolean(currentStatus?.Is_Anomaly)}
            anomalyDelta={anomalyDelta}
            lastSync={getConnectionAge(currentStatus)}
          />
        </div>
      </section>
    </main>
  );
}

function HeroChartCard({
  className,
  title,
  subtitle,
  data,
  yDomain,
  isMounted,
  children,
}: {
  className: string;
  title: string;
  subtitle: string;
  data: TelemetryPacket[];
  yDomain: [number | "auto", number | "auto"];
  isMounted: boolean;
  children: ReactNode;
}) {
  return (
    <section className={`rounded-2xl border border-neutral-800 bg-neutral-900/60 p-4 shadow-2xl shadow-black/30 backdrop-blur ${className}`}>
      <div className="mb-3 flex items-start justify-between gap-4">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-[0.28em] text-neutral-200">{title}</h2>
          <p className="mt-1 text-xs uppercase tracking-[0.22em] text-neutral-500">{subtitle}</p>
        </div>
        <div className="rounded-full border border-neutral-700 bg-neutral-950/70 px-3 py-1 font-mono text-xs text-neutral-400">
          {data.length.toString().padStart(3, "0")} SAMPLES
        </div>
      </div>
      <div className="h-[310px] min-h-[310px] xl:h-[calc(100%-4rem)] xl:min-h-0">
        {!isMounted ? (
          <div className="flex h-full items-center justify-center rounded-xl border border-neutral-800 bg-neutral-950/60 font-mono text-xs uppercase tracking-[0.28em] text-neutral-600">
            Awaiting sensor feed
          </div>
        ) : (
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 12, right: 18, bottom: 0, left: -18 }}>
            <CartesianGrid stroke="#262626" strokeDasharray="3 6" vertical={false} />
            <XAxis
              dataKey="TimeSec"
              stroke="#737373"
              tick={{ fill: "#737373", fontSize: 11, fontFamily: "monospace" }}
              tickFormatter={(value: number) => value.toFixed(1)}
              minTickGap={24}
            />
            <YAxis
              stroke="#737373"
              tick={{ fill: "#737373", fontSize: 11, fontFamily: "monospace" }}
              domain={yDomain}
              width={58}
            />
            <Tooltip
              cursor={{ stroke: "#525252", strokeDasharray: "3 3" }}
              contentStyle={{
                backgroundColor: "rgba(10,10,10,0.94)",
                border: "1px solid #404040",
                borderRadius: "12px",
                color: "#fafafa",
                fontFamily: "monospace",
              }}
              labelStyle={{ color: "#a3a3a3" }}
            />
            {children}
          </AreaChart>
        </ResponsiveContainer>
        )}
      </div>
    </section>
  );
}

function MetricCard({ label, value, unit, tone, icon, sublabel }: MetricCardProps) {
  const toneClass = {
    nominal: "border-emerald-500/30 text-[#00FF66]",
    warning: "border-yellow-500/30 text-[#FFB000]",
    critical: "border-red-500/40 text-[#FF0033]",
    neutral: "border-neutral-700 text-neutral-100",
  }[tone];

  return (
    <section className={`rounded-2xl border bg-neutral-900/60 p-4 shadow-xl shadow-black/20 backdrop-blur ${toneClass}`}>
      <div className="mb-5 flex items-center justify-between text-neutral-400">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-xs font-bold uppercase tracking-[0.25em]">{label}</span>
        </div>
        <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-neutral-500">{sublabel}</span>
      </div>
      <div className="flex items-end gap-2 font-mono">
        <span className="text-4xl font-black leading-none tracking-tight md:text-5xl">{value}</span>
        {unit && <span className="pb-1 text-xs uppercase tracking-[0.22em] text-neutral-500">{unit}</span>}
      </div>
    </section>
  );
}

function StatusPill({
  label,
  value,
  tone,
  icon,
}: {
  label: string;
  value: string;
  tone: "nominal" | "warning" | "critical";
  icon: ReactNode;
}) {
  const toneClass = {
    nominal: "border-emerald-500/30 text-[#00FF66]",
    warning: "border-yellow-500/30 text-[#FFB000]",
    critical: "border-red-500/40 text-[#FF0033]",
  }[tone];

  return (
    <div className={`rounded-2xl border bg-neutral-900/60 p-5 shadow-2xl shadow-black/30 backdrop-blur ${toneClass}`}>
      <div className="mb-3 flex items-center justify-between text-neutral-400">
        <span className="text-xs font-bold uppercase tracking-[0.3em]">{label}</span>
        {icon}
      </div>
      <div className="font-mono text-4xl font-black tracking-tight">{value}</div>
    </div>
  );
}

function StatusTicker({
  streams,
  isAnomaly,
  anomalyDelta,
  lastSync,
}: {
  streams: { name: string; value: number; max: number; active: boolean }[];
  isAnomaly: boolean;
  anomalyDelta: number | undefined;
  lastSync: string;
}) {
  return (
    <section className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-4 shadow-xl shadow-black/20 backdrop-blur xl:col-span-4">
      <div className="mb-4 grid grid-cols-1 gap-3 border-b border-neutral-800 pb-4 md:grid-cols-[1fr_1fr_1fr]">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.28em] text-neutral-500">Sensor Stream Ticker</p>
          <p className="mt-1 font-mono text-sm text-neutral-300">LAST SYNC {lastSync}</p>
        </div>
        <div className="font-mono text-sm uppercase tracking-[0.2em] text-neutral-400">
          THRESHOLD DELTA{" "}
          <span className={isAnomaly ? "text-[#FF0033]" : "text-[#00FF66]"}>
            {anomalyDelta === undefined ? "--" : anomalyDelta.toFixed(3)}
          </span>
        </div>
        <div className={`font-mono text-sm font-bold uppercase tracking-[0.25em] ${isAnomaly ? "text-[#FF0033]" : "text-[#00FF66]"}`}>
          {isAnomaly ? "ALERT: RESIDUAL BREACH" : "ALL CHANNELS NOMINAL"}
        </div>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-5">
        {streams.map((stream) => {
          const width = `${Math.min(100, Math.max(0, (stream.value / stream.max) * 100))}%`;
          const barColor = !stream.active ? "bg-neutral-700" : isAnomaly && stream.name === "LOSS" ? "bg-[#FF0033]" : stream.name === "BRAKE" ? "bg-[#FFB000]" : "bg-[#00FF66]";
          return (
            <div key={stream.name} className="rounded-xl border border-neutral-800 bg-neutral-950/70 p-3">
              <div className="mb-2 flex items-center justify-between font-mono text-xs">
                <span className="text-neutral-400">{stream.name}</span>
                <span className={stream.active ? "text-neutral-100" : "text-neutral-600"}>{stream.active ? stream.value.toFixed(stream.name === "RPM" ? 0 : 2) : "NO DATA"}</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-neutral-800">
                <div className={`h-full rounded-full ${barColor} shadow-[0_0_18px_currentColor]`} style={{ width }} />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
