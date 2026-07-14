"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface ComparisonCarSample {
  car_id: string;
  speed: number;
  predicted_temperature: number;
  actual_temperature: number;
  anomaly_score: number;
  degradation_index: number | null;
}

export interface ComparisonDeltaSample {
  speed: number;
  predicted_temperature: number;
  actual_temperature: number;
  anomaly_score: number;
  degradation_index: number;
}

export interface ComparisonSample {
  time_sec: number;
  car_a: ComparisonCarSample;
  car_b: ComparisonCarSample;
  delta: ComparisonDeltaSample;
}

export interface ComparisonPayload {
  session_id: string;
  car_a: string;
  car_b: string;
  alignment: {
    method: string;
    direction: string;
    tolerance_seconds: number;
  };
  samples: ComparisonSample[];
}

interface ComparisonChartProps {
  payload: ComparisonPayload;
  metric?: "speed" | "actual_temperature" | "predicted_temperature";
  className?: string;
}

interface ChartPoint {
  time_sec: number;
  car_a_value: number;
  car_b_value: number;
  upper_value: number;
  lower_value: number;
  delta_value: number;
}

const SERIES_A = "#00FF66";
const SERIES_B = "#38BDF8";
const DELTA_POSITIVE = "#00FF66";
const DELTA_NEGATIVE = "#FF0033";

const metricLabels = {
  speed: "Speed",
  actual_temperature: "Actual Temp",
  predicted_temperature: "Virtual Temp",
};

export default function ComparisonChart({
  payload,
  metric = "speed",
  className = "",
}: ComparisonChartProps) {
  const chartData = payload.samples.map((sample): ChartPoint => {
    const carAValue = sample.car_a[metric];
    const carBValue = sample.car_b[metric];
    return {
      time_sec: sample.time_sec,
      car_a_value: carAValue,
      car_b_value: carBValue,
      upper_value: Math.max(carAValue, carBValue),
      lower_value: Math.min(carAValue, carBValue),
      delta_value: carAValue - carBValue,
    };
  });

  const latestDelta = chartData.at(-1)?.delta_value ?? 0;
  const fasterCar =
    latestDelta === 0
      ? "EVEN"
      : latestDelta > 0
        ? payload.car_a
        : payload.car_b;

  return (
    <section
      className={`rounded-2xl border border-neutral-800 bg-neutral-900/60 p-4 text-neutral-100 shadow-xl shadow-black/20 backdrop-blur ${className}`}
    >
      <div className="mb-4 flex flex-col justify-between gap-3 border-b border-neutral-800 pb-4 md:flex-row md:items-start">
        <div>
          <p className="font-mono text-xs font-bold uppercase tracking-[0.28em] text-neutral-500">
            Multi-Car Comparison
          </p>
          <h2 className="mt-1 text-xl font-black uppercase tracking-tight text-white">
            {payload.car_a} vs {payload.car_b}
          </h2>
        </div>
        <div className="grid grid-cols-2 gap-2 font-mono text-xs uppercase tracking-[0.16em] text-neutral-300">
          <Badge label="Session" value={payload.session_id} />
          <Badge label="Advantage" value={fasterCar} />
        </div>
      </div>

      <div className="h-[340px] min-h-[340px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 12, right: 18, bottom: 0, left: -12 }}>
            <defs>
              <linearGradient id="comparisonDeltaFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={latestDelta >= 0 ? DELTA_POSITIVE : DELTA_NEGATIVE} stopOpacity={0.28} />
                <stop offset="95%" stopColor={latestDelta >= 0 ? DELTA_POSITIVE : DELTA_NEGATIVE} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#262626" strokeDasharray="3 6" vertical={false} />
            <XAxis
              dataKey="time_sec"
              stroke="#737373"
              tick={{ fill: "#737373", fontSize: 11, fontFamily: "monospace" }}
              tickFormatter={(value: number) => value.toFixed(1)}
              minTickGap={24}
            />
            <YAxis
              stroke="#737373"
              tick={{ fill: "#737373", fontSize: 11, fontFamily: "monospace" }}
              width={58}
              domain={["auto", "auto"]}
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
              formatter={(value, name) => [
                typeof value === "number" ? value.toFixed(3) : String(value ?? "--"),
                String(name),
              ]}
              labelFormatter={(label) =>
                typeof label === "number" ? `T+${label.toFixed(2)}s` : String(label)
              }
            />
            <ReferenceLine y={0} stroke="#404040" strokeDasharray="4 4" />
            <Area
              type="monotone"
              dataKey="upper_value"
              stroke="none"
              fill="url(#comparisonDeltaFill)"
              fillOpacity={1}
              activeDot={false}
              isAnimationActive={false}
              name="Delta Overlay"
            />
            <Area
              type="monotone"
              dataKey="lower_value"
              stroke="none"
              fill="#171717"
              fillOpacity={1}
              activeDot={false}
              isAnimationActive={false}
              name="Delta Base"
            />
            <Line
              type="monotone"
              dataKey="car_a_value"
              stroke={SERIES_A}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              name={`${payload.car_a} ${metricLabels[metric]}`}
            />
            <Line
              type="monotone"
              dataKey="car_b_value"
              stroke={SERIES_B}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              name={`${payload.car_b} ${metricLabels[metric]}`}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 font-mono text-xs uppercase tracking-[0.18em] text-neutral-400 md:grid-cols-3">
        <Badge label="Metric" value={metricLabels[metric]} />
        <Badge label="Latest Delta" value={latestDelta.toFixed(3)} />
        <Badge
          label="Alignment"
          value={`${payload.alignment.method}/${payload.alignment.tolerance_seconds}s`}
        />
      </div>
    </section>
  );
}

function Badge({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-950/70 p-3">
      <p className="text-[10px] text-neutral-500">{label}</p>
      <p className="mt-1 truncate text-sm font-black text-white">{value}</p>
    </div>
  );
}
