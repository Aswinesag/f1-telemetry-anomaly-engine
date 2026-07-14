"use client";

interface TireDegradationPanelProps {
  tireCompound?: string | null;
  stintLapNumber?: number | null;
  degradationIndex?: number | null;
  degradationTrend?: number | null;
  faultType?: string | null;
  className?: string;
}

function formatValue(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return value.toFixed(digits);
}

export default function TireDegradationPanel({
  tireCompound,
  stintLapNumber,
  degradationIndex,
  degradationTrend,
  faultType,
  className = "",
}: TireDegradationPanelProps) {
  const hasDegradationData =
    degradationIndex !== null && degradationIndex !== undefined;
  if (!hasDegradationData) return null;

  const normalizedIndex = Math.min(
    100,
    Math.max(0, (degradationIndex ?? 0) * 100),
  );
  const isPerformanceDecay = faultType === "Tire Degradation";

  return (
    <section
      className={`rounded-2xl border p-4 shadow-xl shadow-black/20 backdrop-blur ${
        isPerformanceDecay
          ? "border-orange-500/40 bg-orange-950/20 text-orange-50 shadow-[0_0_32px_rgba(255,128,0,0.10)]"
          : "border-neutral-800 bg-neutral-900/60 text-neutral-100"
      } ${className}`}
    >
      <div className="mb-4 flex items-start justify-between gap-4 border-b border-neutral-800 pb-3">
        <div>
          <p className="font-mono text-xs font-bold uppercase tracking-[0.28em] text-neutral-500">
            Tire Degradation Model
          </p>
          <h2 className="mt-1 text-xl font-black uppercase tracking-tight text-white">
            {isPerformanceDecay ? "Performance Decay Detected" : "Compound State"}
          </h2>
        </div>
        <div className="rounded-full border border-neutral-700 bg-neutral-950/70 px-3 py-1 font-mono text-xs font-bold text-neutral-300">
          {tireCompound ?? "UNKNOWN"}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 font-mono">
        <Metric label="Stint Lap" value={stintLapNumber?.toString() ?? "--"} />
        <Metric label="Deg Index" value={formatValue(degradationIndex)} />
        <Metric label="Trend" value={formatValue(degradationTrend, 5)} />
      </div>

      <div className="mt-4">
        <div className="mb-2 flex justify-between font-mono text-xs uppercase tracking-[0.2em] text-neutral-400">
          <span>Thermal Decay Load</span>
          <span>{normalizedIndex.toFixed(1)}%</span>
        </div>
        <div className="h-3 overflow-hidden rounded-full bg-neutral-950">
          <div
            className={`h-full rounded-full ${
              isPerformanceDecay
                ? "bg-orange-400 shadow-[0_0_18px_rgba(251,146,60,0.75)]"
                : "bg-[#00FF66] shadow-[0_0_18px_rgba(0,255,102,0.45)]"
            }`}
            style={{ width: `${normalizedIndex}%` }}
          />
        </div>
      </div>

      <p className="mt-4 text-sm leading-6 text-neutral-300">
        {isPerformanceDecay
          ? "Sustained slip/load accumulation is trending upward. Reduce traction-zone wheelspin and consider compound management."
          : "No tire degradation trend shift is currently classified by the inference worker."}
      </p>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-950/70 p-3">
      <p className="text-[10px] uppercase tracking-[0.22em] text-neutral-500">
        {label}
      </p>
      <p className="mt-1 text-lg font-black text-white">{value}</p>
    </div>
  );
}
