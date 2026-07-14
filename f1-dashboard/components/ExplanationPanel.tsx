"use client";

export interface AttributionExplanation {
  top_factor: string;
  importance_score: number;
  fault_type?: string;
  recommendation?: string;
  feature_importance?: Record<string, number>;
}

interface ExplanationPanelProps {
  explanation?: AttributionExplanation | null;
  anomalyScore?: number;
  alertThreshold?: number;
  isAnomaly: boolean;
  className?: string;
}

function formatFactorName(featureName: string) {
  return featureName.replaceAll("_", " ");
}

export default function ExplanationPanel({
  explanation,
  anomalyScore,
  alertThreshold,
  isAnomaly,
  className = "",
}: ExplanationPanelProps) {
  if (!isAnomaly || !explanation) return null;

  const normalizedScore = Math.min(
    100,
    Math.max(0, explanation.importance_score * 100),
  );
  const scoreLabel = `${normalizedScore.toFixed(1)}%`;
  const topFactor = formatFactorName(explanation.top_factor);

  return (
    <section
      className={`rounded-2xl border border-red-500/40 bg-red-950/20 p-4 text-red-50 shadow-[0_0_32px_rgba(255,0,51,0.12)] backdrop-blur ${className}`}
    >
      <div className="mb-4 flex items-start justify-between gap-4 border-b border-red-500/20 pb-3">
        <div>
          <p className="font-mono text-xs font-bold uppercase tracking-[0.28em] text-red-400">
            Pit Recommendation
          </p>
          <h2 className="mt-1 text-xl font-black uppercase tracking-tight text-white">
            Inspect {topFactor}
          </h2>
        </div>
        <div className="rounded-full border border-red-500/40 bg-red-500/10 px-3 py-1 font-mono text-xs font-bold text-red-300">
          {explanation.fault_type ?? "XAI"}
        </div>
      </div>

      <div className="mb-3 flex items-end justify-between gap-3 font-mono">
        <span className="text-sm uppercase tracking-[0.22em] text-red-200">
          Contribution
        </span>
        <span className="text-3xl font-black text-red-400">{scoreLabel}</span>
      </div>

      <div className="h-3 overflow-hidden rounded-full border border-red-500/30 bg-neutral-950">
        <div
          className="h-full rounded-full bg-[#FF0033] shadow-[0_0_20px_rgba(255,0,51,0.75)]"
          style={{ width: scoreLabel }}
        />
      </div>

      <p className="mt-4 text-sm leading-6 text-red-100/90">
        Attribution indicates <span className="font-semibold text-white">{topFactor}</span>{" "}
        is the dominant contributor to the current anomaly signature.{" "}
        {explanation.recommendation ??
          "Prioritize brake thermal load, cooling efficiency, and correlated telemetry channels before the next stint decision."}
      </p>

      <div className="mt-4 grid grid-cols-2 gap-3 font-mono text-xs uppercase tracking-[0.18em] text-red-200/80">
        <div className="rounded-xl border border-red-500/20 bg-neutral-950/60 p-3">
          Score
          <span className="block pt-1 text-base font-bold text-red-300">
            {anomalyScore?.toFixed(3) ?? "--"}
          </span>
        </div>
        <div className="rounded-xl border border-red-500/20 bg-neutral-950/60 p-3">
          Limit
          <span className="block pt-1 text-base font-bold text-red-300">
            {alertThreshold?.toFixed(3) ?? "--"}
          </span>
        </div>
      </div>
    </section>
  );
}
