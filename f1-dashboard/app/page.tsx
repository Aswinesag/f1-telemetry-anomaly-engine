"use client";

import { Activity, AlertTriangle, CheckCircle, Gauge, Thermometer } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

// Define the shape of our incoming AI telemetry packets
interface TelemetryPacket {
  TimeSec: number;
  Speed: number;
  Brake: number;
  Predicted_Temp: number;
  Actual_Temp: number;
  Anomaly_Score: number;
  Alert_Threshold: number;
  Is_Anomaly: boolean;
}

const MAX_HISTORY_LENGTH = 100;

function getTelemetrySnapshotUrl(wsUrl: string) {
  const parsedUrl = new URL(wsUrl);
  parsedUrl.protocol = parsedUrl.protocol === 'wss:' ? 'https:' : 'http:';
  parsedUrl.pathname = '/telemetry/latest';
  parsedUrl.search = '';
  parsedUrl.hash = '';
  return parsedUrl.toString();
}

export default function F1PitwallDashboard() {
  const [history, setHistory] = useState<TelemetryPacket[]>([]);
  const [currentStatus, setCurrentStatus] = useState<TelemetryPacket | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Mount the WebSocket connection to the inference backend
    // Assuming a FastAPI backend running on localhost:8000
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL ?? 'ws://localhost:8000/ws/telemetry';

    fetch(getTelemetrySnapshotUrl(wsUrl))
      .then((response) => (response.ok ? response.json() : null))
      .then((packet: TelemetryPacket | null) => {
        if (!packet) return;
        setCurrentStatus(packet);
        setHistory([packet]);
      })
      .catch((error) => {
        console.error("Telemetry Snapshot Error:", error);
      });

    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onmessage = (event) => {
      const packet: TelemetryPacket = JSON.parse(event.data);
      
      setCurrentStatus(packet);
      
      setHistory((prev) => {
        const newHistory = [...prev, packet];
        if (newHistory.length > MAX_HISTORY_LENGTH) {
          return newHistory.slice(newHistory.length - MAX_HISTORY_LENGTH);
        }
        return newHistory;
      });
    };

    wsRef.current.onerror = (error) => {
      console.error("WebSocket Telemetry Error:", error);
    };

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return (
    <div className="min-h-screen bg-neutral-950 text-white p-6 font-sans">
      
      {/* HEADER */}
      <header className="mb-8 border-b border-neutral-800 pb-4">
        <h1 className="text-3xl font-bold tracking-tight text-red-600 flex items-center gap-3">
          <Activity className="w-8 h-8" />
          F1 VIRTUAL THERMAL SENSOR & ANOMALY ISOLATION SUITE
        </h1>
      </header>

      {/* CRITICAL ALERT BANNER */}
      {currentStatus?.Is_Anomaly === true && (
        <div className="mb-6 bg-red-900/50 border border-red-500 p-4 rounded-lg flex items-center gap-4 animate-pulse">
          <AlertTriangle className="w-8 h-8 text-red-500" />
          <div>
            <h3 className="text-xl font-bold text-red-400">CRITICAL SYSTEM ALARM</h3>
            <p className="text-red-200">
              Anomaly score ({currentStatus?.Anomaly_Score?.toFixed(4) ?? 'N/A'}) breaches background safety limits!
            </p>
          </div>
        </div>
      )}

      {/* METRICS ROW */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <MetricCard 
          icon={<Gauge className="w-6 h-6 text-blue-400" />}
          title="Vehicle Velocity" 
          value={`${currentStatus?.Speed ? Math.round(currentStatus.Speed) : '--'} km/h`} 
        />
        <MetricCard 
          icon={<AlertTriangle className="w-6 h-6 text-orange-400" />}
          title="Brake Application" 
          value={`${currentStatus?.Brake ? Math.round(currentStatus.Brake) : '--'} %`} 
        />
        <MetricCard 
          icon={<Thermometer className="w-6 h-6 text-green-400" />}
          title="Virtual Thermal State" 
          value={`${currentStatus?.Predicted_Temp ? currentStatus.Predicted_Temp.toFixed(1) : '--'} °C`} 
        />
        
        {/* Dynamic Health Card */}
        <div className={`p-6 rounded-xl border flex flex-col justify-center ${currentStatus?.Is_Anomaly ? 'bg-red-950/30 border-red-800' : 'bg-neutral-900 border-neutral-800'}`}>
          <div className="flex items-center gap-3 mb-2 text-neutral-400 font-medium">
            {currentStatus?.Is_Anomaly ? <AlertTriangle className="w-6 h-6 text-red-500" /> : <CheckCircle className="w-6 h-6 text-emerald-500" />}
            Telemetry Health
          </div>
          <div className="text-3xl font-bold tabular-nums tracking-tight">
            {currentStatus?.Is_Anomaly ? <span className="text-red-500">CRITICAL FAILURE</span> : <span className="text-emerald-500">NOMINAL</span>}
          </div>
          <div className="text-sm mt-1 text-neutral-500">
            Score: {currentStatus?.Anomaly_Score ? currentStatus.Anomaly_Score.toFixed(4) : '--'}
          </div>
          <div className="text-sm mt-1 text-neutral-500">
            Threshold: {currentStatus?.Alert_Threshold ? currentStatus.Alert_Threshold.toFixed(4) : '--'}
          </div>
        </div>
      </div>

      {/* CHARTS ROW */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Thermal Chart */}
        <div className="bg-neutral-900 border border-neutral-800 p-6 rounded-xl">
          <h3 className="text-lg font-semibold mb-4 text-neutral-300">Live Performance Tracks: Sensors vs AI</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                <XAxis dataKey="TimeSec" stroke="#666" tick={{fill: '#666'}} tickFormatter={(val) => val.toFixed(1)} />
                <YAxis stroke="#666" tick={{fill: '#666'}} domain={['auto', 'auto']} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#171717', border: '1px solid #333', borderRadius: '8px' }}
                  labelStyle={{ color: '#aaa' }}
                />
                <Line type="monotone" dataKey="Actual_Temp" stroke="#FF1801" strokeWidth={2} dot={false} isAnimationActive={false} name="Actual °C" />
                <Line type="monotone" dataKey="Predicted_Temp" stroke="#00FF66" strokeWidth={2} dot={false} isAnimationActive={false} name="Predicted °C" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Anomaly Chart */}
        <div className="bg-neutral-900 border border-neutral-800 p-6 rounded-xl">
          <h3 className="text-lg font-semibold mb-4 text-neutral-300">Isolation Engine Loss Signature</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                <XAxis dataKey="TimeSec" stroke="#666" tick={{fill: '#666'}} tickFormatter={(val) => val.toFixed(1)} />
                <YAxis stroke="#666" tick={{fill: '#666'}} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#171717', border: '1px solid #333', borderRadius: '8px' }}
                  labelStyle={{ color: '#aaa' }}
                />
                {currentStatus?.Alert_Threshold !== undefined && (
                  <ReferenceLine y={currentStatus.Alert_Threshold} stroke="#FF1801" strokeDasharray="3 3" label={{ position: 'top', value: `Safety Boundary ${currentStatus.Alert_Threshold.toFixed(2)}`, fill: '#FF1801', fontSize: 12 }} />
                )}
                <Line type="stepAfter" dataKey="Anomaly_Score" stroke="#FFA500" strokeWidth={2} dot={false} isAnimationActive={false} name="Loss Intensity" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
}

// Reusable micro-component for the top metrics
function MetricCard({ title, value, icon }: { title: string, value: string | number, icon: React.ReactNode }) {
  return (
    <div className="bg-neutral-900 border border-neutral-800 p-6 rounded-xl flex flex-col justify-center">
      <div className="flex items-center gap-3 mb-2 text-neutral-400 font-medium">
        {icon}
        {title}
      </div>
      <div className="text-3xl font-bold tabular-nums tracking-tight">
        {value}
      </div>
    </div>
  );
}
