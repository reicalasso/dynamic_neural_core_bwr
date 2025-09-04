import React, { useEffect, useRef, useState } from 'react';

type MetricsPayload = {
  timestamp: string;
  memory: {
    levels: { level: number; slots: number; active_slots: number; avg_salience: number; avg_age: number; avg_access: number }[];
    total_slots: number;
    total_active_slots: number;
  };
  generation: {
    total_requests: number;
    total_generated_tokens: number;
    avg_tokens_per_request: number;
    recent_tps: number;
  };
  performance: { gpu_memory_mb: number | null; gpu_utilization: number | null };
  compression: { levels: number; hierarchical_factor: number | null };
};

const MAX_POINTS = 50;

const ResearchDashboard: React.FC = () => {
  const [isClient, setIsClient] = useState(false);
  const [metrics, setMetrics] = useState<MetricsPayload | null>(null);
  const [tpsSeries, setTpsSeries] = useState<number[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const [connectionState, setConnectionState] = useState<'connecting' | 'live' | 'fallback'>('connecting');

  useEffect(() => { setIsClient(true); }, []);

  useEffect(() => {
    if (!isClient) return;
    const url = `ws://localhost:8000/ws/research`;
    try {
      wsRef.current = new WebSocket(url);
      wsRef.current.onopen = () => setConnectionState('live');
      wsRef.current.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === 'research_metrics') {
            updateMetrics(msg.data as MetricsPayload);
          }
        } catch { /* ignore */ }
      };
      wsRef.current.onerror = () => {
        setConnectionState('fallback');
        wsRef.current?.close();
      };
      wsRef.current.onclose = () => {
        if (connectionState === 'live') setConnectionState('fallback');
      };
    } catch {
      setConnectionState('fallback');
    }

    return () => { wsRef.current?.close(); };
  }, [isClient]);

  // Fallback polling
  useEffect(() => {
    if (!isClient || connectionState !== 'fallback') return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:8000/research/metrics');
        const data = await res.json();
        updateMetrics(data as MetricsPayload);
      } catch {/* ignore */}
    }, 3000);
    return () => clearInterval(interval);
  }, [isClient, connectionState]);

  const updateMetrics = (data: MetricsPayload) => {
    setMetrics(data);
    setTpsSeries(prev => [...prev.slice(-(MAX_POINTS-1)), data.generation.recent_tps]);
  };

  if (!isClient) {
    return <div className="text-cyan-400 p-8">Loading Research Dashboard…</div>;
  }

  return (
    <div className="space-y-6">
      <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <h1 className="text-3xl font-bold text-cyan-400">Research Metrics Dashboard</h1>
        <div className="text-sm text-gray-400">
          Connection: {connectionState === 'live' ? <span className="text-green-400">Live WS</span> : connectionState === 'fallback' ? <span className="text-yellow-400">Polling</span> : 'Connecting…'}
        </div>
      </header>

      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <SummaryCard label="Total Slots" value={metrics?.memory.total_slots ?? 0} color="cyan" />
        <SummaryCard label="Active Slots" value={metrics?.memory.total_active_slots ?? 0} color="green" />
        <SummaryCard label="Total Requests" value={metrics?.generation.total_requests ?? 0} color="blue" />
        <SummaryCard label="Tokens Generated" value={metrics?.generation.total_generated_tokens ?? 0} color="purple" />
      </div>

      {/* Memory levels */}
      <section className="bg-gray-800 p-4 rounded-lg">
        <h2 className="text-xl font-semibold text-cyan-400 mb-4">Memory Levels</h2>
        <div className="grid gap-4 md:grid-cols-3">
          {metrics?.memory.levels.map(level => (
            <div key={level.level} className="bg-gray-900 p-4 rounded border border-gray-700">
              <div className="text-gray-300 mb-2 font-medium">Level {level.level}</div>
              <div className="text-sm text-gray-400 mb-2">Slots: {level.slots}</div>
              <Bar label="Active" value={level.active_slots} max={level.slots} color="bg-green-600" />
              <MiniStat label="Avg Salience" value={level.avg_salience.toFixed(3)} />
              <MiniStat label="Avg Age" value={level.avg_age.toFixed(1)} />
              <MiniStat label="Avg Access" value={level.avg_access.toFixed(1)} />
            </div>
          ))}
        </div>
      </section>

      {/* Throughput line (simple sparkline style) */}
      <section className="bg-gray-800 p-4 rounded-lg">
        <h2 className="text-xl font-semibold text-cyan-400 mb-4">Recent Tokens / Sec</h2>
        <div className="h-32 flex items-end gap-1">
          {tpsSeries.map((v, i) => (
            <div
              key={i}
              className="flex-1 bg-blue-600 rounded-sm"
              style={{ height: `${Math.min(100, v * 10)}%` }}
              title={v.toFixed(2)}
            />
          ))}
          {tpsSeries.length === 0 && <div className="text-gray-500 text-sm">Waiting for data…</div>}
        </div>
      </section>

      {/* Compression & Performance */}
      <section className="grid gap-4 md:grid-cols-2">
        <div className="bg-gray-800 p-4 rounded-lg">
          <h2 className="text-lg font-semibold text-cyan-400 mb-3">Compression</h2>
            <MiniStat label="Levels" value={metrics?.compression.levels ?? 0} />
            <MiniStat label="Hierarchical Factor" value={metrics?.compression.hierarchical_factor?.toFixed(2) ?? '-'} />
        </div>
        <div className="bg-gray-800 p-4 rounded-lg">
          <h2 className="text-lg font-semibold text-cyan-400 mb-3">Performance (GPU)</h2>
            <MiniStat label="Memory MB" value={metrics?.performance.gpu_memory_mb?.toFixed(1) ?? '-'} />
            <MiniStat label="Utilization" value={metrics?.performance.gpu_utilization?.toFixed?.(1) ?? '-'} />
        </div>
      </section>
    </div>
  );
};

const SummaryCard: React.FC<{ label: string; value: number; color: string }> = ({ label, value, color }) => (
  <div className="bg-gray-800 p-4 rounded-lg">
    <div className="text-sm text-gray-400">{label}</div>
    <div className={`text-2xl font-bold text-${color}-400`}>{value}</div>
  </div>
);

const MiniStat: React.FC<{ label: string; value: string | number }> = ({ label, value }) => (
  <div className="text-sm text-gray-300 flex justify-between border-b border-gray-700 py-1">
    <span className="text-gray-400">{label}</span>
    <span className="font-mono">{value}</span>
  </div>
);

const Bar: React.FC<{ label: string; value: number; max: number; color: string }> = ({ label, value, max, color }) => {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{label}</span>
        <span>{value}/{max}</span>
      </div>
      <div className="w-full h-3 bg-gray-700 rounded overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
};

export default ResearchDashboard;
