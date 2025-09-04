import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';

interface EfficiencyData {
  flops_per_token: number[];
  latency_history: number[];
  memory_scaling: [number, number][]; // [seq_len, memory_mb]
  summary: {
    avg_flops_per_token: number;
    avg_latency: number;
    memory_scaling: number;
    throughput_estimate: number;
  };
}

interface EfficiencyMonitorProps {
  className?: string;
}

const EfficiencyMonitor: React.FC<EfficiencyMonitorProps> = ({ className = "" }) => {
  const [efficiency, setEfficiency] = useState<EfficiencyData | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [scalingMode, setScalingMode] = useState<'linear' | 'log'>('linear');

  useEffect(() => {
    fetchEfficiencyData();
  }, []);

  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(fetchEfficiencyData, 3000);
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const fetchEfficiencyData = async () => {
    try {
      const response = await fetch('http://localhost:8000/research/efficiency-metrics');
      if (response.ok) {
        const data = await response.json();
        setEfficiency(data);
      } else {
        generateDemoData();
      }
    } catch (error) {
      console.warn('Failed to fetch efficiency data, generating demo data');
      generateDemoData();
    }
  };

  const generateDemoData = () => {
    // Generate realistic efficiency data
    const flopSeries = Array.from({ length: 50 }, (_, i) => 
      250000 + Math.sin(i / 10) * 50000 + Math.random() * 20000
    );
    
    const latencySeries = Array.from({ length: 50 }, (_, i) => 
      0.05 + Math.cos(i / 8) * 0.01 + Math.random() * 0.005
    );
    
    const memoryScaling: [number, number][] = [];
    for (let seqLen = 16; seqLen <= 512; seqLen *= 2) {
      // Memory grows roughly quadratically for traditional attention
      const memoryMb = (seqLen * seqLen * 4) / (1024 * 1024) * 8 + Math.random() * 10;
      memoryScaling.push([seqLen, memoryMb]);
    }
    
    setEfficiency({
      flops_per_token: flopSeries,
      latency_history: latencySeries,
      memory_scaling: memoryScaling,
      summary: {
        avg_flops_per_token: flopSeries.reduce((a, b) => a + b, 0) / flopSeries.length,
        avg_latency: latencySeries.reduce((a, b) => a + b, 0) / latencySeries.length,
        memory_scaling: 1.8, // Scaling factor
        throughput_estimate: 1 / (latencySeries.reduce((a, b) => a + b, 0) / latencySeries.length)
      }
    });
  };

  const formatNumber = (num: number) => {
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num.toFixed(2);
  };

  const formatBytes = (bytes: number) => {
    if (bytes >= 1024) return (bytes / 1024).toFixed(2) + ' GB';
    return bytes.toFixed(2) + ' MB';
  };

  const getScalingColor = (factor: number) => {
    if (factor < 1.2) return 'text-green-400'; // Sublinear - excellent
    if (factor < 1.5) return 'text-yellow-400'; // Linear - good
    if (factor < 2.0) return 'text-orange-400'; // Slightly superlinear - okay
    return 'text-red-400'; // Quadratic or worse - bad
  };

  const getScalingLabel = (factor: number) => {
    if (factor < 1.2) return 'Sublinear (Excellent)';
    if (factor < 1.5) return 'Linear (Good)';
    if (factor < 2.0) return 'Superlinear (Fair)';
    return 'Quadratic+ (Poor)';
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: { color: '#e2e8f0' }
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
      }
    },
    scales: {
      x: {
        display: true,
        grid: { color: '#374151' },
        ticks: { color: '#9ca3af' }
      },
      y: {
        display: true,
        grid: { color: '#374151' },
        ticks: { color: '#9ca3af' }
      }
    }
  };

  const memoryScalingOptions = {
    ...chartOptions,
    scales: {
      x: {
        ...chartOptions.scales.x,
        title: {
          display: true,
          text: 'Sequence Length',
          color: '#9ca3af'
        },
        type: scalingMode === 'log' ? 'logarithmic' as const : 'linear' as const
      },
      y: {
        ...chartOptions.scales.y,
        title: {
          display: true,
          text: 'Memory (MB)',
          color: '#9ca3af'
        },
        type: scalingMode === 'log' ? 'logarithmic' as const : 'linear' as const
      }
    }
  };

  if (!efficiency) {
    return (
      <div className={`bg-gray-800 p-6 rounded-lg ${className}`}>
        <div className="text-cyan-400 text-lg">Loading Efficiency Monitor...</div>
      </div>
    );
  }

  const flopData = {
    labels: efficiency.flops_per_token.map((_, i) => i.toString()),
    datasets: [{
      label: 'FLOPs per Token',
      data: efficiency.flops_per_token,
      borderColor: '#8b5cf6',
      backgroundColor: '#8b5cf620',
      borderWidth: 2,
      fill: false,
      tension: 0.1,
      pointRadius: 0
    }]
  };

  const latencyData = {
    labels: efficiency.latency_history.map((_, i) => i.toString()),
    datasets: [{
      label: 'Inference Latency (s)',
      data: efficiency.latency_history,
      borderColor: '#f59e0b',
      backgroundColor: '#f59e0b20',
      borderWidth: 2,
      fill: false,
      tension: 0.1,
      pointRadius: 0
    }]
  };

  const memoryScalingData = {
    labels: efficiency.memory_scaling.map(([seqLen]) => seqLen.toString()),
    datasets: [{
      label: 'Memory Usage (MB)',
      data: efficiency.memory_scaling.map(([_, memory]) => memory),
      borderColor: '#ef4444',
      backgroundColor: '#ef444420',
      borderWidth: 3,
      fill: false,
      tension: 0.1,
      pointRadius: 4,
      pointHoverRadius: 6
    }]
  };

  return (
    <div className={`bg-gray-800 p-6 rounded-lg space-y-6 ${className}`}>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <h2 className="text-2xl font-bold text-cyan-400">⚡ Efficiency Monitor</h2>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded"
            />
            <span className="text-gray-300">Auto Refresh</span>
          </label>
          <button
            onClick={fetchEfficiencyData}
            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Key Efficiency Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-700 p-4 rounded-lg">
          <div className="text-gray-300 text-sm">Avg FLOPs/Token</div>
          <div className="text-xl font-bold text-purple-400">
            {formatNumber(efficiency.summary.avg_flops_per_token)}
          </div>
          <div className="text-xs text-gray-400">Computational cost</div>
        </div>
        <div className="bg-gray-700 p-4 rounded-lg">
          <div className="text-gray-300 text-sm">Avg Latency</div>
          <div className="text-xl font-bold text-yellow-400">
            {(efficiency.summary.avg_latency * 1000).toFixed(2)}ms
          </div>
          <div className="text-xs text-gray-400">Per inference</div>
        </div>
        <div className="bg-gray-700 p-4 rounded-lg">
          <div className="text-gray-300 text-sm">Throughput</div>
          <div className="text-xl font-bold text-green-400">
            {efficiency.summary.throughput_estimate.toFixed(1)} tok/s
          </div>
          <div className="text-xs text-gray-400">Generation speed</div>
        </div>
        <div className="bg-gray-700 p-4 rounded-lg">
          <div className="text-gray-300 text-sm">Memory Scaling</div>
          <div className={`text-xl font-bold ${getScalingColor(efficiency.summary.memory_scaling)}`}>
            O(n^{efficiency.summary.memory_scaling.toFixed(1)})
          </div>
          <div className="text-xs text-gray-400">Sequence scaling</div>
        </div>
      </div>

      {/* FLOPs History */}
      <div className="bg-gray-900 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-cyan-400 mb-4">FLOPs per Token History</h3>
        <div className="h-48">
          <Line data={flopData} options={chartOptions} />
        </div>
        <div className="mt-2 text-sm text-gray-400">
          Lower is better. NSM should show reduced computational cost compared to full attention.
        </div>
      </div>

      {/* Latency History */}
      <div className="bg-gray-900 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-cyan-400 mb-4">Inference Latency History</h3>
        <div className="h-48">
          <Line data={latencyData} options={chartOptions} />
        </div>
        <div className="mt-2 text-sm text-gray-400">
          Measures end-to-end inference time per token. Stability indicates predictable performance.
        </div>
      </div>

      {/* Memory Scaling Analysis */}
      <div className="bg-gray-900 p-4 rounded-lg">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-cyan-400">Memory Scaling Analysis</h3>
          <div className="flex gap-2">
            <button
              onClick={() => setScalingMode('linear')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                scalingMode === 'linear' ? 'bg-blue-600' : 'bg-gray-600 hover:bg-gray-700'
              }`}
            >
              Linear
            </button>
            <button
              onClick={() => setScalingMode('log')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                scalingMode === 'log' ? 'bg-blue-600' : 'bg-gray-600 hover:bg-gray-700'
              }`}
            >
              Log Scale
            </button>
          </div>
        </div>
        <div className="h-64">
          <Line data={memoryScalingData} options={memoryScalingOptions} />
        </div>
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="text-gray-300 font-semibold mb-2">Scaling Analysis:</h4>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Scaling Factor:</span>
                <span className={`font-mono ${getScalingColor(efficiency.summary.memory_scaling)}`}>
                  {efficiency.summary.memory_scaling.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Classification:</span>
                <span className={`font-semibold ${getScalingColor(efficiency.summary.memory_scaling)}`}>
                  {getScalingLabel(efficiency.summary.memory_scaling)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Max Seq Length:</span>
                <span className="font-mono text-blue-400">
                  {Math.max(...efficiency.memory_scaling.map(([seqLen]) => seqLen))}
                </span>
              </div>
            </div>
          </div>
          <div>
            <h4 className="text-gray-300 font-semibold mb-2">NSM vs Transformer:</h4>
            <div className="space-y-1 text-sm text-gray-400">
              <div>• Transformer: O(n²) memory scaling</div>
              <div>• NSM: Should show O(n) or better</div>
              <div>• State compression reduces memory</div>
              <div>• Hierarchical levels enable long context</div>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Comparison Table */}
      <div className="bg-gray-900 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-cyan-400 mb-4">Performance Benchmarks</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-gray-700">
              <tr className="text-gray-300">
                <th className="text-left p-2">Sequence Length</th>
                <th className="text-left p-2">Memory (MB)</th>
                <th className="text-left p-2">Est. FLOPs</th>
                <th className="text-left p-2">Efficiency Rating</th>
              </tr>
            </thead>
            <tbody>
              {efficiency.memory_scaling.map(([seqLen, memory], index) => {
                const estimatedFlops = seqLen * efficiency.summary.avg_flops_per_token;
                const efficiencyRating = seqLen / memory; // Higher is better
                return (
                  <tr key={seqLen} className="border-b border-gray-800">
                    <td className="p-2 font-mono text-blue-400">{seqLen}</td>
                    <td className="p-2 font-mono">{memory.toFixed(2)}</td>
                    <td className="p-2 font-mono text-purple-400">{formatNumber(estimatedFlops)}</td>
                    <td className="p-2">
                      <span className={`font-semibold ${
                        efficiencyRating > 20 ? 'text-green-400' :
                        efficiencyRating > 10 ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {efficiencyRating > 20 ? 'Excellent' :
                         efficiencyRating > 10 ? 'Good' : 'Poor'}
                      </span>
                      <span className="text-gray-400 ml-2">({efficiencyRating.toFixed(1)})</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Optimization Recommendations */}
      <div className="bg-gray-900 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-cyan-400 mb-4">Optimization Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h4 className="text-gray-300 font-semibold mb-2">Current Performance:</h4>
            <ul className="space-y-1 text-gray-400">
              <li className={`${efficiency.summary.memory_scaling < 1.5 ? 'text-green-400' : 'text-yellow-400'}`}>
                • Memory scaling: {getScalingLabel(efficiency.summary.memory_scaling)}
              </li>
              <li className={`${efficiency.summary.throughput_estimate > 20 ? 'text-green-400' : 'text-yellow-400'}`}>
                • Throughput: {efficiency.summary.throughput_estimate.toFixed(1)} tokens/sec
              </li>
              <li className={`${efficiency.summary.avg_latency < 0.1 ? 'text-green-400' : 'text-yellow-400'}`}>
                • Latency: {(efficiency.summary.avg_latency * 1000).toFixed(1)}ms per token
              </li>
            </ul>
          </div>
          <div>
            <h4 className="text-gray-300 font-semibold mb-2">Optimization Tips:</h4>
            <ul className="space-y-1 text-gray-400">
              <li>• Increase state compression for longer sequences</li>
              <li>• Use gradient checkpointing for memory efficiency</li>
              <li>• Optimize state eviction policies</li>
              <li>• Consider mixed precision training</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EfficiencyMonitor;
