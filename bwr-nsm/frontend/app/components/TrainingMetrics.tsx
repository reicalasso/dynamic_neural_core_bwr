import React, { useState, useEffect } from 'react';

interface MetricData {
  step: number;
  value: number;
  timestamp: string;
}

const TrainingMetrics: React.FC = () => {
  const [metrics, setMetrics] = useState<{
    loss: MetricData[];
    accuracy: MetricData[];
    learning_rate: MetricData[];
    memory_usage: MetricData[];
  }>({
    loss: [],
    accuracy: [],
    learning_rate: [],
    memory_usage: []
  });

  // Generate mock training metrics
  useEffect(() => {
    const generateMetrics = () => {
      const steps = 50;
      const now = new Date();
      
      const loss = Array.from({ length: steps }, (_, i) => ({
        step: i * 10,
        value: Math.max(0.1, 4.0 * Math.exp(-i * 0.1) + Math.random() * 0.3),
        timestamp: new Date(now.getTime() - (steps - i) * 60000).toISOString()
      }));

      const accuracy = Array.from({ length: steps }, (_, i) => ({
        step: i * 10,
        value: Math.min(0.95, 0.1 + (i / steps) * 0.8 + Math.random() * 0.1),
        timestamp: new Date(now.getTime() - (steps - i) * 60000).toISOString()
      }));

      const learning_rate = Array.from({ length: steps }, (_, i) => ({
        step: i * 10,
        value: 3e-4 * Math.cos(i / steps * Math.PI / 2),
        timestamp: new Date(now.getTime() - (steps - i) * 60000).toISOString()
      }));

      const memory_usage = Array.from({ length: steps }, (_, i) => ({
        step: i * 10,
        value: 60 + Math.sin(i / 10) * 20 + Math.random() * 10,
        timestamp: new Date(now.getTime() - (steps - i) * 60000).toISOString()
      }));

      setMetrics({ loss, accuracy, learning_rate, memory_usage });
    };

    generateMetrics();
    
    // Update metrics periodically
    const interval = setInterval(() => {
      setMetrics(prev => {
        const newStep = (prev.loss[prev.loss.length - 1]?.step || 0) + 10;
        const now = new Date().toISOString();
        
        return {
          loss: [...prev.loss.slice(-49), {
            step: newStep,
            value: Math.max(0.1, prev.loss[prev.loss.length - 1].value * 0.995 + Math.random() * 0.05),
            timestamp: now
          }],
          accuracy: [...prev.accuracy.slice(-49), {
            step: newStep,
            value: Math.min(0.95, prev.accuracy[prev.accuracy.length - 1].value + Math.random() * 0.02 - 0.01),
            timestamp: now
          }],
          learning_rate: [...prev.learning_rate.slice(-49), {
            step: newStep,
            value: prev.learning_rate[prev.learning_rate.length - 1].value * 0.9999,
            timestamp: now
          }],
          memory_usage: [...prev.memory_usage.slice(-49), {
            step: newStep,
            value: 60 + Math.sin(newStep / 100) * 20 + Math.random() * 10,
            timestamp: now
          }]
        };
      });
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const renderChart = (data: MetricData[], title: string, color: string, unit: string = '') => {
    const maxValue = Math.max(...data.map(d => d.value));
    const minValue = Math.min(...data.map(d => d.value));
    const range = maxValue - minValue;
    
    return (
      <div className="bg-gray-900 p-4 rounded-lg">
        <h4 className="text-lg font-semibold mb-3" style={{ color }}>{title}</h4>
        
        {/* Current value display */}
        <div className="mb-4">
          <span className="text-2xl font-bold" style={{ color }}>
            {data[data.length - 1]?.value.toFixed(4)}{unit}
          </span>
          <span className="text-sm text-gray-400 ml-2">
            Step {data[data.length - 1]?.step}
          </span>
        </div>
        
        {/* Simple SVG chart */}
        <div className="bg-gray-800 p-2 rounded">
          <svg width="100%" height="120" viewBox="0 0 300 120">
            <defs>
              <linearGradient id={`gradient-${title}`} x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style={{ stopColor: color, stopOpacity: 0.3 }} />
                <stop offset="100%" style={{ stopColor: color, stopOpacity: 0.1 }} />
              </linearGradient>
            </defs>
            
            {/* Grid lines */}
            {[25, 50, 75].map(y => (
              <line 
                key={y} 
                x1="0" 
                y1={y} 
                x2="300" 
                y2={y} 
                stroke="#374151" 
                strokeWidth="1"
                strokeDasharray="2,2"
              />
            ))}
            
            {/* Data line */}
            <polyline
              fill="none"
              stroke={color}
              strokeWidth="2"
              points={data.map((d, i) => 
                `${(i / (data.length - 1)) * 300},${120 - ((d.value - minValue) / range) * 100}`
              ).join(' ')}
            />
            
            {/* Fill area */}
            <polygon
              fill={`url(#gradient-${title})`}
              points={
                data.map((d, i) => 
                  `${(i / (data.length - 1)) * 300},${120 - ((d.value - minValue) / range) * 100}`
                ).join(' ') + ` 300,120 0,120`
              }
            />
          </svg>
        </div>
        
        {/* Min/Max values */}
        <div className="flex justify-between text-xs text-gray-400 mt-2">
          <span>Min: {minValue.toFixed(4)}</span>
          <span>Max: {maxValue.toFixed(4)}</span>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-xl font-semibold mb-4 text-cyan-400">Training Metrics Dashboard</h3>
        
        {/* Summary cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-900 p-4 rounded-lg text-center">
            <div className="text-sm text-gray-400">Current Loss</div>
            <div className="text-2xl font-bold text-red-400">
              {metrics.loss[metrics.loss.length - 1]?.value.toFixed(4) || '---'}
            </div>
          </div>
          <div className="bg-gray-900 p-4 rounded-lg text-center">
            <div className="text-sm text-gray-400">Accuracy</div>
            <div className="text-2xl font-bold text-green-400">
              {((metrics.accuracy[metrics.accuracy.length - 1]?.value || 0) * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-gray-900 p-4 rounded-lg text-center">
            <div className="text-sm text-gray-400">Learning Rate</div>
            <div className="text-2xl font-bold text-blue-400">
              {metrics.learning_rate[metrics.learning_rate.length - 1]?.value.toExponential(2) || '---'}
            </div>
          </div>
          <div className="bg-gray-900 p-4 rounded-lg text-center">
            <div className="text-sm text-gray-400">Memory Usage</div>
            <div className="text-2xl font-bold text-purple-400">
              {metrics.memory_usage[metrics.memory_usage.length - 1]?.value.toFixed(0) || '---'}%
            </div>
          </div>
        </div>
      </div>

      {/* Charts grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {renderChart(metrics.loss, 'Training Loss', '#ef4444')}
        {renderChart(metrics.accuracy, 'Accuracy', '#22c55e', '%')}
        {renderChart(metrics.learning_rate, 'Learning Rate', '#3b82f6')}
        {renderChart(metrics.memory_usage, 'GPU Memory Usage', '#a855f7', '%')}
      </div>

      {/* Training status */}
      <div className="bg-gray-800 p-6 rounded-lg">
        <h4 className="text-lg font-semibold mb-3 text-cyan-400">Training Status</h4>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-green-400 text-sm">Status</div>
            <div className="text-white font-semibold">🟢 Training</div>
          </div>
          <div className="text-center">
            <div className="text-blue-400 text-sm">Epoch</div>
            <div className="text-white font-semibold">3/10</div>
          </div>
          <div className="text-center">
            <div className="text-purple-400 text-sm">Time Elapsed</div>
            <div className="text-white font-semibold">2h 34m</div>
          </div>
          <div className="text-center">
            <div className="text-orange-400 text-sm">ETA</div>
            <div className="text-white font-semibold">5h 12m</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingMetrics;
