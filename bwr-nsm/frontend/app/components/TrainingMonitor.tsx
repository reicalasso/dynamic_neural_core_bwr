import React, { useEffect, useState, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface TrainingMetrics {
  loss_curve: number[];
  accuracy_curve: number[];
  learning_rate: number;
  gradient_norms: number[];
  convergence_speed: number;
}

interface TrainingMonitorProps {
  className?: string;
}

const TrainingMonitor: React.FC<TrainingMonitorProps> = ({ className = "" }) => {
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [isLive, setIsLive] = useState(false);
  const [simulationActive, setSimulationActive] = useState(false);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    fetchTrainingMetrics();
  }, []);

  const fetchTrainingMetrics = async () => {
    try {
      const response = await fetch('http://localhost:8000/research/advanced-metrics');
      if (response.ok) {
        const data = await response.json();
        setMetrics(data.training);
        setIsLive(true);
      } else {
        // If no real metrics, create demo data
        generateDemoData();
      }
    } catch (error) {
      console.warn('Live metrics unavailable, generating demo data');
      generateDemoData();
    }
  };

  const generateDemoData = () => {
    const steps = 100;
    const lossData = [];
    const accData = [];
    const gradNorms = [];
    
    // Simulate training curves
    for (let i = 0; i < steps; i++) {
      // Exponential decay loss with some noise
      const baseLoss = 2.5 * Math.exp(-i / 30) + 0.1;
      const noisyLoss = baseLoss + (Math.random() - 0.5) * 0.1;
      lossData.push(Math.max(0.05, noisyLoss));
      
      // Corresponding accuracy (inverse relationship with some saturation)
      const baseAcc = 1 - Math.exp(-i / 20);
      const noisyAcc = baseAcc + (Math.random() - 0.5) * 0.02;
      accData.push(Math.min(0.95, Math.max(0.1, noisyAcc)));
      
      // Gradient norms - start high, stabilize
      const baseGrad = 5.0 * Math.exp(-i / 25) + 0.5;
      const noisyGrad = baseGrad + (Math.random() - 0.5) * 0.3;
      gradNorms.push(Math.max(0.1, noisyGrad));
    }

    setMetrics({
      loss_curve: lossData,
      accuracy_curve: accData,
      learning_rate: 0.001,
      gradient_norms: gradNorms,
      convergence_speed: 0.02
    });
    setIsLive(false);
  };

  const startSimulation = () => {
    setSimulationActive(true);
    intervalRef.current = setInterval(() => {
      setMetrics(prev => {
        if (!prev) return prev;
        
        // Add new data points with realistic progression
        const lastLoss = prev.loss_curve[prev.loss_curve.length - 1] || 1.0;
        const lastAcc = prev.accuracy_curve[prev.accuracy_curve.length - 1] || 0.5;
        const lastGrad = prev.gradient_norms[prev.gradient_norms.length - 1] || 1.0;
        
        const newLoss = Math.max(0.05, lastLoss * 0.995 + (Math.random() - 0.5) * 0.02);
        const newAcc = Math.min(0.98, lastAcc + 0.001 + (Math.random() - 0.5) * 0.005);
        const newGrad = Math.max(0.1, lastGrad * 0.99 + (Math.random() - 0.5) * 0.1);
        
        return {
          ...prev,
          loss_curve: [...prev.loss_curve.slice(-99), newLoss],
          accuracy_curve: [...prev.accuracy_curve.slice(-99), newAcc],
          gradient_norms: [...prev.gradient_norms.slice(-99), newGrad],
          convergence_speed: (lastLoss - newLoss) / Math.max(lastLoss, 0.001)
        };
      });
    }, 500);
  };

  const stopSimulation = () => {
    setSimulationActive(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const generateChartData = (values: number[], label: string, color: string, yAxisID?: string) => ({
    labels: values.map((_, i) => i.toString()),
    datasets: [{
      label,
      data: values,
      borderColor: color,
      backgroundColor: color + '20',
      borderWidth: 2,
      fill: false,
      tension: 0.1,
      pointRadius: 0,
      pointHoverRadius: 4,
      yAxisID
    }]
  });

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: { color: '#e2e8f0' }
      },
      tooltip: {
        mode: 'index',
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
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  const dualAxisOptions: ChartOptions<'line'> = {
    ...chartOptions,
    scales: {
      ...chartOptions.scales,
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        grid: { color: '#374151' },
        ticks: { color: '#9ca3af' }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        grid: { drawOnChartArea: false },
        ticks: { color: '#9ca3af' }
      }
    }
  };

  const lossAccuracyData = metrics ? {
    labels: metrics.loss_curve.map((_, i) => i.toString()),
    datasets: [
      {
        label: 'Loss',
        data: metrics.loss_curve,
        borderColor: '#ef4444',
        backgroundColor: '#ef444420',
        borderWidth: 2,
        fill: false,
        tension: 0.1,
        pointRadius: 0,
        yAxisID: 'y'
      },
      {
        label: 'Accuracy',
        data: metrics.accuracy_curve,
        borderColor: '#10b981',
        backgroundColor: '#10b98120',
        borderWidth: 2,
        fill: false,
        tension: 0.1,
        pointRadius: 0,
        yAxisID: 'y1'
      }
    ]
  } : null;

  if (!metrics) {
    return (
      <div className={`bg-gray-800 p-6 rounded-lg ${className}`}>
        <div className="text-cyan-400 text-lg">Loading Training Monitor...</div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800 p-6 rounded-lg space-y-6 ${className}`}>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <h2 className="text-2xl font-bold text-cyan-400">🔥 Training Monitor</h2>
        <div className="flex items-center gap-4">
          <div className={`px-3 py-1 rounded text-sm ${isLive ? 'bg-green-600' : 'bg-gray-600'}`}>
            {isLive ? '🔴 Live' : '📊 Demo'}
          </div>
          <div className="flex gap-2">
            <button
              onClick={startSimulation}
              disabled={simulationActive}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded text-sm transition-colors"
            >
              Start Sim
            </button>
            <button
              onClick={stopSimulation}
              disabled={!simulationActive}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 rounded text-sm transition-colors"
            >
              Stop
            </button>
          </div>
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-700 p-4 rounded-lg">
          <div className="text-gray-300 text-sm">Current Loss</div>
          <div className="text-2xl font-bold text-red-400">
            {(metrics.loss_curve[metrics.loss_curve.length - 1] || 0).toFixed(4)}
          </div>
        </div>
        <div className="bg-gray-700 p-4 rounded-lg">
          <div className="text-gray-300 text-sm">Current Accuracy</div>
          <div className="text-2xl font-bold text-green-400">
            {((metrics.accuracy_curve[metrics.accuracy_curve.length - 1] || 0) * 100).toFixed(2)}%
          </div>
        </div>
        <div className="bg-gray-700 p-4 rounded-lg">
          <div className="text-gray-300 text-sm">Learning Rate</div>
          <div className="text-2xl font-bold text-blue-400">
            {metrics.learning_rate.toExponential(2)}
          </div>
        </div>
        <div className="bg-gray-700 p-4 rounded-lg">
          <div className="text-gray-300 text-sm">Convergence Speed</div>
          <div className="text-2xl font-bold text-purple-400">
            {(metrics.convergence_speed * 100).toFixed(3)}%
          </div>
        </div>
      </div>

      {/* Loss & Accuracy Chart */}
      <div className="bg-gray-900 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-cyan-400 mb-4">Loss & Accuracy Curves</h3>
        <div className="h-64">
          {lossAccuracyData && (
            <Line data={lossAccuracyData} options={dualAxisOptions} />
          )}
        </div>
      </div>

      {/* Gradient Norms Chart */}
      <div className="bg-gray-900 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-cyan-400 mb-4">Gradient Norms (Stability Check)</h3>
        <div className="h-48">
          <Line 
            data={generateChartData(metrics.gradient_norms, 'Gradient Norm', '#f59e0b')} 
            options={chartOptions} 
          />
        </div>
        <div className="mt-2 text-sm">
          <span className="text-gray-400">Current: </span>
          <span className={`font-mono ${metrics.gradient_norms[metrics.gradient_norms.length - 1] > 10 ? 'text-red-400' : 'text-green-400'}`}>
            {(metrics.gradient_norms[metrics.gradient_norms.length - 1] || 0).toFixed(3)}
          </span>
          <span className="ml-4 text-gray-400">Status: </span>
          <span className={`font-semibold ${metrics.gradient_norms[metrics.gradient_norms.length - 1] > 10 ? 'text-red-400' : 'text-green-400'}`}>
            {metrics.gradient_norms[metrics.gradient_norms.length - 1] > 10 ? 'Unstable' : 'Stable'}
          </span>
        </div>
      </div>

      {/* Training Progress Indicator */}
      <div className="bg-gray-900 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-cyan-400 mb-4">Training Progress</h3>
        <div className="space-y-3">
          <div>
            <div className="flex justify-between text-sm text-gray-300 mb-1">
              <span>Loss Reduction</span>
              <span>{((1 - (metrics.loss_curve[metrics.loss_curve.length - 1] / Math.max(metrics.loss_curve[0], 0.001))) * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-red-500 h-2 rounded-full transition-all duration-300" 
                style={{ width: `${Math.min(100, (1 - (metrics.loss_curve[metrics.loss_curve.length - 1] / Math.max(metrics.loss_curve[0], 0.001))) * 100)}%` }}
              />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-sm text-gray-300 mb-1">
              <span>Accuracy Progress</span>
              <span>{(metrics.accuracy_curve[metrics.accuracy_curve.length - 1] * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-green-500 h-2 rounded-full transition-all duration-300" 
                style={{ width: `${metrics.accuracy_curve[metrics.accuracy_curve.length - 1] * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingMonitor;
