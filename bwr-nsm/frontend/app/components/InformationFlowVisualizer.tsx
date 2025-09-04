import React, { useState } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Colors } from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Colors);

interface InformationBottleneck {
  layer: number;
  position: number;
  strength: number;
}

interface FlowAnalysisResult {
  flow_matrices: number[][][];
  bottlenecks: InformationBottleneck[];
  total_flow: number;
  input_text: string;
  timestamp: string;
}

const InformationFlowVisualizer: React.FC = () => {
  const [analysisResult, setAnalysisResult] = useState<FlowAnalysisResult | null>(null);
  const [inputText, setInputText] = useState('The quick brown fox jumps over the lazy dog');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyzeInformationFlow = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/research/information-flow', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: inputText
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setAnalysisResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      console.error('Failed to analyze information flow:', err);
    } finally {
      setLoading(false);
    }
  };

  const generateFlowData = () => {
    if (!analysisResult?.flow_matrices || analysisResult.flow_matrices.length === 0) {
      return { labels: [], datasets: [] };
    }

    // Create labels for positions in the sequence
    const maxLength = Math.max(...analysisResult.flow_matrices.map(m => m.length));
    const labels = Array.from({ length: maxLength }, (_, i) => `Pos ${i}`);

    // Create datasets for each layer's flow
    const datasets = analysisResult.flow_matrices.map((matrix, layerIdx) => {
      // Calculate flow strength for each position (sum of incoming flows)
      const flowStrengths = matrix.map(row => 
        row.reduce((sum, val) => sum + Math.abs(val), 0)
      );

      return {
        label: `Layer ${layerIdx + 1}`,
        data: flowStrengths,
        borderColor: `hsl(${layerIdx * 60}, 70%, 50%)`,
        backgroundColor: `hsla(${layerIdx * 60}, 70%, 50%, 0.1)`,
        tension: 0.3,
        pointRadius: 4,
        pointHoverRadius: 6
      };
    });

    return { labels: labels.slice(0, datasets[0]?.data.length || 0), datasets };
  };

  const flowChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Information Flow Across Layers'
      },
      tooltip: {
        callbacks: {
          afterLabel: function(context: any) {
            return `Total Flow: ${context.parsed.y.toFixed(4)}`;
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Sequence Position'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Flow Strength'
        }
      }
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Information Flow Analysis</h2>
      </div>

      {/* Input Section */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold mb-4">Analyze Text</h3>
        <div className="space-y-4">
          <div>
            <label htmlFor="input-text" className="block text-sm font-medium text-gray-700 mb-2">
              Input Text
            </label>
            <textarea
              id="input-text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              rows={3}
              placeholder="Enter text to analyze information flow..."
            />
          </div>
          <button
            onClick={analyzeInformationFlow}
            disabled={loading || !inputText.trim()}
            className="px-6 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Analyzing...' : 'Analyze Information Flow'}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          <strong>Error:</strong> {error}
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          <span className="ml-3 text-gray-600">Analyzing information flow patterns...</span>
        </div>
      )}

      {analysisResult && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Flow Visualization */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <div style={{ height: '400px' }}>
              <Line data={generateFlowData()} options={flowChartOptions} />
            </div>
          </div>

          {/* Bottlenecks Analysis */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-lg font-semibold mb-4">Information Bottlenecks</h3>
            
            {analysisResult.bottlenecks.length > 0 ? (
              <div className="space-y-3">
                {analysisResult.bottlenecks.map((bottleneck, index) => (
                  <div key={index} className="border rounded p-3 bg-gray-50">
                    <div className="flex justify-between items-center">
                      <div>
                        <div className="font-medium">Layer {bottleneck.layer}</div>
                        <div className="text-sm text-gray-600">
                          Position: {bottleneck.position}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-semibold text-red-600">
                          {bottleneck.strength.toFixed(4)}
                        </div>
                        <div className="text-xs text-gray-500">Constraint</div>
                      </div>
                    </div>
                    <div className="mt-2">
                      <div className="bg-red-200 rounded-full h-2">
                        <div
                          className="bg-red-500 h-2 rounded-full"
                          style={{ width: `${Math.min(100, bottleneck.strength * 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-gray-500 text-center py-4">
                No significant bottlenecks detected
              </div>
            )}
          </div>
        </div>
      )}

      {/* Flow Summary */}
      {analysisResult && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold mb-4">Flow Analysis Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 p-4 rounded">
              <div className="text-2xl font-bold text-blue-600">
                {analysisResult.total_flow.toFixed(3)}
              </div>
              <div className="text-sm text-blue-700">Total Information Flow</div>
            </div>
            <div className="bg-red-50 p-4 rounded">
              <div className="text-2xl font-bold text-red-600">
                {analysisResult.bottlenecks.length}
              </div>
              <div className="text-sm text-red-700">Detected Bottlenecks</div>
            </div>
            <div className="bg-green-50 p-4 rounded">
              <div className="text-2xl font-bold text-green-600">
                {analysisResult.flow_matrices.length}
              </div>
              <div className="text-sm text-green-700">Analyzed Layers</div>
            </div>
          </div>
          
          <div className="mt-4 text-sm text-gray-600">
            <p><strong>Input:</strong> "{analysisResult.input_text}"</p>
            <p><strong>Analysis Time:</strong> {new Date(analysisResult.timestamp).toLocaleString()}</p>
          </div>

          <div className="mt-4 text-sm text-gray-600">
            <p><strong>Interpretation:</strong> Information flow analysis reveals how data moves through the model's layers. 
            Bottlenecks indicate positions where information processing is constrained, potentially affecting model performance.</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default InformationFlowVisualizer;
