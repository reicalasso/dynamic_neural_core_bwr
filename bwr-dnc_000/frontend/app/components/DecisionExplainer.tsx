import React, { useState } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

interface DecisionStep {
  layer: number;
  operation: string;
  importance: number;
}

interface ExplanationResult {
  importance_scores: number[];
  state_contributions: number[];
  attention_contributions: number[];
  decision_path: DecisionStep[];
  confidence: number;
  input_text: string;
  target_token_idx: number;
  timestamp: string;
}

const DecisionExplainer: React.FC = () => {
  const [explanationResult, setExplanationResult] = useState<ExplanationResult | null>(null);
  const [inputText, setInputText] = useState('The cat sat on the mat');
  const [targetTokenIdx, setTargetTokenIdx] = useState(-1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const explainDecision = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/research/explain-decision', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: inputText,
          target_token_idx: targetTokenIdx
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setExplanationResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      console.error('Failed to explain decision:', err);
    } finally {
      setLoading(false);
    }
  };

  const generateImportanceData = () => {
    if (!explanationResult?.importance_scores) {
      return { labels: [], datasets: [] };
    }

    const labels = explanationResult.importance_scores.map((_, idx) => `Token ${idx}`);
    
    return {
      labels,
      datasets: [{
        label: 'Token Importance',
        data: explanationResult.importance_scores,
        backgroundColor: explanationResult.importance_scores.map(score => 
          score > 0 ? 'rgba(34, 197, 94, 0.8)' : 'rgba(239, 68, 68, 0.8)'
        ),
        borderColor: explanationResult.importance_scores.map(score => 
          score > 0 ? 'rgba(34, 197, 94, 1)' : 'rgba(239, 68, 68, 1)'
        ),
        borderWidth: 1
      }]
    };
  };

  const generateContributionData = () => {
    if (!explanationResult?.state_contributions || !explanationResult?.attention_contributions) {
      return { labels: [], datasets: [] };
    }

    const labels = explanationResult.state_contributions.map((_, idx) => `Layer ${idx + 1}`);
    
    return {
      labels,
      datasets: [
        {
          label: 'State Contributions',
          data: explanationResult.state_contributions,
          backgroundColor: 'rgba(59, 130, 246, 0.8)',
          borderColor: 'rgba(59, 130, 246, 1)',
          borderWidth: 1
        },
        {
          label: 'Attention Contributions',
          data: explanationResult.attention_contributions,
          backgroundColor: 'rgba(168, 85, 247, 0.8)',
          borderColor: 'rgba(168, 85, 247, 1)',
          borderWidth: 1
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      tooltip: {
        callbacks: {
          afterLabel: function(context: any) {
            return `Value: ${context.parsed.y.toFixed(4)}`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Contribution Score'
        }
      }
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Decision Explainer</h2>
      </div>

      {/* Input Section */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold mb-4">Explain Model Decision</h3>
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
              placeholder="Enter text to explain the model's decision..."
            />
          </div>
          <div>
            <label htmlFor="target-token" className="block text-sm font-medium text-gray-700 mb-2">
              Target Token Index (-1 for last token)
            </label>
            <input
              id="target-token"
              type="number"
              value={targetTokenIdx}
              onChange={(e) => setTargetTokenIdx(parseInt(e.target.value) || -1)}
              className="w-32 p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="-1"
            />
          </div>
          <button
            onClick={explainDecision}
            disabled={loading || !inputText.trim()}
            className="px-6 py-2 bg-purple-500 text-white rounded-md hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Explaining...' : 'Explain Decision'}
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
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>
          <span className="ml-3 text-gray-600">Generating explanation...</span>
        </div>
      )}

      {explanationResult && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Token Importance */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-lg font-semibold mb-4">Token Importance Scores</h3>
            <div style={{ height: '300px' }}>
              <Bar 
                data={generateImportanceData()} 
                options={{
                  ...chartOptions,
                  plugins: {
                    ...chartOptions.plugins,
                    title: {
                      display: true,
                      text: 'Individual Token Contributions'
                    }
                  }
                }} 
              />
            </div>
          </div>

          {/* Layer Contributions */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-lg font-semibold mb-4">Layer-wise Contributions</h3>
            <div style={{ height: '300px' }}>
              <Bar 
                data={generateContributionData()} 
                options={{
                  ...chartOptions,
                  plugins: {
                    ...chartOptions.plugins,
                    title: {
                      display: true,
                      text: 'State vs Attention Contributions by Layer'
                    }
                  }
                }} 
              />
            </div>
          </div>
        </div>
      )}

      {/* Decision Path */}
      {explanationResult && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold mb-4">Decision Path</h3>
          {explanationResult.decision_path.length > 0 ? (
            <div className="space-y-3">
              {explanationResult.decision_path.map((step, index) => (
                <div key={index} className="border rounded p-3 bg-gray-50">
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="font-medium">Layer {step.layer}</div>
                      <div className="text-sm text-gray-600">{step.operation}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-semibold text-blue-600">
                        {step.importance.toFixed(4)}
                      </div>
                      <div className="text-xs text-gray-500">Importance</div>
                    </div>
                  </div>
                  <div className="mt-2">
                    <div className="bg-blue-200 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full"
                        style={{ width: `${Math.min(100, Math.abs(step.importance) * 100)}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-gray-500 text-center py-4">
              No decision path data available
            </div>
          )}
        </div>
      )}

      {/* Explanation Summary */}
      {explanationResult && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold mb-4">Explanation Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-green-50 p-4 rounded">
              <div className="text-2xl font-bold text-green-600">
                {(explanationResult.confidence * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-green-700">Confidence</div>
            </div>
            <div className="bg-blue-50 p-4 rounded">
              <div className="text-2xl font-bold text-blue-600">
                {explanationResult.importance_scores.length}
              </div>
              <div className="text-sm text-blue-700">Analyzed Tokens</div>
            </div>
            <div className="bg-purple-50 p-4 rounded">
              <div className="text-2xl font-bold text-purple-600">
                {explanationResult.decision_path.length}
              </div>
              <div className="text-sm text-purple-700">Decision Steps</div>
            </div>
            <div className="bg-orange-50 p-4 rounded">
              <div className="text-2xl font-bold text-orange-600">
                {explanationResult.target_token_idx === -1 ? 'Last' : explanationResult.target_token_idx}
              </div>
              <div className="text-sm text-orange-700">Target Token</div>
            </div>
          </div>
          
          <div className="mt-4 text-sm text-gray-600">
            <p><strong>Input:</strong> "{explanationResult.input_text}"</p>
            <p><strong>Analysis Time:</strong> {new Date(explanationResult.timestamp).toLocaleString()}</p>
          </div>

          <div className="mt-4 text-sm text-gray-600">
            <p><strong>Interpretation:</strong> This explanation shows how different parts of the input contribute to the model's decision. 
            Higher importance scores indicate tokens that had more influence on the final output. The decision path traces the key computational steps.</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default DecisionExplainer;
