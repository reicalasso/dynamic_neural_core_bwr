import React, { useEffect, useState } from 'react';

interface AttentionAnalysis {
  entropy_history: number[];
  concentration_history: number[];
  state_vs_attention_ratios: number[];
}

interface AttentionHeatmapProps {
  data?: number[][];
  showAnalysis?: boolean;
  className?: string;
}

const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({ 
  data, 
  showAnalysis = true, 
  className = "" 
}) => {
  const [analysisData, setAnalysisData] = useState<AttentionAnalysis | null>(null);
  const [generatedData, setGeneratedData] = useState<number[][]>([]);

  useEffect(() => {
    if (showAnalysis) {
      fetchAnalysisData();
    }
    if (!data) {
      generateDemoHeatmap();
    }
  }, [data, showAnalysis]);

  const fetchAnalysisData = async () => {
    try {
      const response = await fetch('http://localhost:8000/research/attention-analysis');
      if (response.ok) {
        const analysis = await response.json();
        setAnalysisData(analysis);
      }
    } catch (error) {
      console.warn('Failed to fetch attention analysis, generating demo data');
      generateDemoAnalysis();
    }
  };

  const generateDemoHeatmap = () => {
    // Generate a more realistic attention pattern
    const size = 16;
    const matrix: number[][] = [];
    
    for (let i = 0; i < size; i++) {
      const row: number[] = [];
      for (let j = 0; j < size; j++) {
        // Create attention patterns: diagonal emphasis, some local clusters
        let value = 0;
        
        // Diagonal attention (self-attention)
        if (i === j) value += 0.3;
        
        // Local attention (neighboring tokens)
        const distance = Math.abs(i - j);
        if (distance <= 2) value += 0.2 / (distance + 1);
        
        // Random attention + some global patterns
        value += Math.random() * 0.2;
        
        // Add some long-range dependencies occasionally
        if (Math.random() < 0.1) value += 0.3;
        
        row.push(Math.min(1.0, value));
      }
      matrix.push(row);
    }
    setGeneratedData(matrix);
  };

  const generateDemoAnalysis = () => {
    const history = Array.from({ length: 50 }, (_, i) => ({
      entropy: 2.5 + Math.sin(i / 10) * 0.5 + Math.random() * 0.3,
      concentration: 0.4 + Math.cos(i / 8) * 0.2 + Math.random() * 0.1,
      ratio: 0.5 + Math.sin(i / 15) * 0.3 + Math.random() * 0.1
    }));

    setAnalysisData({
      entropy_history: history.map(h => h.entropy),
      concentration_history: history.map(h => h.concentration),
      state_vs_attention_ratios: history.map(h => h.ratio)
    });
  };

  const heatmapData = data || generatedData;
  const maxValue = heatmapData.length > 0 ? Math.max(...heatmapData.flat()) : 1;
  const minValue = heatmapData.length > 0 ? Math.min(...heatmapData.flat()) : 0;
  
  const getColor = (value: number) => {
    const normalized = (value - minValue) / (maxValue - minValue);
    const hue = (1 - normalized) * 240; // Blue to red
    const saturation = 70 + normalized * 30; // 70-100%
    const lightness = 20 + normalized * 60; // 20-80%
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  };

  const getContributionColor = (ratio: number) => {
    // ratio: 0 = all attention, 1 = all state
    if (ratio < 0.4) return 'text-blue-400'; // Attention dominant
    if (ratio > 0.6) return 'text-green-400'; // State dominant
    return 'text-purple-400'; // Balanced
  };

  const getContributionLabel = (ratio: number) => {
    if (ratio < 0.4) return 'Attention Dominant';
    if (ratio > 0.6) return 'State Dominant';
    return 'Balanced';
  };

  return (
    <div className={`bg-gray-800 p-6 rounded-lg space-y-6 ${className}`}>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <h2 className="text-2xl font-bold text-cyan-400">🎯 Attention Analysis</h2>
        <div className="text-sm text-gray-400">
          {data ? 'Live Data' : 'Demo Data'}
        </div>
      </div>

      {/* Attention vs State Contribution Summary */}
      {analysisData && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-700 p-4 rounded-lg">
            <div className="text-gray-300 text-sm">Current Entropy</div>
            <div className="text-xl font-bold text-blue-400">
              {(analysisData.entropy_history[analysisData.entropy_history.length - 1] || 0).toFixed(2)}
            </div>
            <div className="text-xs text-gray-400">Information spread</div>
          </div>
          <div className="bg-gray-700 p-4 rounded-lg">
            <div className="text-gray-300 text-sm">Attention Focus</div>
            <div className="text-xl font-bold text-yellow-400">
              {((analysisData.concentration_history[analysisData.concentration_history.length - 1] || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-400">Concentration level</div>
          </div>
          <div className="bg-gray-700 p-4 rounded-lg">
            <div className="text-gray-300 text-sm">Decision Mode</div>
            <div className={`text-xl font-bold ${getContributionColor(analysisData.state_vs_attention_ratios[analysisData.state_vs_attention_ratios.length - 1] || 0.5)}`}>
              {getContributionLabel(analysisData.state_vs_attention_ratios[analysisData.state_vs_attention_ratios.length - 1] || 0.5)}
            </div>
            <div className="text-xs text-gray-400">State vs Attention</div>
          </div>
        </div>
      )}

      {/* Attention Heatmap */}
      <div className="bg-gray-900 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-cyan-400 mb-4">Token-Token Attention Matrix</h3>
        {heatmapData.length > 0 ? (
          <div className="overflow-auto">
            <div 
              className="grid gap-1 w-fit mx-auto"
              style={{ 
                gridTemplateColumns: `repeat(${heatmapData[0].length}, 16px)`,
                gridTemplateRows: `repeat(${heatmapData.length}, 16px)`
              }}
            >
              {heatmapData.map((row, i) =>
                row.map((value, j) => (
                  <div
                    key={`${i}-${j}`}
                    className="w-4 h-4 border border-gray-700 rounded-sm relative group cursor-pointer transition-transform hover:scale-150 hover:z-10"
                    style={{ backgroundColor: getColor(value) }}
                    title={`Token ${i} → Token ${j}: ${value.toFixed(3)}`}
                  >
                    {/* Hover tooltip */}
                    <div className="invisible group-hover:visible absolute z-20 bg-black text-white text-xs rounded p-2 -top-12 left-1/2 transform -translate-x-1/2 whitespace-nowrap border border-gray-600">
                      <div>From: Token {i}</div>
                      <div>To: Token {j}</div>
                      <div>Weight: {value.toFixed(3)}</div>
                    </div>
                  </div>
                ))
              )}
            </div>
            
            {/* Enhanced Legend */}
            <div className="mt-4 space-y-2">
              <div className="flex items-center justify-center space-x-6">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: getColor(minValue) }}></div>
                  <span className="text-sm text-gray-400">Low ({minValue.toFixed(3)})</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: getColor(maxValue) }}></div>
                  <span className="text-sm text-gray-400">High ({maxValue.toFixed(3)})</span>
                </div>
              </div>
              <div className="text-center text-xs text-gray-500">
                Rows = From tokens, Columns = To tokens. Hover for details.
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center text-gray-400 py-8">
            No attention data available
          </div>
        )}
      </div>

      {/* Time Series Analysis */}
      {analysisData && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Entropy History */}
          <div className="bg-gray-900 p-4 rounded-lg">
            <h4 className="text-md font-semibold text-cyan-400 mb-3">Attention Entropy</h4>
            <div className="h-24 flex items-end gap-1">
              {analysisData.entropy_history.slice(-20).map((entropy, i) => (
                <div
                  key={i}
                  className="flex-1 bg-blue-500 rounded-sm opacity-70 hover:opacity-100 transition-opacity"
                  style={{ height: `${(entropy / 4) * 100}%` }}
                  title={`Entropy: ${entropy.toFixed(2)}`}
                />
              ))}
            </div>
            <div className="text-xs text-gray-400 mt-2">Higher = more spread out</div>
          </div>

          {/* Concentration History */}
          <div className="bg-gray-900 p-4 rounded-lg">
            <h4 className="text-md font-semibold text-cyan-400 mb-3">Attention Focus</h4>
            <div className="h-24 flex items-end gap-1">
              {analysisData.concentration_history.slice(-20).map((concentration, i) => (
                <div
                  key={i}
                  className="flex-1 bg-yellow-500 rounded-sm opacity-70 hover:opacity-100 transition-opacity"
                  style={{ height: `${concentration * 100}%` }}
                  title={`Focus: ${(concentration * 100).toFixed(1)}%`}
                />
              ))}
            </div>
            <div className="text-xs text-gray-400 mt-2">Higher = more focused</div>
          </div>

          {/* State vs Attention Ratio */}
          <div className="bg-gray-900 p-4 rounded-lg">
            <h4 className="text-md font-semibold text-cyan-400 mb-3">Decision Balance</h4>
            <div className="h-24 flex items-end gap-1">
              {analysisData.state_vs_attention_ratios.slice(-20).map((ratio, i) => (
                <div
                  key={i}
                  className={`flex-1 rounded-sm opacity-70 hover:opacity-100 transition-opacity ${
                    ratio < 0.4 ? 'bg-blue-500' : ratio > 0.6 ? 'bg-green-500' : 'bg-purple-500'
                  }`}
                  style={{ height: `${ratio * 100}%` }}
                  title={`Ratio: ${(ratio * 100).toFixed(1)}% (${getContributionLabel(ratio)})`}
                />
              ))}
            </div>
            <div className="text-xs text-gray-400 mt-2">
              <span className="text-blue-400">Blue:</span> Attention, 
              <span className="text-green-400"> Green:</span> State, 
              <span className="text-purple-400"> Purple:</span> Balanced
            </div>
          </div>
        </div>
      )}

      {/* Pattern Analysis */}
      <div className="bg-gray-900 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-cyan-400 mb-4">Pattern Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h4 className="text-gray-300 font-semibold mb-2">Attention Characteristics:</h4>
            <ul className="space-y-1 text-gray-400">
              <li>• Diagonal strength indicates self-attention</li>
              <li>• Nearby tokens show local dependencies</li>
              <li>• Sparse patterns suggest efficient focus</li>
              <li>• Dense rows indicate information hubs</li>
            </ul>
          </div>
          <div>
            <h4 className="text-gray-300 font-semibold mb-2">NSM vs Transformer:</h4>
            <ul className="space-y-1 text-gray-400">
              <li>• State memory reduces attention computation</li>
              <li>• Hierarchical states capture long-range deps</li>
              <li>• Balanced ratio indicates hybrid efficiency</li>
              <li>• Lower entropy suggests more structured focus</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AttentionHeatmap;
