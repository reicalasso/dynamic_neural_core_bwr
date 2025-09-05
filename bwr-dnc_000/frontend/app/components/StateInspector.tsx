import React, { useEffect, useState, useCallback } from 'react';

interface StateLevel {
  level: number;
  slots: number;
  active_slots: number;
  avg_salience: number;
  avg_age: number;
  avg_access: number;
}

interface StateEvolutionData {
  state_changes: number[];
  stability_scores: number[];
  summary: {
    avg_state_change: number;
    avg_stability: number;
    change_trend: string;
    stability_trend: string;
  };
}

interface StateInspectorProps {
  className?: string;
}

const StateInspector: React.FC<StateInspectorProps> = ({ className = "" }) => {
  const [stateData, setStateData] = useState<StateLevel[]>([]);
  const [evolutionData, setEvolutionData] = useState<StateEvolutionData | null>(null);
  const [selectedLevel, setSelectedLevel] = useState<number>(0);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [inspectionText, setInspectionText] = useState("Hello world, this is a test text for DNC state inspection.");

  const fetchStateData = useCallback(async () => {
    try {
      // Fetch current memory state
      const memoryResponse = await fetch('http://localhost:8000/research/metrics');
      if (memoryResponse.ok) {
        const memoryData = await memoryResponse.json();
        setStateData(memoryData.memory.levels || []);
      }

      // Fetch state evolution data
      const evolutionResponse = await fetch('http://localhost:8000/research/state-evolution');
      if (evolutionResponse.ok) {
        const evolution = await evolutionResponse.json();
        setEvolutionData(evolution);
      }
    } catch (error) {
      console.warn('Failed to fetch state data, generating demo data');
      generateDemoData();
    }
  }, []);

  const generateDemoData = () => {
    // Generate realistic state levels data
    const levels: StateLevel[] = [];
    for (let i = 0; i < 4; i++) {
      const slots = Math.floor(2048 / Math.pow(2, i)); // Hierarchical reduction
      const activeSlots = Math.floor(slots * (0.3 + Math.random() * 0.4));
      levels.push({
        level: i,
        slots,
        active_slots: activeSlots,
        avg_salience: 0.2 + Math.random() * 0.6,
        avg_age: Math.random() * 100,
        avg_access: Math.random() * 50
      });
    }
    setStateData(levels);

    // Generate evolution data
    const changeData = Array.from({ length: 50 }, () => Math.random() * 2);
    const stabilityData = Array.from({ length: 50 }, () => 0.7 + Math.random() * 0.3);
    
    setEvolutionData({
      state_changes: changeData,
      stability_scores: stabilityData,
      summary: {
        avg_state_change: changeData.reduce((a, b) => a + b, 0) / changeData.length,
        avg_stability: stabilityData.reduce((a, b) => a + b, 0) / stabilityData.length,
        change_trend: "stable",
        stability_trend: "increasing"
      }
    });
  };

  const analyzeText = async () => {
    try {
      const response = await fetch('http://localhost:8000/research/analyze-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inspectionText })
      });
      
      if (response.ok) {
        const analysis = await response.json();
        console.log('Text analysis result:', analysis);
        // Refresh state data after analysis
        fetchStateData();
      }
    } catch (error) {
      console.warn('Text analysis failed:', error);
    }
  };

  useEffect(() => {
    fetchStateData();
  }, [fetchStateData]);

  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(fetchStateData, 3000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchStateData]);

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'increasing': return '📈';
      case 'decreasing': return '📉';
      default: return '➡️';
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'increasing': return 'text-green-400';
      case 'decreasing': return 'text-red-400';
      default: return 'text-blue-400';
    }
  };

  const selectedLevelData = stateData[selectedLevel];

  return (
    <div className={`bg-gray-800 p-6 rounded-lg space-y-6 ${className}`}>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <h2 className="text-2xl font-bold text-cyan-400">🧠 State Inspector</h2>
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
            onClick={fetchStateData}
            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* State Evolution Summary */}
      {evolutionData && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-700 p-4 rounded-lg">
            <div className="text-gray-300 text-sm">Avg State Change</div>
            <div className="text-xl font-bold text-blue-400">
              {evolutionData.summary.avg_state_change.toFixed(3)}
            </div>
            <div className={`text-sm ${getTrendColor(evolutionData.summary.change_trend)}`}>
              {getTrendIcon(evolutionData.summary.change_trend)} {evolutionData.summary.change_trend}
            </div>
          </div>
          <div className="bg-gray-700 p-4 rounded-lg">
            <div className="text-gray-300 text-sm">Avg Stability</div>
            <div className="text-xl font-bold text-green-400">
              {(evolutionData.summary.avg_stability * 100).toFixed(1)}%
            </div>
            <div className={`text-sm ${getTrendColor(evolutionData.summary.stability_trend)}`}>
              {getTrendIcon(evolutionData.summary.stability_trend)} {evolutionData.summary.stability_trend}
            </div>
          </div>
          <div className="bg-gray-700 p-4 rounded-lg">
            <div className="text-gray-300 text-sm">Change Variance</div>
            <div className="text-xl font-bold text-purple-400">
              {evolutionData.state_changes.length > 0 ? 
                (evolutionData.state_changes.reduce((acc, val, _, arr) => {
                  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
                  return acc + Math.pow(val - mean, 2);
                }, 0) / evolutionData.state_changes.length).toFixed(3) : '0.000'
              }
            </div>
          </div>
          <div className="bg-gray-700 p-4 rounded-lg">
            <div className="text-gray-300 text-sm">Total Levels</div>
            <div className="text-xl font-bold text-cyan-400">
              {stateData.length}
            </div>
          </div>
        </div>
      )}

      {/* Level Selection */}
      <div className="bg-gray-900 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-cyan-400 mb-4">Memory Levels</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {stateData.map((level, index) => (
            <button
              key={level.level}
              onClick={() => setSelectedLevel(index)}
              className={`p-3 rounded-lg border transition-all ${
                selectedLevel === index
                  ? 'border-cyan-400 bg-cyan-400 bg-opacity-20'
                  : 'border-gray-600 bg-gray-700 hover:border-gray-500'
              }`}
            >
              <div className="text-sm font-semibold">Level {level.level}</div>
              <div className="text-xs text-gray-400">{level.slots} slots</div>
              <div className="text-xs text-green-400">{level.active_slots} active</div>
            </button>
          ))}
        </div>
      </div>

      {/* Selected Level Details */}
      {selectedLevelData && (
        <div className="bg-gray-900 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-cyan-400 mb-4">
            Level {selectedLevelData.level} Details
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Slot Utilization */}
            <div>
              <h4 className="text-md font-semibold text-gray-300 mb-3">Slot Utilization</h4>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm text-gray-400 mb-1">
                    <span>Active Slots</span>
                    <span>{selectedLevelData.active_slots} / {selectedLevelData.slots}</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div 
                      className="bg-green-500 h-3 rounded-full transition-all duration-300"
                      style={{ width: `${(selectedLevelData.active_slots / selectedLevelData.slots) * 100}%` }}
                    />
                  </div>
                  <div className="text-right text-xs text-gray-500 mt-1">
                    {((selectedLevelData.active_slots / selectedLevelData.slots) * 100).toFixed(1)}% utilized
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-sm text-gray-400 mb-1">
                    <span>Avg Salience</span>
                    <span>{selectedLevelData.avg_salience.toFixed(3)}</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-yellow-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${selectedLevelData.avg_salience * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* State Statistics */}
            <div>
              <h4 className="text-md font-semibold text-gray-300 mb-3">State Statistics</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Average Age:</span>
                  <span className="font-mono text-blue-400">{selectedLevelData.avg_age.toFixed(1)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Average Access:</span>
                  <span className="font-mono text-purple-400">{selectedLevelData.avg_access.toFixed(1)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Efficiency:</span>
                  <span className="font-mono text-green-400">
                    {((selectedLevelData.active_slots / selectedLevelData.slots) * selectedLevelData.avg_salience * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Memory Pressure:</span>
                  <span className={`font-mono ${
                    (selectedLevelData.active_slots / selectedLevelData.slots) > 0.8 ? 'text-red-400' :
                    (selectedLevelData.active_slots / selectedLevelData.slots) > 0.6 ? 'text-yellow-400' : 'text-green-400'
                  }`}>
                    {(selectedLevelData.active_slots / selectedLevelData.slots) > 0.8 ? 'High' :
                     (selectedLevelData.active_slots / selectedLevelData.slots) > 0.6 ? 'Medium' : 'Low'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* State Evolution Visualization */}
      {evolutionData && (
        <div className="bg-gray-900 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-cyan-400 mb-4">State Evolution Timeline</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* State Changes */}
            <div>
              <h4 className="text-md font-semibold text-gray-300 mb-3">State Changes (L2 Norm)</h4>
              <div className="h-32 flex items-end gap-1">
                {evolutionData.state_changes.slice(-30).map((change, i) => (
                  <div
                    key={i}
                    className="flex-1 bg-blue-500 rounded-sm opacity-70 hover:opacity-100 transition-opacity"
                    style={{ height: `${Math.min(100, (change / 4) * 100)}%` }}
                    title={`Change: ${change.toFixed(3)}`}
                  />
                ))}
              </div>
              <div className="text-xs text-gray-400 mt-2">Last 30 steps</div>
            </div>
            
            {/* Stability Scores */}
            <div>
              <h4 className="text-md font-semibold text-gray-300 mb-3">Stability Scores</h4>
              <div className="h-32 flex items-end gap-1">
                {evolutionData.stability_scores.slice(-30).map((stability, i) => (
                  <div
                    key={i}
                    className="flex-1 bg-green-500 rounded-sm opacity-70 hover:opacity-100 transition-opacity"
                    style={{ height: `${stability * 100}%` }}
                    title={`Stability: ${(stability * 100).toFixed(1)}%`}
                  />
                ))}
              </div>
              <div className="text-xs text-gray-400 mt-2">Last 30 steps</div>
            </div>
          </div>
        </div>
      )}

      {/* Interactive Text Inspection */}
      <div className="bg-gray-900 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-cyan-400 mb-4">🔍 Interactive Text Inspection</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-300 mb-2">
              Enter text to analyze state evolution:
            </label>
            <textarea
              value={inspectionText}
              onChange={(e) => setInspectionText(e.target.value)}
              rows={3}
              className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-cyan-400 focus:outline-none"
              placeholder="Type your text here to see how DNC states evolve..."
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={analyzeText}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
            >
              🔬 Analyze Text
            </button>
            <button
              onClick={() => setInspectionText("The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.")}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition-colors text-sm"
            >
              Load Sample
            </button>
          </div>
          <div className="text-xs text-gray-400">
            Analyze how text input affects DNC state evolution patterns and memory utilization across hierarchical levels.
          </div>
        </div>
      </div>
    </div>
  );
};

export default StateInspector;
