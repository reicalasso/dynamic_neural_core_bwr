import React, { useState, useEffect, useRef } from 'react';
import SlotCard from './SlotCard';
import AttentionHeatmap from './AttentionHeatmap';
import MemoryLevelChart from './MemoryLevelChart';
import TrainingMetrics from './TrainingMetrics';

// Enhanced mock data for demonstration
const generateMockSlots = (count: number = 20) => {
  return Array.from({ length: count }, (_, i) => ({
    id: i,
    salience: Math.random(),
    age: Math.floor(Math.random() * 100),
    access_count: Math.floor(Math.random() * 50),
    content: `Slot ${i}: ${['memory', 'pattern', 'association', 'context', 'knowledge'][Math.floor(Math.random() * 5)]}`,
    level: Math.floor(Math.random() * 3), // 0, 1, or 2 for memory levels
    compression_ratio: [1, 2, 4][Math.floor(Math.random() * 3)]
  }));
};

const generateMockAttentionData = () => {
  const size = 20;
  return Array.from({ length: size }, (_, i) => 
    Array.from({ length: size }, (_, j) => Math.random() * 0.8 + 0.1)
  );
};

const StateVisualizer: React.FC = () => {
  const [isClient, setIsClient] = useState(false);
  const [slots, setSlots] = useState<any[]>([]);
  const [tokenStream, setTokenStream] = useState<string[]>([]);
  const [compactionEvents, setCompactionEvents] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'attention' | 'memory' | 'metrics'>('overview');
  const [attentionData, setAttentionData] = useState<number[][]>([]);
  const [isLiveMode, setIsLiveMode] = useState(true);
  const [selectedLevel, setSelectedLevel] = useState<number>(0);
  const wsRef = useRef<WebSocket | null>(null);

  // Initialize client-side only to prevent hydration errors
  useEffect(() => {
    setIsClient(true);
    setSlots(generateMockSlots());
    setAttentionData(generateMockAttentionData());
  }, []);

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (!isClient || !isLiveMode) return;
    
    // Mock real-time updates for demo
    const interval = setInterval(() => {
      // Simulate new token
      const newToken = `tk_${Math.random().toString(36).substring(2, 7)}`;
      setTokenStream(prev => [...prev, newToken].slice(-30));

      // Simulate compaction events
      if (Math.random() < 0.15) {
        const event = `Level ${Math.floor(Math.random() * 3)} compaction - ${new Date().toLocaleTimeString()}`;
        setCompactionEvents(prev => [...prev, event].slice(-8));
      }
      
      // Update slot states
      setSlots(prevSlots => prevSlots.map(s => ({
        ...s, 
        salience: Math.max(0, s.salience - 0.02 + (Math.random() * 0.08)),
        access_count: s.access_count + (Math.random() < 0.3 ? 1 : 0)
      })));

      // Update attention heatmap
      if (Math.random() < 0.4) {
        setAttentionData(generateMockAttentionData());
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [isClient, isLiveMode]);

  // Don't render until client-side hydration is complete
  if (!isClient) {
    return (
      <div className="bg-gray-900 text-white min-h-screen p-4 flex items-center justify-center">
        <div className="text-xl text-cyan-400">Loading BWR-NSM Visualizer...</div>
      </div>
    );
  }

  const filteredSlots = slots.filter(slot => slot.level === selectedLevel);
  const sortedSlots = filteredSlots.sort((a, b) => b.salience - a.salience);

  const memoryStats = {
    totalSlots: slots.length,
    activeSlots: slots.filter(s => s.salience > 0.1).length,
    compressionRatio: slots.reduce((acc, s) => acc + s.compression_ratio, 0) / slots.length,
    averageAge: slots.reduce((acc, s) => acc + s.age, 0) / slots.length
  };

  return (
    <div className="space-y-6">
      {/* Header with controls */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-cyan-400">Neural State Monitor</h2>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setIsLiveMode(!isLiveMode)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              isLiveMode 
                ? 'bg-green-600 hover:bg-green-700 text-white' 
                : 'bg-gray-600 hover:bg-gray-700 text-gray-300'
            }`}
          >
            {isLiveMode ? '🟢 Live' : '⏸️ Paused'}
          </button>
          
          <select 
            value={selectedLevel}
            onChange={(e) => setSelectedLevel(Number(e.target.value))}
            className="bg-gray-700 text-white px-3 py-2 rounded-lg border border-gray-600"
          >
            <option value={0}>Memory Level 0 (1x)</option>
            <option value={1}>Memory Level 1 (2x)</option>
            <option value={2}>Memory Level 2 (4x)</option>
          </select>
        </div>
      </div>

      {/* Stats overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="text-sm text-gray-400">Total Slots</div>
          <div className="text-2xl font-bold text-cyan-400">{memoryStats.totalSlots}</div>
        </div>
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="text-sm text-gray-400">Active Slots</div>
          <div className="text-2xl font-bold text-green-400">{memoryStats.activeSlots}</div>
        </div>
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="text-sm text-gray-400">Avg Compression</div>
          <div className="text-2xl font-bold text-purple-400">{memoryStats.compressionRatio.toFixed(1)}x</div>
        </div>
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="text-sm text-gray-400">Avg Age</div>
          <div className="text-2xl font-bold text-yellow-400">{memoryStats.averageAge.toFixed(0)}</div>
        </div>
      </div>

      {/* Tab navigation */}
      <div className="flex space-x-1 bg-gray-800 p-1 rounded-lg">
        {(['overview', 'attention', 'memory', 'metrics'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-md font-medium transition-colors capitalize ${
              activeTab === tab
                ? 'bg-cyan-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Token stream */}
          <div className="bg-gray-800 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-cyan-400 mb-3">Token Stream</h3>
            <div className="flex flex-wrap gap-2">
              {tokenStream.slice(-15).map((token, idx) => (
                <span 
                  key={idx}
                  className="bg-blue-900 text-blue-200 px-2 py-1 rounded text-sm font-mono"
                >
                  {token}
                </span>
              ))}
            </div>
          </div>

          {/* Compaction events */}
          <div className="bg-gray-800 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-cyan-400 mb-3">Recent Compaction Events</h3>
            <div className="space-y-1">
              {compactionEvents.slice(-5).map((event, idx) => (
                <div key={idx} className="text-sm text-gray-300 font-mono">
                  {event}
                </div>
              ))}
            </div>
          </div>

          {/* Memory slots preview */}
          <div className="bg-gray-800 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-cyan-400 mb-3">
              Memory Level {selectedLevel} ({filteredSlots.length} slots)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {sortedSlots.slice(0, 6).map((slot) => (
                <SlotCard key={slot.id} {...slot} />
              ))}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'attention' && (
        <div className="bg-gray-800 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-cyan-400 mb-3">Attention Heatmap</h3>
          <AttentionHeatmap data={attentionData} />
        </div>
      )}

      {activeTab === 'memory' && (
        <div className="space-y-6">
          <MemoryLevelChart 
            level0Count={slots.filter(s => s.level === 0).length}
            level1Count={slots.filter(s => s.level === 1).length}
            level2Count={slots.filter(s => s.level === 2).length}
          />
          
          <div className="bg-gray-800 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-cyan-400 mb-3">
              All Slots - Level {selectedLevel}
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 max-h-96 overflow-y-auto">
              {sortedSlots.map((slot) => (
                <SlotCard key={slot.id} {...slot} />
              ))}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'metrics' && (
        <TrainingMetrics />
      )}
    </div>
  );
};

export default StateVisualizer;